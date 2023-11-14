import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from typing import Any, Optional, Dict, List, Tuple, Union, Sequence, Callable
from ..util.schedule import Config, conf2scheduler, conf2optimizer
from ..util.util import asnumpy, get_tiling_slices, fetch_model, load_model
from torch import optim, Tensor, nn
from collections import OrderedDict, ChainMap
from ..data.instance_eval import LabelMatcher, LabelMatcherList
from ..data.cpn import contours2labels
from ..data.misc import channels_first2channels_last
from ..ops.cpn import remove_border_contours
from . import cpn
from ..visualization.images import show_detection, imshow_row
from torch.distributed import is_available, all_gather_object, get_world_size, is_initialized, get_rank
from itertools import chain
import numpy as np
from os.path import isfile
from torchvision.ops.boxes import remove_small_boxes, nms
from torch.optim.lr_scheduler import SequentialLR
from ..optim.lr_scheduler import WarmUp
from ..util.util import GpuStats, to_device, dict2model, model2dict

STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]

GPU_STATS = GpuStats() if torch.cuda.is_available() else None


class LitCpn(pl.LightningModule):
    def __init__(
            self,
            model: Union[str, nn.Module],
            losses_prog_bar=True,
            optimizer=None,
            scheduler=None,
            scheduler_conf=None,
            warmup_steps=512,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(model, str):
            model = self.build_model(model, **kwargs)
            if kwargs.pop('resolve_model_hparam', True):
                self.hparams['model'] = model2dict(model)
        elif isinstance(model, dict):
            model = dict2model(model)
        self.model = model
        self.inputs_key = 'inputs'
        self.targets_key = 'targets'
        self.val_iou_thresholds = kwargs.get('val_iou_thresholds', (.5, .6, .7, .8, .9))
        self.test_iou_thresholds = kwargs.get('test_iou_thresholds', (.5, .6, .7, .8, .9))
        self.losses_prog_bar = losses_prog_bar
        self.warmup_steps = warmup_steps
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_conf = scheduler_conf
        self._lr_multiplier = kwargs.get('lr_multiplier')
        self._weight_decay_multiplier = kwargs.get('weight_decay_multiplier')
        self.figsize = kwargs.get('figsize', (32, 18))
        self._log_figures = kwargs.get('log_figures', True)
        self._eval_zero_division = kwargs.get('eval_zero_division', 0.)
        self._validation_outputs = None
        self._test_outputs = None
        self._predict_outputs = None
        self._val_mean_keys = (
            'f1',

            'f1_np',
            'jaccard_np',
            'fowlkes_mallows_np',
            'recall',
            'precision',

            'avg_recall',
            'avg_precision',
            'avg_f1',
            'avg_jaccard',
            'avg_fowlkes_mallows',
        )
        self._val_sum_keys = ('true_positives', 'false_negatives', 'false_positives')
        self._test_mean_keys = tuple(self._val_mean_keys)
        self._test_sum_keys = tuple(self._val_sum_keys)
        self.max_imsize = kwargs.get('max_imsize', 2048)

    @staticmethod
    def build_model(model: str, *args, **kwargs):
        if model in dir(cpn):
            assert 'cpn' in model.lower()
            return getattr(cpn, model)(*args, **kwargs)
        elif isfile(model):
            return load_model(model, **kwargs)
        else:
            return fetch_model(model, **kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        inputs = batch[self.inputs_key]
        if batch_idx == 0:
            logger = None if self.logger is None else self.logger.experiment
            if logger is not None:
                logger.add_images('inputs', inputs, self.global_step)
        outputs: dict = self.model(inputs, targets=batch)
        log_d = {} if GPU_STATS is None else GPU_STATS.dict(prefix='gpus/gpu')
        losses = outputs.get('losses')
        losses['loss'] = losses.get('loss', outputs['loss'])
        if losses is not None and isinstance(losses, dict):
            log_d.update({f'losses/{k}': v for k, v in losses.items() if v is not None})
            self.log_dict(log_d, prog_bar=self.losses_prog_bar, logger=True, on_step=True)
        return outputs

    def training_step_end(self, training_step_outputs):
        return training_step_outputs['loss']

    def configure_optimizers(self):
        if isinstance(self._optimizer, (dict, str, type(None))):
            optimizer = dict(AdamW=dict()) if self._optimizer is None else self._optimizer
            optimizer = conf2optimizer(optimizer, filter(lambda p: p.requires_grad, self.parameters()))
        else:
            optimizer = self._optimizer

        if self._lr_multiplier is not None:
            for gr in optimizer.param_groups:
                print('Update learning rate from', gr['lr'], end='')
                gr['lr'] *= self._lr_multiplier
                print('to', gr['lr'], flush=True)
        if self._weight_decay_multiplier is not None:
            for gr in optimizer.param_groups:
                if 'weight_decay' in gr:
                    print('Update learning rate from', gr['weight_decay'], end='')
                    gr['weight_decay'] *= self._weight_decay_multiplier
                    print('to', gr['weight_decay'], flush=True)

        if self._scheduler is None:
            return optimizer

        if isinstance(self._scheduler, (dict, str)):
            scheduler = conf2scheduler(self._scheduler, optimizer)
        else:
            scheduler = self._scheduler

        if self.warmup_steps and scheduler is not None:
            scheduler = SequentialLR(optimizer, [WarmUp(optimizer, self.warmup_steps), scheduler],
                                     milestones=[self.warmup_steps])

        scheduler = {
            **dict(
                interval='step',
                frequency=1,
                scheduler=scheduler,
                strict=True,
                name=None,
            ),
            **self._scheduler_conf
        }

        return [optimizer], [scheduler]

    def log_figures(self, tag, inputs, contours, close=True):
        logger = None if self.logger is None else self.logger.experiment
        if logger is not None:
            figures = []
            for i in range(len(inputs)):
                img = channels_first2channels_last(asnumpy(inputs[i]))
                cons = asnumpy(contours[i])
                imshow_row(img, img, figsize=self.figsize)
                figures.append(plt.gcf())
                show_detection(contours=cons)
            logger.add_figure(
                tag=tag,
                figure=figures,
                global_step=self.global_step,
                close=close
            )
            plt.close('all')  # should be done above

    def evaluation_step(self, batch: dict, batch_idx: int, prefix: str):
        inputs = batch[self.inputs_key]
        indices = asnumpy(batch['indices'])
        outputs: dict = self(inputs)  # TODO: Add val loss
        contours = asnumpy(outputs['contours'])
        targets = asnumpy(batch[self.targets_key])
        if self._log_figures:
            self.log_figures(tag=f'{prefix}/batch{batch_idx}', inputs=inputs, contours=contours)
        matches = {}
        for i, (cons, target, index) in enumerate(zip(contours, targets, indices)):
            prediction = contours2labels(cons, size=inputs[i].shape[-2:], initial_depth=3)
            target = channels_first2channels_last(target)
            matches[index] = LabelMatcher(prediction, target, zero_division=0.)
        return matches

    def evaluation_epoch_end(self, outputs, prefix, iou_thresholds, mean_keys, sum_keys) -> None:
        """

        Note:
            `outputs` is a nested sequence of dictionaries, mapping data_index to data_item. Use of dictionary prevents
            double calculations of examples during evaluation. Note that double calculations are the default
            behaviour in PyTorch's DistributedSampler and example omission the only provided
            alternative (drop_last). Both would give wrong evaluation results.
            The use of dictionaries prevents this, yielding correct evaluation results.

        Args:
            outputs:
            prefix:
            iou_thresholds:
            mean_keys:
            sum_keys:

        Returns:

        """
        if isinstance(outputs, list):
            if is_available() and is_initialized():
                o = ([None] * get_world_size())
                all_gather_object(obj=outputs, object_list=o)  # give every rank access to results
                outputs = o
            if outputs is not None:
                while len(outputs) and isinstance(outputs[0], list):
                    outputs = list(chain.from_iterable(outputs))
                assert isinstance(outputs[0], dict)
                outputs = ChainMap(*outputs)
        else:
            raise ValueError(type(outputs))

        if outputs is not None:
            res = LabelMatcherList(list(outputs.values()))

            # Gather results for different IoU thresholds
            primary = {k: [] for k in mean_keys + sum_keys}
            primary['score_thresh'] = self.model.__dict__.get('score_thresh')
            primary['nms_thresh'] = self.model.__dict__.get('nms_thresh')
            primary['num_examples'] = len(res)
            secondary = {}
            for res.iou_thresh in iou_thresholds:
                for k in mean_keys + sum_keys:
                    v = getattr(res, k)
                    primary[k].append(v)
                    secondary[f'{k}_{int(res.iou_thresh * 100)}'] = v

            # Reduce results
            for k in mean_keys:
                primary[k] = np.mean(primary[k])
            for k in sum_keys:
                primary[k] = np.sum(primary[k])

            self.log_dict({f'{prefix}/{k}': float(v) for k, v in primary.items()}, logger=True, sync_dist=True,
                          reduce_fx='max')  # ranks are assumed to have same results
            self.log_dict({f'{prefix}_detail/{k}': float(v) for k, v in secondary.items()}, logger=True, sync_dist=True,
                          reduce_fx='max')

    def on_validation_epoch_start(self) -> None:
        self._validation_outputs = []

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self.evaluation_step(batch, batch_idx, 'val')
        self._validation_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        assert self._validation_outputs is not None
        outputs = self._validation_outputs
        self.evaluation_epoch_end(outputs, 'val', iou_thresholds=self.val_iou_thresholds,
                                  sum_keys=self._val_sum_keys, mean_keys=self._val_mean_keys)

    def on_test_epoch_start(self) -> None:
        self._test_outputs = []

    def test_step(self, batch: dict, batch_idx: int):
        outputs = self.evaluation_step(batch, batch_idx, 'test')
        self._test_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        outputs = self._test_outputs
        self.evaluation_epoch_end(outputs, 'test', iou_thresholds=self.test_iou_thresholds,
                                  sum_keys=self._test_sum_keys, mean_keys=self._test_mean_keys)

    def on_predict_epoch_start(self) -> None:
        self._predict_outputs = []

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, Tensor):
            return super().predict_step(batch, batch_idx, dataloader_idx)
        assert isinstance(batch, dict)
        inputs = batch.pop(self.inputs_key)
        assert inputs is not None
        out = OrderedDict(batch)
        out.update(self(inputs, **batch))
        self._predict_outputs.append(out)
        return out

    def on_predict_epoch_end(self) -> None:
        outputs = self._predict_outputs
        return outputs

    def forward(
            self,
            inputs: Tensor,
            targets: Optional[Dict[str, Tensor]] = None,
            max_imsize: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        max_imsize = self.max_imsize if max_imsize is None else max_imsize
        if max_imsize and max(inputs.shape[2:]) > max_imsize:
            return self.forward_tiled(inputs, targets=targets, **kwargs)

        device = self.device
        if inputs.device != device:
            inputs = inputs.to(device)

        return self.model(inputs, targets=targets, **kwargs)

    def forward_tiled(
            self,
            inputs: Tensor,
            crop_size: Union[int, Sequence[int]] = 768,
            stride: Union[int, Sequence[int]] = 384,
            **kwargs
    ) -> Dict[str, List[Tensor]]:
        assert np.array(crop_size) <= np.array(stride) * 2
        slices, slices_by_dim = get_tiling_slices(inputs.shape[2:], crop_size, stride)  # ordered
        prod = np.prod(slices_by_dim)
        results: List[List[Dict[str, Tensor]]] = [[None] * prod for _ in torch.arange(0, inputs.shape[0])]
        h_tiles, w_tiles = slices_by_dim
        device = self.device
        extra_keys = kwargs.get('extra_keys', ())  # extra output keys
        extra_nms = kwargs.get('extra_nms', {})  # specify which extra outputs need to be filtered by nms
        border_removal = kwargs.get('border_removal', 6)
        box_min_size = kwargs.get('min_box_size', 1.)
        nms_thresh = kwargs.get('nms_thresh', self.model.__dict__.get('nms_thresh', None))
        inputs_mask = kwargs.get('inputs_mask')
        assert nms_thresh is not None, 'Could not retrieve nms_thresh from model. Please specify it in forward method.'
        for i, slices_ in enumerate(slices):
            crop = inputs[(...,) + tuple(slices_)].to(device)
            if inputs_mask is not None:
                crop_m = inputs_mask[(...,) + tuple(slices_)].to(device)
                if not torch.any(crop_m):
                    continue  # skip masked out tile
            outputs = self.forward(crop, targets=kwargs.get('targets'), max_imsize=False)
            h_i, w_i = np.unravel_index(i, slices_by_dim)
            h_start, w_start = [s.start for s in slices_]

            top, bottom = h_i > 0, h_i < (h_tiles - 1)
            right, left = w_i < (w_tiles - 1), w_i > 0
            for j in torch.arange(0, inputs.shape[0]):
                contours = outputs['contours'][j]
                boxes = outputs['boxes'][j]
                scores = outputs['scores'][j]
                extra = [outputs[k][j] for k in extra_keys]

                # Remove small boxes (default min_size: 1.)
                keep = remove_small_boxes(boxes, box_min_size)
                contours, scores, boxes = (c[keep] for c in (contours, scores, boxes))
                extra = [e[keep] for e in extra]

                # Remove partial detections to avoid tiling artifacts
                keep = remove_border_contours(contours, crop.shape[2:], border_removal,
                                              top=top, right=right, bottom=bottom, left=left)
                contours, scores, boxes = (c[keep] for c in (contours, scores, boxes))
                extra = [e[keep] for e in extra]

                # Add offset  # TODO: Replace with cpn internal offsets
                contours[..., 1] += h_start
                contours[..., 0] += w_start
                boxes[..., [0, 2]] += w_start
                boxes[..., [1, 3]] += h_start

                results[j][i] = dict(
                    contours=contours,
                    boxes=boxes,
                    scores=scores,
                    extra=extra,  # list
                    keep=torch.ones(contours.shape[0], dtype=torch.bool)
                )

        final = OrderedDict(
            contours=[torch.cat([res_['contours'] for res_ in res if res_ is not None]) for res in results],
            scores=[torch.cat([res_['scores'] for res_ in res if res_ is not None]) for res in results],
            boxes=[torch.cat([res_['boxes'] for res_ in res if res_ is not None]) for res in results],
            **{k: [torch.cat([res_['extra'][i] for res_ in res if res_ is not None]) for res in results] for i, k in
               enumerate(extra_keys)}
        )

        if not self.training:
            for n in torch.arange(inputs.shape[0]):  # TODO
                boxes = final['boxes'][n]
                reference = boxes.shape[0]
                keep = nms(boxes, final['scores'][n], iou_threshold=nms_thresh)
                for k in final:
                    v = final[k][n]
                    if extra_nms.get(k, True):
                        assert v.shape[0] == reference, f'Output `{k}` is not compatible with nms. ' \
                                                        f'Specify extra_nms=dict({k}=False) when calling ' \
                                                        f'forward method.'
                        final[k][n] = v[keep]
        return final
