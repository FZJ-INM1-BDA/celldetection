import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from typing import Any, Optional, Dict, List, Tuple, Union, Sequence, Callable
from ..util.schedule import Config, conf2scheduler, conf2optimizer
from ..util.util import asnumpy, get_tiling_slices, fetch_model, load_model
from torch import optim, Tensor, nn
from collections import OrderedDict
from ..data.instance_eval import LabelMatcher
from ..data.cpn import contours2labels
from ..data.misc import channels_first2channels_last
from ..ops.cpn import remove_border_contours
from . import cpn
from ..visualization.images import show_detection, imshow_row
import pandas as pd
from torch.distributed import is_available, all_gather_object, get_world_size, is_initialized
from itertools import chain
import numpy as np
from os.path import isfile
from torchvision.ops.boxes import remove_small_boxes, nms
from torch.optim.lr_scheduler import SequentialLR
from ..optim.lr_scheduler import WarmUp

STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]


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
        self.model = model
        self.inputs_key = 'inputs'
        self.targets_key = 'targets'
        self.val_iou_thresholds = (.5, .6, .7, .8, .9)
        self.test_iou_thresholds = (.5, .6, .7, .8, .9)
        self.losses_prog_bar = losses_prog_bar
        self.local_tabs: Dict[str, pd.DataFrame] = {}  # local to process
        self.tabs: Dict[str, pd.DataFrame] = {}  # synced tabs
        self.sync_tabs = is_available() and is_initialized()
        self.warmup_steps = warmup_steps
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_conf = scheduler_conf
        self.figsize = kwargs.get('figsize', (32, 18))
        self._log_figures = kwargs.get('log_figures', True)
        self._eval_zero_division = kwargs.get('eval_zero_division', 0.)
        self._validation_outputs = None
        self._test_outputs = None

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
        losses = outputs.get('losses')
        losses['loss'] = losses.get('loss', outputs['loss'])
        if losses is not None and isinstance(losses, dict):
            self.log_dict({f'losses/{k}': v for k, v in losses.items() if v is not None},
                          prog_bar=self.losses_prog_bar, logger=True, on_step=True)
        return outputs

    def training_step_end(self, training_step_outputs):
        return training_step_outputs['loss']

    def configure_optimizers(self):
        if isinstance(self._optimizer, (dict, str, type(None))):
            optimizer = dict(AdamW=dict()) if self._optimizer is None else self._optimizer
            optimizer = conf2optimizer(optimizer, filter(lambda p: p.requires_grad, self.parameters()))
        else:
            optimizer = self._optimizer

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

    def evaluation_step(self, batch: dict, batch_idx: int, prefix: str,
                        iou_thresholds: Sequence[float]) -> List[pd.DataFrame]:
        inputs = batch[self.inputs_key]
        batch_size = inputs.shape[0]
        outputs: dict = self(inputs)  # TODO: Add val loss
        contours = asnumpy(outputs['contours'])
        targets = asnumpy(batch[self.targets_key])
        if self._log_figures:
            self.log_figures(tag=f'{prefix}/batch{batch_idx}', inputs=inputs, contours=contours)
        matches = []
        tabs = []
        mean_keys = ('f1', 'jaccard', 'fowlkes_mallows', 'recall', 'precision', 'score_thresh', 'nms_thresh')
        sum_keys = ('tp', 'fn', 'fp')
        for i, (cons, target) in enumerate(zip(contours, targets)):
            tab = None
            prediction = contours2labels(cons, size=inputs[i].shape[-2:], initial_depth=3)
            target = channels_first2channels_last(target)
            match = LabelMatcher(prediction, target, zero_division=0.)
            matches.append(match)
            for match.iou_thresh in iou_thresholds:
                tab_ = pd.DataFrame(data=[dict(
                    epoch=self.trainer.current_epoch,
                    f1=match.f1,
                    jaccard=match.jaccard,
                    fowlkes_mallows=match.fowlkes_mallows,
                    recall=match.recall,
                    precision=match.precision,
                    tp=match.true_positives,
                    fp=match.false_positives,
                    fn=match.false_negatives,
                    score_thresh=self.model.__dict__.get('score_thresh'),
                    nms_thresh=self.model.__dict__.get('nms_thresh'),
                    iou_thresh=match.iou_thresh,
                )])
                tab = pd.concat((tab, tab_), ignore_index=True)

                # Log results with mean reduce
                self.log_dict(
                    {f'{prefix}_detail_mean/{k}_{int(float(tab_.iou_thresh) * 100)}': float(tab.get(k).mean()) for k in
                     mean_keys},
                    sync_dist=True, logger=True, on_epoch=True, batch_size=batch_size
                )

                # Log results with sum reduce
                self.log_dict(
                    {f'{prefix}_detail_sum/{k}_{int(float(tab_.iou_thresh) * 100)}': float(tab.get(k).mean()) for k in
                     sum_keys},
                    sync_dist=True, reduce_fx='sum', logger=True, on_epoch=True, batch_size=batch_size
                )
            tabs.append(tab)

            # Log average results with mean reduce
            self.log_dict({f'{prefix}/{k}': float(tab.get(k).mean()) for k in mean_keys}, sync_dist=True,
                          prog_bar=True,
                          logger=True, on_epoch=True, batch_size=batch_size)

            # Log average results with sum reduce
            self.log_dict({f'{prefix}/{k}': float(tab.get(k).mean()) for k in sum_keys}, sync_dist=True,
                          reduce_fx='sum', logger=True, on_epoch=True, batch_size=batch_size)
        self.local_tabs[prefix] = pd.concat([self.local_tabs.get(prefix, None)] + tabs)

        return tabs

    def evaluation_epoch_end(self, outputs, prefix) -> None:
        from ..mpi import get_comm
        comm, rank, ranks = get_comm(None, True)

        if isinstance(outputs, list):
            if self.sync_tabs and is_available() and is_initialized():
                o = [None] * get_world_size()
                all_gather_object(o, outputs)
                outputs = o
            while len(outputs) and isinstance(outputs[0], list):
                outputs = list(chain.from_iterable(outputs))
            tabs = outputs
        elif isinstance(outputs, pd.DataFrame):
            tabs: List[pd.DataFrame] = [outputs]
        else:
            raise ValueError(type(outputs))

        if tabs is not None:
            self.tabs[prefix] = pd.concat([self.tabs.get(prefix, None)] + tabs)

    def on_validation_epoch_start(self) -> None:
        self._validation_outputs = []

    def validation_step(self, batch: dict, batch_idx: int) -> List[pd.DataFrame]:
        outputs = self.evaluation_step(batch, batch_idx, 'val', iou_thresholds=self.val_iou_thresholds)
        self._validation_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        assert self._validation_outputs is not None
        outputs = self._validation_outputs
        self.evaluation_epoch_end(outputs, 'val')

    def on_test_epoch_start(self) -> None:
        self._test_outputs = []

    def test_step(self, batch: dict, batch_idx: int) -> List[pd.DataFrame]:
        outputs = self.evaluation_step(batch, batch_idx, 'test', iou_thresholds=self.test_iou_thresholds)
        self._test_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        outputs = self._test_outputs
        self.evaluation_epoch_end(outputs, 'test')

    def forward(
            self,
            inputs: Tensor,
            targets: Optional[Dict[str, Tensor]] = None,
            max_imsize: Optional[int] = 2048,
            **kwargs
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        if max_imsize is not None and max(inputs.shape[2:]) > max_imsize:
            return self.forward_tiled(inputs, targets=targets, **kwargs)

        device = self.device
        if inputs.device != device:
            inputs = inputs.to(device)
        
        return self.model(inputs, targets=targets)

    def forward_tiled(
            self,
            inputs: Tensor,
            crop_size: Union[int, Sequence[int]] = 512,
            stride: Union[int, Sequence[int]] = 256,
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
        border_removal = kwargs.get('border_removal', 8)
        box_min_size = kwargs.get('min_box_size', 1.)
        nms_thresh = kwargs.get('nms_thresh', self.model.__dict__.get('nms_thresh', None))
        assert nms_thresh is not None, 'Could not retrieve nms_thresh from model. Please specify it in forward method.'
        for i, slices_ in enumerate(slices):
            crop = inputs[(...,) + tuple(slices_)].to(device)
            outputs = self.forward(crop, targets=kwargs.get('targets'), max_imsize=None)
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

                # Add offset
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
            contours=[torch.cat([res_['contours'] for res_ in res]) for res in results],
            scores=[torch.cat([res_['scores'] for res_ in res]) for res in results],
            boxes=[torch.cat([res_['boxes'] for res_ in res]) for res in results],
            **{k: [torch.cat([res_['extra'][i] for res_ in res]) for res in results] for i, k in enumerate(extra_keys)}
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
