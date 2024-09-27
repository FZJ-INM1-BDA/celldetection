import torch
from typing import Any, Dict, List, Union, Sequence
from ..util.util import asnumpy, get_tiling_slices
from torch import Tensor, nn
from collections import OrderedDict
from lightning_fabric.utilities.rank_zero import rank_zero_only
from ..data.instance_eval import LabelMatcher
from ..data.cpn import contours2labels
from ..data.misc import channels_first2channels_last
from ..ops.cpn import remove_border_contours
import numpy as np
from torchvision.ops.boxes import remove_small_boxes, nms
import pytorch_lightning as pl
from .lightning_base import LitBase

__all__ = ['LitCpn']


STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]


class LitCpn(LitBase):
    def __init__(
            self,
            model: Union[str, dict, nn.Module],
            losses_prog_bar=True,
            optimizer=None,
            scheduler=None,
            scheduler_conf=None,
            warmup_steps=512,
            lr_scale='sqrt',
            weight_decay_scale=None,
            **kwargs
    ):
        kwargs['val_hparams'] = {
            'score_thresh': [.5, .86, .88, .9, .92],
            **(kwargs.get('val_hparams') or {})
        }

        super().__init__(
            model=model,
            losses_prog_bar=losses_prog_bar,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_conf=scheduler_conf,
            warmup_steps=warmup_steps,
            lr_scale=lr_scale,
            weight_decay_scale=weight_decay_scale,
            **kwargs
        )

        self.scores_lower_bound_key = 'scores_lower_bound'
        self.scores_upper_bound_key = 'scores_upper_bound'

    def _training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        inputs = batch.pop(self.inputs_key)
        outputs: dict = self.model(inputs, targets=batch, rank=self.global_rank)
        return outputs

    def _evaluation_log(self, prefix, batch_idx, inputs, contours, global_step):
        self.log_contour_figures(tag=f'{prefix}/batch{batch_idx}', inputs=inputs, contours=contours,
                                 global_step=global_step)

    def _evaluation_step(self, batch: dict, batch_idx: int, prefix: str, hparams_key, inputs, indices, matches,
                         log_step: bool):
        with torch.no_grad():
            outputs: dict = self(inputs, targets=batch, feature_cache=True)  # todo: val loss
        contours = asnumpy(outputs['contours'])
        targets = asnumpy(batch[self.targets_key])
        if log_step and self._log_figures:
            self._evaluation_log(prefix, batch_idx, inputs=inputs, contours=contours, global_step=self.global_step)
        matches[hparams_key] = matches_ = matches.get(hparams_key, {})

        for i, (cons, target, index) in enumerate(zip(contours, targets, indices)):
            prediction = contours2labels(cons, size=inputs[i].shape[-2:], initial_depth=3)
            target = channels_first2channels_last(target)
            matches_[index] = LabelMatcher(prediction, target, zero_division=self._eval_zero_division)

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Note: Remove everything that is not required for postprocessing from batch (e.g. via batch.pop).
        inputs = batch.pop(self.inputs_key)  # remove inputs
        assert inputs is not None
        scores_lower_bound = batch.pop(self.scores_lower_bound_key, None)
        scores_upper_bound = batch.pop(self.scores_upper_bound_key, None)
        return self(inputs, scores_upper_bound=scores_upper_bound, scores_lower_bound=scores_lower_bound, **batch)

    def forward_tiled(
            self,
            inputs: Tensor,
            crop_size: Union[int, Sequence[int]] = 1024,
            stride: Union[int, Sequence[int]] = 512,
            **kwargs
    ) -> Dict[str, List[Tensor]]:
        assert np.array(crop_size) <= np.array(stride) * 2
        slices, slices_by_dim = get_tiling_slices(inputs.shape[2:], crop_size, stride)  # ordered
        prod = np.prod(slices_by_dim)
        results: List[List[Dict[str, Tensor]]] = [[None] * prod for _ in torch.arange(0, inputs.shape[0])]
        h_tiles, w_tiles = slices_by_dim
        device = self.device
        kwargs.pop('max_imsize', None)
        extra_keys = kwargs.get('extra_keys', ())  # extra output keys
        extra_nms = kwargs.get('extra_nms', {})  # specify which extra outputs need to be filtered by nms
        border_removal = kwargs.get('border_removal', 6)
        box_min_size = kwargs.get('min_box_size', 1.)
        nms_thresh = kwargs.get('nms_thresh', self.model.__dict__.get('nms_thresh', None))
        inputs_mask = kwargs.get('inputs_mask')
        assert nms_thresh is not None, 'Could not retrieve nms_thresh from model. Please specify it in forward method.'
        targets = kwargs.pop('targets', None)
        for i, slices_ in enumerate(slices):
            crop = inputs[(...,) + tuple(slices_)].to(device)
            if inputs_mask is not None:
                crop_m = inputs_mask[(...,) + tuple(slices_)].to(device)
                if not torch.any(crop_m):
                    continue  # skip masked out tile
            outputs = self.forward(crop, targets=targets, max_imsize=False, **kwargs)
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
            nms_method = self.model.__dict__.get('nms_method', 'nms')
            for n in torch.arange(inputs.shape[0]):
                boxes = final['boxes'][n]
                reference = boxes.shape[0]
                if nms_method == 'nms':
                    keep = nms(boxes, final['scores'][n], iou_threshold=nms_thresh)
                else:
                    raise ValueError(f'Unknown nms method: {self.nms_method}')
                for k in final:
                    v = final[k][n]
                    if extra_nms.get(k, True):
                        assert v.shape[0] == reference, f'Output `{k}` is not compatible with nms. ' \
                                                        f'Specify extra_nms=dict({k}=False) when calling ' \
                                                        f'forward method.'
                        final[k][n] = v[keep]
        return final

    @rank_zero_only
    def log_batch(self: 'pl.LightningModule', batch: dict, stage: str, keys=('inputs', 'labels'), global_step=None):
        if global_step is None:
            global_step = self.global_step
        super().log_batch(batch=batch, stage=stage, keys=keys)
        for logger in self._iter_loggers():
            if hasattr(logger, 'add_image'):
                if 'sampled_contours' in batch:
                    # Prepare contours (contours may be zero padded)
                    cons = [c[:cl.max()] for c, cl in zip(batch['sampled_contours'], batch['labels'])]
                    self.log_contour_figures(f'{stage}/contours', batch['inputs'], cons, logger=logger,
                                             global_step=global_step)
                if 'resampled_contours' in batch:
                    # Prepare contours (contours may be zero padded)
                    cons = [c[:cl.max()] for c, cl in zip(batch['resampled_contours'], batch['labels'])]
                    self.log_contour_figures(f'{stage}/contours_resampled', batch['inputs'], cons, logger=logger,
                                             global_step=global_step)
