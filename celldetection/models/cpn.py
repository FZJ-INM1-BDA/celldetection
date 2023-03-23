import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Union
import warnings
from pytorch_lightning.core.mixins import HyperparametersMixin
from ..util.util import add_to_loss_dict, reduce_loss_dict, fetch_model, update_dict_
from .commons import ScaledTanh, ReadOut, Fuse2d
from .loss import IoULoss, BoxNpllLoss
from ..ops.commons import downsample_labels
from ..ops import boxes as bx
from ..ops.cpn import rel_location2abs_location, fouriers2contours, scale_contours, scale_fourier, batched_box_nmsi, \
    order_weighting, resolve_refinement_buckets
from .unet import U22, SlimU22, WideU22, ResUNet, ResNet50UNet, ResNet34UNet, ResNet18UNet, ResNet101UNet, \
    ResNeXt50UNet, ResNeXt101UNet, ResNet152UNet, ResNeXt152UNet, ConvNeXtTinyUNet, ConvNeXtSmallUNet, \
    ConvNeXtLargeUNet, ConvNeXtBaseUNet, SmpUNet, TimmUNet
from .fpn import ResNet34FPN, ResNet18FPN, ResNet50FPN, ResNet101FPN, ResNet152FPN, ResNeXt50FPN, \
    ResNeXt101FPN, ResNeXt152FPN, WideResNet50FPN, WideResNet101FPN, MobileNetV3LargeFPN, MobileNetV3SmallFPN
from .manet import MaNet, SmpMaNet, TimmMaNet

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


def resolve_batch_index(inputs: dict, n, b) -> dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for batch_index in range(n):
        sel = b == batch_index
        for k, v in inputs.items():
            o = outputs[k]
            if o is not None:
                o.append(v[sel])
    return outputs


def resolve_keep_indices(inputs: dict, keep: list) -> dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for j, indices in enumerate(keep):
        for k, v in inputs.items():
            o = outputs[k]
            if o is not None:
                o.append(v[j][indices])
    return outputs


def local_refinement(det_indices, refinement, num_loops, num_buckets, original_size, sampling, b):
    for _ in torch.arange(0, num_loops):
        det_indices = torch.round(det_indices.detach())  # Tensor[num_contours, samples, 2]
        det_indices[..., 0].clamp_(0, original_size[1] - 1)
        det_indices[..., 1].clamp_(0, original_size[0] - 1)
        indices = det_indices.detach().long()  # Tensor[-1, samples, 2]
        if num_buckets == 1:
            responses = refinement[b[:, None], :, indices[:, :, 1], indices[:, :, 0]]  # Tensor[-1, samples, 2]
        else:
            buckets = resolve_refinement_buckets(sampling, num_buckets)
            responses = None
            for bucket_indices, bucket_weights in buckets:
                bckt_idx = torch.stack((bucket_indices * 2, bucket_indices * 2 + 1), -1)
                cur_ref = refinement[b[:, None, None], bckt_idx, indices[:, :, 1, None], indices[:, :, 0, None]]
                cur_ref = cur_ref * bucket_weights[..., None]
                if responses is None:
                    responses = cur_ref
                else:
                    responses = responses + cur_ref
        det_indices = det_indices + responses
    return det_indices


def _resolve_channels(encoder_channels, backbone_channels, keys: Union[list, tuple, str], encoder_prefix: str):
    channels = 0
    reference = None
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for k in keys:
        if k.startswith(encoder_prefix):
            channels += encoder_channels[int(k[len(encoder_prefix):])]
        else:
            channels += backbone_channels[int(k)]
        if reference is None:
            reference = channels
    return channels, reference, len(keys)


def _resolve_features(features, keys):
    if isinstance(keys, (tuple, list)):
        return [features[k] for k in keys]
    return features[keys]


class CPNCore(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_channels,
            order,
            score_channels: int,
            refinement: bool = True,
            refinement_margin: float = 3.,
            uncertainty_head=False,
            contour_features='1',
            location_features='1',
            uncertainty_features='1',
            score_features='1',
            refinement_features='0',
            contour_head_channels=None,
            contour_head_stride=1,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',
            refinement_buckets=1,
            refinement_full_res=True,
            encoder_channels=None,
            **kwargs,
    ):
        super().__init__()
        self.order = order
        self.backbone = backbone
        self.refinement_interpolation = refinement_interpolation
        assert refinement_buckets >= 1
        self.refinement_buckets = refinement_buckets

        if encoder_channels is None:
            encoder_channels = backbone_channels  # assuming same channels
        channels = encoder_channels, backbone_channels
        kw = {'encoder_prefix': kwargs.get('encoder_prefix', 'encoder.')}
        self.contour_features = contour_features
        self.location_features = location_features
        self.score_features = score_features
        self.refinement_features = refinement_features
        self.uncertainty_features = uncertainty_features
        self.refinement_full_res = refinement_full_res
        fourier_channels, fourier_channels_, num_fourier_inputs = _resolve_channels(*channels, contour_features, **kw)
        loc_channels, loc_channels_, num_loc_inputs = _resolve_channels(*channels, location_features, **kw)
        sco_channels, sco_channels_, num_score_inputs = _resolve_channels(*channels, score_features, **kw)
        ref_channels, ref_channels_, num_ref_inputs = _resolve_channels(*channels, refinement_features, **kw)
        unc_channels, unc_channels_, num_unc_inputs = _resolve_channels(*channels, uncertainty_features, **kw)
        fuse_kw = kwargs.get('fuse_kwargs', {})

        # Score
        self.score_fuse = Fuse2d(sco_channels, sco_channels_, **fuse_kw) if num_score_inputs > 1 else None
        self.score_head = ReadOut(
            sco_channels_, score_channels,
            kernel_size=kwargs.get('kernel_size_score', 7),
            padding=kwargs.get('kernel_size_score', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride
        )

        # Location
        self.location_fuse = Fuse2d(loc_channels, loc_channels_, **fuse_kw) if num_loc_inputs > 1 else None
        self.location_head = ReadOut(
            loc_channels_, 2,
            kernel_size=kwargs.get('kernel_size_location', 7),
            padding=kwargs.get('kernel_size_location', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride
        )

        # Fourier
        self.fourier_fuse = Fuse2d(fourier_channels, fourier_channels_, **fuse_kw) if num_fourier_inputs > 1 else None
        self.fourier_head = ReadOut(
            fourier_channels_, order * 4,
            kernel_size=kwargs.get('kernel_size_fourier', 7),
            padding=kwargs.get('kernel_size_fourier', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride
        )

        # Uncertainty
        if uncertainty_head:
            self.uncertainty_fuse = Fuse2d(unc_channels, unc_channels_, **fuse_kw) if num_unc_inputs > 1 else None
            self.uncertainty_head = ReadOut(
                unc_channels_, 4,
                kernel_size=kwargs.get('kernel_size_uncertainty', 7),
                padding=kwargs.get('kernel_size_uncertainty', 7) // 2,
                channels_mid=contour_head_channels,
                stride=contour_head_stride,
                final_activation='sigmoid'
            )
        else:
            self.uncertainty_fuse = self.uncertainty_head = None

        # Refinement
        if refinement:
            self.refinement_fuse = Fuse2d(ref_channels, ref_channels_, **fuse_kw) if num_ref_inputs > 1 else None
            self.refinement_head = ReadOut(
                ref_channels_, 2 * refinement_buckets,
                kernel_size=kwargs.get('kernel_size_refinement', 7),
                padding=kwargs.get('kernel_size_refinement', 7) // 2,
                final_activation=ScaledTanh(refinement_margin),
                channels_mid=refinement_head_channels,
                stride=refinement_head_stride
            )
        else:
            self.refinement_fuse = self.refinement_head = None

    def forward(self, inputs):
        features = self.backbone(inputs)

        if isinstance(features, torch.Tensor):
            score_features = fourier_features = location_features = unc_features = ref_features = features
        else:
            score_features = _resolve_features(features, self.score_features)
            fourier_features = _resolve_features(features, self.contour_features)
            location_features = _resolve_features(features, self.location_features)
            unc_features = _resolve_features(features, self.uncertainty_features)
            ref_features = _resolve_features(features, self.refinement_features)

        # Scores
        if self.score_fuse is not None:
            score_features = self.score_fuse(score_features)
        scores = self.score_head(score_features)

        # Locations
        if self.location_fuse is not None:
            location_features = self.location_fuse(location_features)
        locations = self.location_head(location_features)

        # Fourier
        if self.fourier_fuse is not None:
            fourier_features = self.fourier_fuse(fourier_features)
        fourier = self.fourier_head(fourier_features)

        # Uncertainty
        if self.uncertainty_head is not None:
            if self.uncertainty_fuse is not None:
                unc_features = self.uncertainty_fuse(unc_features)
            uncertainty = self.uncertainty_head(unc_features)
        else:
            uncertainty = None

        # Refinement
        if self.refinement_head is not None:
            if self.refinement_fuse is not None:
                ref_features = self.refinement_fuse(ref_features)
            if self.refinement_full_res:
                ref_features = F.interpolate(ref_features, inputs.shape[-2:], mode=self.refinement_interpolation,
                                             align_corners=False)
            refinement = self.refinement_head(ref_features)
            if refinement.shape[-2:] != inputs.shape[-2:]:  # 337 ns
                # bilinear: 3.79 ms for (128, 128) to (512, 512)
                # bicubic: 11.5 ms for (128, 128) to (512, 512)
                refinement = F.interpolate(refinement, inputs.shape[-2:],
                                           mode=self.refinement_interpolation, align_corners=False)
        else:
            refinement = None

        return scores, locations, refinement, fourier, uncertainty


@register
class CPN(nn.Module, HyperparametersMixin):
    def __init__(
            self,
            backbone: nn.Module,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            certainty_thresh: float = None,
            samples: int = 32,
            classes: int = 2,

            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,

            contour_features='1',
            location_features='1',
            uncertainty_features='1',
            score_features='1',
            refinement_features='0',

            uncertainty_head=False,
            uncertainty_nms=False,
            uncertainty_factor=7.,

            contour_head_channels=None,
            contour_head_stride=1,
            order_weights=True,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',

            **kwargs
    ):
        """CPN base class.

        This is the base class for the Contour Proposal Network.

        References:
            https://www.sciencedirect.com/science/article/pii/S136184152200024X

        Args:
            backbone: A backbone network. E.g. ``cd.models.U22(in_channels, 0)``.
            order: Contour order. The higher, the more complex contours can be proposed.
                ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
                ``order=8`` allows even finer detail.
            nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
                ``iou > nms_thresh`` to be identical.
            score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
                have ``score > score_thresh`` to be proposed as a result.
            samples: Number of samples. This sets the number of coordinates with which a contour is defined.
                This setting can be changed on the fly, e.g. small for training and large for inference.
                Small settings reduces computational costs, while larger settings capture more detail.
            classes: Number of classes. Default: 2 (object vs. background).
            refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
                the proposed contours.
            refinement_iterations: Number of refinement iterations.
            refinement_margin: Maximum refinement margin (step size) per iteration.
            refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
                with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
                ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
            contour_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict contours.
            refinement_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict the refinement tensor.
            contour_head_channels: Number of intermediate channels in contour ``ReadOut`` Modules. By default, this is
                the number of incoming feature channels.
            contour_head_stride: Stride used for the contour prediction. Larger stride means less contours can
                be proposed in total, which speeds up execution times.
            order_weights: Whether to use order specific weights.
            refinement_head_channels: Number of intermediate channels in refinement ``ReadOut`` Modules. By default,
                this is the number of incoming feature channels.
            refinement_head_stride: Stride used for the refinement prediction. Larger stride means less detail, but
                speeds up execution times.
            refinement_interpolation: Interpolation mode that is used to ensure that refinement tensor and input
                image have the same shape.
            score_encoder_features: Whether to use encoder-head skip connections for the score head.
            refinement_encoder_features: Whether to use encoder-head skip connections for the refinement head.
        """
        super().__init__()
        self.order = order
        self.nms_thresh = nms_thresh
        self.samples = samples
        self.score_thresh = score_thresh
        self.score_channels = 1 if classes in (1, 2) else classes
        self.refinement = refinement
        self.refinement_iterations = refinement_iterations
        self.refinement_margin = refinement_margin
        self.functional = False
        self.full_detail = False
        self.score_target_dtype = None
        self.certainty_thresh = certainty_thresh
        self.uncertainty_nms = uncertainty_nms

        if not hasattr(backbone, 'out_channels'):
            raise ValueError('Backbone should have an attribute out_channels that states the channels of its output.')

        self.core = CPNCore(
            backbone=backbone,
            backbone_channels=backbone.out_channels,
            order=order,
            score_channels=self.score_channels,
            refinement=refinement,
            refinement_margin=refinement_margin,
            contour_features=contour_features,
            location_features=location_features,
            uncertainty_features=uncertainty_features,
            score_features=score_features,
            refinement_features=refinement_features,
            contour_head_channels=contour_head_channels,
            contour_head_stride=contour_head_stride,
            refinement_head_channels=refinement_head_channels,
            refinement_head_stride=refinement_head_stride,
            refinement_interpolation=refinement_interpolation,
            refinement_buckets=refinement_buckets,
            uncertainty_head=uncertainty_head,
            **kwargs
        )

        if isinstance(order_weights, bool):
            if order_weights:
                self.register_buffer('order_weights', order_weighting(self.order))
            else:
                self.order_weights = 1.
        else:
            self.order_weights = order_weights

        self.objectives = OrderedDict({
            'score': nn.CrossEntropyLoss() if self.score_channels > 1 else nn.BCEWithLogitsLoss(),
            'fourier': nn.L1Loss(reduction='none'),
            'location': nn.L1Loss(),
            'contour': nn.L1Loss(),
            'refinement': nn.L1Loss() if refinement else None,
            'boxes': None,
            'iou': IoULoss(min_size=1.),
            'uncertainty': BoxNpllLoss(uncertainty_factor, min_size=1., sigmoid=False) if uncertainty_head else None
        })
        self.weights = {
            'fourier': 1.,  # note: fourier has order specific weights
            'location': 1.,
            'contour': 3.,
            'score_bg': 1.,
            'score_fg': 1.,
            'refinement': 1.,
            'boxes': .88,
            'iou': 1.,
            'uncertainty': 1.,
        }

        self._rel_location2abs_location_cache: Dict[str, Tensor] = {}
        self._fourier2contour_cache: Dict[str, Tensor] = {}
        self._warn_iou = False

    def compute_loss(
            self,
            uncertainty,
            fourier,
            locations,
            contours,
            refined_contours,
            boxes,
            raw_scores,
            targets: dict,
            labels,
            fg_masks,
            b
    ):
        assert targets is not None

        fourier_targets = targets.get('fourier')
        location_targets = targets.get('locations')
        contour_targets = targets.get('sampled_contours')
        hires_contour_targets = targets.get('hires_sampled_contours')
        box_targets = targets.get('boxes')
        class_targets = targets.get('classes')

        losses = OrderedDict({
            'fourier': None,
            'location': None,
            'contour': None,
            'score': None,
            'refinement': None,
            'boxes': None,
            'iou': None,
            'uncertainty': None,
        })

        bg_masks = labels == 0
        fg_n, fg_y, fg_x = torch.where(fg_masks)
        bg_n, bg_y, bg_x = torch.where(bg_masks)
        objectives = self.objectives

        fg_scores = raw_scores[fg_n, :, fg_y, fg_x]  # Tensor[-1, classes]
        bg_scores = raw_scores[bg_n, :, bg_y, bg_x]  # Tensor[-1, classes]
        fg_indices = labels[fg_n, fg_y, fg_x].long() - 1  # -1 because fg labels start at 1, but indices at 0
        fg_num = fg_indices.numel()
        bg_num = bg_scores.numel()

        if box_targets is not None:
            if fg_num:
                box_targets = box_targets[b, fg_indices]
        elif not self._warn_iou and self.objectives.get('iou') is not None and self.samples < 32:
            self._warn_iou = True
            warnings.warn('The iou loss option of the CPN is enabled, but the `samples` setting is rather low. '
                          'This may impair detection performance. '
                          'Increase `samples`, provide box targets manually or set model.objectives["iou"] = False.')
        if fg_num and contour_targets is not None:
            c_tar = contour_targets[b, fg_indices]  # Tensor[num_pixels, samples, 2]

            if box_targets is None:
                box_targets = bx.contours2boxes(c_tar, axis=1)

        if self.score_target_dtype is None:
            if isinstance(objectives['score'], nn.CrossEntropyLoss):
                self.score_target_dtype = torch.int64
            else:
                self.score_target_dtype = fg_scores.dtype

        if fg_num:
            if class_targets is None:
                ones = torch.broadcast_tensors(torch.ones((), dtype=self.score_target_dtype, device=fg_scores.device),
                                               fg_scores[..., 0])[0]
            else:
                ones = class_targets[b, fg_indices].to(self.score_target_dtype)
            if self.score_channels == 1:
                fg_scores = torch.squeeze(fg_scores, 1)
            add_to_loss_dict(losses, 'score', objectives['score'](fg_scores, ones), self.weights['score_fg'])

        if bg_num:
            zeros = torch.broadcast_tensors(torch.zeros((), dtype=self.score_target_dtype, device=bg_scores.device),
                                            bg_scores[..., 0])[0]
            if self.score_channels == 1:
                bg_scores = torch.squeeze(bg_scores, 1)
            add_to_loss_dict(losses, 'score', objectives['score'](bg_scores, zeros), self.weights['score_bg'])

        if fg_num:
            if fourier_targets is not None:
                f_tar = fourier_targets[b, fg_indices]  # Tensor[num_pixels, order, 4]
                add_to_loss_dict(losses, 'fourier',
                                 (objectives['fourier'](fourier, f_tar) * self.order_weights).mean(),
                                 self.weights['fourier'])
            if location_targets is not None:
                l_tar = location_targets[b, fg_indices]  # Tensor[num_pixels, 2]
                assert len(locations) == len(l_tar)
                add_to_loss_dict(losses, 'location', objectives['location'](locations, l_tar), self.weights['location'])
            if contour_targets is not None:
                add_to_loss_dict(losses, 'contour', objectives['contour'](contours, c_tar), self.weights['contour'])

                if self.refinement and self.refinement_iterations > 0:
                    if hires_contour_targets is None:
                        cc_tar = c_tar
                    else:
                        cc_tar = hires_contour_targets[b, fg_indices]  # Tensor[num_pixels, samples', 2]

                    add_to_loss_dict(losses, 'refinement', objectives['refinement'](refined_contours, cc_tar),
                                     self.weights['refinement'])

                if (uncertainty is not None and boxes.nelement() > 0 and box_targets is not None and
                        box_targets.nelement() > 0):
                    add_to_loss_dict(losses, 'uncertainty',
                                     objectives['uncertainty'](uncertainty, boxes.detach(), box_targets),
                                     self.weights['uncertainty'])

            if box_targets is not None:
                if objectives.get('iou') is not None:
                    add_to_loss_dict(losses, 'iou', objectives['iou'](boxes, box_targets), self.weights['iou'])
                if objectives.get('boxes') is not None:
                    add_to_loss_dict(losses, 'boxes', objectives['boxes'](boxes, box_targets), self.weights['boxes'])
        loss = reduce_loss_dict(losses, 1)
        return loss, losses

    def forward(
            self,
            inputs,
            targets: Dict[str, Tensor] = None,
            nms=True,
            **kwargs
    ):
        # Presets
        original_size = inputs.shape[-2:]

        # Core
        scores, locations, refinement, fourier, uncertainty = self.core(inputs)

        # Apply optional score bounds
        scores_upper_bound = kwargs.get('scores_upper_bound')
        scores_lower_bound = kwargs.get('scores_lower_bound')
        if scores_upper_bound is not None:
            scores = torch.minimum(scores, scores_upper_bound)
        if scores_lower_bound is not None:
            scores = torch.maximum(scores, scores_lower_bound)

        # Scores
        raw_scores = scores
        if self.score_channels == 1:
            scores = torch.sigmoid(scores)
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels == 2:
            scores = F.softmax(scores, dim=1)[:, 1:2]
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels > 2:
            scores = F.softmax(scores, dim=1)
            classes = torch.argmax(scores, dim=1).long()
        else:
            raise ValueError

        actual_size = fourier.shape[-2:]
        n, c, h, w = fourier.shape
        if self.functional:
            fourier = fourier.view((n, c // 2, 2, h, w))
        else:
            fourier = fourier.view((n, c // 4, 4, h, w))

        # Maybe apply changed order
        if self.order < self.core.order:
            fourier = fourier[:, :self.order]

        # Fetch sampling and labels
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            sampling = targets.get('sampling')
            labels = targets['labels']
        else:
            sampling = None
            labels = classes.detach()
        labels = downsample_labels(labels[:, None], actual_size)[:, 0]

        # Locations
        # raw_locations = locations.detach()
        locations = rel_location2abs_location(locations, cache=self._rel_location2abs_location_cache)

        # Extract proposals
        fg_mask = labels > 0
        if self.certainty_thresh is not None and uncertainty is not None:
            fg_mask &= uncertainty.mean(1) < (1 - self.certainty_thresh)

        b, y, x = torch.where(fg_mask)
        selected_fourier = fourier[b, :, :, y, x]  # Tensor[-1, order, 4]
        selected_locations = locations[b, :, y, x]  # Tensor[-1, 2]
        selected_classes = classes[b, y, x]

        if self.score_channels in (1, 2):
            selected_scores = scores[b, 0, y, x]  # Tensor[-1]
        elif self.score_channels > 2:
            selected_scores = scores[b, selected_classes, y, x]  # Tensor[-1]
        else:
            raise ValueError

        selected_uncertainties = None
        if uncertainty is not None:
            selected_uncertainties = uncertainty[b, :, y, x]

        if sampling is not None:
            sampling = sampling[b]

        # Convert to pixel space
        selected_contour_proposals, sampling = fouriers2contours(selected_fourier, selected_locations,
                                                                 samples=self.samples, sampling=sampling,
                                                                 cache=self._fourier2contour_cache)

        # Rescale in case of multi-scale
        selected_contour_proposals = scale_contours(actual_size=actual_size, original_size=original_size,
                                                    contours=selected_contour_proposals)
        selected_fourier, selected_locations = scale_fourier(actual_size=actual_size, original_size=original_size,
                                                             fourier=selected_fourier, location=selected_locations)

        if self.refinement and self.refinement_iterations > 0:
            num_loops = self.refinement_iterations
            if self.training and num_loops > 1:
                num_loops = torch.randint(low=1, high=num_loops + 1, size=())
            selected_contours = local_refinement(
                selected_contour_proposals, refinement, num_loops=num_loops, num_buckets=self.core.refinement_buckets,
                original_size=original_size, sampling=sampling, b=b
            )
        else:
            selected_contours = selected_contour_proposals
        selected_contours[..., 0].clamp_(0, original_size[1] - 1)
        selected_contours[..., 1].clamp_(0, original_size[0] - 1)

        # Bounding boxes
        if selected_contours.numel() > 0:
            selected_boxes = torch.cat((selected_contours.min(1).values,
                                        selected_contours.max(1).values), 1)  # 43.3 µs ± 290 ns for Tensor[2203, 32, 2]
        else:
            selected_boxes = torch.empty((0, 4), device=selected_contours.device)

        # Loss
        loss, losses = None, None
        if self.training or targets is not None:
            loss, losses = self.compute_loss(
                uncertainty=selected_uncertainties,
                fourier=selected_fourier,
                locations=selected_locations,
                contours=selected_contour_proposals,
                refined_contours=selected_contours,
                boxes=selected_boxes,
                raw_scores=raw_scores,
                targets=targets,
                labels=labels,
                fg_masks=fg_mask,
                b=b
            )

        if self.training and not self.full_detail:
            return OrderedDict({
                'loss': loss,
                'losses': losses,
            })

        outputs = OrderedDict(
            contours=selected_contours,
            boxes=selected_boxes,
            scores=selected_scores,
            classes=selected_classes,
            locations=selected_locations,
            fourier=selected_fourier,
            contour_proposals=selected_contour_proposals,
            box_uncertainties=selected_uncertainties,
        )
        outputs = resolve_batch_index(outputs, inputs.shape[0], b=b)

        if not self.training and nms:
            if self.uncertainty_nms and outputs['box_uncertainties'] is not None:
                nms_weights = [s * (1. - u.mean(1)) for s, u in zip(outputs['scores'], outputs['box_uncertainties'])]
            else:
                nms_weights = outputs['scores']
            keep_indices: list = batched_box_nmsi(outputs['boxes'], nms_weights, self.nms_thresh)
            outputs = resolve_keep_indices(outputs, keep_indices)

        if loss is not None:
            outputs['loss'] = loss
            outputs['losses'] = losses

        return outputs


def _make_cpn_doc(title, text, backbone):
    return f"""{title}
    
    {text}
    
    References:
        https://www.sciencedirect.com/science/article/pii/S136184152200024X
    
    Args:
        in_channels: Number of input channels.
        order: Contour order. The higher, the more complex contours can be proposed.
            ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
            ``order=8`` allows even finer detail.
        nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
            ``iou > nms_thresh`` to be identical.
        score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
            have ``score > score_thresh`` to be proposed as a result.
        samples: Number of samples. This sets the number of coordinates with which a contour is defined.
            This setting can be changed on the fly, e.g. small for training and large for inference.
            Small settings reduces computational costs, while larger settings capture more detail.
        classes: Number of classes. Default: 2 (object vs. background).
        refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
            the proposed contours.
        refinement_iterations: Number of refinement iterations.
        refinement_margin: Maximum refinement margin (step size) per iteration.
        refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
            with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
            ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
        backbone_kwargs: Additional backbone keyword arguments. See docstring of ``{backbone}``.
        **kwargs: Additional CPN keyword arguments. See docstring of ``cd.models.CPN``.
    
    """


@register
class CpnU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=U22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with U-Net 22 backbone.',
        'A Contour Proposal Network that uses a U-Net with 22 convolutions as a backbone.',
        'cd.models.U22'
    )


@register
class CpnResUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResUNet(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Residual U-Net backbone.',
        'A Contour Proposal Network that uses a U-Net build with residual blocks.',
        'cd.models.ResUNet'
    )


@register
class CpnSlimU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=SlimU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Slim U-Net 22 backbone.',
        'A Contour Proposal Network that uses a Slim U-Net as a backbone. '
        'Slim U-Net has 22 convolutions with less feature channels than normal U22.',
        'cd.models.SlimU22'
    )


@register
class CpnWideU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Wide U-Net 22 backbone.',
        'A Contour Proposal Network that uses a Wide U-Net as a backbone. '
        'Wide U-Net has 22 convolutions with more feature channels than normal U22.',
        'cd.models.WideU22'
    )


@register
class CpnResNeXt101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 101 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNeXt 101 U-Net as a backbone.',
        'cd.models.ResNeXt101UNet'
    )


@register
class CpnResNeXt152UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt152UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 152 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 152 U-Net as a backbone.',
        'cd.models.ResNeXt152UNet'
    )


@register
class CpnResNet152UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet152UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 152 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 152 U-Net as a backbone.',
        'cd.models.ResNet152UNet'
    )


@register
class CpnResNet101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 101 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 101 U-Net as a backbone.',
        'cd.models.ResNet101UNet'
    )


@register
class CpnResNeXt50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt50UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 50 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNeXt 50 U-Net as a backbone.',
        'cd.models.ResNeXt50UNet'
    )


@register
class CpnResNet50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet50UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 50 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 50 U-Net as a backbone.',
        'cd.models.ResNet50UNet'
    )


@register
class CpnResNet34UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet34UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 34 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 34 U-Net as a backbone.',
        'cd.models.ResNet34UNet'
    )


@register
class CpnResNet18UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet18UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 18 U-Net backbone.',
        'A Contour Proposal Network that uses a ResNet 18 U-Net as a backbone.',
        'cd.models.ResNet18UNet'
    )


@register
class CpnResNet18FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet18FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 18 FPN backbone.',
        'A Contour Proposal Network that uses a ResNet 18 Feature Pyramid Network as a backbone.',
        'cd.models.ResNet18FPN'
    )


@register
class CpnResNet34FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet34FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 34 FPN backbone.',
        'A Contour Proposal Network that uses a ResNet 34 Feature Pyramid Network as a backbone.',
        'cd.models.ResNet34FPN'
    )


@register
class CpnResNet50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 50 FPN backbone.',
        'A Contour Proposal Network that uses a ResNet 50 Feature Pyramid Network as a backbone.',
        'cd.models.ResNet50FPN'
    )


@register
class CpnResNet101FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet101FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 101 FPN backbone.',
        'A Contour Proposal Network that uses a ResNet 101 Feature Pyramid Network as a backbone.',
        'cd.models.ResNet101FPN'
    )


@register
class CpnResNet152FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet152FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNet 152 FPN backbone.',
        'A Contour Proposal Network that uses a ResNet 152 Feature Pyramid Network as a backbone.',
        'cd.models.ResNet152FPN'
    )


@register
class CpnResNeXt50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNeXt50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 50 FPN backbone.',
        'A Contour Proposal Network that uses a ResNeXt 50 Feature Pyramid Network as a backbone.',
        'cd.models.ResNeXt50FPN'
    )


@register
class CpnResNeXt101FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNeXt101FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 101 FPN backbone.',
        'A Contour Proposal Network that uses a ResNeXt 101 Feature Pyramid Network as a backbone.',
        'cd.models.ResNeXt101FPN'
    )


@register
class CpnResNeXt152FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNeXt152FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ResNeXt 152 FPN backbone.',
        'A Contour Proposal Network that uses a ResNeXt 152 Feature Pyramid Network as a backbone.',
        'cd.models.ResNeXt152FPN'
    )


@register
class CpnWideResNet50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideResNet50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Wide ResNet 50 FPN backbone.',
        'A Contour Proposal Network that uses a Wide ResNet 50 Feature Pyramid Network as a backbone.',
        'cd.models.WideResNet50FPN'
    )


@register
class CpnWideResNet101FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideResNet101FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Wide ResNet 101 FPN backbone.',
        'A Contour Proposal Network that uses a Wide ResNet 101 Feature Pyramid Network as a backbone.',
        'cd.models.WideResNet101FPN'
    )


@register
class CpnMobileNetV3SmallFPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=MobileNetV3SmallFPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Small MobileNetV3 FPN backbone.',
        'A Contour Proposal Network that uses a Small MobileNetV3 Feature Pyramid Network as a backbone.',
        'cd.models.MobileNetV3SmallFPN'
    )


@register
class CpnMobileNetV3LargeFPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=MobileNetV3LargeFPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Large MobileNetV3 FPN backbone.',
        'A Contour Proposal Network that uses a Large MobileNetV3 Feature Pyramid Network as a backbone.',
        'cd.models.MobileNetV3LargeFPN'
    )


@register
class CpnMiTB5MaNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=SmpMaNet(in_channels=in_channels, out_channels=0, model_name='mit_b5',
                              **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with Mix Transformer encoder and Multi-Scale Attention Network decoder as backbone.',
        'A Contour Proposal Network that uses a Mix Transformer B5 encoder with the Multi-Scale Attention Network '
        'decoder as a backbone.',
        'cd.models.CpnMiTB5MaNet'
    )


@register
class CpnConvNeXtSmallUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ConvNeXtSmallUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ConvNeXt Small U-Net backbone.',
        'A Contour Proposal Network that uses a ConvNeXt Small U-Net as a backbone.',
        'cd.models.ConvNeXtSmallUNet'
    )


@register
class CpnConvNeXtLargeUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ConvNeXtLargeUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ConvNeXt Large U-Net backbone.',
        'A Contour Proposal Network that uses a ConvNeXt Large U-Net as a backbone.',
        'cd.models.ConvNeXtLargeUNet'
    )


@register
class CpnConvNeXtBaseUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ConvNeXtBaseUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ConvNeXt Base U-Net backbone.',
        'A Contour Proposal Network that uses a ConvNeXt Base U-Net as a backbone.',
        'cd.models.ConvNeXtBaseUNet'
    )


@register
class CpnConvNeXtTinyUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ConvNeXtTinyUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with ConvNeXt Tiny U-Net backbone.',
        'A Contour Proposal Network that uses a ConvNeXt Tiny U-Net as a backbone.',
        'cd.models.ConvNeXtTinyUNet'
    )


@register
class CpnSmpMaNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        update_dict_(backbone_kwargs, dict(model_name=kwargs.get('model_name')))
        super().__init__(
            backbone=SmpMaNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with MA-Net and a backbone from the smp package.',
        'A Contour Proposal Network that uses MA-Net and a backbone from the smp package.',
        'cd.models.SmpMaNet'
    )


@register
class CpnSmpUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        update_dict_(backbone_kwargs, dict(model_name=kwargs.get('model_name')))
        super().__init__(
            backbone=SmpUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with a U-Net and a backbone from the smp package.',
        'A Contour Proposal Network that uses a U-Net and a backbone from the smp package.',
        'cd.models.SmpUNet'
    )


@register
class CpnTimmUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        update_dict_(backbone_kwargs, dict(model_name=kwargs.get('model_name')))
        super().__init__(
            backbone=TimmUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with a U-Net and a backbone from the timm package.',
        'A Contour Proposal Network that uses a U-Net and a backbone from the timm package.',
        'cd.models.TimmUNet'
    )


@register
class CpnTimmMaNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        update_dict_(backbone_kwargs, dict(model_name=kwargs.get('model_name')))
        super().__init__(
            backbone=TimmMaNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

    __init__.__doc__ = _make_cpn_doc(
        'Contour Proposal Network with MA-Net and a backbone from the timm package.',
        'A Contour Proposal Network that uses a MA-Net and a backbone from the timm package.',
        'cd.models.TimmMaNet'
    )


models_by_name = {
    'cpn_u22': 'cpn_u22'
}


def get_cpn(name):
    return fetch_model(models_by_name[name])
