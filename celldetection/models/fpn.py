import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection import backbone_utils
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import feature_pyramid_network
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock as _ExtraFPNBlock
from ..util.util import lookup_nn, dict2model, update_model_hparams_, resolve_model
from .commons import ConvNorm, _ni_3d, Normalize
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50_32x4d, ResNeXt101_32x8d, \
    ResNeXt152_32x8d, WideResNet50_2, WideResNet101_2
from .mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from typing import Optional, Callable, List, Tuple, Dict
from functools import partial
from pytorch_lightning.core.mixins import HyperparametersMixin
from .smp import SmpEncoder
from .timmodels import TimmEncoder
from .convnext import ConvNeXtLarge, ConvNeXtSmall, ConvNeXtBase, ConvNeXtTiny
import copy

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


class ExtraFPNBlock(_ExtraFPNBlock):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def adapt_out_channel_list(self, channel_list):  # must be implemented if extra block changes channels
        return channel_list


class LastLevelMaxPool(ExtraFPNBlock):
    def __init__(self, nd=2):
        """
        This is an adapted class from torchvision to support n-dimensional data.

        References:
            https://github.com/pytorch/vision/blob/d2d448c71b4cb054d160000a0f63eecad7867bdb/torchvision/ops/feature_pyramid_network.py#L207

        Notes:
            This class just applies stride 2 to spatial dimensions, and uses pytorch's max_pool function to do it.
        """
        super().__init__()
        self._fn = lookup_nn('max_pool2d', nd=nd, call=False, src=F)

    def adapt_out_channel_list(self, channel_list):  # must be implemented if extra block changes channels
        return channel_list + channel_list[-1:]

    def forward(
            self,
            x: List[Tensor],
            y: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(self._fn(  # this actually just does x[(...,) + (slice(None, None, 2),) * nd)]
            x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names


class FeaturePyramidNetwork(feature_pyramid_network.FeaturePyramidNetwork):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            block_cls=None,
            block_kwargs: dict = None,
            extra_blocks: Optional['ExtraFPNBlock'] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            nd: int = 2,
    ):
        """Feature Pyramid Network

        Note:
            Model uses output convolutions, mapping decoder features to the returned output.
            If output features are not used (e.g. by prediction heads), the parameters of the output convolutions
            will not contribute to the loss, which is to be avoided and may lead to a `RuntimeError`.

        Args:
            in_channels_list:
            out_channels:
            block_cls:
            block_kwargs:
            extra_blocks:
            norm_layer:
            nd:
        """
        super(feature_pyramid_network.FeaturePyramidNetwork, self).__init__()

        block = partial(ConvNorm, nd=nd) if block_cls is None else block_cls
        block_kwargs = {} if block_kwargs is None else block_kwargs
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = block(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, nd=nd, **block_kwargs
            )
            layer_block_module = block(
                out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, nd=nd, **block_kwargs
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")
        self.extra_blocks = extra_blocks


class BackboneWithFPN(backbone_utils.BackboneWithFPN):
    def __init__(
            self,
            backbone: nn.Module,
            return_layers: Dict[str, str],
            in_channels_list: List[int],
            out_channels: int,
            out_channel_list: List[int],
            extra_blocks: Optional['ExtraFPNBlock'] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ilg=None,
            nd: int = 2,
            **kwargs
    ) -> None:
        super(backbone_utils.BackboneWithFPN, self).__init__()
        if ilg is None:
            ilg = isinstance(backbone, nn.Sequential)
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool(nd=nd)
            if hasattr(extra_blocks, 'adapt_out_channel_list'):
                out_channel_list = extra_blocks.adapt_out_channel_list(out_channel_list)

        pretrained_cfg = backbone.__dict__.get('pretrained_cfg', {})
        if not len(pretrained_cfg) and hasattr(backbone, 'hparams') and isinstance(backbone.hparams, dict):
            pretrained_cfg = backbone.hparams.get('pretrained_cfg', pretrained_cfg)

        if kwargs.pop('normalize', True):
            self.normalize = Normalize(mean=kwargs.get('inputs_mean', pretrained_cfg.get('mean', 0.)),
                                       std=kwargs.get('inputs_std', pretrained_cfg.get('std', 1.)),
                                       assert_range=kwargs.get('assert_range', (0., 1.)))
        else:
            self.normalize = None
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) if ilg else backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
            nd=nd
        )
        # self.out_channels = out_channels
        self.out_channels = out_channel_list

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.body(x)
        x = self.fpn(x)
        return x


@register
class FPN(BackboneWithFPN, HyperparametersMixin):
    def __init__(self, backbone, channels=256, return_layers: dict = None, **kwargs):
        """

        Examples:
            >>> from celldetection.models import ResNet18, FPN
            ... import torch
            >>> model = FPN(ResNet18(in_channels=1))
            >>> for k, v in model(torch.rand(1, 1, 128, 128)).items():
            ...     print(k, "\t", v.shape)
            0 	     torch.Size([1, 256, 32, 32])
            1 	     torch.Size([1, 256, 16, 16])
            2 	     torch.Size([1, 256, 8, 8])
            3 	     torch.Size([1, 256, 4, 4])
            pool 	 torch.Size([1, 256, 2, 2])

        Args:
            backbone: Backbone module
                Note that `backbone.out_channels` must be defined.
            channels: Channels in the upsampling branch.
            return_layers: Dictionary like `{backbone_layer_name: out_name}`.
                Note that this influences how outputs are computed, as the input for the upsampling
                is gathered by `IntermediateLayerGetter` based on given dict keys.
        """
        self.save_hyperparameters(ignore=['backbone'])
        update_model_hparams_(self, resolve=kwargs.pop('resolve_model_hparam', True), backbone=backbone)
        backbone = resolve_model(backbone, **kwargs.pop('backbone_kwargs', {}))
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
            out_channel_list = [channels] * len(list(backbone.out_channels))
        else:
            out_channel_list = [channels] * len(return_layers)
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        super(FPN, self).__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=list(backbone.out_channels),
            out_channels=channels,
            out_channel_list=out_channel_list,  # [channels] * len(layers),
            **kwargs
        )


def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class ResNet18FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(ResNet18(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


@register
class ResNet34FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(ResNet34(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


@register
class ResNet50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(ResNet50(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


@register
class ResNet101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(ResNet101(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


@register
class ResNet152FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(ResNet152(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


@register
class ResNeXt50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ResNeXt50_32x4d(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, **kwargs)


@register
class ResNeXt101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ResNeXt101_32x8d(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, **kwargs)


@register
class ResNeXt152FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ResNeXt152_32x8d(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, **kwargs)


@register
class WideResNet50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            WideResNet50_2(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, **kwargs)


@register
class WideResNet101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            WideResNet101_2(in_channels=in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, **kwargs)


def _default_smp_kwargs(backbone_kwargs, pretrained=False):
    if pretrained is True:
        pretrained = 'imagenet'
    elif pretrained is False:
        pretrained = None
    kw = dict(weights=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class SmpFPN(FPN):
    def __init__(self, in_channels, model_name, fpn_channels=256, backbone_kwargs=None, pretrained=True, **kwargs):
        super().__init__(SmpEncoder(model_name=model_name, in_channels=in_channels,
                                    **_default_smp_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


def _default_timm_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class TimmFPN(FPN):
    def __init__(self, in_channels, model_name, fpn_channels=256, backbone_kwargs=None, pretrained=True, **kwargs):
        super().__init__(TimmEncoder(model_name=model_name, in_channels=in_channels,
                                     **_default_timm_kwargs(backbone_kwargs, pretrained)),
                         channels=fpn_channels, **kwargs)


def _default_convnext_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class ConvNeXtSmallFPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ConvNeXtSmall(in_channels=in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, block_cls=block_cls, **kwargs)


@register
class ConvNeXtLargeFPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ConvNeXtLarge(in_channels=in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, block_cls=block_cls, **kwargs)


@register
class ConvNeXtBaseFPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ConvNeXtBase(in_channels=in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, block_cls=block_cls, **kwargs)


@register
class ConvNeXtTinyFPN(FPN):
    def __init__(self, in_channels, fpn_channels=256, backbone_kwargs=None, pretrained=False, block_cls=None, nd=2,
                 **kwargs):
        super().__init__(
            ConvNeXtTiny(in_channels=in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
            channels=fpn_channels, block_cls=block_cls, **kwargs)


@register
class MobileNetV3SmallFPN(FPN):
    """Feature Pyramid Network with MobileNetV3Small.

    Examples:
        >>> import torch
        >>> from celldetection import models
        >>> model = models.MobileNetV3SmallFPN(in_channels=3)
        >>> out: dict = model(torch.rand(1, 3, 256, 256))
        >>> for k, v in out.items():
        ...     print(k, v.shape)
        0 torch.Size([1, 256, 128, 128])
        1 torch.Size([1, 256, 64, 64])
        2 torch.Size([1, 256, 32, 32])
        3 torch.Size([1, 256, 16, 16])
        4 torch.Size([1, 256, 8, 8])
        pool torch.Size([1, 256, 4, 4])
    """

    def __init__(self, in_channels, fpn_channels=256, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(MobileNetV3Small(in_channels=in_channels, **kwargs), channels=fpn_channels, **kwargs)


@register
class MobileNetV3LargeFPN(FPN):
    """Feature Pyramid Network with MobileNetV3Large.

    Examples:
        >>> import torch
        >>> from celldetection import models
        >>> model = models.MobileNetV3LargeFPN(in_channels=3)
        >>> out: dict = model(torch.rand(1, 3, 256, 256))
        >>> for k, v in out.items():
        ...     print(k, v.shape)
        0 torch.Size([1, 256, 128, 128])
        1 torch.Size([1, 256, 64, 64])
        2 torch.Size([1, 256, 32, 32])
        3 torch.Size([1, 256, 16, 16])
        4 torch.Size([1, 256, 8, 8])
        pool torch.Size([1, 256, 4, 4])
    """

    def __init__(self, in_channels, fpn_channels=256, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(MobileNetV3Large(in_channels=in_channels, **kwargs), channels=fpn_channels, **kwargs)
