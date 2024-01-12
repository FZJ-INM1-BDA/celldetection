import torch
import torch.nn as nn
from torch.nn.functional import softmax
from typing import Optional, Union, List, Type, Dict
from torch import Tensor
from torch.nn import functional as F
from functools import partial
from ..util.util import lookup_nn, dict2model
from .unet import UNet, IntermediateUNetBlock
from .commons import ConvNormRelu, SqueezeExcitation as SE, _ni_3d
from .smp import SmpEncoder
from .timmodels import TimmEncoder

__all__ = [
    'MaNet', 'MultiscaleFusionAttention', 'PositionWiseAttention',
    'SmpMaNet', 'TimmMaNet'
]


class SqueezeExcitation(SE):
    def forward(self, inputs):
        scale = super(SE, self).forward(inputs)
        return scale


class PositionWiseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64, kernel_size=3, padding=1, beta=False, nd=2):
        super().__init__()
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        self.beta = nn.Parameter(torch.zeros(1)) if beta else 1.
        if in_channels != out_channels:
            self.in_conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.in_conv = None
        self.proj_b, self.proj_a = [Conv(out_channels, mid_channels, kernel_size=1) for _ in range(2)]
        self.proj = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.out_conv = Conv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = inputs if self.in_conv is None else self.in_conv(inputs)
        a = self.proj_a(x).flatten(2)
        b = self.proj_b(x).flatten(2)  # Tensor[n, c', hw]
        p = torch.matmul(a.permute(0, 2, 1), b)  # Tensor[n, hw, hw]
        p = softmax(p.flatten(1), dim=1).view(p.shape)  # Tensor[n, hw, hw]
        c = self.proj(x).flatten(2)  # Tensor[n, c, hw]
        out = torch.matmul(p, c.permute(0, 2, 1)).view(*c.shape[:2], *inputs.shape[2:])  # T[n, c, hw] -> T[n, c, h, w]
        out = self.out_conv(self.beta * out + x)
        return out


class PAB(IntermediateUNetBlock):
    def __init__(self, in_channels, out_channels, mid_channels=64, kernel_size=3, padding=1, nd=2, replace=False):
        kwargs = dict(out_channels=(out_channels,), out_strides=(1,))
        if replace:
            kwargs = dict(out_channels=(), out_strides=())
        super().__init__(**kwargs)
        self.module = PositionWiseAttention(in_channels=in_channels, out_channels=out_channels,
                                            mid_channels=mid_channels, kernel_size=kernel_size, padding=padding, nd=nd)
        self.replace = replace

    def forward(
            self,
            x: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        in_key = list(x.keys())[-1]
        out_key = in_key if self.replace else str(len(x))
        x[out_key] = self.module(x[in_key])
        return x


class MultiscaleFusionAttention(nn.Module):
    def __init__(
            self,
            in_channels,  # main
            in_channels2,  # lateral
            out_channels,
            norm_layer='BatchNorm2d',
            activation='relu',
            compression=16,
            interpolation=None,
            nd=2,
    ):
        super().__init__()
        kw = dict(activation=activation, norm_layer=norm_layer, nd=nd, bias=False)
        self.in_block = nn.Sequential(
            ConvNormRelu(in_channels, in_channels, **kw),
            ConvNormRelu(in_channels, in_channels2, kernel_size=1, padding=0, **kw),
        )
        self.se_high = SqueezeExcitation(in_channels2, compression=compression, activation=activation, nd=nd)
        self.se_low = SqueezeExcitation(in_channels2, compression=compression, activation=activation, nd=nd)
        self.out_block = nn.Sequential(
            ConvNormRelu(in_channels2 + in_channels2, out_channels, **kw),
            ConvNormRelu(out_channels, out_channels, **kw),
        )
        if interpolation is True:
            interpolation = 'nearest'
        elif interpolation is False:
            interpolation = None
        self.interpolation = interpolation

    def forward(self, x, x2=None):
        if isinstance(x, (tuple, list)):
            assert x2 is None
            x, x2 = x
        x = self.in_block(x)
        if self.interpolation is not None:
            x = F.interpolate(x, x2.shape[2:], mode=self.interpolation)
        if x2 is not None:
            a = self.se_high(x)  # main
            b = self.se_low(x2)  # lateral
            x = x * (a + b)  # scaled x
            x = torch.cat((x, x2), 1)
        return self.out_block(x)


class MaNet(UNet):
    def __init__(self, backbone, out_channels: int = 0, pab_channels=64,
                 block: Type[nn.Module] = None, block_kwargs: dict = None, final_activation=None,
                 interpolate='nearest', nd=2, **kwargs):
        """Multi-Scale Attention Network.

        A U-Net variant using a generic encoder and a special decoder that includes a
        Position-wise Attention Block (PAB) and several Multi-scale Fusion Attention Blocks (MFAB).

        References:
            - https://ieeexplore.ieee.org/document/9201310

        Args:
            backbone: Backbone instance.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            pab_channels: Channels of the Position-wise Attention Block (PAB).
            block: Main block. Default: Multi-scale Fusion Attention Block (MFAB).
            block_kwargs: Block keyword arguments.
            final_activation: Final activation function.
            interpolate: Interpolation.
            nd: Spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(backbone, dict):
            backbone = dict2model(backbone)

        oc = backbone.out_channels
        intermediate_blocks = None
        if pab_channels:
            intermediate_blocks = PAB(oc[-1], oc[-1], mid_channels=pab_channels, nd=nd,
                                      replace=True, **kwargs.get('pwa_kwargs', {}))
        kwargs['block_interpolate'] = block_interpolate = kwargs.get('block_interpolate', True)
        if block is None:
            block = partial(MultiscaleFusionAttention, interpolation=block_interpolate if block_interpolate else None)
            kwargs['block_cat'] = kwargs.get('block_cat', True)
        super().__init__(backbone=backbone, out_channels=out_channels, block=block,
                         block_kwargs=block_kwargs, final_activation=final_activation, interpolate=interpolate,
                         nd=nd, intermediate_blocks=intermediate_blocks, **kwargs)


def _default_timm_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


class TimmMaNet(MaNet):
    def __init__(self, in_channels, out_channels, model_name, final_activation=None, backbone_kwargs=None,
                 pretrained=True, block_cls=None, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(TimmEncoder(model_name=model_name, in_channels=in_channels,
                                     **_default_timm_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)
        self.save_hyperparameters()


def _default_smp_kwargs(backbone_kwargs, pretrained=False):
    if pretrained is True:
        pretrained = 'imagenet'
    elif pretrained is False:
        pretrained = None
    kw = dict(weights=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


class SmpMaNet(MaNet):
    def __init__(self, in_channels, out_channels, model_name, final_activation=None, backbone_kwargs=None,
                 pretrained=True, block_cls=None, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(SmpEncoder(model_name=model_name, in_channels=in_channels,
                                    **_default_smp_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)
        self.save_hyperparameters()
