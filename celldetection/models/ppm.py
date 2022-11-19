import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torchvision.models.segmentation.deeplabv3 import ASPP
from ..util.util import lookup_nn

__all__ = ['Ppm']


class Ppm(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            scales: Union[list, tuple] = (1, 2, 3, 6),
            kernel_size=1,
            norm='BatchNorm2d',
            activation='relu',
            concatenate=True,
            **kwargs
    ):
        """Pyramid Pooling Module.

        References:
            - https://ieeexplore.ieee.org/document/8100143

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per pyramid scale.
            scales: Pyramid scales. Default: (1, 2, 3, 6).
            kernel_size: Kernel size.
            norm: Normalization.
            activation: Activation.
            concatenate: Whether to concatenate module inputs to pyramid pooling output before returning results.
            **kwargs: Keyword arguments for ``nn.Conv2d``.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.concatenate = concatenate
        self.out_channels = out_channels * len(scales) + in_channels * int(concatenate)
        norm = lookup_nn(norm, call=False)
        activation = lookup_nn(activation, call=False)
        for scale in scales:
            self.blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
                norm(out_channels),
                activation(),
            ))

    def forward(self, x):
        prefix = [x] if self.concatenate else []
        return torch.cat(prefix + [
            F.interpolate(m(x), x.shape[-2:], mode='bilinear', align_corners=False) for m in self.blocks
        ], 1)


def append_pyramid_pooling_(module: nn.Sequential, out_channels, scales=(1, 2, 3, 6), method='ppm', in_channels=None,
                            **kwargs):
    if in_channels is None:
        in_channels = module.out_channels[-1]
    method = method.lower()
    if method == 'ppm':
        assert (out_channels % len(scales)) == 0
        p = Ppm(in_channels, out_channels, scales=scales, **kwargs)
        out_channels = p.out_channels
    elif method == 'aspp':
        scales = sorted(tuple(set(scales) - {1}))
        p = ASPP(in_channels, scales, out_channels, **kwargs)
    else:
        raise ValueError
    module.append(p)
    if hasattr(module, 'out_channels'):
        module.out_channels += (out_channels,)
