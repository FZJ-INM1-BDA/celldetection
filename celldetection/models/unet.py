import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from collections import OrderedDict
from typing import List, Dict
from .commons import TwoConvBnRelu
from .resnet import *
from .mobilenetv3 import *

__all__ = [
    'UNetEncoder', 'UNet', 'U12', 'U17', 'U22', 'SlimU22', 'WideU22', 'MobileNetV3Large', 'MobileNetV3Small',
    'ResNet18UNet', 'ResNet34UNet', 'ResNet50UNet', 'ResNet101UNet', 'ResNet152UNet', 'ResNeXt50UNet', 'ResNeXt101UNet',
    'ResNeXt152UNet', 'WideResNet50UNet', 'WideResNet101UNet'
]


class UNetEncoder(nn.Sequential):
    def __init__(self, in_channels, depth=5, base_channels=64, factor=2, pool=True):
        """

        Args:
            in_channels: Input channels.
            depth: Model depth.
            base_channels: Base channels.
            factor: Growth factor of base_channels.
            pool: Whether to use max pooling or stride 2 for downsampling.
        """
        layers = []
        self.out_channels = []
        for i in range(depth):
            in_c = base_channels * int(factor ** (i - 1)) * int(i > 0) + int(i <= 0) * in_channels
            out_c = base_channels * (factor ** i)
            self.out_channels.append(out_c)
            block = TwoConvBnRelu(in_c, out_c, stride=int((not pool and i > 0) + 1))
            if i > 0 and pool:
                block = nn.Sequential(nn.MaxPool2d(2, stride=2), block)
            layers.append(block)
        super().__init__(*layers)


class GeneralizedUNet(FeaturePyramidNetwork):
    def __init__(
            self,
            in_channels_list,
            out_channels: int,
            block: nn.Module
    ):
        super().__init__([], 0)
        for j, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            oc = out_channels if j <= 0 else in_channels_list[j - 1]
            inner_block_module = nn.Identity() if oc <= 0 else nn.Conv2d(in_channels, oc, 1)
            self.inner_blocks.append(inner_block_module)
            if j <= 0 or oc <= 0:
                layer_block_module = nn.Identity()
            else:
                layer_block_module = block(in_channels, oc)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, Tensor], size: List[int]) -> Dict[str, Tensor]:
        """

        Args:
            x: Input dictionary. E.g. {
                    0: Tensor[1, 64, 128, 128]
                    1: Tensor[1, 128, 64, 64]
                    2: Tensor[1, 256, 32, 32]
                    3: Tensor[1, 512, 16, 16]
                }
            size: Desired final output size. If set to None output remains as it is.

        Returns:
            Output dictionary. For each key in `x` a corresponding output is returned; the final output
            has the key `'out'`.
            E.g. {
                out: Tensor[1, 2, 128, 128]
                0: Tensor[1, 64, 128, 128]
                1: Tensor[1, 128, 64, 64]
                2: Tensor[1, 256, 32, 32]
                3: Tensor[1, 512, 16, 16]
            }
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        last_inner = x[-1]
        results = [last_inner]
        idx = -1
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = x[idx]
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")  # adjust size
            inner_top_down = self.get_result_from_inner_blocks(inner_top_down, idx + 1)  # reduce channels
            last_inner = torch.cat((inner_lateral, inner_top_down), 1)  # concat
            last_inner = self.get_result_from_layer_blocks(last_inner, idx + 1)  # apply layer
            results.insert(0, last_inner)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        if size is None:
            final = results[0]
        else:
            final = F.interpolate(last_inner, size=size, mode="bilinear", align_corners=False)
        final = self.get_result_from_inner_blocks(final, idx)
        results.insert(0, final)
        names.insert(0, 'out')
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class BackboneAsUNet(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, block):
        super(BackboneAsUNet, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.unet = GeneralizedUNet(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block=block,
            # extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = list(in_channels_list)

    def forward(self, inputs):
        x = self.body(inputs)
        x = self.unet(x, size=inputs.shape[-2:])
        return x


class UNet(BackboneAsUNet):
    def __init__(self, backbone, out_channels: int, return_layers: dict = None, block: nn.Module = TwoConvBnRelu):
        """
        Examples:
            ```python
            >>> model = UNet(UNetEncoder(in_channels=3), out_channels=2)
            ```

            ```python
            >>> model = UNet(UNetEncoder(in_channels=3, base_channels=16), out_channels=2)
            >>> o = model(torch.rand(1, 3, 256, 256))
            >>> for k, v in o.items():
            ...     print(k, "\t", v.shape)
            out 	 torch.Size([1, 2, 256, 256])
            0 	 torch.Size([1, 16, 256, 256])
            1 	 torch.Size([1, 32, 128, 128])
            2 	 torch.Size([1, 64, 64, 64])
            3 	 torch.Size([1, 128, 32, 32])
            4 	 torch.Size([1, 256, 16, 16])
            ```

        Args:
            backbone: Backbone instance.
            out_channels: Output channels.
            return_layers: Dictionary like `{backbone_layer_name: out_name}`.
                Note that this influences how outputs are computed, as the input for the upsampling
                is gathered by `IntermediateLayerGetter` based on given dict keys.
            block: Module class. Default is `block=TwoConvBnRelu`. Must be callable: block(in_channels, out_channels).
        """
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        super(UNet, self).__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=list(backbone.out_channels),
            out_channels=out_channels,
            block=block
        )


class U22(UNet):
    def __init__(self, in_channels, out_channels):
        """ U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.

        References:
            https://arxiv.org/abs/1505.04597

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
        """
        super().__init__(UNetEncoder(in_channels=in_channels), out_channels=out_channels)


class SlimU22(UNet):
    def __init__(self, in_channels, out_channels):
        """ Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels reduce by half.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
        """
        super().__init__(UNetEncoder(in_channels=in_channels, base_channels=32), out_channels=out_channels)


class WideU22(UNet):
    def __init__(self, in_channels, out_channels):
        """ Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels doubled.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
        """
        super().__init__(UNetEncoder(in_channels=in_channels, base_channels=128), out_channels=out_channels)


class U17(UNet):
    def __init__(self, in_channels, out_channels):
        """ U-Net 17.

        U-Net with 17 convolutions on 4 feature resolutions (1, 1/2, 1/4, 1/8) and one final output layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
        """
        super().__init__(UNetEncoder(in_channels=in_channels, depth=4), out_channels=out_channels)


class U12(UNet):
    def __init__(self, in_channels, out_channels):
        """ U-Net 12.

        U-Net with 17 convolutions on 3 feature resolutions (1, 1/2, 1/4) and one final output layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
        """
        super().__init__(UNetEncoder(in_channels=in_channels, depth=3), out_channels=out_channels)


class ResNet18UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNet18(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNet34UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNet34(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNet50(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNet101(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNet152UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNet152(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNeXt50UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNeXt50_32x4d(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNeXt101UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNeXt101_32x8d(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class ResNeXt152UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(ResNeXt152_32x8d(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class WideResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(WideResNet50_2(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class WideResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(WideResNet101_2(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class MobileNetV3SmallUNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(MobileNetV3Small(in_channels, **(res_kwargs or {})), out_channels, **kwargs)


class MobileNetV3LargeUNet(UNet):
    def __init__(self, in_channels, out_channels, res_kwargs=None, **kwargs):
        super().__init__(MobileNetV3Large(in_channels, **(res_kwargs or {})), out_channels, **kwargs)
