import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, ExtraFPNBlock
from collections import OrderedDict
from typing import List, Dict, Type, Union, Tuple
from .commons import TwoConvNormRelu, ResBlock, Normalize, _ni_3d
from .resnet import *
from .mobilenetv3 import *
from .convnext import *
from .timmodels import TimmEncoder
from .smp import SmpEncoder
from ..util.util import lookup_nn, get_nd_max_pool, get_nd_conv, get_nd_linear, update_dict_, get_nn
from functools import partial
import numpy as np
from pytorch_lightning.core.mixins import HyperparametersMixin

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


@register
class UNetEncoder(nn.Sequential):
    def __init__(self, in_channels, depth=5, base_channels=64, factor=2, pool=True, block_cls: Type[nn.Module] = None,
                 nd=2):
        """U-Net Encoder.

        Args:
            in_channels: Input channels.
            depth: Model depth.
            base_channels: Base channels.
            factor: Growth factor of base_channels.
            pool: Whether to use max pooling or stride 2 for downsampling.
            block_cls: Block class. Callable as `block_cls(in_channels, out_channels, stride=stride)`.
        """
        if block_cls is None:
            block_cls = partial(TwoConvNormRelu, nd=nd)
        else:
            block_cls = get_nn(block_cls, nd=nd)
        MaxPool = get_nd_max_pool(nd)
        layers = []
        self.out_channels = []
        self.out_strides = list(range(1, depth + 1))
        for i in range(depth):
            in_c = base_channels * int(factor ** (i - 1)) * int(i > 0) + int(i <= 0) * in_channels
            out_c = base_channels * (factor ** i)
            self.out_channels.append(out_c)
            block = block_cls(in_c, out_c, stride=int((not pool and i > 0) + 1))
            if i > 0 and pool:
                block = nn.Sequential(MaxPool(2, stride=2), block)
            layers.append(block)
        super().__init__(*layers)


@register
class GeneralizedUNet(FeaturePyramidNetwork):
    def __init__(
            self,
            in_channels_list,
            out_channels: int,
            block_cls: nn.Module,
            block_kwargs: dict = None,
            final_activation=None,
            interpolate='nearest',
            final_interpolate=None,
            initialize=True,
            keep_features=True,
            bridge_strides=True,
            bridge_block_cls: 'nn.Module' = None,
            bridge_block_kwargs: dict = None,
            secondary_block: 'nn.Module' = None,
            in_strides_list: Union[List[int], Tuple[int]] = None,
            out_channels_list: Union[List[int], Tuple[int]] = None,
            nd=2,
            **kwargs
    ):
        super().__init__([], 0, extra_blocks=kwargs.get('extra_blocks'))
        block_kwargs = {} if block_kwargs is None else block_kwargs
        Conv = get_nd_conv(nd)
        if out_channels_list is None:
            out_channels_list = in_channels_list
        if in_strides_list is None or bridge_strides is False:  # if not provided, act as if it is starting at stride 1
            in_strides_list = [2 ** i for i in range(len(in_channels_list))]

        # Optionally bridge stride gaps
        self.bridges = np.log2(in_strides_list[0])
        assert self.bridges % 1 == 0
        self.bridges = int(self.bridges)
        if bridge_block_cls is None:
            bridge_block_cls = partial(TwoConvNormRelu, bias=False)
        else:
            bridge_block_cls = get_nn(bridge_block_cls, nd=nd)
        bridge_block_kwargs = {} if bridge_block_kwargs is None else bridge_block_kwargs
        update_dict_(bridge_block_kwargs, block_kwargs, ('activation', 'norm_layer'))
        if self.bridges:
            num = len(in_channels_list)
            for _ in range(self.bridges):
                in_channels_list = (0,) + tuple(in_channels_list)
                if len(out_channels_list) < num + self.bridges - 1:
                    # out_channels_list = (2 ** int(np.log2(out_channels_list[0]) - 1e-8),) + tuple(out_channels_list)
                    out_channels_list = (out_channels_list[0],) + tuple(out_channels_list)

        # Build decoder
        self.cat_order = kwargs.get('cat_order', 0)
        assert self.cat_order in (0, 1)
        self.block_channel_reduction = kwargs.get('block_channel_reduction', False)  # whether block reduces in_channels
        self.block_interpolate = kwargs.get('block_interpolate', False)  # whether block handles interpolation
        self.block_cat = kwargs.get('block_cat', False)  # whether block handles cat
        self.bridge_block_interpolate = kwargs.get('bridge_block_interpolate', False)  # whether block handles interpol.
        self.apply_cat = {}
        self.has_lat = {}
        len_in_channels_list = len(in_channels_list)
        for i in range(len_in_channels_list):
            # Inner conv
            if i > 0:
                inner_ouc = out_channels_list[i - 1]
                inner_inc = out_channels_list[i] if i < len_in_channels_list - 1 else in_channels_list[i]
                if not self.block_channel_reduction and inner_inc > 0 and inner_ouc < inner_inc:
                    inner = Conv(inner_inc, inner_ouc, 1)
                else:
                    inner = nn.Identity()
                self.inner_blocks.append(inner)

            if i < len_in_channels_list - 1:
                # Layer block channels
                lat = in_channels_list[i]
                if self.block_channel_reduction:
                    inc = out_channels_list[i + 1] if i < len_in_channels_list - 2 else in_channels_list[i + 1]
                else:
                    inc = min(out_channels_list[i:i + 2])
                ouc = out_channels_list[i]

                # Build decoder block
                self.apply_cat[i] = False
                self.has_lat[i] = has_lat = lat > 0
                cls, kw = block_cls, block_kwargs
                if not has_lat:  # bridge block
                    self.has_lat[i] = False
                    cls, kw = bridge_block_cls, bridge_block_kwargs
                    inp = inc,
                elif self.block_cat:  # block_cls handles merging
                    inp = inc, lat
                else:  # normal cat
                    self.apply_cat[i] = True
                    inp = inc + lat,
                layer_block = cls(*inp, ouc, nd=nd, **kw)
                if secondary_block is not None:  # must be preconfigured and not change channels
                    layer_block = nn.Sequential(layer_block, secondary_block(ouc, nd=nd))
                self.layer_blocks.append(layer_block)

        self.depth = len(self.layer_blocks)
        self.interpolate = interpolate

        self.keep_features = keep_features
        self.features_prefix = 'encoder'
        self.out_layer = Conv(out_channels_list[0], out_channels, 1) if out_channels > 0 else None
        self.nd = nd
        self.final_interpolate = final_interpolate
        if self.final_interpolate is None:
            self.final_interpolate = get_nd_linear(nd)
        self.final_activation = None if final_activation is None else lookup_nn(final_activation)
        self.out_channels_list = out_channels_list
        self.out_channels = out_channels if out_channels else out_channels_list

        if initialize:
            for m in self.modules():
                if isinstance(m, Conv):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, Tensor], size: List[int]) -> Union[Dict[str, Tensor], Tensor]:
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
        features = x
        names = list(x.keys())
        x = list(x.values())
        last_inner = x[-1]
        results = [last_inner]
        kw = {} if self.interpolate == 'nearest' else {'align_corners': False}
        for i in range(self.depth - 1, -1, -1):
            lateral = lateral_size = None
            if self.has_lat[i]:
                lateral = x[i - self.bridges]
                lateral_size = lateral.shape[2:]
            inner_top_down = last_inner
            if self.interpolate and ((not self.block_interpolate and lateral is not None) or (
                    not self.bridge_block_interpolate and lateral is None)):
                inner_top_down = F.interpolate(  # TODO: scale factor entails shape assumption
                    inner_top_down, **(dict(scale_factor=2) if lateral_size is None else dict(size=lateral_size)),
                    mode=self.interpolate, **kw)
            inner_top_down = self.get_result_from_inner_blocks(inner_top_down, i)
            if self.apply_cat[i]:
                if self.cat_order == 0:
                    cat = lateral, inner_top_down
                else:
                    cat = inner_top_down, lateral
                layer_block_inputs = torch.cat(cat, 1)
            elif lateral is None:
                layer_block_inputs = inner_top_down
            else:
                layer_block_inputs = inner_top_down, lateral
            last_inner = self.get_result_from_layer_blocks(layer_block_inputs, i)
            results.insert(0, last_inner)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        if size is None:
            final = results[0]
        else:
            final = F.interpolate(last_inner, size=size, mode=self.final_interpolate, align_corners=False)
        if self.out_layer is not None:
            final = self.out_layer(final)
        if self.final_activation is not None:
            final = self.final_activation(final)
        if self.out_layer is not None:
            return final
        results.insert(0, final)
        names.insert(0, 'out')
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        if self.keep_features:
            out.update(OrderedDict([('.'.join([self.features_prefix, k]), v) for k, v in features.items()]))
        return out


@register
class BackboneAsUNet(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, block, block_kwargs: dict = None,
                 final_activation=None, interpolate='nearest', ilg=None, nd=2, in_strides_list=None, **kwargs):
        super(BackboneAsUNet, self).__init__()
        if ilg is None:
            ilg = isinstance(backbone, nn.Sequential)
        if block is None:
            block = TwoConvNormRelu  # it's called with nd
        else:
            block = get_nn(block, nd=nd)
        self.nd = nd
        pretrained_cfg = backbone.__dict__.get('pretrained_cfg', {})
        if kwargs.pop('normalize', True):
            self.normalize = Normalize(mean=kwargs.get('inputs_mean', pretrained_cfg.get('mean', 0.)),
                                       std=kwargs.get('inputs_std', pretrained_cfg.get('std', 1.)),
                                       assert_range=kwargs.get('assert_range', (0., 1.)))
        else:
            self.normalize = None
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) if ilg else backbone

        self.intermediate_blocks = kwargs.get('intermediate_blocks')
        if self.intermediate_blocks is not None:
            in_channels_list = in_channels_list + type(in_channels_list)(self.intermediate_blocks.out_channels)
            if in_strides_list is not None:
                in_strides_list = in_strides_list + type(in_strides_list)(
                    [i * in_strides_list[-1] for i in self.intermediate_blocks.out_strides])

        self.unet = GeneralizedUNet(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block_cls=block,
            block_kwargs=block_kwargs,
            # extra_blocks=LastLevelMaxPool(),
            final_activation=final_activation,
            interpolate=interpolate,
            in_strides_list=in_strides_list,
            nd=nd,
            **kwargs
        )
        self.out_channels = list(self.unet.out_channels_list)  # list(in_channels_list)
        # self.out_strides = kwargs.get('in_stride_list')
        self.nd = nd

    def forward(self, inputs):
        x = inputs
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.body(x)
        if self.intermediate_blocks is not None:
            x = self.intermediate_blocks(x)
        x = self.unet(x, size=inputs.shape[-self.nd:])
        return x


@register
class ExtraUNetBlock(ExtraFPNBlock):
    def __init__(self, out_channels: Tuple[int], out_strides: Tuple[int]):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides

    def forward(
            self,
            results: List[Tensor],
            x: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


@register
class IntermediateUNetBlock(nn.Module):
    def __init__(self, out_channels: Tuple[int], out_strides: Tuple[int]):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides

    def forward(
            self,
            x: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        pass


@register
class UNet(BackboneAsUNet, HyperparametersMixin):
    def __init__(self, backbone, out_channels: int, return_layers: dict = None,
                 block: Type[nn.Module] = None, block_kwargs: dict = None, final_activation=None,
                 interpolate='nearest', nd=2, **kwargs):
        """U-Net.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            backbone: Backbone instance.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            return_layers: Return layers used to extract layer features from backbone.
                Dictionary like `{backbone_layer_name: out_name}`.
                Note that this influences how outputs are computed, as the input for the upsampling
                is gathered by `IntermediateLayerGetter` based on given dict keys.
            block: Main block. Default: ``TwoConvNormRelu``.
            block_kwargs: Block keyword arguments.
            final_activation: Final activation function.
            interpolate: Interpolation.
            nd: Spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if block is None:
            block = partial(TwoConvNormRelu, nd=nd)
        else:
            block = get_nn(block, nd=nd)
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        in_channels_list = list(backbone.out_channels)
        in_strides_list = backbone.__dict__.get('out_strides')
        extra_blocks = kwargs.get('extra_blocks')
        if extra_blocks is not None:
            in_channels_list = in_channels_list + type(in_channels_list)(extra_blocks.out_channels)
            if in_strides_list is not None:
                in_strides_list = in_strides_list + type(in_strides_list)(
                    [i * in_strides_list[-1] for i in extra_blocks.out_strides])
        super().__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block=block,
            block_kwargs=block_kwargs,
            final_activation=final_activation if out_channels else None,
            interpolate=interpolate,
            nd=nd,
            in_strides_list=in_strides_list,
            **kwargs
        )


def _ni_pretrained(pretrained):
    if pretrained:
        raise NotImplementedError('The `pretrained` option is not yet available for this model.')


def _default_unet_kwargs(backbone_kwargs, pretrained=False):
    _ni_pretrained(pretrained)
    kw = dict()
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class U22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


@register
class ResUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Residual U-Net.

        U-Net with residual blocks.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``ResBlock``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        if block_cls is None:
            block_cls = partial(ResBlock, nd=nd)
        else:
            block_cls = get_nn(block_cls, nd=nd)
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


@register
class SlimU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels reduce by half.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=32, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


@register
class WideU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels doubled.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=128, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


@register
class U17(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 17.

        U-Net with 17 convolutions on 4 feature resolutions (1, 1/2, 1/4, 1/8) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=4, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


@register
class U12(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 12.

        U-Net with 12 convolutions on 3 feature resolutions (1, 1/2, 1/4) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=3, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class ResNet18UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """ResNet 18 U-Net.

        A U-Net with ResNet 18 encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(ResNet18(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)


@register
class ResNet34UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet34(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 34')


@register
class ResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet50(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 50')


@register
class ResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet101(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 101')


@register
class ResNet152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet152(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 152')


@register
class ResNeXt50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt50_32x4d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 50')


@register
class ResNeXt101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt101_32x8d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 101')


@register
class ResNeXt152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt152_32x8d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 152')


@register
class WideResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            WideResNet50_2(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 50')


@register
class WideResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            WideResNet101_2(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 101')


@register
class MobileNetV3SmallUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        _ni_pretrained(pretrained)
        _ni_3d(nd)
        super().__init__(MobileNetV3Small(in_channels, **(backbone_kwargs or {})), out_channels,
                         final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Small MobileNet V3')


@register
class MobileNetV3LargeUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        _ni_pretrained(pretrained)
        _ni_3d(nd)
        super().__init__(MobileNetV3Large(in_channels, **(backbone_kwargs or {})), out_channels,
                         final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Large MobileNet V3')


def _default_convnext_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class ConvNeXtSmallUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtSmall(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Small')


@register
class ConvNeXtLargeUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtLarge(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Large')


@register
class ConvNeXtBaseUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtBase(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Base')


@register
class ConvNeXtTinyUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtTiny(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Tiny')


def _default_timm_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class TimmUNet(UNet):
    def __init__(self, in_channels, out_channels, model_name, final_activation=None, backbone_kwargs=None,
                 pretrained=True, block_cls=None, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(TimmEncoder(model_name=model_name, in_channels=in_channels,
                                     **_default_timm_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'TimmEncoder')


def _default_smp_kwargs(backbone_kwargs, pretrained=False):
    if pretrained is True:
        pretrained = 'imagenet'
    elif pretrained is False:
        pretrained = None
    kw = dict(weights=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


@register
class SmpUNet(UNet):
    def __init__(self, in_channels, out_channels, model_name, final_activation=None, backbone_kwargs=None,
                 pretrained=True, block_cls=None, nd=2, **kwargs):
        _ni_3d(nd)
        super().__init__(SmpEncoder(model_name=model_name, in_channels=in_channels,
                                    **_default_smp_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'SmpEncoder')
