import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.densenet import _DenseLayer, _DenseBlock, _Transition
from torchvision.models import densenet
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
import numpy as np
from ..util.util import get_nn, lookup_nn
from .ppm import append_pyramid_pooling_
from .resnet import _apply_mapping_rules

__all__ = ['DenseNet', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']

default_model_urls = {
    'DenseNet121': densenet.DenseNet121_Weights.IMAGENET1K_V1.url,
    'DenseNet161': densenet.DenseNet161_Weights.IMAGENET1K_V1.url,
    'DenseNet169': densenet.DenseNet169_Weights.IMAGENET1K_V1.url,
    'DenseNet201': densenet.DenseNet201_Weights.IMAGENET1K_V1.url,
}


def map_state_dict(in_channels, state_dict, fused_initial):
    mapping = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            k = k[len('features.'):]
        if k.startswith('conv0.') and v.data.shape[1] != in_channels:  # initial layer, img channels might differ
            v.data = F.interpolate(v.data[None], (in_channels,) + v.data.shape[2:]).squeeze(0)
        if fused_initial:
            rules = {'conv0': '0.0', 'norm0': '0.1', 'denseblock1': '0.4.block'}
        else:
            rules = {'conv0': '0.0', 'norm0': '0.1'}

        k = _apply_mapping_rules(k, rules)

        if k.startswith('denseblock'):
            sp = k.split('.')
            ki = int(sp[0].replace('denseblock', ''))
            if fused_initial:
                k = f'{ki - 1}.block.' + '.'.join(sp[1:])
            else:
                k = f'{ki}.block.' + '.'.join(sp[1:])
        if k.startswith('transition'):
            k = k[len('transition'):]
            sp = k.split('.')
            ki = int(sp[0])
            if not fused_initial:
                ki += 1
            k = '.'.join([str(ki), 'transition'] + sp[1:])
        if not fused_initial:
            k = _apply_mapping_rules(k, {'1.block.denselayer': '1.1.block.denselayer'})

        mapping[k] = v
    return mapping


class DenseLayer(_DenseLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            drop_rate=0.,
            memory_efficient=False,
            bias=False,
            mid_channels=None  # typically bn_size * growth_rate
    ):
        super(_DenseLayer, self).__init__()
        norm = get_nn(norm_layer, nd=nd)
        conv = get_nn(nn.Conv2d, nd=nd)

        self.memory_efficient = memory_efficient
        self.drop_rate = drop_rate

        if mid_channels is None:
            mid_channels = out_channels

        self.norm1 = norm(in_channels)
        self.relu1 = lookup_nn(activation, call=True, inplace=True)
        self.conv1 = conv(in_channels, mid_channels, kernel_size=1, bias=bias)

        self.norm2 = norm(mid_channels)
        self.relu2 = lookup_nn(activation, call=True, inplace=True)
        self.conv2 = conv(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                          bias=bias)


class DenseBlock(_DenseBlock):
    def __init__(
            self,
            in_channels,
            num_layers,
            bn_size: int,
            growth_rate: int,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            drop_rate=0.,
            memory_efficient=False,
            bias=False,

    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                norm_layer=norm_layer,
                activation=activation,
                nd=nd,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                bias=bias,
                mid_channels=bn_size * growth_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class Transition(_Transition):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            bias=False,
            nd=2,
    ) -> None:
        super(_Transition, self).__init__()
        norm = get_nn(norm_layer, nd=nd)
        conv = get_nn(nn.Conv2d, nd=nd)
        pool = get_nn(nn.AvgPool2d, nd=nd)

        self.norm = norm(in_channels)
        self.relu = lookup_nn(activation, call=True, inplace=True)
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.pool = pool(kernel_size=2, stride=2)


class DenseNet(nn.Sequential):  # Adaptation of torchvision.models.densenet.DenseNet for compatibility and customization
    def __init__(
            self,
            in_channels=3,
            out_channels=0,
            growth_rate: int = 32,
            layers=(6, 12, 24, 16),
            base_channels=64,
            bn_size: int = 4,
            drop_rate: float = 0,
            memory_efficient: bool = False,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            bias=False,
            pretrained=False,
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            fused_initial=True,
            secondary_block=None,
    ) -> None:
        norm = get_nn(norm_layer, nd=nd)
        conv = get_nn(nn.Conv2d, nd=nd)

        assert len(layers)

        # Each denseblock
        self.out_channels = (base_channels,) * (not fused_initial)
        self.out_strides = tuple(2 ** np.arange(1 + fused_initial, len(layers) + 2))
        body = []
        num_features = base_channels
        trans = None
        for i, num_layers in enumerate(layers):
            block = DenseBlock(
                in_channels=num_features,
                num_layers=num_layers,
                bn_size=bn_size,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                norm_layer=norm_layer,
                activation=activation,
                nd=nd,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                bias=bias,
            )
            num_features = num_features + num_layers * growth_rate
            self.out_channels += (num_features,)
            drop_in = {}
            if secondary_block is not None:
                drop_in['extra'] = secondary_block(num_features, nd=nd)  # must be preconfigured and not change channels
            if trans is None:
                body.append(nn.Sequential(OrderedDict(block=block, **drop_in)))
            else:
                body.append(nn.Sequential(OrderedDict(transition=trans, block=block, **drop_in)))

            if i != len(layers) - 1:
                trans = Transition(
                    in_channels=num_features, out_channels=num_features // 2, nd=nd, bias=bias,
                    activation=activation, norm_layer=norm
                )
                num_features = num_features // 2
            else:
                trans = None

        # Initial
        initial = [
            conv(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            norm(base_channels),
            lookup_nn(activation, call=True, inplace=True),
        ]
        pool = get_nn(nn.MaxPool2d, nd=nd)(kernel_size=3, stride=2, padding=1)
        if fused_initial:
            initial += [pool, body[0]]
        else:
            body[0] = nn.Sequential(pool, body[0])
        initial = nn.Sequential(*initial)

        # Putting everything together
        components = [initial] + list(body[1:] if fused_initial else body)
        super().__init__(*components)

        # Linear layer
        if out_channels:
            self.classifier = nn.Sequential(
                norm(num_features),
                lookup_nn(activation, call=True, inplace=True),
                get_nn(nn.AdaptiveAvgPool2d, nd=nd)((1, 1)),
                nn.Flatten(),
                nn.Linear(num_features, out_channels)
            )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, conv):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if pretrained:
            if isinstance(pretrained, str):
                state_dict = load_state_dict_from_url(pretrained)
                if '.pytorch.org' in pretrained:
                    state_dict = map_state_dict(in_channels, state_dict, fused_initial=fused_initial)
                self.load_state_dict(state_dict)
            else:
                raise ValueError('There is no default set of weights for this model. '
                                 'Please specify a URL using the `pretrained` argument.')
        if pyramid_pooling:
            pyramid_pooling_kwargs = {} if pyramid_pooling_kwargs is None else pyramid_pooling_kwargs
            append_pyramid_pooling_(self, pyramid_pooling_channels, nd=nd, **pyramid_pooling_kwargs)


DOC_TEMPLATE = """DenseNet-{layer} Model.

        This class implements the DenseNet-{layer} architecture, a variant of the DenseNet model
        that has {layer} layers. It is particularly efficient in terms of parameter usage,
        offering substantial reduction in computational cost while maintaining or improving
        model accuracy. The model is characterized by its dense connectivity pattern,
        with each layer receiving inputs from all preceding layers and passing on its
        own feature-maps to all subsequent layers, enhancing feature propagation and reuse.
        
        References:
            - https://arxiv.org/abs/1608.06993

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            out_channels (int): Number of output classes (e.g., 1000 for ImageNet).
            growth_rate (int): Number of filters to add per dense layer.
            layers (tuple of int): Numbers of layers in each dense block.
            base_channels (int): Number of filters in the first convolution layer.
            bn_size (int): Multiplicative factor for number of bottle neck layers.
            drop_rate (float): Dropout rate.
            memory_efficient (bool): Enables memory-efficient implementation.
            kernel_size (int): Convolution kernel size.
            padding (int): Padding size.
            stride (int): Convolution stride.
            norm_layer: Normalization layer (e.g., BatchNorm2d).
            activation (str): Activation function (e.g., 'relu').
            nd (int): Number of dimensions.
            bias (bool): Whether to use bias.
            pretrained (bool): If True, returns a model pretrained on ImageNet.
            pyramid_pooling (bool): Whether to use pyramid pooling module.
            pyramid_pooling_channels (int): Number of channels for pyramid pooling.
            pyramid_pooling_kwargs (dict or None): Additional arguments for pyramid pooling.
            fused_initial (bool): Whether to use fused initial layer.
        """


class DenseNet121(DenseNet):
    def __init__(
            self,
            in_channels=3,
            out_channels=0,
            growth_rate=32,
            layers=(6, 12, 24, 16),
            base_channels=64,
            bn_size=4,
            drop_rate=0,
            memory_efficient=False,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            bias=False,
            pretrained=False,
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            fused_initial=True
    ) -> None:
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('DenseNet121', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=growth_rate,
            layers=layers,
            base_channels=base_channels,
            bn_size=bn_size,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_layer=norm_layer,
            activation=activation,
            nd=nd,
            bias=bias,
            pretrained=pretrained,
            pyramid_pooling=pyramid_pooling,
            pyramid_pooling_channels=pyramid_pooling_channels,
            pyramid_pooling_kwargs=pyramid_pooling_kwargs,
            fused_initial=fused_initial
        )

    __init__.__doc__ = DOC_TEMPLATE.format(layer=121)


class DenseNet161(DenseNet):
    def __init__(
            self,
            in_channels=3,
            out_channels=1000,
            bn_size=4,
            drop_rate=0,
            memory_efficient=False,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            bias=False,
            pretrained=False,
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            fused_initial=True
    ) -> None:
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('DenseNet161', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=48,
            layers=(6, 12, 36, 24),
            base_channels=96,
            bn_size=bn_size,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_layer=norm_layer,
            activation=activation,
            nd=nd,
            bias=bias,
            pretrained=pretrained,
            pyramid_pooling=pyramid_pooling,
            pyramid_pooling_channels=pyramid_pooling_channels,
            pyramid_pooling_kwargs=pyramid_pooling_kwargs,
            fused_initial=fused_initial
        )

    __init__.__doc__ = DOC_TEMPLATE.format(layer=161)


class DenseNet169(DenseNet):
    def __init__(
            self,
            in_channels=3,
            out_channels=1000,
            bn_size=4,
            drop_rate=0,
            memory_efficient=False,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            bias=False,
            pretrained=False,
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            fused_initial=True
    ) -> None:
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('DenseNet169', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=32,
            layers=(6, 12, 32, 32),
            base_channels=64,
            bn_size=bn_size,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_layer=norm_layer,
            activation=activation,
            nd=nd,
            bias=bias,
            pretrained=pretrained,
            pyramid_pooling=pyramid_pooling,
            pyramid_pooling_channels=pyramid_pooling_channels,
            pyramid_pooling_kwargs=pyramid_pooling_kwargs,
            fused_initial=fused_initial
        )

    __init__.__doc__ = DOC_TEMPLATE.format(layer=169)


class DenseNet201(DenseNet):
    def __init__(
            self,
            in_channels=3,
            out_channels=1000,
            bn_size=4,
            drop_rate=0,
            memory_efficient=False,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation='relu',
            nd=2,
            bias=False,
            pretrained=False,
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            fused_initial=True
    ) -> None:
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('DenseNet201', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=32,
            layers=(6, 12, 48, 32),
            base_channels=64,
            bn_size=bn_size,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_layer=norm_layer,
            activation=activation,
            nd=nd,
            bias=bias,
            pretrained=pretrained,
            pyramid_pooling=pyramid_pooling,
            pyramid_pooling_channels=pyramid_pooling_channels,
            pyramid_pooling_kwargs=pyramid_pooling_kwargs,
            fused_initial=fused_initial
        )

    __init__.__doc__ = DOC_TEMPLATE.format(layer=201)
