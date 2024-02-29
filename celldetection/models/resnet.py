import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet as tvr
from os.path import isfile
from ..util.util import Dict, lookup_nn, get_nd_conv, get_nn, resolve_pretrained
from torch.hub import load_state_dict_from_url
from .ppm import append_pyramid_pooling_
from typing import Type, Union, Optional
from pytorch_lightning.core.mixins import HyperparametersMixin

__all__ = ['get_resnet', 'ResNet50', 'ResNet34', 'ResNet18', 'ResNet152', 'ResNet101', 'WideResNet101_2',
           'WideResNet50_2', 'ResNeXt152_32x8d', 'ResNeXt101_32x8d', 'ResNeXt50_32x4d']

default_model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',  # IMAGENET1K_V1
    'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',  # IMAGENET1K_V1
    'ResNet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',  # IMAGENET1K_V2
    'ResNet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth',  # IMAGENET1K_V2
    'ResNet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth',  # IMAGENET1K_V2
    'ResNeXt50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',  # IMAGENET1K_V2
    'ResNeXt101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',  # IMAGENET1K_V2
    'WideResNet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',  # IMAGENET1K_V2
    'WideResNet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',  # IMAGENET1K_V2
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, kernel_size=3,
            nd=2) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd

    # Calculate padding for 'same' padding
    padding = tuple((ks - 1) * dil // 2 for ks, dil in zip(kernel_size, dilation))

    return get_nd_conv(nd)(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, nd=2) -> nn.Conv2d:
    """1x1 convolution"""
    return get_nd_conv(nd)(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = tvr.BasicBlock.expansion
    forward = tvr.BasicBlock.forward

    def __init__(  # Port from torchvision (to support 3d and add more features)
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer='batchnorm2d',
            kernel_size=3,
            nd=2
    ) -> None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, nd=nd, kernel_size=kernel_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, nd=nd)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


class Bottleneck(nn.Module):
    expansion: int = tvr.Bottleneck.expansion
    forward = tvr.Bottleneck.forward

    def __init__(  # Port from torchvision (to support 3d and add more features)
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer='batchnorm2d',
            kernel_size=3,
            nd=2
    ) -> None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, nd=nd)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, kernel_size=kernel_size, nd=nd)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, nd=nd)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


def _make_layer(  # Port from torchvision (to support 3d)
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        kernel_size: int = 3,
        nd=2,
        secondary_block=None,
        downsample_method=None,
) -> nn.Sequential:
    """

    References:
        - [1] https://arxiv.org/abs/1812.01187.pdf

    Args:
        self:
        block:
        planes:
        blocks:
        stride:
        dilate:
        kernel_size:
        nd:
        secondary_block:
        downsample_method: Downsample method. None: 1x1Conv with stride, Norm (standard ResNet),
            'avg': AvgPool, 1x1Conv, Norm (ResNet-D in [1])

    Returns:

    """
    if secondary_block is not None:
        secondary_block = get_nn(secondary_block, nd=nd)
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
        self.dilation *= stride
        stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
        if downsample_method is None or stride <= 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, nd=nd),
                norm_layer(planes * block.expansion),
            )
        elif downsample_method == 'avg':
            downsample = nn.Sequential(
                get_nn(nn.AvgPool2d, nd=nd)(2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion, nd=nd),
                norm_layer(planes * block.expansion),
            )
        else:
            raise ValueError(f'Unknown downsample_method: {downsample_method}')

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
              kernel_size=kernel_size, nd=nd))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(
            self.inplanes,
            planes,
            groups=self.groups,
            base_width=self.base_width,
            dilation=self.dilation,
            norm_layer=norm_layer,
            kernel_size=kernel_size,
            nd=nd,
        ))
    if secondary_block is not None:
        layers.append(secondary_block(self.inplanes, nd=nd))  # must be preconfigured and not change channels
    return nn.Sequential(*layers)


def make_res_layer(block, inplanes, planes, blocks, norm_layer=nn.BatchNorm2d, base_width=64, groups=1, stride=1,
                   dilation=1, dilate=False, nd=2, secondary_block=None, downsample_method=None, kernel_size=3,
                   **kwargs) -> nn.Module:
    """

    Args:
        block: Module class. For example `BasicBlock` or `Bottleneck`.
        inplanes: Number of in planes
        planes: Number of planes
        blocks: Number of blocks
        norm_layer: Norm Module class
        base_width: Base width. Acts as a factor of the bottleneck size of the Bottleneck block and is used with groups.
        groups:
        stride:
        dilation:
        dilate:
        nd:
        secondary_block:
        downsample_method:
        kernel_size:
        kwargs:

    Returns:

    """
    norm_layer = lookup_nn(norm_layer, nd=nd, call=False)
    d = Dict(inplanes=inplanes, _norm_layer=norm_layer, base_width=base_width,
             groups=groups, dilation=dilation)  # almost a ResNet

    return _make_layer(self=d, block=block, planes=planes, blocks=blocks, stride=stride, dilate=dilate, nd=nd,
                       secondary_block=secondary_block, downsample_method=downsample_method, kernel_size=kernel_size)


def _apply_mapping_rules(key, rules: dict):
    for prefix, repl in rules.items():
        if key.startswith(prefix):
            key = key.replace(prefix, repl, 1)
    return key


def map_state_dict(in_channels, state_dict, fused_initial):
    """Map state dict.

    Map state dict from torchvision format to celldetection format.

    Args:
        in_channels: Number of input channels.
        state_dict: State dict.
        fused_initial:

    Returns:
        State dict in celldetection format.
    """
    mapping = {}
    for k, v in state_dict.items():
        if 'fc' in k:  # skip fc
            continue
        if k.startswith('conv1.') and v.data.shape[1] != in_channels:  # initial layer, img channels might differ
            v.data = F.interpolate(v.data[None], (in_channels,) + v.data.shape[-2:]).squeeze(0)
        if fused_initial:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '0.4.', 'layer2.': '1.', 'layer3.': '2.',
                     'layer4.': '3.', 'layer5.': '4.'}
        else:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '1.1.', 'layer2.': '2.', 'layer3.': '3.',
                     'layer4.': '4.', 'layer5.': '5.'}
        mapping[_apply_mapping_rules(k, rules)] = v
    return mapping


class ResNet(nn.Sequential, HyperparametersMixin):
    def __init__(self, in_channels, *body: nn.Module, initial_strides=2, base_channel=64, initial_pooling=True,
                 final_layer=None, final_activation=None, fused_initial=True, pretrained=False,
                 pyramid_pooling=False, pyramid_pooling_channels=64, pyramid_pooling_kwargs=None, nd=2, **kwargs):
        assert len(body) > 0
        body = list(body)
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        MaxPool = lookup_nn(nn.MaxPool2d, nd=nd, call=False)
        initial = [
            Conv(in_channels, base_channel, 7, padding=3, bias=False, stride=initial_strides),
            Norm(base_channel),
            nn.ReLU(inplace=True)
        ]
        pool = MaxPool(kernel_size=3, stride=2, padding=1) if initial_pooling else nn.Identity()
        if fused_initial:
            initial += [pool, body[0]]
        else:
            body[0] = nn.Sequential(pool, body[0])
        initial = nn.Sequential(*initial)
        components = [initial] + list(body[1:] if fused_initial else body)
        if final_layer is not None:
            components += [final_layer]
        if final_activation is not None:
            components += [lookup_nn(final_activation)]
        super(ResNet, self).__init__(*components)
        if pretrained:
            state_dict = resolve_pretrained(pretrained, in_channels=in_channels, fused_initial=fused_initial,
                                            state_dict_mapper=map_state_dict)
            self.load_state_dict(state_dict, strict=kwargs.get('pretrained_strict', True))
        if pyramid_pooling:
            pyramid_pooling_kwargs = {} if pyramid_pooling_kwargs is None else pyramid_pooling_kwargs
            append_pyramid_pooling_(self, pyramid_pooling_channels, nd=nd, **pyramid_pooling_kwargs)


class VanillaResNet(ResNet):
    def __init__(self, in_channels, out_channels=0, layers=(2, 2, 2, 2), base_channel=64, fused_initial=True,
                 kernel_size=3, per_layer_kernel_sizes: dict = None, nd=2, **kwargs):
        if per_layer_kernel_sizes is None:
            per_layer_kernel_sizes = {}
        if isinstance(per_layer_kernel_sizes, (tuple, list)):
            per_layer_kernel_sizes = {i: v for i, v in enumerate(per_layer_kernel_sizes)}
        self.save_hyperparameters()
        self.out_channels = oc = (base_channel, base_channel * 2, base_channel * 4, base_channel * 8)
        self.out_strides = (4, 8, 16, 32)
        if out_channels and 'final_layer' not in kwargs.keys():
            kwargs['final_layer'] = get_nd_conv(nd)(self.out_channels[-1], out_channels, 1)

        super(VanillaResNet, self).__init__(
            in_channels,
            make_res_layer(BasicBlock, base_channel, oc[0], layers[0], stride=1, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(0, kernel_size), **kwargs),
            make_res_layer(BasicBlock, oc[0], oc[1], layers[1], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(1, kernel_size), **kwargs),
            make_res_layer(BasicBlock, oc[1], oc[2], layers[2], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(2, kernel_size), **kwargs),
            make_res_layer(BasicBlock, oc[2], oc[3], layers[3], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(3, kernel_size), **kwargs),
            base_channel=base_channel, fused_initial=fused_initial, nd=nd, **kwargs
        )
        if not fused_initial:
            self.out_channels = (base_channel,) + self.out_channels
            self.out_strides = (2,) + self.out_strides


class ResNet18(VanillaResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        """ResNet 18.

        ResNet 18 encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_layer: Final output layer. Default: 1x1 Conv2d if ``out_channels >= 1``, ``None`` otherwise.
            final_activation: Final activation layer (e.g. ``nn.ReLU`` or ``'relu'``). Default: ``None``.
            pretrained: Whether to load weights from a pretrained network. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet18']
        super(ResNet18, self).__init__(in_channels, out_channels=out_channels, layers=(2, 2, 2, 2),
                                       pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()


class ResNet34(VanillaResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet34']
        super(ResNet34, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3),
                                       pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 34')


class BottleResNet(ResNet):
    def __init__(self, in_channels, out_channels=0, layers=(3, 4, 6, 3), base_channel=64, fused_initial=True,
                 kernel_size=3, per_layer_kernel_sizes: dict = None, nd=2, **kwargs):
        if per_layer_kernel_sizes is None:
            per_layer_kernel_sizes = {}
        if isinstance(per_layer_kernel_sizes, (tuple, list)):
            per_layer_kernel_sizes = {i: v for i, v in enumerate(per_layer_kernel_sizes)}
        self.save_hyperparameters()
        ex = Bottleneck.expansion
        self.out_channels = oc = (base_channel * 4, base_channel * 8, base_channel * 16, base_channel * 32)
        self.out_strides = (4, 8, 16, 32)
        if out_channels and 'final_layer' not in kwargs.keys():
            kwargs['final_layer'] = nn.Conv2d(self.out_channels[-1], out_channels, 1)
        super(BottleResNet, self).__init__(
            in_channels,
            make_res_layer(Bottleneck, base_channel, oc[0] // ex, layers[0], stride=1, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(0, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 4, oc[1] // ex, layers[1], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(1, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 8, oc[2] // ex, layers[2], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(2, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 16, oc[3] // ex, layers[3], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(3, kernel_size), **kwargs),
            base_channel=base_channel, fused_initial=fused_initial, nd=nd, **kwargs
        )
        if not fused_initial:
            self.out_channels = (base_channel,) + self.out_channels
            self.out_strides = (2,) + self.out_strides


class ResNet50(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet50']
        super(ResNet50, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3),
                                       pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 50')


class ResNet101(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet101']
        super(ResNet101, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3),
                                        pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 101')


class ResNet152(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet152']
        super(ResNet152, self).__init__(in_channels, out_channels=out_channels, layers=(3, 8, 36, 3),
                                        pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 152')


class ResNeXt50_32x4d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNeXt50_32x4d']
        super(ResNeXt50_32x4d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3), groups=32,
                                              base_width=4, pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 50')


class ResNeXt101_32x8d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNeXt101_32x8d']
        super(ResNeXt101_32x8d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3), groups=32,
                                               base_width=8, pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 101')


class ResNeXt152_32x8d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, nd=2, **kwargs):
        super(ResNeXt152_32x8d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 8, 36, 3), groups=32,
                                               base_width=8, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 152')


class WideResNet50_2(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['WideResNet50_2']
        super(WideResNet50_2, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3),
                                             base_width=64 * 2, pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 50')


class WideResNet101_2(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['WideResNet101_2']
        super(WideResNet101_2, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3),
                                              base_width=64 * 2, pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 101')


models_by_name = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'resnext50_32x4d': ResNeXt50_32x4d,
    'resnext101_32x8d': ResNeXt101_32x8d,
    'resnext152_32x8d': ResNeXt152_32x8d,
    'wideresnet50_2': WideResNet50_2,
    'wideresnet101_2': WideResNet101_2
}


def get_resnet(name, in_channels, **kwargs):
    return models_by_name[name](in_channels=in_channels, **kwargs)
