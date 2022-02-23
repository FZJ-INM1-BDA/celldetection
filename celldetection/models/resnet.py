import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.resnet import ResNet as RN, Bottleneck, BasicBlock
from ..util.util import Dict, lookup_nn

__all__ = ['get_resnet', 'ResNet50', 'ResNet34', 'ResNet18', 'ResNet152', 'ResNet101', 'WideResNet101_2',
           'WideResNet50_2', 'ResNeXt152_32x8d', 'ResNeXt101_32x8d', 'ResNeXt50_32x4d']


def make_res_layer(block, inplanes, planes, blocks, norm_layer=nn.BatchNorm2d, base_width=64, groups=1, stride=1,
                   dilation=1, dilate=False, **kwargs) -> nn.Module:
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

    Returns:

    """
    d = Dict(inplanes=inplanes, _norm_layer=norm_layer, base_width=base_width,
             groups=groups, dilation=dilation)  # almost a ResNet
    return RN._make_layer(self=d, block=block, planes=planes, blocks=blocks, stride=stride, dilate=dilate)


class ResNet(nn.Sequential):
    def __init__(self, in_channels, *body: nn.Module, initial_strides=2, base_channel=64, initial_pooling=True,
                 final_layer=None, final_activation=None, **kwargs):
        assert len(body) > 0
        initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, 7, padding=3, bias=False, stride=initial_strides),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if initial_pooling else nn.Identity(),
            body[0]
        )
        components = [initial] + list(body[1:])
        if final_layer is not None:
            components += [final_layer]
        if final_activation is not None:
            components += [lookup_nn(final_activation)]
        super(ResNet, self).__init__(*components)


class VanillaResNet(ResNet):
    def __init__(self, in_channels, out_channels=0, layers=(2, 2, 2, 2), base_channel=64, **kwargs):
        self.out_channels = oc = (base_channel, base_channel * 2, base_channel * 4, base_channel * 8)
        if out_channels and 'final_layer' not in kwargs.keys():
            kwargs['final_layer'] = nn.Conv2d(self.out_channels[-1], out_channels, 1)
        super(VanillaResNet, self).__init__(
            in_channels,
            make_res_layer(BasicBlock, base_channel, oc[0], layers[0], stride=1, **kwargs),
            make_res_layer(BasicBlock, oc[0], oc[1], layers[1], stride=2, **kwargs),
            make_res_layer(BasicBlock, oc[1], oc[2], layers[2], stride=2, **kwargs),
            make_res_layer(BasicBlock, oc[2], oc[3], layers[3], stride=2, **kwargs),
            base_channel=base_channel, **kwargs
        )


class ResNet18(VanillaResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        """ResNet 18.

        ResNet 18 encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_layer: Final output layer. Default: 1x1 Conv2d if ``out_channels >= 1``, ``None`` otherwise.
            final_activation: Final activation layer (e.g. ``nn.ReLU`` or ``'relu'``). Default: ``None``.
            **kwargs: Additional keyword arguments.
        """
        super(ResNet18, self).__init__(in_channels, out_channels=out_channels, layers=(2, 2, 2, 2), **kwargs)


class ResNet34(VanillaResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNet34, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3), **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 34')


class BottleResNet(ResNet):
    def __init__(self, in_channels, out_channels=0, layers=(3, 4, 6, 3), base_channel=64, **kwargs):
        ex = Bottleneck.expansion
        self.out_channels = oc = (base_channel * 4, base_channel * 8, base_channel * 16, base_channel * 32)
        if out_channels and 'final_layer' not in kwargs.keys():
            kwargs['final_layer'] = nn.Conv2d(self.out_channels[-1], out_channels, 1)
        super(BottleResNet, self).__init__(
            in_channels,
            make_res_layer(Bottleneck, base_channel, oc[0] // ex, layers[0], stride=1, **kwargs),
            make_res_layer(Bottleneck, base_channel * 4, oc[1] // ex, layers[1], stride=2, **kwargs),
            make_res_layer(Bottleneck, base_channel * 8, oc[2] // ex, layers[2], stride=2, **kwargs),
            make_res_layer(Bottleneck, base_channel * 16, oc[3] // ex, layers[3], stride=2, **kwargs),
            base_channel=base_channel, **kwargs
        )


class ResNet50(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNet50, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3), **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 50')


class ResNet101(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNet101, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3), **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 101')


class ResNet152(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNet152, self).__init__(in_channels, out_channels=out_channels, layers=(3, 8, 36, 3), **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNet 152')


class ResNeXt50_32x4d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNeXt50_32x4d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3), groups=32,
                                              base_width=4, **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 50')


class ResNeXt101_32x8d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNeXt101_32x8d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3), groups=32,
                                               base_width=8, **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 101')


class ResNeXt152_32x8d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(ResNeXt152_32x8d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 8, 36, 3), groups=32,
                                               base_width=8, **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'ResNeXt 152')


class WideResNet50_2(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(WideResNet50_2, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3),
                                             base_width=64 * 2, **kwargs)

    __init__.__doc__ = ResNet18.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 50')


class WideResNet101_2(BottleResNet):
    def __init__(self, in_channels, out_channels=0, **kwargs):
        super(WideResNet101_2, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3),
                                              base_width=64 * 2, **kwargs)

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
