import torch.nn as nn
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual, _mobilenet_v3_conf
from torchvision.models.mobilenetv2 import ConvBNActivation
from typing import Any, Callable, List, Optional, Sequence
from functools import partial

__all__ = ['MobileNetV3Large', 'MobileNetV3Small']


def init_modules_(mod: nn.Module):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


class MobileNetV3Base(nn.Sequential):
    """Adaptation of torchvision.models.mobilenetv3.MobileNetV3"""

    def __init__(
            self,
            in_channels,
            inverted_residual_setting: List[InvertedResidualConfig],
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Sequential] = [nn.Sequential()]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.out_channels = [firstconv_output_channels]
        layers[-1].add_module(str(len(layers[-1])),
                              ConvBNActivation(in_channels, firstconv_output_channels, kernel_size=3, stride=2,
                                               norm_layer=norm_layer, activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            if cnf.stride > 1:
                layers.append(nn.Sequential())
                self.out_channels.append(cnf.out_channels)
            else:
                self.out_channels[-1] = cnf.out_channels
            layers[-1].add_module(str(len(layers[-1])), block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_channels[-1] = lastconv_output_channels
        assert len(self.out_channels) == len(layers)
        layers[-1].add_module(str(len(layers[-1])),
                              ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                               norm_layer=norm_layer, activation_layer=nn.Hardswish))

        super().__init__(*layers)
        init_modules_(self)


class MobileNetV3Large(MobileNetV3Base):
    def __init__(self, in_channels, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False):
        super().__init__(in_channels=in_channels, inverted_residual_setting=_mobilenet_v3_conf(
            'mobilenet_v3_large', width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated)[0])


class MobileNetV3Small(MobileNetV3Base):
    def __init__(self, in_channels, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False):
        super().__init__(in_channels=in_channels, inverted_residual_setting=_mobilenet_v3_conf(
            'mobilenet_v3_small', width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated)[0])
