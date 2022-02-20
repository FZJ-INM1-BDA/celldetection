import torch
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50_32x4d, ResNeXt101_32x8d, \
    ResNeXt152_32x8d, WideResNet50_2, WideResNet101_2
from .mobilenetv3 import MobileNetV3Large, MobileNetV3Small

__all__ = ['FPN', 'ResNeXt50FPN', 'ResNeXt101FPN', 'ResNet18FPN', 'ResNet34FPN', 'ResNeXt152FPN', 'WideResNet50FPN',
           'WideResNet101FPN', 'ResNet50FPN', 'ResNet101FPN', 'ResNet152FPN', 'MobileNetV3SmallFPN',
           'MobileNetV3LargeFPN']


class FPN(BackboneWithFPN):
    def __init__(self, backbone, channels=256, return_layers: dict = None):
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
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        super(FPN, self).__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=list(backbone.out_channels),
            out_channels=channels
        )


class ResNet18FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNet18(in_channels=in_channels), channels=fpn_channels)


class ResNet34FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNet34(in_channels=in_channels), channels=fpn_channels)


class ResNet50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNet50(in_channels=in_channels), channels=fpn_channels)


class ResNet101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNet101(in_channels=in_channels), channels=fpn_channels)


class ResNet152FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNet152(in_channels=in_channels), channels=fpn_channels)


class ResNeXt50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNeXt50_32x4d(in_channels=in_channels), channels=fpn_channels)


class ResNeXt101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNeXt101_32x8d(in_channels=in_channels), channels=fpn_channels)


class ResNeXt152FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(ResNeXt152_32x8d(in_channels=in_channels), channels=fpn_channels)


class WideResNet50FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(WideResNet50_2(in_channels=in_channels), channels=fpn_channels)


class WideResNet101FPN(FPN):
    def __init__(self, in_channels, fpn_channels=256):
        super().__init__(WideResNet101_2(in_channels=in_channels), channels=fpn_channels)


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

    def __init__(self, in_channels, fpn_channels=256, **kwargs):
        super().__init__(MobileNetV3Small(in_channels=in_channels, **kwargs), channels=fpn_channels)


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

    def __init__(self, in_channels, fpn_channels=256, **kwargs):
        super().__init__(MobileNetV3Large(in_channels=in_channels, **kwargs), channels=fpn_channels)
