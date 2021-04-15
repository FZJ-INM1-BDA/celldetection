import torch
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class FPN(BackboneWithFPN):
    def __init__(self, backbone, channels=256, return_layers: dict = None):
        """

        Examples:
            >>> from juvision.models import ResNet18, FPN
            >>> m = FPN(ResNet18(in_channels=1))
            >>> for k, v in m(torch.rand(1, 1, 128, 128)).items():
            >>>     print(k, "\t", v.shape)
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
