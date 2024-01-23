from torchvision.models.convnext import CNBlockConfig
from torchvision.models import convnext as cnx
from typing import List, Optional, Callable, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from functools import partial
from torchvision.ops import misc, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from ..util.util import lookup_nn
from .commons import LayerNorm1d, LayerNorm2d, LayerNorm3d, channels_last_permute, channels_first_permute
from .convnext import CNBlock, ConvNeXt, map_state_dict
from torch.hub import load_state_dict_from_url
from timm.models.layers import trunc_normal_

__all__ = ['ConvNeXtV2', 'ConvNeXtV2Atto', 'ConvNeXtV2Femto', 'ConvNeXtV2Pico', 'ConvNeXtV2Nano', 'ConvNeXtV2Tiny',
           'ConvNeXtV2Base', 'ConvNeXtV2Large', 'ConvNeXtV2Huge']

default_model_urls = dict(

)


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class GRN(nn.Module):
    def __init__(self, channels, nd=2, channels_axis=-1, epsilon=1e-6):
        """Global Response Normalization.

        References:
            - https://arxiv.org/abs/2301.00808

        Note:
            - Expects channels last format

        Args:
            channels: Number of channels.
            nd: Number of spatial dimensions.
            channels_axis: Channels axis. Expects channels-last format by default.
        """
        super().__init__()

        self.channels_axis = channels_axis
        dims = [1] * (nd + 2)
        dims[self.channels_axis] = channels

        self.spatial_dims = tuple(range(1, nd + 1))
        self.nd = nd
        self.gamma = nn.Parameter(torch.zeros(*dims))
        self.beta = nn.Parameter(torch.zeros(*dims))
        self.epsilon = epsilon

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=self.spatial_dims, keepdim=True)
        nx = gx / (gx.mean(dim=self.channels_axis, keepdim=True) + self.epsilon)
        return self.gamma * (x * nx) + self.beta + x


class CNBlockV2(CNBlock):
    def __init__(self, in_channels, out_channels=None, layer_scale: float = None, stochastic_depth_prob: float = 0,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, activation='gelu', stride: int = 1,
                 identity_norm_layer=None, nd: int = 2, conv_kwargs=None) -> None:
        """ConvNeXt Block V2.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels:
            out_channels:
            layer_scale:
            stochastic_depth_prob:
            norm_layer:
            activation:
            stride:
            identity_norm_layer:
            nd:
            conv_kwargs:
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, layer_scale=layer_scale,
                         stochastic_depth_prob=stochastic_depth_prob, norm_layer=norm_layer, activation=activation,
                         stride=stride, identity_norm_layer=identity_norm_layer, nd=nd, conv_kwargs=conv_kwargs)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if conv_kwargs is None:
            conv_kwargs = {}
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        out_channels = in_channels if out_channels is None else out_channels

        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=conv_kwargs.pop('kernel_size', 7),
                 padding=conv_kwargs.pop('padding', 3), groups=conv_kwargs.pop('groups', out_channels),
                 bias=conv_kwargs.pop('bias', True), **conv_kwargs),
            Permute(list(channels_last_permute(nd))),
            norm_layer(out_channels),
            nn.Linear(in_features=out_channels, out_features=4 * out_channels, bias=True),
            lookup_nn(activation),
            GRN(4 * out_channels, nd=nd, channels_axis=-1),
            nn.Linear(in_features=4 * out_channels, out_features=out_channels, bias=True),
            Permute(list(channels_first_permute(nd))),
        )


class ConvNeXtV2(ConvNeXt):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            block_kwargs: dict = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            pretrained=False,
            fused_initial=True,
            final_layer=None,
            secondary_block=None,
            nd=2,
    ):
        """ConvNeXt V2.

        Notes:
            - Initialized with `trunc_normal_(tensor, mean=0., std=0.02, a=-2., b=2.)` by default.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels:
            out_channels:
            block_setting:
            stochastic_depth_prob:
            layer_scale:
            block:
            block_kwargs:
            norm_layer:
            pretrained:
            fused_initial:
            final_layer:
            nd:
        """
        if block is None:
            block = partial(CNBlockV2, nd=nd)
        super().__init__(in_channels=in_channels, out_channels=out_channels, block_setting=block_setting,
                         stochastic_depth_prob=stochastic_depth_prob, layer_scale=layer_scale, block=block,
                         block_kwargs=block_kwargs, norm_layer=norm_layer, pretrained=False,
                         fused_initial=fused_initial, final_layer=final_layer, secondary_block=secondary_block, nd=nd)
        self.apply(_init_weights)

        if pretrained:
            if isinstance(pretrained, str):
                state_dict = load_state_dict_from_url(pretrained)
                if '.pytorch.org' in pretrained:
                    state_dict = map_state_dict(in_channels, self.state_dict(), state_dict, nd=nd,
                                                fused_initial=fused_initial)
                self.load_state_dict(state_dict)
            else:
                raise ValueError('There is no default set of weights for this model. '
                                 'Please specify a URL using the `pretrained` argument.')


class ConvNeXtV2Atto(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Atto.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtAtto', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(40, 80, 2),
                CNBlockConfig(80, 160, 2),
                CNBlockConfig(160, 320, 6),
                CNBlockConfig(320, None, 2),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Femto(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Femto.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtFemto', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(48, 96, 2),
                CNBlockConfig(96, 192, 2),
                CNBlockConfig(192, 384, 6),
                CNBlockConfig(384, None, 2),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Pico(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Pico.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtPico', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(64, 128, 2),
                CNBlockConfig(128, 256, 2),
                CNBlockConfig(256, 512, 6),
                CNBlockConfig(512, None, 2),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Nano(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Nano.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtNano', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(80, 160, 2),
                CNBlockConfig(160, 320, 2),
                CNBlockConfig(320, 640, 8),
                CNBlockConfig(640, None, 2),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Tiny(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Tiny.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtTiny', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 9),
                CNBlockConfig(768, None, 3),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Base(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Base.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtBase', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(128, 256, 3),
                CNBlockConfig(256, 512, 3),
                CNBlockConfig(512, 1024, 27),
                CNBlockConfig(1024, None, 3),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Large(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Large.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtLarge', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 3),
                CNBlockConfig(768, 1536, 27),
                CNBlockConfig(1536, None, 3),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )


class ConvNeXtV2Huge(ConvNeXtV2):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ConvNeXtV2 Huge.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels: Input channels.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            stochastic_depth_prob: Stochastic depth probability.
                Base probability of randomly dropping residual connections. Actual probability in blocks is given by
                ``stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)``.
            pretrained: Whether to use pretrained weights. By default, weights from ``torchvision`` are used.
            nd: Number of spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if pretrained is True and nd == 2:
            pretrained = default_model_urls.get('ConvNeXtHuge', pretrained)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(352, 704, 3),
                CNBlockConfig(704, 1408, 3),
                CNBlockConfig(1408, 2816, 27),
                CNBlockConfig(2816, None, 3),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )
