from torchvision.models.convnext import CNBlockConfig
from torchvision.models import convnext as cnx
from typing import List, Optional, Callable, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from functools import partial
from torchvision.ops import misc, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from ..util.util import lookup_nn, get_nn
from .ppm import append_pyramid_pooling_
from .commons import LayerNorm1d, LayerNorm2d, LayerNorm3d, channels_last_permute, channels_first_permute
from torch.hub import load_state_dict_from_url
from pytorch_lightning.core.mixins import HyperparametersMixin


__all__ = ['ConvNeXtBase', 'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtLarge', 'ConvNeXt', 'CNBlock']

default_model_urls = dict(
    ConvNeXtLarge=cnx.ConvNeXt_Large_Weights.IMAGENET1K_V1.url,
    ConvNeXtBase=cnx.ConvNeXt_Base_Weights.IMAGENET1K_V1.url,
    ConvNeXtSmall=cnx.ConvNeXt_Small_Weights.IMAGENET1K_V1.url,
    ConvNeXtTiny=cnx.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
)


def map_state_dict(in_channels, current_state_dict, state_dict, nd=2, fused_initial=False):
    # Only keep params of features branch
    selection = {k: v for k, v in state_dict.items() if k.startswith('features.')}
    assert len(selection) == len(current_state_dict), (len(selection), len(current_state_dict))
    # Rename
    mapping = {}
    for a, b in zip(selection, current_state_dict):
        params = state_dict[a]
        if b == ('0.0.0.weight' if fused_initial else '0.0.weight') and params.shape[1] != in_channels:
            params.data = F.interpolate(params.data[None], (in_channels,) + params.data.shape[-nd:]).squeeze(0)
        mapping[b] = params
    return mapping


class ConvNormActivation(misc.ConvNormActivation):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
            nd=2,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            lookup_nn('Conv2d', nd=nd, call=False),
        )


class CNBlock(nn.Module):  # ported from torchvision.models.convnext to support n-dimensions and add more features
    def __init__(self, in_channels, out_channels=None, layer_scale: float = 1e-6, stochastic_depth_prob: float = 0,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, activation='gelu', stride: int = 1,
                 identity_norm_layer=None, nd: int = 2, conv_kwargs=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if conv_kwargs is None:
            conv_kwargs = {}
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        out_channels = in_channels if out_channels is None else out_channels
        self.identity = None
        if in_channels != out_channels or stride != 1:
            if identity_norm_layer is None:
                identity_norm_layer = [LayerNorm1d, LayerNorm2d, LayerNorm3d][nd - 1]
            self.identity = nn.Sequential(  # following option (b) in He et al. (2015)
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                identity_norm_layer(out_channels)
            )
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=conv_kwargs.pop('kernel_size', 7),
                 padding=conv_kwargs.pop('padding', 3), groups=conv_kwargs.pop('groups', out_channels),
                 bias=conv_kwargs.pop('bias', True), **conv_kwargs),
            Permute(list(channels_last_permute(nd))),
            norm_layer(out_channels),
            nn.Linear(in_features=out_channels, out_features=4 * out_channels, bias=True),
            lookup_nn(activation),
            nn.Linear(in_features=4 * out_channels, out_features=out_channels, bias=True),
            Permute(list(channels_first_permute(nd))),
        )
        if layer_scale is None:
            self.layer_scale = 1
        else:
            self.layer_scale = nn.Parameter(torch.ones(out_channels, *(1,) * nd) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs if self.identity is None else self.identity(inputs)
        result = self.layer_scale * self.block(inputs)
        result = self.stochastic_depth(result)
        result += identity
        return result


class ConvNeXt(nn.Sequential, HyperparametersMixin):
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
            pyramid_pooling=False,
            pyramid_pooling_channels=64,
            pyramid_pooling_kwargs=None,
            secondary_block=None,
            nd=2,
    ):
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if secondary_block is not None:
            secondary_block = get_nn(secondary_block, nd=nd)

        block_kwargs = {} if block_kwargs is None else block_kwargs
        if block is None:
            block = partial(CNBlock, nd=nd)

        if norm_layer is None:
            norm_layer = partial([LayerNorm1d, LayerNorm2d, LayerNorm3d][nd - 1], eps=1e-6)

        layers: List[nn.Module] = []
        firstconv_output_channels = block_setting[0].input_channels
        fi = 1 + (1 - fused_initial)
        self.out_channels = [firstconv_output_channels] * fi + [s.out_channels for s in block_setting if
                                                                s.out_channels is not None]
        num = len([b for b in block_setting if b.out_channels is not None])
        self.out_strides = [4] * fi + [4 * (2 ** i) for i in range(1, num + 1)]

        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        initial = ConvNormActivation(
            in_channels,
            firstconv_output_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
            bias=True,
            nd=nd
        )
        if not fused_initial:
            layers.append(initial)
            initial = None

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        down = cnf = None
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            if initial is not None:
                stage += [initial]  # fused initial
                initial = None
            if down is not None:
                stage += [down]  # downsampling is part of the stage
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale=layer_scale, stochastic_depth_prob=sd_prob,
                                   **block_kwargs))
                stage_block_id += 1
            if secondary_block is not None:
                stage.append(secondary_block(cnf.input_channels, nd=nd))
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                down = nn.Sequential(
                    norm_layer(cnf.input_channels),
                    Conv(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                )

        if out_channels or final_layer is not None:

            if final_layer is None:
                final_layer = Conv(cnf.out_channels or cnf.input_channels, out_channels, 1)
            layers.append(final_layer)
        super().__init__(*layers)

        for m in self.modules():
            if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        if pyramid_pooling:
            pyramid_pooling_kwargs = {} if pyramid_pooling_kwargs is None else pyramid_pooling_kwargs
            append_pyramid_pooling_(self, pyramid_pooling_channels, nd=nd, **pyramid_pooling_kwargs)


class ConvNeXtTiny(ConvNeXt):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ContNeXt Tiny.

        References:
            - https://arxiv.org/abs/2201.03545

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
        self.save_hyperparameters()
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ConvNeXtTiny']
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


class ConvNeXtSmall(ConvNeXt):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        self.save_hyperparameters()
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ConvNeXtSmall']
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 27),
                CNBlockConfig(768, None, 3),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
            pretrained=pretrained,
            nd=nd,
            **kwargs
        )

    __init__.__doc__ = ConvNeXtTiny.__init__.__doc__.replace('ContNeXt Tiny.', 'ContNeXt Small.')


class ConvNeXtBase(ConvNeXt):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ContNeXt Base.

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
        self.save_hyperparameters()
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ConvNeXtBase']
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

    __init__.__doc__ = ConvNeXtTiny.__init__.__doc__.replace('ContNeXt Tiny.', 'ContNeXt Base.')


class ConvNeXtLarge(ConvNeXt):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 0,
            stochastic_depth_prob: float = .1,
            pretrained: bool = False,
            nd: int = 2,
            **kwargs
    ):
        """ContNeXt Large.

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
        self.save_hyperparameters()
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ConvNeXtLarge']
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

    __init__.__doc__ = ConvNeXtTiny.__init__.__doc__.replace('ContNeXt Tiny.', 'ContNeXt Large.')
