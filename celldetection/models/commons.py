import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tanh, sigmoid
from torchvision import transforms as trans
from torch.nn.common_types import _size_2_t
from ..util.util import lookup_nn, tensor_to, ensure_num_tuple, get_nd_conv
from ..ops.commons import split_spatially, minibatch_std_layer
from typing import Type, Union
from functools import partial

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


def _ni_3d(nd):
    if nd != 2:
        raise NotImplementedError('The `nd` option is not yet available for this model.')


@register
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        """Dynamic Tanh.

        Dynamic Tanh (DyT) is a replacement for commonly used Layer Norm or RMSNorm layers [1].

        References:
            - [1] https://arxiv.org/abs/2503.10622v1

        Args:
            normalized_shape (int or tuple): Shape of the normalized channels.
            channels_last (bool): If True, expects input in (..., C) format.
                If False, expects input in (B, C, ...) format.
            alpha_init_value (float): Initial value for the learnable scalar alpha.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        weight = self.weight
        bias = self.bias
        if not self.channels_last:
            extra_dims = x.dim() - 1 - len(self.normalized_shape)
            new_shape = self.weight.shape + (1,) * extra_dims
            weight = weight.view(new_shape)
            bias = bias.view(new_shape)
        return x * weight + bias

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, channels_last={self.channels_last}, "
                f"alpha_init_value={self.alpha.item()}")


@register
class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 nd=2, **kwargs):
        """ConvNorm.

        Just a convolution and a normalization layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(
            Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(out_channels),
        )


@register
class ConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 activation='relu', nd=2, **kwargs):
        """ConvNormReLU.

        Just a convolution, normalization layer and an activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(
            Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(out_channels),
            lookup_nn(activation)
        )


@register
class TwoConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None,
                 norm_layer=nn.BatchNorm2d, activation='relu', nd=2, **kwargs):
        """TwoConvNormReLU.

        A sequence of conv, norm, activation, conv, norm, activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            mid_channels: Mid-channels. Default: Same as ``out_channels``.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            Conv(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(mid_channels),
            lookup_nn(activation),
            Conv(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs),
            Norm(out_channels),
            lookup_nn(activation)
        )


@register
class TwoConvNormLeaky(TwoConvNormRelu):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None,
                 norm_layer=nn.BatchNorm2d, nd=2, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         mid_channels=mid_channels, norm_layer=norm_layer, activation='leakyrelu', nd=nd, **kwargs)


class ScaledX(nn.Module):
    def __init__(self, fn, factor, shift=0.):
        super().__init__()
        self.factor = factor
        self.shift = shift
        self.fn = fn

    def forward(self, inputs: Tensor) -> Tensor:
        return self.fn(inputs) * self.factor + self.shift

    def extra_repr(self) -> str:
        return 'factor={}, shift={}'.format(self.factor, self.shift)


@register
class ScaledTanh(ScaledX):
    def __init__(self, factor, shift=0.):
        """Scaled Tanh.

        Computes the scaled and shifted hyperbolic tangent:

        .. math:: tanh(x) * factor + shift

        Args:
            factor: Scaling factor.
            shift: Shifting constant.
        """
        super().__init__(tanh, factor, shift)


@register
class ScaledSigmoid(ScaledX):
    def __init__(self, factor, shift=0.):
        """Scaled Sigmoid.

        Computes the scaled and shifted sigmoid:

        .. math:: sigmoid(x) * factor + shift

        Args:
            factor: Scaling factor.
            shift: Shifting constant.
        """
        super().__init__(sigmoid, factor, shift)


@register
class ReplayCache:
    def __init__(self, size=128):
        """Replay Cache.

        Typical cache that can be used for experience replay in GAN training.

        Notes:
            - Items remain on their current device.

        Args:
            size: Number of batch items that fit in cache.
        """
        self.cache = []
        self.size = size

    def __len__(self):
        return len(self.cache)

    def is_empty(self):
        return len(self) <= 0

    def add(self, x, fraction=.5):
        """Add.

        Add a ``fraction`` of batch ``x`` to cache.
        Drop random items if cache is full.

        Args:
            x: Batch Tensor[n, ...].
            fraction: Fraction in 0..1.

        """
        lx = len(x)
        for i in np.random.choice(np.arange(lx), int(lx * fraction), replace=False):
            self.cache.append(x[i].detach())
        while len(self) > self.size:
            del self.cache[np.random.randint(0, len(self))]

    def __call__(self, num):
        """Call.

        Args:
            num: Batch size / number of returned items.

        Returns:
            Tensor[num, ...]
        """
        if self.is_empty():
            return None
        return torch.stack([self.cache[i] for i in np.random.randint(0, len(self), num)], 0)


class _ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            block,
            activation='ReLU',
            stride=1,
            downsample=None,
            norm_layer='BatchNorm2d',
            nd=2,
    ) -> None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            https://arxiv.org/abs/1512.03385

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            nd: Number of spatial dimensions.
        """
        super().__init__()
        downsample = downsample or partial(ConvNorm, nd=nd, norm_layer=norm_layer)
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample(in_channels, out_channels, 1, stride=stride, bias=False, padding=0)
        else:
            self.downsample = nn.Identity()
        self.block = block
        self.activation = lookup_nn(activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return self.activation(out)


@register
class ResBlock(_ResBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer='BatchNorm2d',
            activation='ReLU',
            stride=1,
            downsample=None,
            nd=2,
            **kwargs
    ) -> None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90

        Notes:
            - Similar to ``torchvision.models.resnet.BasicBlock``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                     **kwargs),
                Norm(out_channels),
                lookup_nn(activation),
                Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **kwargs),
                Norm(out_channels),
            ),
            activation=activation, stride=stride, downsample=downsample, nd=nd, norm_layer=norm_layer
        )


@register
class BottleneckBlock(_ResBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            mid_channels=None,
            compression=4,
            base_channels=64,
            norm_layer='BatchNorm2d',
            activation='ReLU',
            stride=1,
            downsample=None,
            nd=2,
            **kwargs
    ) -> None:
        """Bottleneck Block.

        Typical Bottleneck Block with variable kernel size and an included mapping of the identity to correct
        dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90
            - https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch

        Notes:
            - Similar to ``torchvision.models.resnet.Bottleneck``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.
            - Stride handled in bottleneck.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            mid_channels:
            compression: Compression rate of the bottleneck. The default 4 compresses 256 channels to 64=256/4.
            base_channels: Minimum number of ``mid_channels``.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        mid_channels = mid_channels or np.max([base_channels, out_channels // compression, in_channels // compression])
        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                Conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                Norm(mid_channels),
                lookup_nn(activation),

                Conv(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                     **kwargs),
                Norm(mid_channels),
                lookup_nn(activation),

                Conv(mid_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                Norm(out_channels)
            ),
            activation=activation, stride=stride, downsample=downsample
        )


@register
class NoAmp(nn.Module):
    def __init__(self, module: Type[nn.Module]):
        """No AMP.

        Wrap a ``Module`` object and disable ``torch.cuda.amp.autocast`` during forward pass if it is enabled.

        Examples:
            >>> import celldetection as cd
            ... model = cd.models.CpnU22(1)
            ... # Wrap all ReadOut modules in model with NoAmp, thus disabling autocast for those modules
            ... cd.wrap_module_(model, cd.models.ReadOut, cd.models.NoAmp)

        Args:
            module: Module.
        """
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                result = self.module(*tensor_to(args, torch.float32), **tensor_to(kwargs, torch.float32))
        else:
            result = self.module(*args, **kwargs)
        return result


@register
class ReadOut(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=3,
            padding=1,
            activation='relu',
            norm='batchnorm2d',
            final_activation=None,
            dropout=0.1,
            channels_mid=None,
            stride=1,
            nd=2,
            attention=None,
    ):
        super().__init__()
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm, nd=nd, call=False)
        Dropout = lookup_nn(nn.Dropout2d, nd=nd, call=False)
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in

        self.attention = None
        if attention is not None:
            if isinstance(attention, dict):
                attention_kwargs, = list(attention.values())
                attention, = list(attention.keys())
            else:
                attention_kwargs = {}
            self.attention = lookup_nn(attention, nd=nd, call=False)(channels_in, **attention_kwargs)

        self.block = nn.Sequential(
            Conv(channels_in, channels_mid, kernel_size, padding=padding, stride=stride),
            Norm(channels_mid),
            lookup_nn(activation),
            Dropout(p=dropout) if dropout else nn.Identity(),
            Conv(channels_mid, channels_out, 1),
        )

        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        if self.attention is not None:
            x = self.attention(x)
        out = self.block(x)
        return self.activation(out)


@register
class SpatialSplit(nn.Module):
    def __init__(self, height, width=None):
        """Spatial split.

        Splits spatial dimensions of input Tensor into patches of size ``(height, width)`` and adds the patches
        to the batch dimension.

        Args:
            height: Patch height.
            width: Patch width.
        """
        super().__init__()
        self.height = height
        self.width = width or height

    def forward(self, x):
        return split_spatially(x, self.height, self.width)


@register
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, channels=1, group_channels=None, epsilon=1e-8):
        """Minibatch standard deviation layer.

        The minibatch standard deviation layer first splits the batch dimension into slices of size ``group_channels``.
        The channel dimension is split into ``channels`` slices. For the groups the standard deviation is calculated and
        averaged over spatial dimensions and channel slice depth. The result is broadcasted to the spatial dimensions,
        repeated for the batch dimension and then concatenated to the channel dimension of ``x``.

        References:
            - https://arxiv.org/pdf/1710.10196.pdf

        Args:
            channels: Number of averaged standard deviation channels.
            group_channels: Number of channels per group. Default: batch size.
            epsilon: Epsilon.
        """
        super().__init__()
        self.channels = channels
        self.group_channels = group_channels
        self.epsilon = epsilon

    def forward(self, x):
        return minibatch_std_layer(x, self.channels, self.group_channels, epsilon=self.epsilon)

    def extra_repr(self) -> str:
        return f'channels={self.channels}, group_channels={self.group_channels}'


class _AdditiveNoise(nn.Module):
    def __init__(self, in_channels, noise_channels=1, mean=0., std=1., weighted=False, nd=2):
        super().__init__()
        self.noise_channels = noise_channels
        self.in_channels = in_channels
        self.reps = (1, self.in_channels // self.noise_channels) + (1,) * nd
        self.weighted = weighted
        self.weight = nn.Parameter(torch.zeros((1, in_channels) + (1,) * nd)) if weighted else 1.
        self.constant = False
        self.mean = mean
        self.std = std
        self._constant = None

    def sample_noise(self, shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype) * self.std + self.mean

    def forward(self, x):
        shape = x.shape
        constant = getattr(self, 'constant', False)
        _constant = getattr(self, '_constant', None)
        if (constant and _constant is None) or not constant:
            noise = self.sample_noise((shape[0], self.noise_channels) + shape[2:], x.device, x.dtype)
            if constant and _constant is None:
                self._constant = noise
        else:
            noise = _constant
        return x + noise.repeat(self.reps) * self.weight

    def extra_repr(self):
        s = f"in_channels={self.in_channels}, noise_channels={self.noise_channels}, mean={self.mean}, " \
            f"std={self.std}, weighted={self.weighted}"
        if getattr(self, 'constant', False):
            s += ', constant=True'
        return s


@register
class AdditiveNoise2d(_AdditiveNoise):
    def __init__(self, in_channels, noise_channels=1, weighted=True, **kwargs):
        super().__init__(in_channels=in_channels, noise_channels=noise_channels, weighted=weighted, nd=2, **kwargs)


@register
class AdditiveNoise3d(_AdditiveNoise):
    def __init__(self, in_channels, noise_channels=1, weighted=True, **kwargs):
        super().__init__(in_channels=in_channels, noise_channels=noise_channels, weighted=weighted, nd=3, **kwargs)


class _Stride(nn.Module):
    def __init__(self, stride, start=0, nd=2):
        super().__init__()
        self.stride = ensure_num_tuple(stride, nd)
        self.start = start

    def forward(self, x):
        return x[(...,) + tuple((slice(self.start, None, s) for s in self.stride))]


@register
class Stride1d(_Stride):
    def __init__(self, stride, start=0):
        super().__init__(stride, start, 1)


@register
class Stride2d(_Stride):
    def __init__(self, stride, start=0):
        super().__init__(stride, start, 2)


@register
class Stride3d(_Stride):
    def __init__(self, stride, start=0):
        super().__init__(stride, start, 3)


class _Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm2d',
                 nd=2, dim=1, **kwargs):
        super().__init__()
        modules = [get_nd_conv(nd)(in_channels, out_channels, kernel_size, padding=padding, **kwargs)]
        if norm_layer is not None:
            modules.append(lookup_nn(norm_layer, out_channels, nd=nd))
        if activation is not None:
            modules.append(lookup_nn(activation, inplace=False))
        self.block = nn.Sequential(*modules)
        self.nd = nd
        self.dim = dim

    def forward(self, x: tuple):
        x = tuple(x)
        target_size = x[0].shape[-self.nd:]
        x = torch.cat([(F.interpolate(x_, target_size) if x_.shape[-self.nd:] != target_size else x_) for x_ in x],
                      dim=self.dim)
        return self.block(x)


@register
class Fuse1d(_Fuse):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm1d',
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                         activation=activation, norm_layer=norm_layer, nd=1, **kwargs)


@register
class Fuse2d(_Fuse):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm2d',
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                         activation=activation, norm_layer=norm_layer, nd=2, **kwargs)


@register
class Fuse3d(_Fuse):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm3d',
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                         activation=activation, norm_layer=norm_layer, nd=3, **kwargs)


@register
class Normalize(nn.Module):
    def __init__(self, mean=0., std=1., assert_range=(0., 1.)):
        super().__init__()
        self.assert_range = assert_range
        self.transform = trans.Compose([
            trans.Normalize(mean=mean, std=std)
        ])

    def forward(self, inputs: Tensor):
        if self.assert_range is not None:
            assert torch.all(inputs >= self.assert_range[0]) and torch.all(
                inputs <= self.assert_range[1]), f'Inputs should be in interval {self.assert_range}'
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs

    def extra_repr(self) -> str:
        s = ''
        if self.assert_range is not None:
            s += f'(assert_range): {self.assert_range}\n'
        s += f'(norm): {repr(self.transform)}'
        return s


@register
class SqueezeExcitation(nn.Sequential):
    def __init__(self, in_channels, squeeze_channels=None, compression=16, activation='relu',
                 scale_activation='sigmoid', residual=True, nd=2):
        Pool = lookup_nn('AdaptiveAvgPool2d', nd=nd, call=False)
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        self.residual = residual
        if squeeze_channels is None:
            squeeze_channels = max(in_channels // compression, 1)
        super().__init__(
            Pool(1),
            Conv(in_channels, squeeze_channels, 1),
            lookup_nn(activation),
            Conv(squeeze_channels, in_channels, 1),
            lookup_nn(scale_activation)
        )

    def forward(self, inputs):
        scale = super().forward(inputs)
        scaled = inputs * scale
        if self.residual:
            return inputs + scaled
        return scaled


@register
class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, mid_channels=None, kernel_size=1, padding=0, beta=True, nd=2):
        """Self-Attention.

        References:
            - https://arxiv.org/pdf/1805.08318.pdf

        Args:
            in_channels:
            out_channels: Equal to `in_channels` by default.
            mid_channels: Set to `in_channels // 8` by default.
            kernel_size:
            padding:
            beta:
            nd:
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 8
        if out_channels is None:
            out_channels = in_channels
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        self.beta = nn.Parameter(torch.zeros(1)) if beta else 1.
        if in_channels != out_channels:
            self.in_conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.in_conv = None
        self.proj_b, self.proj_a = [Conv(out_channels, mid_channels, kernel_size=1) for _ in range(2)]
        self.proj = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.out_conv = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, inputs):
        x = inputs if self.in_conv is None else self.in_conv(inputs)
        a = self.proj_a(x).flatten(2)
        b = self.proj_b(x).flatten(2)  # Tensor[n, c', hw]
        p = torch.matmul(a.permute(0, 2, 1), b)  # Tensor[n, hw, hw]
        p = F.softmax(p, dim=1)  # Tensor[n, hw, hw]
        c = self.proj(x).flatten(2)  # Tensor[n, c, hw]
        out = torch.matmul(p, c.permute(0, 2, 1)).view(*c.shape[:2], *inputs.shape[2:])  # T[n, c, hw] -> T[n, c, h, w]
        out = self.out_conv(self.beta * out + x)
        return out


def channels_last_permute(nd):
    return (0,) + tuple(range(2, nd + 2)) + (1,)


def channels_first_permute(nd):
    return (0, nd + 1,) + tuple(range(1, nd + 1))


class LayerNormNd(nn.LayerNorm):  # Generalized version of torchvision.models.convnext.LayerNorm2d
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, nd=2,
                 device=None, dtype=None) -> None:
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype)
        self._perm0 = channels_last_permute(nd)
        self._perm1 = channels_first_permute(nd)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(*self._perm0)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(*self._perm1)
        return x


@register
class LayerNorm1d(LayerNormNd):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None,
                 dtype=None) -> None:
        """Layer Norm.

        By default, ``LayerNorm1d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype, nd=1)


@register
class LayerNorm2d(LayerNormNd):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None,
                 dtype=None) -> None:
        """Layer Norm.

        By default, ``LayerNorm2d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype, nd=2)


@register
class LayerNorm3d(LayerNormNd):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None,
                 dtype=None) -> None:
        """Layer Norm.

        By default, ``LayerNorm3d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype, nd=3)
