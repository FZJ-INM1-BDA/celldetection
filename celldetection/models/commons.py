import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tanh, sigmoid
from ..util.util import lookup_nn, tensor_to
from ..ops.commons import split_spatially, minibatch_std_layer
from typing import Type

__all__ = ['TwoConvNormRelu', 'ScaledTanh', 'ScaledSigmoid', 'ReplayCache', 'ConvNormRelu', 'ConvNorm',
           'ResBlock', 'NoAmp', 'ReadOut', 'BottleneckBlock', 'SpatialSplit', 'MinibatchStdLayer']


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 **kwargs):
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
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            norm_layer(out_channels),
        )


class ConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 activation='relu', **kwargs):
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
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            norm_layer(out_channels),
            lookup_nn(activation)
        )


class TwoConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None,
                 norm_layer=nn.BatchNorm2d, activation='relu', **kwargs):
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
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            norm_layer(mid_channels),
            lookup_nn(activation),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs),
            norm_layer(out_channels),
            lookup_nn(activation)
        )


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
    ) -> None:
        super().__init__()
        downsample = downsample or ConvNorm
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
        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False,
                          stride=stride, **kwargs),
                lookup_nn(norm_layer, out_channels),
                lookup_nn(activation),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **kwargs),
                lookup_nn(norm_layer, out_channels)
            ),
            activation=activation, stride=stride, downsample=downsample
        )


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
        mid_channels = mid_channels or np.max([base_channels, out_channels // compression, in_channels // compression])
        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                lookup_nn(norm_layer, mid_channels),
                lookup_nn(activation),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False,
                          stride=stride, **kwargs),
                lookup_nn(norm_layer, mid_channels),
                lookup_nn(activation),

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                lookup_nn(norm_layer, out_channels)
            ),
            activation=activation, stride=stride, downsample=downsample
        )


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
            stride=1
    ):
        super().__init__()
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in

        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding, stride=stride),
            lookup_nn(norm, channels_mid),
            lookup_nn(activation),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels_mid, channels_out, 1),
        )

        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        out = self.block(x)
        return self.activation(out)


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
