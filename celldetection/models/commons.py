import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tanh, no_grad, as_tensor, sigmoid
from ..util.util import gaussian_kernel, lookup_nn

__all__ = ['TwoConvNormRelu', 'ScaledTanh', 'ScaledSigmoid', 'GaussianBlur', 'ReplayCache', 'ConvNormRelu', 'ConvNorm',
           'ResBlock']


class GaussianBlur(nn.Conv2d):
    def __init__(self, in_channels, kernel_size=3, sigma=-1, padding='same', padding_mode='reflect',
                 requires_grad=False, **kwargs):
        self._kernel = gaussian_kernel(kernel_size, sigma)
        super().__init__(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding,
                         padding_mode=padding_mode, bias=False, requires_grad=requires_grad, **kwargs)

    def reset_parameters(self):
        with no_grad():
            as_tensor(self._kernel, dtype=self.weight.dtype)


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


class ResBlock(nn.Module):
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
            **kwargs: Keyword arguments for Conv2d layers.
        """
        super().__init__()
        downsample = downsample or ConvNorm
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample(in_channels, out_channels, 1, stride=stride, bias=False, padding=0)
        else:
            self.downsample = nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                      **kwargs),
            lookup_nn(norm_layer, out_channels),
            lookup_nn(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **kwargs),
            lookup_nn(norm_layer, out_channels)
        )
        self.activation = lookup_nn(activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return self.activation(out)
