import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union
from torch.nn.common_types import _size_2_t
from ..ops.commons import strided_upsampling2d

__all__ = ['Filter2d', 'PascalFilter2d', 'ScharrFilter2d', 'SobelFilter2d', 'BoxFilter2d', 'UpFilter2d',
           'LaplaceFilter2d', 'EdgeFilter2d']


class Filter2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            kernel,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=True
    ) -> None:
        """Filter 2d.

        Applies a 2d filter to all channels of input.

        Examples:
            >>> sobel = torch.as_tensor([
            ...     [1, 0, -1],
            ...     [2, 0, -2],
            ...     [1, 0, -1],
            ... ], dtype=torch.float32)
            ... sobel_layer = Filter2d(in_channels=3, kernel=sobel, padding=1, trainable=False)
            ... sobel_layer, sobel_layer.weight
            (Filter2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False),
             tensor([[ 1.,  0., -1.],
                     [ 2.,  0., -2.],
                     [ 1.,  0., -1.]]))

        Args:
            in_channels: Number of input channels.
            kernel: Filter matrix.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
        """
        self._padding_mode = padding_mode.replace('zeros', 'constant')  # for F.pad
        kernel_size = len(kernel)
        self.pad = [0, 1, 0, 1] if (odd_padding and kernel_size % 2 == 0) else None
        self._kernel, self._shape, self._trainable = kernel, None, trainable
        super().__init__(in_channels=in_channels, out_channels=in_channels, kernel_size=(kernel_size, kernel_size),
                         stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False,
                         padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.pad is not None:
            x = F.pad(x, self.pad, mode=self._padding_mode)
        x = self._conv_forward(x, self.weight.broadcast_to(self._shape), self.bias)
        return x

    def reset_parameters(self):
        if self._shape is None:
            self._shape = self.weight.shape
        kernel = self._kernel.to(self.weight.dtype)
        with torch.no_grad():
            if self._trainable:
                self.weight = nn.Parameter(kernel)
            else:
                del self.weight
                self.register_buffer('weight', kernel)
            if self.bias is not None:
                self.bias.data.zero_()


class PascalFilter2d(Filter2d):
    def __init__(
            self,
            in_channels: int,
            kernel_size,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=False,
            normalize=True
    ) -> None:
        """Pascal Filter 2d.

        Applies a 2d pascal filter to all channels of input.

        References:
            - https://en.wikipedia.org/wiki/Pascal%27s_triangle

        Args:
            in_channels: Number of input channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
            normalize: Whether to normalize the kernel to retain magnitude.
        """
        super().__init__(in_channels=in_channels, kernel=self.get_kernel2d(kernel_size, normalize),
                         stride=stride, padding=padding, dilation=dilation, odd_padding=odd_padding,
                         trainable=trainable, padding_mode=padding_mode, device=device, dtype=dtype)

    @staticmethod
    def get_kernel1d(kernel_size, normalize=True):
        triangle = []
        for k in range(1, kernel_size + 1):
            triangle.append(np.ones(k))
            for n in range(k // 2):
                triangle[-1][[n + 1, -n - 2]] = triangle[-2][n:n + 2].sum()
        return triangle[-1] / (triangle[-1].sum() if normalize else 1)

    @staticmethod
    def get_kernel2d(kernel_size, normalize=True):
        k = PascalFilter2d.get_kernel1d(kernel_size, normalize)
        return torch.as_tensor(np.outer(k, k))


class ScharrFilter2d(Filter2d):
    def __init__(
            self,
            in_channels: int,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=False,
            transpose=False
    ) -> None:
        """Scharr Filter 2d.

        Applies the Scharr gradient operator, a 3x3 kernel optimized for rotational symmetry.

        References:
            - https://archiv.ub.uni-heidelberg.de/volltextserver/962/
            - https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators

        Args:
            in_channels: Number of input channels.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
            transpose: ``False`` for :math:`h_x` kernel, ``True`` for :math:`h_y` kernel.
        """
        super().__init__(in_channels=in_channels, kernel=self.get_kernel2d(transpose),
                         stride=stride, padding=padding, dilation=dilation, odd_padding=odd_padding,
                         trainable=trainable, padding_mode=padding_mode, device=device, dtype=dtype)

    @staticmethod
    def get_kernel2d(transpose=False):
        kernel = torch.as_tensor([
            [47, 0, -47],
            [162, 0, -162],
            [47, 0, -47.],
        ])
        if transpose:
            kernel = kernel.T
        return kernel


class SobelFilter2d(Filter2d):
    def __init__(
            self,
            in_channels: int,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=False,
            transpose=False
    ) -> None:
        """Sobel Filter 2d.

        Applies the 3x3 Sobel image gradient operator.

        References:
            - https://en.wikipedia.org/wiki/Sobel_operator

        Args:
            in_channels: Number of input channels.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
            transpose: ``False`` for :math:`h_x` kernel, ``True`` for :math:`h_y` kernel.
        """
        super().__init__(in_channels=in_channels, kernel=self.get_kernel2d(transpose),
                         stride=stride, padding=padding, dilation=dilation, odd_padding=odd_padding,
                         trainable=trainable, padding_mode=padding_mode, device=device, dtype=dtype)

    @staticmethod
    def get_kernel2d(transpose=False):
        sobel = torch.as_tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1.],
        ])
        if transpose:
            sobel = sobel.T
        return sobel


class BoxFilter2d(Filter2d):
    def __init__(
            self,
            in_channels: int,
            kernel_size,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=False,
            normalize=True
    ) -> None:
        """Box Filter 2d.

        Args:
            in_channels: Number of input channels.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
            normalize: Whether to normalize the kernel to retain magnitude.
        """
        super().__init__(in_channels=in_channels, kernel=self.get_kernel2d(kernel_size, normalize),
                         stride=stride, padding=padding, dilation=dilation, odd_padding=odd_padding,
                         trainable=trainable, padding_mode=padding_mode, device=device, dtype=dtype)

    @staticmethod
    def get_kernel2d(kernel_size, normalize=True):
        return torch.ones((kernel_size, kernel_size)) / (kernel_size ** 2 if normalize else 1)


class LaplaceFilter2d(Filter2d):
    def __init__(
            self,
            in_channels: int,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            odd_padding=True,
            trainable=False,
            diagonal=False
    ) -> None:
        """Laplace Filter 2d.

        Applies the 3x3 Laplace operator.

        References:
            - https://en.wikipedia.org/wiki/Discrete_Laplace_operator

        Args:
            in_channels: Number of input channels.
            stride: Stride.
            padding: Padding.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
            device: Device.
            dtype: Data type.
            odd_padding: Whether to apply one-sided padding to account for even kernel sizes.
            trainable: Whether the kernel should be trainable.
            diagonal: Whether to use a kernel that includes diagonals.
        """
        super().__init__(in_channels=in_channels, kernel=self.get_kernel2d(diagonal),
                         stride=stride, padding=padding, dilation=dilation, odd_padding=odd_padding,
                         trainable=trainable, padding_mode=padding_mode, device=device, dtype=dtype)

    @staticmethod
    def get_kernel2d(diagonal=False):
        if diagonal:
            laplace = torch.as_tensor([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1.],
            ])
        else:
            laplace = torch.as_tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0.],
            ])
        return laplace


class UpFilter2d(nn.Module):
    def __init__(
            self,
            module,
            scale_factor: int = 2,
    ) -> None:
        """Upsample Filter 2d.

        This Module performs the upsampling step of a typical image pyramid.
        First, it upsamples the input Tensor by injecting zeros as columns and rows, then it applies the given
        ``module``, which could be for example a ``cd.models.PascalFilter2d``.

        Args:
            module: Filter module.
            scale_factor: Scale factor.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.filter = module

    def forward(self, x: Tensor) -> Tensor:
        x = strided_upsampling2d(x, self.scale_factor)
        return self.filter(x)


class EdgeFilter2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 1,
            dilation: _size_2_t = 1,
            padding_mode: str = 'replicate',
            device=None,
            dtype=None,
            magnitude=True,
            trainable=False,
            **kwargs
    ):
        """Edge Filter 2d.

        Find edges in an image using the Sobel filter.

        Args:
            in_channels: Number of input channels.
            stride: Stride.
            padding: Padding. Default: 1.
            dilation: Spacing between kernel elements.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'replicate'``.
            device: Device.
            dtype: Data type.
            trainable: Whether the kernel should be trainable.
            magnitude: Whether to compute the magnitude image.
            **kwargs: Additional keyword arguments for ``cd.models.Filter2d``.
        """
        super().__init__()
        self.axis0 = Filter2d(in_channels, SobelFilter2d.get_kernel2d(False) / 4,
                              stride=stride, dilation=dilation, device=device, dtype=dtype, trainable=trainable,
                              padding=padding, padding_mode=padding_mode, **kwargs)
        self.axis1 = Filter2d(in_channels, SobelFilter2d.get_kernel2d(True) / 4,
                              stride=stride, dilation=dilation, device=device, dtype=dtype, trainable=trainable,
                              padding=padding, padding_mode=padding_mode, **kwargs)
        self.magnitude = magnitude

    def forward(self, x):
        exp = int(self.magnitude) + 1
        return ((self.axis0(x) ** exp + self.axis1(x) ** exp) ** (1 / exp)) / np.sqrt(exp)
