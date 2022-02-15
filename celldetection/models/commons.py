import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tanh, no_grad, as_tensor
from ..util.util import gaussian_kernel

__all__ = ['TwoConvBnRelu', 'ScaledTanh', 'GaussianBlur', 'ReplayCache']


class GaussianBlur(nn.Conv2d):
    def __init__(self, in_channels, kernel_size=3, sigma=-1, padding='same', padding_mode='reflect',
                 requires_grad=False, **kwargs):
        self._kernel = gaussian_kernel(kernel_size, sigma)
        super().__init__(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding,
                         padding_mode=padding_mode, bias=False, requires_grad=requires_grad, **kwargs)

    def reset_parameters(self):
        with no_grad():
            as_tensor(self._kernel, dtype=self.weight.dtype)


class TwoConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None, **kwargs):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ScaledTanh(nn.Module):
    def __init__(self, factor):
        super(ScaledTanh, self).__init__()
        self.factor = factor

    def forward(self, inputs: Tensor) -> Tensor:
        return tanh(inputs) * self.factor


class ReplayCache:
    def __init__(self, size=128):
        self.cache = []
        self.size = size

    def __len__(self):
        return len(self.cache)

    def is_empty(self):
        return len(self) <= 0

    def append(self, x, fraction=.5):
        lx = len(x)
        for i in np.random.choice(np.arange(lx), int(lx * fraction), replace=False):
            self.cache.append(x[i].detach())
        while len(self) > self.size:
            del self.cache[np.random.randint(0, len(self))]

    def __call__(self, num):
        if self.is_empty():
            return None
        return torch.stack([self.cache[i] for i in np.random.randint(0, len(self), num)], 0)

    add = append
