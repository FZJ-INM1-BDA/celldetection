import torch.nn as nn
from torch import Tensor, tanh


__all__ = ['TwoConvBnRelu', 'ScaledTanh']


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
