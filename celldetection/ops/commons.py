import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List

__all__ = ['downsample_labels', 'padded_stack2d', 'split_spatially', 'minibatch_std_layer']


def downsample_labels(inputs, size: List[int]):
    """

    Down-sample via max-pooling and interpolation

    Notes:
        - Downsampling can lead to loss of labeled instances, both during max pooling and interpolation.
        - Typical timing: 0.08106 ms for 256x256

    Args:
        inputs: Label Tensor to resize. Shape (n, c, h, w)
        size: Tuple containing target height and width.

    Returns:

    """
    sizeh, sizew = size  # for torchscript
    if inputs.shape[-2:] == (sizeh, sizew):
        return inputs
    if inputs.dtype != torch.float32:
        inputs = inputs.float()
    h, w = inputs.shape[-2:]
    th, tw = size
    k = h // th, w // tw
    r = F.max_pool2d(inputs, k, k)
    if r.shape[-2:] != (sizeh, sizew):
        r = F.interpolate(r, size, mode='nearest')
    return r


def padded_stack2d(*images, dim=0) -> Tensor:
    """Padding stack.

    Stacks 2d images along given axis.
    Spatial dimensions are padded according to largest height/width.

    Args:
        *images: Tensor[..., h, w]
        dim: Stack dimension.

    Returns:
        Tensor
    """
    ts = tuple(max((i.shape[j] for i in images)) for j in range(-2, 0))
    images = [F.pad(i, [0, ts[1] - i.shape[-1], 0, ts[0] - i.shape[-2]]) for i in images]
    return torch.stack(images, dim=dim)


def split_spatially(x, height, width=None):
    """Split spatially.

    Splits spatial dimensions of Tensor ``x`` into patches of size ``(height, width)`` and adds the patches
    to the batch dimension.

    Args:
        x: Input Tensor[n, c, h, w].
        height: Patch height.
        width: Patch width.

    Returns:
        Tensor[n * h//height * w//width, c, height, width].
    """
    width = width or height
    n, c, h, w = x.shape
    h_, w_ = h // height, w // width
    return x.view(n, c, h_, height, w_, width).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, height, width)


def minibatch_std_layer(x, channels=1, group_channels=None):
    """Minibatch standard deviation layer.

    The minibatch standard deviation layer first splits the batch dimension into slices of size ``group_channels``.
    The channel dimension is split into ``channels`` slices. For the groups the standard deviation is calculated and
    averaged over spatial dimensions and channel slice depth. The result is broadcasted to the spatial dimensions,
    repeated for the batch dimension and then concatenated to the channel dimension of ``x``.

    References:
        - https://arxiv.org/pdf/1710.10196.pdf

    Args:
        x: Input Tensor[n, c, h, w].
        channels: Number of averaged standard deviation channels.
        group_channels: Number of channels per group. Default: batch size.

    Returns:
        Tensor[n, c + channels, h, w].
    """
    n, c, h, w = x.shape
    gc = min(group_channels or n, n)
    cc, g = c // channels, n // gc
    y = x.view(gc, g, channels, cc, h, w)
    return torch.cat([x, y.std(0, False).mean([2, 3, 4], True).squeeze(-1).repeat(gc, 1, h, w)], 1)
