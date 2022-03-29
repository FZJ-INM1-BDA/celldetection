import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List

__all__ = ['downsample_labels', 'padded_stack2d', 'split_spatially']


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
