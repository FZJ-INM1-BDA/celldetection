import torch
import torch.nn.functional as F
from typing import List

__all__ = ['downsample_labels']


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
