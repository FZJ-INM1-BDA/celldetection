import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List

__all__ = ['downsample_labels', 'padded_stack2d', 'split_spatially', 'minibatch_std_layer', 'strided_upsampling2d',
           'interpolate_vector', 'pad_to_div', 'pad_to_size']


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


def split_spatially(x, size):
    """Split spatially.

    Splits spatial dimensions of Tensor ``x`` into patches of given ``size`` and adds the patches
    to the batch dimension.

    Args:
        x: Input Tensor[n, c, h, w, ...].
        size: Patch size of the splits.

    Returns:
        Tensor[n * h//height * w//width, c, height, width].
    """
    n, c = x.shape[:2]
    spatial = x.shape[2:]
    nd = len(spatial)
    assert len(spatial) == len(size)
    v = n, c
    for cur, new in zip(spatial, size):
        v += (cur // new, new)
    perm = (0,) + tuple(range(2, nd * 2 + 1, 2)) + tuple(range(1, nd * 3, 2))
    return x.view(v).permute(perm).reshape((-1, c) + tuple(size))


def minibatch_std_layer(x, channels=1, group_channels=None, epsilon=1e-8):
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
        epsilon: Epsilon.

    Returns:
        Tensor[n, c + channels, h, w].
    """
    n, c, h, w = x.shape
    gc = min(group_channels or n, n)
    cc, g = c // channels, n // gc
    y = x.view(gc, g, channels, cc, h, w)
    y = y.var(0, False).add(epsilon).sqrt().mean([2, 3, 4], True).squeeze(-1).repeat(gc, 1, h, w)
    return torch.cat([x, y], 1)


def strided_upsampling2d(x, factor=2, const=0):
    """Strided upsampling.

    Upsample by inserting rows and columns filled with ``constant``.

    Args:
        x: Tensor[n, c, h, w].
        factor: Upsampling factor.
        const: Constant used to fill inserted rows and columns.

    Returns:
        Tensor[n, c, h*factor, w*factor].
    """
    n, c, h, w = x.shape
    x_ = torch.zeros((n, c, h * factor, w * factor), dtype=x.dtype, device=x.device)
    if const != 0:
        x_.fill_(const)
    x_[..., ::factor, ::factor] = x
    return x_


def interpolate_vector(v, size, **kwargs):
    """Interpolate vector.

    Args:
        v: Vector as ``Tensor[d]``.
        size: Target size.
        **kwargs: Keyword arguments for ``F.interpolate``

    Returns:

    """
    return torch.squeeze(torch.squeeze(
        F.interpolate(v[None, None], size, **kwargs), 0
    ), 0)


def pad_to_size(v, size, return_pad=False, **kwargs):
    """Pad tp size.

    Applies padding to end of each dimension.

    Args:
        v: Input Tensor.
        size: Size tuple. Last element corresponds to last dimension of input `v`.
        return_pad: Whether to return padding values.
        **kwargs: Additional keyword arguments for `F.pad`.

    Returns:
        Padded Tensor.
    """
    pad = []
    for a, b in zip(size, v.shape[-len(size):]):
        pad += [max(0, a - b), 0]
    if any(pad):
        v = F.pad(v, pad[::-1], **kwargs)
    if return_pad:
        return v, pad
    return v


def pad_to_div(v, div=32, nd=2, return_pad=False, **kwargs):
    """Pad to div.

    Applies padding to input Tensor to make it divisible by `div`.

    Args:
        v: Input Tensor.
        div: Div tuple. If single integer, `nd` is used to define number of dimensions to pad.
        nd: Number of dimensions to pad. Only used if `div` is not a tuple or list.
        return_pad: Whether to return padding values.
        **kwargs: Additional keyword arguments for `F.pad`.

    Returns:
        Padded Tensor.
    """
    if not isinstance(div, (tuple, list)):
        div = (div,) * nd
    size = [(i // d + bool(i % d)) * d for i, d in zip(v.shape[-len(div):], div)]
    return pad_to_size(v, size, return_pad=return_pad, **kwargs)
