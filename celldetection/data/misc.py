import numpy as np
import torch
from collections import OrderedDict
from skimage import img_as_ubyte


def transpose_spatial(inputs: np.ndarray, inputs_channels_last=True, spatial_dims=2, has_batch=False):
    if spatial_dims == 0:
        return inputs
    has_batch = bool(has_batch)
    a = ([0] * has_batch)
    if inputs_channels_last:
        # e. g. (0, 3, 1, 2)
        b = list(range(spatial_dims + has_batch, inputs.ndim))  # n channels
        c = list(range(has_batch, spatial_dims + has_batch))  # spatial dims
    else:
        b = list(range(inputs.ndim - spatial_dims, inputs.ndim))  # spatial dims
        c = list(range(has_batch, inputs.ndim - spatial_dims - (1 - has_batch)))  # n channels
        # e.g. (0, 2, 3, 1)
    print(a, b, c)
    return np.transpose(inputs, a + b + c)


def channels_last2channels_first(inputs: np.ndarray, spatial_dims=2, has_batch=False) -> np.ndarray:
    """Channels last to channels first.

    Args:
        inputs: Input array.
        spatial_dims: Number of spatial dimensions.
        has_batch: Whether inputs has a batch dimension.

    Returns:
        Transposed array.
    """
    return transpose_spatial(inputs, inputs_channels_last=True, spatial_dims=spatial_dims, has_batch=has_batch)


def channels_first2channels_last(inputs: np.ndarray, spatial_dims=2, has_batch=False) -> np.ndarray:
    """Channels first to channels last.

    Args:
        inputs: Input array.
        spatial_dims: Number of spatial dimensions.
        has_batch: Whether inputs has a batch dimension.

    Returns:
        Transposed array.
    """
    return transpose_spatial(inputs, inputs_channels_last=False, spatial_dims=spatial_dims, has_batch=has_batch)


def to_tensor(inputs: np.ndarray, spatial_dims=2, transpose=False, has_batch=False, dtype=None,
              device=None) -> torch.Tensor:
    """Array to Tensor.

    Converts numpy array to Tensor and optionally transposes from channels last to channels first.

    Args:
        inputs: Input array.
        transpose: Whether to transpose channels from channels last to channels first.
        spatial_dims: Number of spatial dimensions.
        has_batch: Whether inputs has a batch dimension.
        dtype: Data type of output Tensor.
        device: Device of output Tensor.

    Returns:
        Tensor.
    """
    return torch.as_tensor(
        channels_last2channels_first(inputs, spatial_dims=bool(transpose) * spatial_dims, has_batch=has_batch),
        device=device, dtype=dtype)


def universal_dict_collate_fn(batch) -> OrderedDict:
    results = OrderedDict({})
    ref = batch[0]
    for k in ref.keys():
        if isinstance(ref[k], (list, tuple)):
            max_dim = np.max([b[k][0].shape[0] for b in batch])
            results[k] = np.stack(
                [np.pad(b[k][0], ((0, max_dim - b[k][0].shape[0]),) + ((0, 0),) * (b[k][0].ndim - 1)) for b in batch],
                axis=0)
            results[k] = to_tensor(results[k], transpose=False, spatial_dims=0, has_batch=True)
        else:
            results[k] = np.stack([b[k] for b in batch], axis=0)
            results[k] = to_tensor(results[k], transpose=True, spatial_dims=2, has_batch=True)
    return results


def normalize_percentile(image, percentile=99.9, to_uint8=True):
    low, high = np.percentile(image, (100 - percentile, percentile))
    img = (np.clip(image, low, high) - low) / (high - low)
    return img_as_ubyte(img) if to_uint8 else img


def random_crop(*arrays, height, width=None):
    """Random crop.

    Args:
        *arrays: Input arrays that are to be cropped. None values accepted.
            The shape of the first element is used as reference.
        height: Output height.
        width: Output width. Default is same as height.

    Returns:
        Cropped array if `arrays` contains a single array; a list of cropped arrays otherwise
    """
    if len(arrays) <= 0:
        return None
    if width is None:
        width = height
    h, w = arrays[0].shape[:2]
    hh, ww = h - height, w - width
    a, b = np.random.randint(0, hh) if hh > 0 else 0, np.random.randint(0, ww) if ww > 0 else 0
    slices = (
        slice(a, a + height),
        slice(b, b + width)
    )
    results = [(None if v is None else v[slices]) for v in arrays]
    if len(results) == 1:
        results, = results
    return results


def rle2mask(code, shape, transpose=True, min_index=1, constant=1) -> np.ndarray:
    """Run length encoding to mask.

    Convert run length encoding to mask image.

    Args:
        code: Run length code.
            As ndarray: array([idx0, len0, idx1, len1, ...]) or array([[idx0, len0], [idx1, len1], ...])
            As list: [idx0, len0, idx1, len1, ...] or [[idx0, len0], [idx1, len1], ...]
            As str: 'idx0 len0 idx1 len1 ...'
        shape: Mask shape.
        transpose: If True decode row by row, otherwise decode column by column.
        min_index: Smallest pixel index. Depends on rle encoding.
        constant: Pixels marked by rle are set to this value.

    Returns:
        Mask image.

    """
    image = np.zeros(np.multiply.reduce(shape))
    code = np.array([int(i) for i in code.split(' ')] if isinstance(code, str) else code).ravel()
    c0 = code.shape[0]
    assert c0 % 2 == 0
    for i in range(0, c0, 2):
        idx, le = code[i:i + 2]
        idx -= min_index
        image[idx:idx + le] = constant
    image = np.reshape(image, shape)
    if transpose:
        image = image.T
    return image
