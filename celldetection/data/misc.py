import numpy as np
import torch
from collections import OrderedDict
from skimage import img_as_ubyte


def transpose_spatial(inputs, spatial_dims=2, has_batch=False):
    if spatial_dims == 0:
        return inputs
    return np.transpose(inputs, ([0] * has_batch) + list(range(spatial_dims + has_batch, inputs.ndim)) + list(
        range(has_batch, spatial_dims + has_batch)))


def to_tensor(inputs, spatial_dims=2, has_batch=False):
    return torch.as_tensor(transpose_spatial(inputs, spatial_dims=spatial_dims, has_batch=has_batch))


def universal_dict_collate_fn(batch):
    results = OrderedDict({})
    ref = batch[0]
    for k in ref.keys():
        if isinstance(ref[k], (list, tuple)):
            max_dim = np.max([b[k][0].shape[0] for b in batch])
            results[k] = np.stack(
                [np.pad(b[k][0], ((0, max_dim - b[k][0].shape[0]),) + ((0, 0),) * (b[k][0].ndim - 1)) for b in batch],
                axis=0)
            results[k] = to_tensor(results[k], 0, True)
        else:
            results[k] = np.stack([b[k] for b in batch], axis=0)
            results[k] = to_tensor(results[k], 2, True)
    return results


def normalize_percentile(image, percentile=99.9):
    low, high = np.percentile(image, (100 - percentile, percentile))
    return img_as_ubyte((np.clip(image, low, high) - low) / (high - low))


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
