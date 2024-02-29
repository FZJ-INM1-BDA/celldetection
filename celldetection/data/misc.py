import numpy as np
import torch
from collections import OrderedDict
from skimage import img_as_ubyte
from ..util.util import get_device
import cv2
from skimage import measure
import pandas as pd
import traceback

__all__ = ['to_tensor', 'transpose_spatial', 'universal_dict_collate_fn', 'normalize_percentile', 'random_crop',
           'channels_last2channels_first', 'channels_first2channels_last', 'ensure_tensor', 'rgb_to_scalar',
           'padding_stack', 'labels2crops', 'labels2properties', 'rle2mask', 'resample_contours',
           'labels2property_table', 'pad_to_size', 'pad_to_div', 'regionprops2d', 'split']


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
        # e.g. (0, 2, 3, 1)
        b = list(range(inputs.ndim - spatial_dims, inputs.ndim))  # spatial dims
        c = list(range(has_batch, inputs.ndim - spatial_dims))  # n channels
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


def ensure_tensor(x, device=None, dtype=torch.float32):
    """Ensure tensor.

    Mapping ndarrays to Tensor.
    Possible shape mappings:
    - (h, w) -> (1, 1, h, w)
    - (h, w, c) -> (1, c, h, w)
    - (b, c, h, w) -> (b, c, h, w)

    Args:
        x: Inputs.
        device: Either Device or a Module or Tensor to retrieve the device from.
        dtype: Data type.

    Returns:
        Tensor
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = x[:, :, None]
        b = to_tensor(x, device=None if device is None else get_device(device),
                      transpose=x.ndim == 3, has_batch=x.ndim == 4)
        if b.ndim == 3:
            b = b[None]
        if b.dtype != dtype:
            b = b.to(dtype=dtype)
    else:
        b = x
    return b


def padding_stack(*images, axis=0) -> np.ndarray:
    """Padding stack.

    Stack images along `axis`. If images have different shapes all images are padded to larges shape.

    Args:
        *images: Images.
        axis: Axis used for stacking.

    Returns:
        Array
    """
    if len(images) == 1 and isinstance(images[0], (list, tuple)):
        images, = images
    shapes = np.array([i.shape for i in images])
    pad = np.any([np.unique(shapes[:, col].size > 1 for col in range(shapes.shape[1]))])
    if pad:
        target_shape = np.max(shapes, 0)
        images = [np.pad(i, [(0, ts - s) for s, ts in zip(i.shape, target_shape)]) for i in images]
    return np.stack(images, axis=axis)


def universal_dict_collate_fn(batch, check_padding=True) -> OrderedDict:
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
            items = [b[k] for b in batch]
            if check_padding:
                results[k] = padding_stack(*items, axis=0)
            else:
                results[k] = np.stack(items, axis=0)
            results[k] = to_tensor(results[k], transpose=True, spatial_dims=2, has_batch=True)
    return results


def normalize_percentile(image, percentile=99.9, to_uint8=True):
    if not isinstance(percentile, (list, tuple)):
        percentile = (100 - percentile, percentile)
    low, high = np.percentile(image, percentile)
    img = (np.clip(image, low, high) - low) / (high - low)
    return img_as_ubyte(img) if to_uint8 else img


def _legacy_random_crop(*arrays, height, width=None):
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


def random_crop(inputs, size=None, *args, return_coords=False, return_slices=False, **kwargs):
    if 'height' in kwargs or 'width' in kwargs:
        if size is None:
            return _legacy_random_crop(inputs, *args, **kwargs)
        else:
            return _legacy_random_crop(inputs, size, *args, **kwargs)
    assert size is not None, 'Specify a targeted size for cropping.'
    reference_size = (inputs[0] if isinstance(inputs, (tuple, list)) else inputs).shape[:len(size)]
    size = [(np.random.randint(*i) if isinstance(i, tuple) else i) for i in size]
    diffs = [a - b for a, b in zip(reference_size, size)]
    coords = [(np.random.randint(0, d) if d > 0 else 0) for d in diffs]
    slices = tuple(slice(a, a + s) for a, s in zip(coords, size))

    if isinstance(inputs, (list, tuple)):
        res = tuple((None if i is None else i[slices]) for i in inputs)
    else:
        res = inputs[slices]

    meta = tuple(i for i, c in ((coords, return_coords), (slices, return_slices)) if c)
    if len(meta):
        return res, meta
    return res


def random_pad(*arrays, size, mode='constant', **kwargs):
    if len(arrays) <= 0:
        return None
    reference = arrays[0].shape[:len(size)]
    padding = [max(size[i] - reference[i], 0) for i in range(len(size))]
    start = [int(np.random.uniform() * p) for p in padding]
    end = [p - s for p, s in zip(padding, start)]
    p = [[a, b] for a, b in zip(start, end)]
    results = [np.pad(i, p + [[0, 0]] * (i.ndim - len(p)), mode=mode, **kwargs) for i in arrays]
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


def rgb_to_scalar(inputs: np.ndarray, dtype='int32'):
    """RGB to scalar.

    Convert RGB data to scalar, while maintaining color uniqueness.

    Args:
        inputs: Input array. Shape ([d1, d2, ..., dn,] 3)
        dtype: Data type

    Returns:
        Output array. Shape ([d1, d2, ..., dn])
    """
    red, green, blue = inputs[..., 0], inputs[..., 1], inputs[..., 2]
    rgb = red.astype(dtype)
    rgb = (rgb << 8) + green
    rgb = (rgb << 8) + blue
    return rgb


def _labels2properties(p, results, label, properties):
    if p.label > 0:
        results.append([getattr(p, k) for k in properties])
        label.append(p.label)


def labels2properties(labels: 'np.ndarray', *properties, iter_channels=True, **kwargs):
    """Labels to properties.

    References:
        [1] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        labels: Label image.
        *properties: Property names. See [1] for details.
        iter_channels: Whether to iterate channel axis of label image. If `False` label image is processed as is.
        **kwargs: Keyword arguments for `skimage.measure.regionprops`.

    Returns:
        List of property lists.
    """
    if len(properties) == 1 and isinstance(properties[0], (list, tuple)):
        properties, = properties
    if labels.ndim == 2 and iter_channels:
        labels = labels[..., None]
    label = []
    results = []
    if iter_channels:
        for z in range(labels.shape[2]):
            for p in measure.regionprops(labels[..., z], **kwargs):
                _labels2properties(p, results, label, properties)
    else:
        for p in measure.regionprops(labels, **kwargs):
            _labels2properties(p, results, label, properties)
    return [a for _, a in sorted(zip(label, results))]


def labels2property_table(labels: 'np.ndarray', *properties, iter_channels=True, **kwargs) -> 'pd.DataFrame':
    """Labels to property table.

    References:
        [1] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        labels: Label image.
        *properties: Property names. See [1] for details.
        iter_channels: Whether to iterate channel axis of label image. If `False` label image is processed as is.
        **kwargs: Keyword arguments for `skimage.measure.regionprops_table`.

    Returns:
        Table (pd.DataFrame) of properties.
    """
    if len(properties) == 1 and isinstance(properties[0], (list, tuple)):
        properties, = properties
    if labels.ndim == 2 and iter_channels:
        labels = labels[..., None]
    df_kwargs = kwargs.pop('df_kwargs', {})
    if iter_channels:
        tab = None
        for z in range(labels.shape[2]):
            tab_ = pd.DataFrame(measure.regionprops_table(labels[..., z], properties=properties, **kwargs), **df_kwargs)
            tab = pd.concat((tab, tab_))
    else:
        tab = pd.DataFrame(measure.regionprops_table(labels, properties=properties, **kwargs), **df_kwargs)
    return tab


def labels2crops(labels: np.ndarray, image: np.ndarray):
    """Labels to crops.

    Crop all objects that are represented in ``labels`` from given ``image`` and return a list of all image crops,
    and a list of masks, each marking object pixels for respective crop.

    Args:
        labels: Label image. Array[h, w(, c)].
        image: Image. Array[h, w, ...].

    Returns:
        (crop_list, mask_list)
    """
    crops = []
    masks = []
    for (y0, x0, y1, x1), mask in labels2properties(labels, 'bbox', 'image'):
        crops.append(image[y0:y1, x0:x1])
        masks.append(mask)
    return crops, masks


def resample_contours(contours, num=None, close=True, epsilon=1e-6):
    """Resample contour.

    Sample ´´num´´ equidistant points on each contour in ``contours``.

    Notes:
        - Works for closed and open contours.

    Args:
        contours: Contours to sample from. Array[..., num', 2] or list of Arrays.
        num: Number of points.
        close: Set True if ``contours`` contains closed contours, with the end point not being equal to the start point.
            Set False otherwise.
        epsilon: Epsilon.

    Returns:
        Array[..., num, 2] or list of Arrays.
    """
    if isinstance(contours, (list, tuple)):
        return type(contours)([resample_contours(c, num=num, close=close, epsilon=epsilon) for c in contours])
    if close:  # should be closed for this implementation to work
        contours = np.concatenate((contours, contours[..., :1, :]), -2)
    dxy = np.diff(contours, axis=-2)  # shape: (..., p, d)
    dt = np.sqrt(np.sum(np.square(dxy), axis=-1)) + epsilon  # shape: (..., p)
    cumsum = np.cumsum(dt, axis=-1)  # shape: (..., p)
    if num is None or isinstance(num, float):
        num = int(np.max(np.round(cumsum[..., -1])) * (num if isinstance(num, float) else 1))
    cumsum0 = np.concatenate((np.zeros_like(cumsum[..., :1]), cumsum), -1)
    ts = np.linspace(0, cumsum[..., -1], num + 1, axis=-1)[..., :-1]
    v = ts[..., :, None] <= cumsum[..., None, :]
    idx = np.where(v.max(-1))[:-1] + (np.argmax(v, axis=-1).ravel(),)
    alpha = ((ts - cumsum0[idx].reshape(*ts.shape)) / dt[idx].reshape(*ts.shape))[..., None]
    shape = contours.shape[:-2] + (num, 2)
    sample = contours[idx].reshape(shape) * (1 - alpha) + contours[idx[:-1] + (idx[-1] + 1,)].reshape(shape) * alpha
    return sample


def rescale_image(img, scale, **kwargs):
    target_size = tuple(np.round(np.array(img.shape[:2]) * scale).astype('int'))
    return cv2.resize(img, target_size[::-1], **kwargs)


def pad_to_size(v, size, **kwargs):
    """Pad tp size.

    Applies padding to end of each dimension.

    Args:
        v: Input array.
        size: Size tuple. First element corresponds to first dimension of input `v`.
        **kwargs: Additional keyword arguments for `np.pad`.

    Returns:
        Padded Array.
    """
    pad = [[0, max(0, a - b)] for a, b in zip(size, v.shape)]
    pad += [[0, 0]] * (len(v.shape) - len(pad))
    return np.pad(v, pad, **kwargs)


def pad_to_div(v, div=32, nd=2, **kwargs):
    """Pad to div.

    Applies padding to input Array to make it divisible by `div`.

    Args:
        v: Input array.
        div: Div tuple. If single integer, `nd` is used to define number of dimensions to pad.
        nd: Number of dimensions to pad. Only used if `div` is not a tuple or list.
        **kwargs: Additional keyword arguments for `np.pad`.

    Returns:
        Padded Array.
    """
    if not isinstance(div, (tuple, list)):
        div = (div,) * nd
    size = [(i // d + bool(i % d)) * d for i, d in zip(v.shape, div)]
    return pad_to_size(v, size, **kwargs)


def regionprops2d(
        label_image,
        intensity_image=None,
        cache=True,
        *,
        extra_properties=None,
        spacing=None,
        offset=None,
):
    """Regionprops 2d.

    Helper function that allows to use `skimage.measure.regionprops` with label images that have channels.

    Note:
        Labels may not yield in order!

    Args:
        label_image: Array[h, w] or Array[h, w, c].
        intensity_image:
        cache:
        extra_properties:
        spacing:
        offset:

    Returns:

    """
    assert label_image.ndim in (2, 3)
    if label_image.ndim == 2:
        label_image = label_image[..., None]
    for z in range(label_image.shape[2]):
        label_image_ = label_image[..., z]
        for p in measure.regionprops(label_image_, intensity_image=intensity_image, cache=cache,
                                     extra_properties=extra_properties,
                                     spacing=spacing, offset=offset):
            yield p


def split(n: int, *splits, shuffle=True, seed=None):
    """Split.

    Splits a range of indices into multiple sets based on the given fractions.

    Args:
        n: The total number of indices.
        *splits: Variable length list of floats representing the fraction of the dataset for each split.
        shuffle: Whether to shuffle the indices before splitting.
        seed: Seed for the random number generator.

    Returns:
        Split indices.
    """
    if sum(splits) != 1:
        raise ValueError("The sum of splits must be equal to 1.")

    indices = np.arange(n)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    split_indices = []
    start = 0
    for i, sp in enumerate(splits):
        end = n if (i == len(splits) - 1) else start + int(round(n * sp))
        split_indices.append(indices[start:end])
        start = end

    assert sum([len(i) for i in split_indices]) == n
    assert np.unique(np.concatenate(split_indices)).size == n

    return split_indices


def pad_arrays(arrays):
    if not arrays:
        return []

    # Finding the maximum shape
    max_shape = np.max([np.array(a.shape) for a in arrays], axis=0)

    # Padding each array
    padded_arrays = []
    for a in arrays:
        # Calculate the padding needed for each dimension
        padding = [(0, max_s - s) for s, max_s in zip(a.shape, max_shape)]
        padded_a = np.pad(a, padding, mode='constant')
        padded_arrays.append(padded_a)

    return padded_arrays