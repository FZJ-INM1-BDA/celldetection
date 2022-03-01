import numpy as np
import torch
from collections import OrderedDict
from skimage import img_as_ubyte
from ..util.util import get_device
from skimage import measure

__all__ = ['to_tensor', 'transpose_spatial', 'universal_dict_collate_fn', 'normalize_percentile', 'random_crop',
           'channels_last2channels_first', 'channels_first2channels_last', 'ensure_tensor', 'rgb_to_scalar',
           'padding_stack', 'labels2crops', 'rle2mask']


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
    if labels.ndim == 2:
        labels = labels[..., None]
    crops = []
    masks = []
    for z in range(labels.shape[2]):
        for p in measure.regionprops(labels[:, :, z]):
            if p.label <= 0:
                continue
            y0, x0, y1, x1 = p.bbox  # half-open interval [min_row; max_row) and [min_col; max_col)
            crops.append(image[y0:y1, x0:x1])
            masks.append(p.image)
    return crops, masks
