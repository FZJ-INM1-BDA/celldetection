import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple, List, Dict


def rel_location2abs_location(locations, cache: Dict[str, Tensor] = None, cache_size: int = 16):
    """

    Args:
        locations: Tensor[..., 2, h, w]. In xy format.
        cache: can be None.
        cache_size:

    Returns:

    """
    d = locations.device
    (h, w) = locations.shape[-2:]
    offset = None
    if cache is not None:
        key = str((h, w, d))
        if key in cache.keys():
            offset = cache[key]
    if offset is None:
        offset = torch.stack((torch.arange(w, device=d)[None] + torch.zeros(h, device=d)[:, None],
                              torch.zeros(w, device=d)[None] + torch.arange(h, device=d)[:, None]), 0)
        if cache is not None:
            cache[str((h, w, d))] = offset
    if cache is not None and len(cache) > cache_size:
        del cache[list(cache.keys())[0]]
    r = locations + offset
    return r


def fouriers2contours(fourier, locations, samples=64, sampling=None, cache: Dict[str, Tensor] = None,
                      cache_size: int = 16):
    """

    Args:
        fourier: Tensor[..., order, 4]
        locations: Tensor[..., 2]
        samples: Number of samples. Only used for default sampling, ignored otherwise.
        sampling: Sampling t. Default is linspace 0..1. Device should match `fourier` and `locations`.
        cache: Cache for initial zero tensors. When fourier shapes are consistent this can increase execution times.
        cache_size: Cache size.

    Returns:
        Contours.
    """

    if isinstance(fourier, (tuple, list)):
        if sampling is None:
            sampling = [sampling] * len(fourier)
        return [fouriers2contours(f, l, samples=samples, sampling=s) for f, l, s in zip(fourier, locations, sampling)]

    order = fourier.shape[-2]
    d = fourier.device
    sampling_ = sampling
    if sampling is None:
        sampling = sampling_ = torch.linspace(0, 1.0, samples, device=d)
    samples = sampling.shape[-1]
    sampling = sampling[..., None, :]

    # shape: (order, samples)
    c = float(np.pi) * 2 * (torch.arange(1, order + 1, device=d)[..., None]) * sampling

    # shape: (order, samples)
    c_cos = torch.cos(c)
    c_sin = torch.sin(c)

    # shape: fourier.shape[:-2] + (samples, 2)
    con = None
    con_shape = fourier.shape[:-2] + (samples, 2)
    con_key = str(tuple(con_shape) + (d,))
    if cache is not None:
        con = cache.get(con_key, None)
    if con is None:
        con = torch.zeros(fourier.shape[:-2] + (samples, 2), device=d)  # 40.1 ms for size (520, 696, 64, 2) to cuda
        if cache is not None:
            if len(cache) >= cache_size:
                del cache[next(iter(cache.keys()))]
            cache[con_key] = con
    con = con + locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con, sampling_


def get_scale(actual_size, original_size, flip=True, dtype=torch.float):
    scale = (torch.as_tensor(original_size, dtype=dtype) /
             torch.as_tensor(actual_size, dtype=dtype))
    if flip:
        scale = scale.flip(-1)
    return scale


def scale_contours(actual_size, original_size, contours):
    """

    Args:
        actual_size: Image size. E.g. (256, 256)
        original_size: Original image size. E.g. (512, 512)
        contours: Contours that are to be scaled to from `actual_size` to `original_size`.
            E.g. array of shape (1, num_points, 2) for a single contour or tuple/list of (num_points, 2) arrays.
            Last dimension is interpreted as (x, y).

    Returns:
        Rescaled contours.
    """

    assert len(actual_size) == len(original_size)
    scale = get_scale(actual_size, original_size, flip=True)

    if isinstance(contours, Tensor):
        contours = contours * scale.to(contours.device)
    else:
        assert isinstance(contours, (tuple, list))
        scale = scale.to(contours[0].device)
        for i in range(len(contours)):
            contours[i] = contours[i] * scale
    return contours


def _scale_fourier(fourier, location, scale):
    fourier[..., [0, 1]] = fourier[..., [0, 1]] * scale[0]
    fourier[..., [2, 3]] = fourier[..., [2, 3]] * scale[1]
    location = location * scale
    return fourier, location


def scale_fourier(actual_size, original_size, fourier, location):
    """

    Args:
        actual_size: Image size. E.g. (256, 256)
        original_size: Original image size. E.g. (512, 512)
        fourier: Fourier descriptor. E.g. array of shape (..., order, 4).
        location: Location. E.g. array of shape (..., 2). Last dimension is interpreted as (x, y).

    Returns:
        Rescaled fourier, rescaled location
    """
    assert len(actual_size) == len(original_size)
    scale = get_scale(actual_size, original_size, flip=True)

    if isinstance(fourier, Tensor):
        return _scale_fourier(fourier, location, scale.to(fourier.device))
    else:
        assert isinstance(fourier, (list, tuple))
        scale = scale.to(fourier[0].device)
        rfo, rlo = [], []
        for fo, lo in zip(fourier, location):
            a, b = _scale_fourier(fo, lo, scale)
            rfo.append(a)
            rlo.append(b)
        return rfo, rlo


def batched_box_nms(
        boxes: List[Tensor], scores: List[Tensor],
        *args,
        iou_threshold: float
) -> Tuple[List[Tensor], ...]:
    assert len(scores) == len(boxes)
    cons = []
    scos = []
    further = ()
    if len(args) > 0:
        further = [[] for _ in range(len(args))]
    for it in zip(*(boxes, scores) + tuple(args)):
        con, sco = it[:2]
        indices = torch.ops.torchvision.nms(con, sco, iou_threshold=iou_threshold)
        cons.append(con[indices])
        scos.append(sco[indices])
        for j, res_ in enumerate(it[2:]):
            further[j].append(res_[indices])
    return (cons, scos) + tuple(further)


def order_weighting(order, max_w=5, min_w=1, spread=None) -> torch.Tensor:
    x = torch.arange(order).float()
    if spread is None:
        spread = order - 1
    y = min_w + (max_w - min_w) * (1 - (x / spread).clamp(0., 1.)) ** 2
    return y[:, None]  # broadcastable to (n, order, 4)


def refinement_bucket_weight(index, base_index):
    dist = torch.abs(index + 0.5 - base_index)
    sel = dist > 1
    dist = 1. - dist
    dist[sel] = 0
    dist.detach_()
    return dist


def resolve_refinement_buckets(samplings, num_buckets):
    base_index = samplings * num_buckets
    base_index_int = base_index.long()
    a, b, c = base_index_int - 1, base_index_int, base_index_int + 1
    return (
        (a % num_buckets, refinement_bucket_weight(a, base_index)),
        (b % num_buckets, refinement_bucket_weight(b, base_index)),
        (c % num_buckets, refinement_bucket_weight(c, base_index))
    )
