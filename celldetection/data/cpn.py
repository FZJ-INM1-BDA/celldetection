import numpy as np
import cv2
import torch
from skimage.measure import regionprops
from collections import OrderedDict
from .segmentation import filter_instances_
from .misc import labels2properties
from ..util.util import asnumpy

__all__ = [
    'CPNTargetGenerator', 'contours2labels', 'render_contour', 'clip_contour_', 'masks2labels',
    'contours2boxes', 'contours2properties', 'resolve_label_channels',
    'filter_contours_by_intensity', 'draw_contours'
]


def efd(contour, order=10, epsilon=1e-6):
    """Elliptic fourier descriptor.

    Computes elliptic fourier descriptors from contour data.

    Args:
        contour: Tensor of shape (..., num_points, 2). Should be set of `num_points` 2D points that describe the contour
            of an object. Based on each contour a descriptor of shape (order, 4) is computed. The result has thus
            a shape of (..., order, 4).
            As `num_points` may differ from one contour to another a list of (num_points, 2) arrays may be passed
            as a numpy array with `object` as its data type, i.e. `np.array(list_of_contours)`.
        order: Order of resulting descriptor. The higher the order, the more detail can be preserved. An order of 1
            produces ellipses.
        epsilon: Epsilon value. Used to avoid division by zero.

    Notes:
        Locations may contain NaN if `contour` only contains a single point.

    Returns:
        Tensor of shape (..., order, 4).
    """
    if isinstance(contour, np.ndarray) and contour.dtype == object:
        r = [efd(c, order=order, epsilon=epsilon) for c in contour]
        if all([isinstance(r_, tuple) and len(r_) == len(r[0]) for r_ in r]):
            res = [[] for _ in range(len(r[0]))]
            for r_ in r:
                for i in range(len(res)):
                    res[i].append(r_[i])
            return tuple(map(np.array, res))
    dxy = np.diff(contour, axis=-2)  # shape: (..., p, d)
    dt = np.sqrt(np.sum(np.square(dxy), axis=-1)) + epsilon  # shape: (..., p)
    cumsum = np.cumsum(dt, axis=-1)  # shape: (..., p)
    zero = np.zeros(cumsum.shape[:-1] + (1,))
    t = np.concatenate([zero, cumsum], axis=-1)  # shape: (..., p + 1)
    sampling = t[..., -1:]  # shape: (..., 1)
    T_ = t[..., -1]  # shape: (...,)
    phi = (2 * np.pi * t) / sampling  # shape: (..., p + 1)
    orders = np.arange(1, order + 1, dtype=phi.dtype)  # shape: (order,)
    constants = sampling / (2. * np.square(orders) * np.square(np.pi))
    phi = np.expand_dims(phi, -2) * np.expand_dims(orders, -1)
    d_cos_phi = np.cos(phi[..., 1:]) - np.cos(phi[..., :-1])
    d_sin_phi = np.sin(phi[..., 1:]) - np.sin(phi[..., :-1])

    dxy0_dt = np.expand_dims(dxy[..., 0] / dt, axis=-2)
    dxy1_dt = np.expand_dims(dxy[..., 1] / dt, axis=-2)
    coefficients = np.stack([
        constants * np.sum(dxy0_dt * d_cos_phi, axis=-1),
        constants * np.sum(dxy0_dt * d_sin_phi, axis=-1),
        constants * np.sum(dxy1_dt * d_cos_phi, axis=-1),
        constants * np.sum(dxy1_dt * d_sin_phi, axis=-1),
    ], axis=-1)

    xi = np.cumsum(dxy[..., 0], axis=-1) - (dxy[..., 0] / dt) * t[..., 1:]
    delta = np.cumsum(dxy[..., 1], axis=-1) - (dxy[..., 1] / dt) * t[..., 1:]
    t_diff = np.diff(t ** 2, axis=-1)
    dt2 = 2 * dt
    a0 = (1 / T_) * np.sum(((dxy[..., 0] / dt2) * t_diff) + xi * dt, axis=-1)
    c0 = (1 / T_) * np.sum(((dxy[..., 1] / dt2) * t_diff) + delta * dt, axis=-1)
    return np.array(coefficients), np.stack((contour[..., 0, 0] + a0, contour[..., 0, 1] + c0), axis=-1)


def labels2contours(labels, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE, flag_fragmented_inplace=False,
                    raise_fragmented=True, constant=-1) -> dict:
    """Labels to contours.

    Notes:
        - If ``flag_fragmented_inplace is True``, ``labels`` may be modified inplace.

    Args:
        labels:
        mode:
        method: Contour method. CHAIN_APPROX_NONE must be used if contours are used for CPN.
        flag_fragmented_inplace: Whether to flag fragmented labels. Flagging sets labels that consist of more than one
            connected component to ``constant``.
        constant: Flagging constant.
        raise_fragmented: Whether to raise ValueError when encountering fragmented labels.

    Returns:
        dict
    """
    crops = []
    contours = OrderedDict()
    for channel in np.split(labels, labels.shape[2], 2):
        crops += [(p.label, p.image) + p.bbox[:2] for p in regionprops(channel)]
    for label, crop, oy, ox in crops:
        crop.dtype = np.uint8
        r = cv2.findContours(crop, mode=mode, method=method, offset=(ox, oy))
        if len(r) == 3:  # be compatible with both existing versions of findContours
            _, c, _ = r
        elif len(r) == 2:
            c, _ = r
        else:
            raise NotImplementedError('try different cv2 version')
        try:
            c, = c
        except ValueError as ve:
            if flag_fragmented_inplace:
                labels[labels == label] = constant
            elif raise_fragmented:
                raise ValueError('Object labeled with multiple connected components.')
            continue
        if len(c) == 1:
            c = np.concatenate((c, c), axis=0)  # min len for other functions to work properly
        contours[label] = c
    if labels.shape[2] > 1:
        return OrderedDict(sorted(contours.items()))
    return contours


def labels2contour_list(labels, **kwargs) -> list:
    if labels.ndim == 2:
        labels = labels[..., None]
    return [np.squeeze(i, 1) for i in list(labels2contours(labels, **kwargs).values())]


def masks2labels(masks, connectivity=8, label_axis=2, count=False, reduce=np.max, keepdims=True, **kwargs):
    """Masks to labels.

    Notes:
        ~ 11.7 ms for Array[25, 256, 256]. For same array skimage.measure.label takes ~ 17.9 ms.

    Args:
        masks: List[Array[height, width]] or Array[num_masks, height, width]
        connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
        label_axis: Axis used for stacking label maps. One per mask.
        count: Whether to count and return the number of components.
        reduce: Callable used to reduce `label_axis`. If set to None, `label_axis` will not be reduced.
            Can be used if instances do not overlap.
        **kwargs: Kwargs for cv2.connectedComponents.

    Returns:
        labels or (labels, count)
    """
    labels = []
    cnt = 0
    for m in masks:
        a, b = cv2.connectedComponents(m, connectivity=connectivity, **kwargs)
        if cnt > 0:
            b[b > 0] += cnt
        cnt += a - (1 if (a > 1 and 0 in b) else 0)
        labels.append(b)
    labels = np.stack(labels, label_axis)
    if reduce is not None:
        labels = reduce(labels, axis=label_axis, keepdims=keepdims)
    return (labels, cnt) if count else labels


def fourier2contour(fourier, locations, samples=64, sampling=None):
    """

    Args:
        fourier: Array[..., order, 4]
        locations: Array[..., 2]
        samples: Number of samples.
        sampling: Array[samples] or Array[(fourier.shape[:-2] + (samples,)].
            Default is linspace from 0 to 1 with `samples` values.

    Returns:
        Contours.
    """
    order = fourier.shape[-2]
    if sampling is None:
        sampling = np.linspace(0, 1.0, samples)
    samples = sampling.shape[-1]
    sampling = sampling[..., None, :]

    # shape: (order, samples)
    c = float(np.pi) * 2 * (np.arange(1, order + 1)[..., None]) * sampling

    # shape: (order, samples)
    c_cos = np.cos(c)
    c_sin = np.sin(c)

    # shape: fourier.shape[:-2] + (samples, 2)
    con = np.zeros(fourier.shape[:-2] + (samples, 2))
    con += locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con


def contours2fourier(contours: dict, order=5, dtype=np.float32):
    if len(contours) > 0:
        max_label = np.max(list(contours.keys()))
    else:
        max_label = 0

    fouriers = np.zeros((max_label, order, 4), dtype=dtype)
    locations = np.zeros((max_label, 2), dtype=dtype)
    for key, contour in contours.items():
        if contour.ndim == 3:
            contour = contour.squeeze(1)
        fourier, location = efd(contour, order)
        fouriers[key - 1] = fourier  # labels start at 1, but indices at 0
        locations[key - 1] = location  # labels start at 1, but indices at 0
    return fouriers, locations


def contours2boxes(contours):
    """Contours to boxes.

    Args:
        contours: Array[num_contours, num_points, 2]. (x, y) format.

    Returns:
        Array[num_contours, 4]. (x0, y0, x1, y1) format.
    """
    if len(contours):
        boxes = np.concatenate((contours.min(1), contours.max(1)), 1)
    else:
        boxes = np.empty((0, 4))
    return boxes


def render_contour(contour, val=1, dtype='int32', round=False, reference=None):
    if reference is None:
        reference = contour
    xmin, ymin = np.floor(np.min(reference, axis=0)).astype('int')
    xmax, ymax = np.ceil(np.max(reference, axis=0)).astype('int')
    a = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=dtype)
    if round:
        contour = contour.round()
    a = cv2.drawContours(a, [np.array(contour, dtype=np.int32).reshape((-1, 1, 2))], 0, val, -1,
                         offset=(-xmin, -ymin))
    return a, (xmin, xmax), (ymin, ymax)


def draw_contours(canvas, contours, val=(51, 255, 51), round=True, contour_idx=-1, thickness=2, **kwargs):
    if isinstance(contours, torch.Tensor):
        contours = asnumpy(contours)
    if canvas.ndim == 2 and isinstance(val, (list, tuple, np.ndarray)) and len(val) == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    if contours.dtype.kind == 'f':
        if round:
            contours = contours.round()
        contours = contours.astype(int)
    return cv2.drawContours(canvas, contours, contour_idx, val, thickness, **kwargs)


def filter_contours_by_intensity(img, contours, min_intensity=None, max_intensity=200, aggregate='mean'):
    keep = np.ones(len(contours), dtype=bool)
    for idx, con in enumerate(contours):
        m, (xmin, xmax), (ymin, ymax) = render_contour(con, dtype='uint8')
        img_crop = img[ymin:ymin + m.shape[0], xmin:xmin + m.shape[1]]
        m = m[:img_crop.shape[0], :img_crop.shape[1]]
        assert m.dtype == np.uint8
        m.dtype = bool
        mean_intensity = getattr(np, aggregate)(img_crop[m])
        if max_intensity is not None and mean_intensity > max_intensity:
            keep[idx] = False
        elif min_intensity is not None and mean_intensity < min_intensity:
            keep[idx] = False
    return keep


def clip_contour_(contour, size):
    np.clip(contour[..., 0], 0, size[1], out=contour[..., 0])
    np.clip(contour[..., 1], 0, size[0], out=contour[..., 1])


def contours2labels(contours, size, rounded=True, clip=True, initial_depth=1, gap=3, dtype='int32', ioa_thresh=None,
                    sort_by=None, sort_descending=True, return_indices=False):
    """Contours to labels.

    Convert contours to label image.

    Notes:
        - ~137 ms for contours.shape=(1284, 128, 2), size=(1000, 1000).
        - Label images come with channels, as contours may assign pixels to multiple objects.
          Since such multi-assignments cannot be easily encoded in a channel-free label image, channels are used.
          To remove channels refer to `resolve_label_channels`.

    Args:
        contours: Contours of a single image. Array[num_contours, num_points, 2] or List[Array[num_points, 2]].
        size: Label image size. (height, width).
        rounded: Whether to round contour coordinates.
        clip: Whether to clip contour coordinates to given `size`.
        initial_depth: Initial number of channels. More channels are used if necessary.
        gap: Gap between instances.
        dtype: Data type of label image.
        ioa_thresh: Intersection over area threshold. Skip contours that have an intersection over own area
            (i.e. area of contour that already contains a label vs. area of contour) greater `ioa_thresh`,
            compared to the union of all contours painted before. Note that the order of `contours` is
            relevant, as contours are processed iteratively. IoA of 0 means no labels present so far, IoA of 1. means
            the entire contour area is already covered by other contours.
        sort_by: Optional Array used to sort contours. Note, that if this option is used, labels and contour indices no
            longer correspond.
        sort_descending: Whether to sort by descending.
        return_indices: Whether to return indices.

    Returns:
        Array[height, width, channels]. Since contours may assign pixels to multiple objects, the label image comes
        with channels. To remove channels refer to `resolve_label_channels`.
    """
    contours_ = contours
    if sort_by is not None:
        indices = np.argsort(sort_by)
        if sort_descending:
            indices = reversed(indices)
        contours_ = (contours[i] for i in indices)
    labels = np.zeros(tuple(size) + (initial_depth,), dtype=dtype)
    lbl = 1
    keep = []
    for idx, contour in enumerate(contours_):
        if rounded:
            contour = np.round(contour)
        if clip:
            clip_contour_(contour, np.array(size) - 1)
        a, (xmin, xmax), (ymin, ymax) = render_contour(contour, val=lbl, dtype=dtype)
        if ioa_thresh is not None:
            m = a > 0
            crp = (labels[ymin:ymin + a.shape[0], xmin:xmin + a.shape[1]] > 0).any(-1)
            ioa = crp[m].sum() / m.sum()
            if ioa > ioa_thresh:
                continue
            else:
                keep.append(idx)
        lbl += 1
        s = (labels[np.maximum(0, ymin - gap): gap + ymin + a.shape[0],
             np.maximum(0, xmin - gap): gap + xmin + a.shape[1]] > 0).sum((0, 1))
        i = next(i for i in range(labels.shape[2] + 1) if ~ (i < labels.shape[2] and np.any(s[i])))
        if i >= labels.shape[2]:
            labels = np.concatenate((labels, np.zeros(size, dtype=dtype)[..., None]), axis=-1)
        labels[ymin:ymin + a.shape[0], xmin:xmin + a.shape[1], i] += a
    if return_indices:
        return labels, keep
    return labels


def resolve_label_channels(labels, method='dilation', max_iter=999, kernel=(3, 3)):
    """Resolve label channels.

    Remove channels from a label image.
    Pixels that are assigned to exactly one foreground label remain as is.
    Pixels that are assigned to multiple foreground labels present a conflict, as they cannot be described by a
    channel-less label image. Such conflicts are resolved by `method`.

    Args:
        labels: Label image. Array[h, w, c].
        method: Method to resolve overlapping regions.
        max_iter: Max iteration.
        kernel: Kernel.

    Returns:
        Labels with channels removed. Array[h, w].
    """
    if isinstance(kernel, (tuple, list)):
        kernel = cv2.getStructuringElement(1, kernel)
    mask_sm = np.sum(labels > 0, axis=-1)
    mask = mask_sm > 1  # all overlaps
    if mask.any():
        if method == 'dilation':
            mask_ = mask_sm == 1  # all cores
            lbl = np.zeros(labels.shape[:2], dtype='float64')
            lbl[mask_] = labels.max(-1)[mask_]
            for _ in range(max_iter):
                lbl_ = np.copy(lbl)
                m = mask & (lbl <= 0)
                if not np.any(m):
                    break
                lbl[m] = cv2.dilate(lbl, kernel=kernel)[m]
                if np.allclose(lbl_, lbl):
                    break
        else:
            raise ValueError(f'Invalid method: {method}')
    else:
        lbl = labels.max(-1)
    return lbl.astype(labels.dtype)


def contours2properties(contours, *properties, round=True, **kwargs):
    """Contours to properties.

    References:
        [1] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        contours: Contours.
        *properties: Property names. See [1] for details.
        round: Whether to round contours. Default is `True`.
        **kwargs: Keyword arguments for `skimage.measure.regionprops`.

    Returns:
        List of property lists.
    """
    results = []
    for idx, con in enumerate(contours):
        m, (xmin, xmax), (ymin, ymax) = render_contour(con, dtype='int32', round=round)
        results += labels2properties(m, *properties, offset=kwargs.pop('offset', (ymin, xmin)), **kwargs)
    return results


def mask_labels_by_distance_(labels, distances, max_bg_dist, min_fg_dist):
    # Set instance labels to 0 if their distance is <= max_bg_dist
    labels[np.logical_and(np.any(labels > 0, 2), distances <= max_bg_dist)] = 0

    # Set all labels to -1 that have a distance d with `max_bg_dist < d < min_fg_dist`
    labels[np.logical_and(distances > max_bg_dist, distances < min_fg_dist)] = -1


def _labels2distances_fg(labels, fg_mask_wo_overlap, distance_type):
    # Distance transform
    fg_mask_wo_overlap.dtype = np.uint8
    dist = cv2.distanceTransform(fg_mask_wo_overlap, distance_type, 3)
    if labels.size > 0:
        for p in regionprops(labels):
            c = p.coords
            indices = (c[:, 0], c[:, 1])
            dist[indices] /= np.maximum(dist[indices].max(), .000001)
    return dist


def _labels2distances_instance(labels, fg_mask_wo_overlap, distance_type, protected_size=6*6):
    dist = np.zeros_like(fg_mask_wo_overlap, dtype='float32')
    if labels.size > 0:
        for p in regionprops(labels):
            y0, x0, _, y1, x1, _ = p.bbox
            box_slices = (slice(y0, y1), slice(x0, x1))
            mask = np.any(p.image, 2) & fg_mask_wo_overlap[box_slices]
            d_ = cv2.distanceTransform(np.pad(mask.astype('uint8'), 1), distance_type, 3)[1:-1, 1:-1]
            if mask.sum() > protected_size:
                d_max = d_.max()
                if d_max > 0:
                    d_ /= d_max
            d_ = d_.clip(0., 1.)
            dist[box_slices][mask] = d_[mask]
    return dist


def labels2distances(labels, distance_type=cv2.DIST_L2, overlap_zero=True, per_instance=True, **kwargs):
    """Label stacks to distances.

    Measures distances from pixel to closest border, relative to largest distance.
    Values as percentage. Overlap is zero.

    Notes:
        54.9 ms ± 3.41 ms (shape (576, 576, 3); 762 instances in three channels)

    Args:
        labels: Label stack. (height, width, channels)
        distance_type: opencv distance type.
        overlap_zero: Whether to set overlapping regions to zero.
        per_instance: Performs the distance transform per instance if ``True``.

    Returns:
        Distance map of shape (height, width). All overlapping pixels are 0. Instance centers are 1.
        Also, labels are returned. They are altered if `overlap_zero is True`.
    """
    labels = np.copy(labels)
    mask = labels > 0

    # Mask out overlap
    if overlap_zero:
        overlap_mask = np.sum(mask, 2) > 1
        labels[overlap_mask] = -1
        fg_mask_wo_overlap = np.sum(mask, 2) == 1
    else:
        fg_mask_wo_overlap = np.any(mask, 2)

    # Fg mask
    if per_instance:
        dist = _labels2distances_instance(labels, fg_mask_wo_overlap, distance_type, **kwargs)
    else:
        dist = _labels2distances_fg(labels, fg_mask_wo_overlap, distance_type, **kwargs)

    return dist.clip(0., 1.), labels  # 332 µs ± 24.5 µs for (576, 576)


class CPNTargetGenerator:
    def __init__(self, samples, order, random_sampling=True, remove_partials=False, min_fg_dist=.75, max_bg_dist=.5,
                 flag_fragmented=True, flag_fragmented_constant=-1):
        self.samples = samples
        self.order = order
        self.random_sampling = random_sampling
        self.remove_partials = remove_partials
        self.min_fg_dist = min_fg_dist
        self.max_bg_dist = max_bg_dist
        self.flag_fragmented = flag_fragmented
        self.flag_fragmented_constant = flag_fragmented_constant

        self.labels = None
        self.labels_red = None
        self.distances = None
        self.partials_mask = None
        self._sampling = self._contours = self._fourier = self._locations = self._sampled_contours = None
        self._sampled_sizes = None
        self._reset()

    def _reset(self):
        self._sampling = None
        self._contours = None
        self._fourier = None
        self._locations = None
        self._sampled_contours = None
        self._sampled_sizes = None

    def feed(self, labels, border=1, min_area=1, max_area=None, **kwargs):
        """

        Notes:
            - May apply inplace changes to ``labels``.

        Args:
            labels: Single label image. E.g. of shape (height, width, channels).
            border:
            min_area:
            max_area:
        """
        self._reset()
        if labels.ndim == 2:
            labels = labels[..., None]

        filter_instances_(labels, partials=self.remove_partials, partials_border=border,
                          min_area=min_area, max_area=max_area, constant=-1, continuous=True)

        self.labels = labels
        _ = self.contours  # compute contours

        self.distances, self.labels_red = labels2distances(labels, **kwargs)
        mask_labels_by_distance_(self.labels_red, self.distances, self.max_bg_dist, self.min_fg_dist)

    @property
    def reduced_labels(self):
        if self.flag_fragmented:
            _ = self.contours  # Since labels2contours may filter instances, it has to be done before returning labels
        return self.labels_red.max(2)

    @property
    def sampling(self):
        if self._sampling is None:
            if self.random_sampling:
                self._sampling = np.random.uniform(0., 1., self.samples)

            else:
                self._sampling = np.linspace(0., 1., self.samples)
            self._sampling.sort()
        return self._sampling

    @property
    def contours(self):
        if self._contours is None:
            self._contours: dict = labels2contours(self.labels, flag_fragmented_inplace=self.flag_fragmented,
                                                   constant=self.flag_fragmented_constant, raise_fragmented=False)
        return self._contours

    @property
    def fourier(self):
        if self._fourier is None:
            self._fourier, self._locations = contours2fourier(self.contours, order=self.order)
        return self._fourier

    @property
    def locations(self):
        if self._locations is None:
            self._fourier, self._locations = contours2fourier(self.contours, order=self.order)
        return self._locations

    @property
    def sampled_contours(self):
        """
        Returns:
            Tensor[num_contours, num_points, 2]
        """
        if self._sampled_contours is None:
            self._sampled_contours = fourier2contour(self.fourier, self.locations, samples=self.samples,
                                                     sampling=self.sampling)
        return self._sampled_contours

    @property
    def sampled_sizes(self):
        """
        Notes:
            The quality of `sizes` depends on how accurate `sampled_contours` represents the actual contours.

        Returns:
            Tensor[num_contours, 2]. Contains height and width for each contour.
        """
        if self._sampled_sizes is None:
            c = self.sampled_contours
            self._sampled_sizes = c.max(1) - c.min(1)
        return self._sampled_sizes
