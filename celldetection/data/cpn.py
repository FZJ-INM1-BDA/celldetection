import numpy as np
import warnings
import cv2
from skimage.measure import regionprops
from collections import OrderedDict
from .segmentation import filter_instances_


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


def labels2contours(labels, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) -> dict:
    """

    Args:
        labels:
        mode:
        method: Contour method. CHAIN_APPROX_NONE must be used if contours are used for CPN.

    Returns:

    """
    crops = []
    contours = OrderedDict({})
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
        c, = c
        if len(c) == 1:
            c = np.concatenate((c, c), axis=0)  # min len for other functions to work properly
        contours[label] = c
    return contours


def labels2contour_list(labels) -> list:
    if labels.ndim == 2:
        labels = labels[..., None]
    return [np.squeeze(i, 1) for i in list(labels2contours(labels).values())]


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


def render_contour(contour, val=1, dtype='int32'):
    xmin, ymin = np.floor(np.min(contour, axis=0)).astype('int')
    xmax, ymax = np.ceil(np.max(contour, axis=0)).astype('int')
    a = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=dtype)
    a = cv2.drawContours(a, [np.array(contour, dtype=np.int32).reshape((-1, 1, 2))], 0, val, -1,
                         offset=(-xmin, -ymin))
    return a, (xmin, xmax), (ymin, ymax)


def clip_contour_(contour, size):
    np.clip(contour[..., 0], 0, size[1], out=contour[..., 0])
    np.clip(contour[..., 1], 0, size[0], out=contour[..., 1])


def contours2labels(contours, size, rounded=True, clip=True, initial_depth=1, gap=3, dtype='int32'):
    """Contours to labels.

    Converts contours to label image.

    Notes:
        ~137 ms for contours.shape=(1284, 128, 2), size=(1000, 1000).

    Args:
        contours: Contours. Array[num_contours, num_points, 2] or List[Array[num_points, 2]].
        size: Label image size. (height, width).
        rounded: Whether to round contour coordinates.
        clip: Whether to clip contour coordinates to given `size`.
        initial_depth: Initial number of channels. More channels are used if necessary.
        gap: Gap between instances.
        dtype: Data type of label image.

    Returns:
        Array[height, width, channels]. Channels are used to model overlap.
    """
    labels = np.zeros(tuple(size) + (initial_depth,), dtype=dtype)
    lbl = 1
    for contour in contours:
        if rounded:
            contour = np.round(contour)
        if clip:
            clip_contour_(contour, np.array(size) - 1)
        a, (xmin, xmax), (ymin, ymax) = render_contour(contour, val=lbl, dtype=dtype)
        lbl += 1
        s = (labels[np.maximum(0, ymin - gap): gap + ymin + a.shape[0],
             np.maximum(0, xmin - gap): gap + xmin + a.shape[1]] > 0).sum((0, 1))
        i = next(i for i in range(labels.shape[2] + 1) if ~ (i < labels.shape[2] and np.any(s[i])))
        if i >= labels.shape[2]:
            labels = np.concatenate((labels, np.zeros(size, dtype=dtype)[..., None]), axis=-1)
        labels[ymin:ymin + a.shape[0], xmin:xmin + a.shape[1], i] += a
    return labels


def mask_labels_by_distance_(labels, distances, max_bg_dist, min_fg_dist):
    # Set instance labels to 0 if their distance is <= max_bg_dist
    labels[np.logical_and(np.any(labels > 0, 2), distances <= max_bg_dist)] = 0

    # Set all labels to -1 that have have a distance d with `max_bg_dist < d < min_fg_dist`
    labels[np.logical_and(distances > max_bg_dist, distances < min_fg_dist)] = -1


def labels2distances(labels, distance_type=cv2.DIST_L2, overlap_zero=True):
    """Label stacks to distances.

    Measures distances from pixel to closest border, relative to largest distance.
    Values as percentage. Overlap is zero.

    Notes:
        54.9 ms ± 3.41 ms (shape (576, 576, 3); 762 instances in three channels)

    Args:
        labels: Label stack. (height, width, channels)
        distance_type: opencv distance type.
        overlap_zero: Whether to set overlapping regions to zero.

    Returns:
        Distance map of shape (height, width). All overlapping pixels are 0. Instance centers are 1.
        Also labels are returned. They are altered if `overlap_zero is True`.
    """
    labels = np.copy(labels)
    mask = labels > 0

    # Mask out overlap
    if overlap_zero:
        overlap_mask = np.sum(mask, 2) > 1
        labels[overlap_mask] = -1

    # Fg mask
    fg_mask_wo_overlap = np.sum(mask, 2) == 1
    fg_mask_wo_overlap.dtype = np.uint8

    # Distance transform
    dist = cv2.distanceTransform(fg_mask_wo_overlap, distance_type, 3)
    if labels.size > 0:
        for p in regionprops(labels):
            c = p.coords
            indices = (c[:, 0], c[:, 1])
            dist[indices] /= np.maximum(dist[indices].max(), .000001)
    return dist.clip(0., 1.), labels  # 332 µs ± 24.5 µs for (576, 576)


class CPNTargetGenerator:
    def __init__(self, samples, order, random_sampling=True, remove_partials=False, min_fg_dist=.75, max_bg_dist=.5):
        self.samples = samples
        self.order = order
        self.random_sampling = random_sampling
        self.remove_partials = remove_partials
        self.min_fg_dist = min_fg_dist
        self.max_bg_dist = max_bg_dist

        self.labels = None
        self.reduced_labels = None
        self.distances = None
        self.partials_mask = None
        self._sampling, self._contours, self._fourier, self._locations, self._sampled_contours = (None,) * 5
        self._sampled_sizes = None
        self._reset()

    def _reset(self):
        self._sampling = None
        self._contours = None
        self._fourier = None
        self._locations = None
        self._sampled_contours = None
        self._sampled_sizes = None

    def feed(self, labels, border=1):
        """

        Args:
            labels: Single label image. E.g. of shape (height, width, channels).
            classes:
            border:

        """
        if labels.ndim == 2:
            labels = labels[..., None]

        filter_instances_(labels, partials=self.remove_partials, partials_border=border,
                          min_area=5, max_area=None, constant=-1, continuous=True)

        self.labels = labels

        self.distances, labels = labels2distances(labels)
        mask_labels_by_distance_(labels, self.distances, self.max_bg_dist, self.min_fg_dist)

        self.reduced_labels = labels.max(2)
        self._reset()

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
            self._contours: dict = labels2contours(self.labels)
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
