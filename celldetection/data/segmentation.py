import numpy as np
from skimage import morphology
import cv2


def remove_partials_(label_stack, border=1, constant=-1):
    if border < 1:
        return label_stack, None
    bad_labels = set(np.unique(label_stack[:, :border]))
    bad_labels.update(np.unique(label_stack[:, -border:]))
    bad_labels.update(np.unique(label_stack[:border, :]))
    bad_labels.update(np.unique(label_stack[-border:, :]))
    mask = np.isin(label_stack, list(bad_labels - {0}))
    label_stack[mask] = constant
    return label_stack, mask


def fill_label_gaps_(labels):
    """Fill label gaps.

    Ensure that labels greater zero are within interval [1, num_unique_labels_greater_zero].
    Works fast if gaps are unlikely, slow otherwise. Alternatively consider using np.vectorize.
    Labels <= 0 are preserved as is.

    Args:
        labels:

    Returns:

    """
    uni = np.unique(labels)
    uniques = list(set(uni) - set(uni[uni <= 0]))  # ignore zeros and negative values
    uniques.sort()
    gaps = list(set(range(1, len(uniques) + 1)) - set(uniques))
    while len(gaps) > 0:
        labels[labels == uniques.pop()] = gaps.pop()


def filter_instances_(labels, partials=True, partials_border=1, min_area=4, max_area=None, constant=-1,
                      continuous=True):
    """Filter instances from label image.

    Note:
        Filtered instance labels are set to `constant`.
        Labels might not be continuous afterwards.

    Args:
        labels:
        partials:
        partials_border:
        min_area:
        max_area:
        constant:
        continuous:

    Returns:

    """
    if partials:
        remove_partials_(labels, border=partials_border, constant=constant)

    if max_area is not None or min_area is not None:
        # Unique labels with counts: 13.5 ms ± 242 µs ((576, 576, 3), 763 instances)
        uni_labels, uni_counts = np.unique(labels, return_counts=True)
        uni_labels, uni_counts = uni_labels[1:], uni_counts[1:]
        bad_labels = []
        if max_area:
            bad_labels += list(uni_labels[uni_counts > max_area].ravel())
        if min_area:
            bad_labels += list(uni_labels[uni_counts < min_area].ravel())
        for label in bad_labels:
            labels[labels == label] = constant

    if continuous:
        fill_label_gaps_(labels)


def relabel_(label_stack, axis=2):
    """Relabel.

    Inplace relabeling of a label stack.
    After applying this op the labels in label_stack are continuous, starting at 1.
    Negative labels remain untouched.

    Notes:
        - Uses label function from sklearn.morphology

    Args:
        label_stack: Array[height, width, channels].
        axis: Channel axis.
    """
    assert label_stack.ndim == 3
    neg_m = label_stack < 0
    cur_max = 0
    for channel in range(label_stack.shape[axis]):
        stack_ = morphology.label(label_stack[(slice(None),) * axis + (channel,)])
        for u in set(np.unique(stack_)) - {0}:
            mask = stack_ == u
            if np.any(np.logical_and(mask, neg_m[(slice(None),) * axis + (channel,)])):
                continue
            cur_max += 1
            label_stack[mask, channel] = cur_max


def unary_masks2labels(unary_masks, transpose=True):
    """Unary masks to labels.

    Args:
        unary_masks: List[Array[height, width]] or Array[num_objects, height, width]
            List of masks. Each mask is assumed to contain exactly one object.
        transpose: If True label images are in channels last format, otherwise channels first.

    Returns:
        Label image. Array[height, width, num_objects] if `transpose` else Array[num_objects, height, width].
    """
    lbl = (unary_masks > 0) * np.arange(1, len(unary_masks) + 1)[:, None, None]
    if transpose:
        lbl = lbl.transpose((1, 2, 0))
    return lbl


def boxes2masks(boxes, size):
    masks = [None] * len(boxes)
    for idx, b in enumerate(boxes):
        mask = masks[idx] = np.zeros(size, dtype='uint8')
        xmin, ymin, xmax, ymax = b
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), idx + 1, thickness=-1)
    return masks
