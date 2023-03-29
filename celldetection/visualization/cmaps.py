from matplotlib.pyplot import cm
import numpy as np
from numpy import ndarray
from typing import Union
import cv2

__all__ = ['label_cmap', 'random_colors_hsv']


def random_colors_hsv(num, hue_range=(0, 180), saturation_range=(60, 133), value_range=(180, 256)):
    colors, = cv2.cvtColor(np.stack((
        np.random.randint(*hue_range, num),
        np.random.randint(*saturation_range, num),
        np.random.randint(*value_range, num),
    ), 1).astype('uint8')[None], cv2.COLOR_HSV2RGB) / 255
    return colors


def label_cmap(labels: ndarray, colors: Union[str, ndarray] = 'rand', zero_val: Union[float, tuple, list] = 0.,
               rgba: bool = True, alpha: float = None):
    """Label colormap.

    Applies a colormap to a label image.

    Args:
        labels: Label image. Typically Array[h, w].
        colors: Either 'rand' or one of ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'] (see matplotlib's qualitative colormaps) or Array[n, c].
        zero_val: Special color for the zero label (usually background).
        rgba: Whether to add an alpha channel to rgb colors.
        alpha: Specific alpha value.

    Returns:
        Mapped labels. E.g. rgba mapping Array[h, w] -> Array[h, w, 4].
    """
    assert issubclass(labels.dtype.type, np.integer), 'Pass labels as an integer array.'
    if isinstance(colors, str):
        if colors == 'rand':
            colors = random_colors_hsv(max(1, min(9999, labels.max())))
        else:
            colors = cm.get_cmap(colors).colors
    if isinstance(colors, (tuple, list)):
        colors = np.array(colors)
    if rgba and colors.shape[1] == 3:
        colors = np.concatenate((colors, np.ones((len(colors), 1))), -1)
    if alpha is not None and colors.shape[1] == 4:
        colors[:, -1] = alpha
    if zero_val is not None:
        labels = np.copy(labels)
        m = labels != 0
        labels[m] = labels[m] % len(colors) + 1
        colors = np.concatenate((np.ones_like(colors[:1]) * zero_val, colors))
    return colors[labels]
