from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
from matplotlib.axes import SubplotBase
import numpy as np
import seaborn as sbn
from torch import Tensor, squeeze, as_tensor
from torchvision.utils import make_grid
from typing import Union, List
from ..util.util import asnumpy

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


def _fig(figsize):
    if figsize is not None:
        plt.figure(None, figsize)


def _title(titles, idx=None):
    if isinstance(titles, str):
        plt.title(titles)
    elif idx is not None and isinstance(titles, (list, tuple)):
        plt.title(titles[idx])
    else:
        plt.title(str(titles))


def _tight(tight=True):
    if tight is True:
        tight = {}
    if tight:
        plt.tight_layout(**tight)


def _im_args(images):
    n = len(images)
    if n == 1 and isinstance(images[0], (list, tuple)):
        images, = images
        n = len(images)
    return n, images


def _unpack(image):
    from ..data.misc import channels_first2channels_last, channels_last2channels_first
    if isinstance(image, Tensor):
        if image.ndim == 4:  # n, c, h, w
            if image.shape[0] == 1:
                image = squeeze(image, dim=0)
            else:
                image = make_grid(image)
        image = asnumpy(image)
        if image.ndim == 3:  # c, h, w
            image = channels_first2channels_last(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 4:  # n, h, w, c
            if image.shape[0] == 1:
                image = np.squeeze(image, 0)
            else:
                image = _unpack(as_tensor(channels_last2channels_first(image, has_batch=True)))
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, 2)
    return image


@register
def get_axes(fig=None) -> List['SubplotBase']:
    """Get current pyplot axes.

    Args:
        fig: Optional Figure.

    Returns:
        List of Axes.
    """
    if fig is None:
        fig = plt.gcf()
    return fig.get_axes()


@register
def imshow(image: Union[np.ndarray, Tensor], figsize=None, **kw):
    """Imshow.

    PyPlot's imshow function with benefits.

    Args:
        image: Image. Valid Formats:
            Array[h, w], Array[h, w, c] or Array[n, h, w, c],
            Tensor[h, w], Tensor[c, h, w] or Tensor[n, c, h, w].
            Images without channels or just one channel are plotted as grayscale images by default.
        figsize: Figure size. If specified, a new ``plt.figure(figsize=figsize)`` is created.
        **kw: Imshow keyword arguments.

    """
    _fig(figsize)
    image = _unpack(image)
    if image.ndim == 2 and 'cmap' not in kw.keys():
        kw['cmap'] = 'gray'
    plt.imshow(image, **kw)
    plt.grid(0)


@register
def plot_text(text, x, y, color='black', stroke_width=5, stroke_color='w'):
    ax = plt.gca()
    txt = ax.text(x, y, str(text), color=color)
    if stroke_width is not None and stroke_width > 0:
        txt.set_path_effects([path_effects.withStroke(linewidth=stroke_width, foreground=stroke_color)])


def _score_text(score, cls, cls_names):
    if isinstance(score, Tensor):
        score = asnumpy(score)
    if isinstance(cls, int) and cls_names is not None:
        cls = cls_names[cls]
    txt = ''
    if cls is not None:
        txt += f'{cls}: '
    txt += f'{np.round(score * 100, 1)}%'
    return txt


def _score_texts(scores, classes: Union[List[str], List[int]] = None, class_names: dict = None):
    texts = []
    for i in range(len(scores)):
        cls = None if classes is None else classes[i]
        texts.append(_score_text(scores[i], cls, class_names))
    return texts


@register
def plot_score(score: float, x, y, cls: Union[int, str] = None, cls_names: dict = None, **kwargs):
    plot_text(text=_score_text(score, cls=cls, cls_names=cls_names), x=x, y=y, **kwargs)


@register
def plot_box(x_min, y_min, x_max, y_max, linewidth=1, edgecolor='#4AF626',
             facecolor='none', text=None, **kwargs):
    ax = plt.gca()
    ax.add_patch(
        patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=linewidth, edgecolor=edgecolor,
                          facecolor=facecolor, **kwargs))
    if text is not None:
        plot_text(text, x_min, y_min, **kwargs.get('text_kwargs', {}))


@register
def plot_boxes(boxes, texts: List[str] = None, **kwargs):
    for box_i, box in enumerate(boxes):
        if isinstance(box, Tensor):
            box = asnumpy(box)
        text = None if texts is None else texts[box_i]
        plot_box(*box, text=text, **kwargs)


@register
def plot_mask(mask, alpha=1):
    mask_im = ((mask * 255)).astype('uint8')
    zm = np.zeros_like(mask_im)
    mask_im = np.stack((zm, mask_im, zm, mask_im), axis=2)
    plt.imshow(mask_im, alpha=alpha)
    plt.grid(0)


@register
def plot_contours(contours, contour_line_width=2, contour_linestyle='-', fill=.2, color=None,
                  texts: list = None, **kwargs):
    if isinstance(contours, Tensor):
        contours = asnumpy(contours)
    text_kwargs = dict(kwargs.get('text_kwargs', {}))
    for c_i, c_ in enumerate(contours):
        kw = dict(kwargs)
        color_ = color
        if isinstance(color, list):
            color_ = color[0]
        if fill:
            fill_kw = dict(kwargs.get('fill_kwargs', {}))
            if color_ is not None:
                fill_kw['color'] = color_
            f, = plt.fill(c_[:, 0], c_[:, 1], alpha=fill, **fill_kw)
            if color_ is None:
                color_ = f.get_facecolor()[:3]
        c_ = np.concatenate((c_, c_[:1]), 0)  # close contour
        plt.plot(c_[:, 0], c_[:, 1], linestyle=contour_linestyle, linewidth=contour_line_width, color=color_, **kw)
        if texts is not None:
            c_min = np.argmin(c_[:, 1])
            plot_text(texts[c_i], c_[c_min, 0], c_[c_min, 1], **text_kwargs)


@register
def show_detection(image=None, contours=None, coordinates=None, boxes=None, scores=None, masks=None,
                   figsize=None, label_stack=None, classes: Union[List[str], List[int]] = None,
                   class_names: dict = None, contour_line_width=2, contour_linestyle='-',
                   fill=.2, cmap=..., **kwargs):
    _fig(figsize)
    plt.grid(0)
    if image is not None:
        imshow(image, **({} if cmap is ... else dict(cmap=cmap)))
    if scores is not None:
        score_texts = _score_texts(scores, classes=classes, class_names=class_names)
    if contours is not None:
        con_kw = kwargs.get('contour_kwargs', {})
        if scores is not None and boxes is None and 'texts' not in con_kw:
            con_kw['texts'] = score_texts
        plot_contours(contours, contour_line_width=contour_line_width, contour_linestyle=contour_linestyle, fill=fill,
                      **con_kw)
    if masks is not None:
        for mask in masks:
            plot_mask(np.squeeze(mask))
    if label_stack is not None:
        for lbl in np.unique(label_stack[label_stack > 0]):
            mask = label_stack == lbl
            if mask.ndim == 3:
                mask = np.logical_or.reduce(mask, axis=2)
            plot_mask(mask)
    if coordinates is not None:
        plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='+')
    if boxes is not None:
        box_kw = kwargs.get('box_kwargs', {})
        if scores is not None and 'texts' not in box_kw:
            box_kw['texts'] = score_texts
        plot_boxes(boxes, **box_kw)


@register
def save_fig(filename, close=True):
    """Save Figure.

    Save current Figure to disk.

    Args:
        filename: Filename, e.g. ``image.png``.
        close: Whether to close all unhandled Figures. Do not close them if you intend to call ``plt.show()``.

    """
    plt.savefig(filename, bbox_inches='tight')
    if close:
        plt.close('all')


@register
def quiver_plot(vector_field, image=None, cmap='gray', figsize=None, qcmap='twilight', linewidth=.125, width=.19,
                alpha=.7):
    """Quiver plot.

    Plots a 2d vector field.
    Can be used to visualize local refinement tensor.

    Args:
        vector_field: Array[2, w, h].
        image: Array[h, w(, 3)].
        cmap: Image color map.
        figsize: Figure size.
        qcmap: Quiver color map. Consider seaborn's: `qcmap = ListedColormap(sns.color_palette("husl", 8).as_hex())`
        linewidth: Quiver line width.
        width: Quiver width.
        alpha: Quiver alpha.

    """
    angles = np.arctan2(vector_field[0], vector_field[1])
    X, Y = np.meshgrid(np.arange(0, vector_field.shape[2]), np.arange(0, vector_field.shape[1]))
    U, V = vector_field[0], vector_field[1]
    if figsize is not None:
        plt.figure(None, figsize)
    if image is not None:
        imshow(image, cmap=cmap)
    plt.quiver(X, Y, U, V, angles, width=width, alpha=alpha, linewidth=linewidth, angles='xy', units='xy', scale=1,
               cmap=qcmap)
    plt.quiver(X, Y, U, V, width=width, edgecolor='k', alpha=alpha, facecolor='None', linewidth=linewidth, angles='xy',
               units='xy', scale=1, cmap=qcmap)
    plt.grid(0)


@register
def imshow_grid(*images, titles=None, figsize=(3, 3), tight=True, **kwargs):
    """Imshow grid.

    Display a list of images in a NxN grid.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    n = int(np.ceil(np.sqrt(n)))
    if figsize is not None:
        plt.figure(None, (figsize[0] * n, figsize[1] * n))
    for i, img in enumerate(images):
        plt.subplot(n, n, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)


@register
def imshow_row(*images, titles=None, figsize=(3, 3), tight=True, **kwargs):
    """Imshow row.

    Display a list of images in a row.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    if figsize is not None:
        plt.figure(None, (figsize[0], figsize[1] * n))
    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)


@register
def imshow_col(*images, titles=None, figsize=(3, 3), tight=True, **kwargs):
    """Imshow row.

    Display a list of images in a row.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    if figsize is not None:
        plt.figure(None, (figsize[0] * n, figsize[1]))
    for i, img in enumerate(images):
        plt.subplot(n, 1, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)
