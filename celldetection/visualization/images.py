from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.image import AxesImage
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.axes import SubplotBase
import numpy as np
import seaborn as sbn
from torch import Tensor, squeeze, as_tensor
from torchvision.utils import make_grid
from typing import Union, List
import tempfile
from os import remove
from os.path import isfile
from ..util.util import asnumpy, is_ipython

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


def _fig(figsize, constrained_layout=True):
    if figsize is not None:
        return plt.figure(None, figsize, constrained_layout=constrained_layout)
    return plt.gcf()


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


def _unpack(image, normalize=True):
    from ..data.misc import channels_first2channels_last, channels_last2channels_first
    if isinstance(image, Tensor):
        if image.ndim == 4:  # n, c, h, w
            if image.shape[0] == 1:
                image = squeeze(image, dim=0)
            else:
                image = make_grid(image, normalize=normalize)
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
def imshow(image: Union[np.ndarray, Tensor], figsize=None, constrained_layout=True, **kw) -> AxesImage:
    """Imshow.

    PyPlot's imshow function with benefits.

    Args:
        image: Image. Valid Formats:
            Array[h, w], Array[h, w, c] or Array[n, h, w, c],
            Tensor[h, w], Tensor[c, h, w] or Tensor[n, c, h, w].
            Images without channels or just one channel are plotted as grayscale images by default.
        figsize: Figure size. If specified, a new ``plt.figure(figsize=figsize)`` is created.
        constrained_layout: A constrained layout reduces empty spaces between and around plots.
        **kw: Imshow keyword arguments.

    """
    _fig(figsize, constrained_layout=constrained_layout)
    image = _unpack(image)
    if image.ndim == 2 and 'cmap' not in kw.keys():
        kw['cmap'] = 'gray'
    ret = plt.imshow(image, **kw)
    plt.grid(0)
    return ret


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
                   fill=.2, cmap=..., constrained_layout=True, **kwargs):
    _fig(figsize, constrained_layout=constrained_layout)
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
                alpha=.7, constrained_layout=True):
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
        constrained_layout: A constrained layout reduces empty spaces between and around plots.

    """
    angles = np.arctan2(vector_field[0], vector_field[1])
    X, Y = np.meshgrid(np.arange(0, vector_field.shape[2]), np.arange(0, vector_field.shape[1]))
    U, V = vector_field[0], vector_field[1]
    _fig(figsize, constrained_layout=constrained_layout)
    if image is not None:
        imshow(image, cmap=cmap)
    plt.quiver(X, Y, U, V, angles, width=width, alpha=alpha, linewidth=linewidth, angles='xy', units='xy', scale=1,
               cmap=qcmap)
    plt.quiver(X, Y, U, V, width=width, edgecolor='k', alpha=alpha, facecolor='None', linewidth=linewidth, angles='xy',
               units='xy', scale=1, cmap=qcmap)
    plt.grid(0)


@register
def imshow_grid(*images, titles=None, figsize=(3, 3), tight=True, constrained_layout=True, **kwargs):
    """Imshow grid.

    Display a list of images in a NxN grid.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        constrained_layout: A constrained layout reduces empty spaces between and around plots.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    n = int(np.ceil(np.sqrt(n)))
    if figsize is not None:
        plt.figure(None, (figsize[0] * n, figsize[1] * n), constrained_layout=constrained_layout)
    for i, img in enumerate(images):
        plt.subplot(n, n, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)


@register
def imshow_row(*images, titles=None, figsize=(3, 3), tight=True, constrained_layout=True, **kwargs):
    """Imshow row.

    Display a list of images in a row.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        constrained_layout: A constrained layout reduces empty spaces between and around plots.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    if figsize is not None:
        plt.figure(None, (figsize[0], figsize[1] * n), constrained_layout=constrained_layout)
    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)


@register
def imshow_col(*images, titles=None, figsize=(3, 3), tight=True, constrained_layout=True, **kwargs):
    """Imshow row.

    Display a list of images in a column.

    Args:
        *images: Images.
        titles: Titles. Either string or list of strings (one for each image).
        figsize: Figure size per image.
        tight: Whether to use tight layout.
        constrained_layout: A constrained layout reduces empty spaces between and around plots.
        **kwargs: Keyword arguments for ``cd.imshow``.

    """
    n, images = _im_args(images)
    if figsize is not None:
        plt.figure(None, (figsize[0] * n, figsize[1]), constrained_layout=constrained_layout)
    for i, img in enumerate(images):
        plt.subplot(n, 1, i + 1)
        imshow(img, **kwargs)
        _tight(tight)
        _title(titles, i)


@register
def plot_zstack(stack, view_axes=None, project=None, titles=None, figsize=None, tight=True, constrained_layout=True,
                **kwargs):
    axis = np.argmin(stack.shape[:3])
    views = {}
    if view_axes is None:
        view_axes = np.array(stack.shape[:3]) // 2
    if project is None:
        views['main'] = stack[(slice(None),) * axis + (view_axes[axis],)]
    else:
        raise ValueError
    for i, ax in enumerate(list({2, 1, 0} - {axis})[::-1]):  # order: width, height
        views[ax] = stack[(slice(None),) * ax + (view_axes[ax],)]
        if (axis == 0 and ax == 2) or (axis == 1 and ax == 1) or (axis == 2 and ax == 0):
            views[ax] = np.transpose(views[ax], (1, 0) + tuple(range(2, views[ax].ndim)))
    if figsize is not None:
        plt.figure(None, (figsize[0] * 2, 2 * figsize[1]), constrained_layout=constrained_layout)
    _tight(tight)
    axes = get_axes()
    if len(axes) <= 0:
        axes = [None] * 3
    for i, (ax, view) in enumerate(zip(axes, views.values())):
        if ax is None:
            ax = plt.subplot(2, 2, i + 1)
        plt.sca(ax)
        imshow(view, **kwargs)
        _tight(True)
        _title(titles, i)


@register
def plot_gif(*frames, figsize=None, interval=200, blit=True, ipython=None, display=True, fn=None, save_kwargs=None,
             axis=0, constrained_layout=True, **kwargs):
    assert len(frames) >= 1, 'Provide at least one frame.'
    if len(frames) == 1:
        f0 = frames[0]
        if isinstance(f0, (list, tuple)):
            frames, = frames
        elif isinstance(f0, np.ndarray):
            if axis < 0:
                axis += f0.ndim
            if axis == 0:
                frames = list(f0)  # faster
            else:
                frames = [f0[(slice(None),) * axis + (i,)] for i in range(f0.shape[axis])]
        else:
            raise ValueError('Could not handle dtype:', type(f0))

    if fn is None:
        fn = imshow

    fig = _fig(figsize, constrained_layout=constrained_layout)
    ani_kwargs = kwargs.pop('animation_kwargs', {})
    ani = animation.ArtistAnimation(
        fig, [[fn(frame, animated=True, **kwargs)] for frame in frames],
        interval=interval, blit=blit, **ani_kwargs
    )

    if ipython is None:
        ipython = is_ipython()

    disp_image = None
    if ipython:
        from IPython.display import Image, display as disp
        plt.close(fig)
        temp_gif = None
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix='.gif') as temp_gif:
                if save_kwargs is None:
                    save_kwargs = {}
                ani.save(temp_gif.name, writer='pillow', **save_kwargs)
                temp_gif.flush()
                disp_image = Image(filename=temp_gif.name)
                if display:
                    disp(disp_image)
        except Exception as e:
            if temp_gif is not None and isfile(temp_gif.name):
                remove(temp_gif.name)
            raise e

    return ani, disp_image


@register
def figure2img(fig, close=True, transpose=False, remove_alpha=False, crop=True, padding=8):
    """Figure to image.

    Converts pyplot Figure to numpy array.
    In contrast to other implementations, only the window extend is returned (removing irrelevant background regions
    of the Figure).

    Args:
        fig: Figure or list of Figures.
        close: Whether to close the figure after converting it to an image.
        transpose: Whether to convert HWC format to CHW. This is a convenience feature for logging.
        remove_alpha: Whether to remove the alpha channel from RGBA images.
        crop: Whether to crop irrelevant background regions of the Figure.
        padding: Padding applied around relevant region of the Figure. Only used for cropping.

    Returns:
        Array[h, w, 4].
    """
    from ..data.misc import padding_stack

    if isinstance(fig, (list, tuple)):
        return padding_stack(
            *[figure2img(f, close=close, transpose=transpose, remove_alpha=remove_alpha, crop=crop) for f in fig])

    # Render the Figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get image of whole Figure
    w_fig, h_fig = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    img = buffer.reshape(h_fig, w_fig, 4)

    # Crop relevant region
    if crop:
        x0_ = None
        y0_ = None
        x1_ = None
        y1_ = None

        for ax in fig.get_axes():  # faster without numpy for most cases
            bbox = ax.get_tightbbox(renderer)  # renderer accounts for dpi
            x0, y0, width, height = bbox.bounds
            x0, y0, width, height = map(int, [x0, y0, width, height])
            x1 = x0 + width
            y1 = y0 + height

            x0_ = x0 if x0_ is None else min(x0, x0_)
            y0_ = y0 if y0_ is None else min(y0, y0_)
            x1_ = x1 if x1_ is None else max(x1, x1_)
            y1_ = y1 if y1_ is None else max(y1, y1_)

        y0_, y1_ = max(y0_ - padding, 0), min(img.shape[0], y1_ + padding)
        x0_, x1_ = max(x0_ - padding, 0), min(img.shape[1], x1_ + padding)
        img = img[y0_:y1_, x0_:x1_]
        if remove_alpha and img.ndim == 3:
            img = img[..., :3]

    # Close Figure
    if close:
        plt.close(fig)

    # Optionally transpose
    if transpose and img.ndim == 3:
        img = np.moveaxis(img, source=2, destination=0)
    return img
