from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sbn

__all__ = ['imshow', 'plot_mask', 'plot_box', 'plot_text', 'quiver_plot', 'show_detection', 'save_fig']


def imshow(image, figsize=None, **kw):
    """Imshow.

    PyPlot's imshow function with benefits.

    Args:
        image: Image as Array[h, w], Array[h, w, c], Array[1, h, w] or Array[1, h, w, c]. Images without channels or
            just one channel are plotted as grayscale images by default.
        figsize: Figsize. If specified a new ``Figure(figsize=figsize)`` with is created.
        **kw: Imshow keyword arguments.

    Returns:

    """
    if figsize is not None:
        plt.figure(None, figsize)
    if image.ndim == 3 and 1 in image.shape:
        if image.shape[-1] == 1:
            image = np.squeeze(image, -1)
        elif image.shape[0] == 1:
            image = np.squeeze(image, 0)
    if image.ndim == 2 and 'cmap' not in kw.keys():
        kw['cmap'] = 'gray'
    plt.imshow(image, **kw)
    plt.grid(0)


def plot_text(class_name, x, y, score=None, color='black', stroke_width=5, stroke_color='w'):
    ax = plt.gca()
    name = f'{class_name}'
    if score is not None:
        name += f': {np.round(score * 100, 1)}%'
    txt = ax.text(x, y, name, color=color)
    if stroke_width is not None and stroke_width > 0:
        txt.set_path_effects([path_effects.withStroke(linewidth=stroke_width, foreground=stroke_color)])


def plot_box(x_min, y_min, x_max, y_max, score=None, class_name='score'):
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='#4AF626',
                                   facecolor='none'))
    if score is not None:
        plot_text(class_name, x_min, y_min - 5, score=score)


def plot_mask(mask, alpha=1):
    mask_im = ((mask * 255)).astype('uint8')
    zm = np.zeros_like(mask_im)
    mask_im = np.stack((zm, mask_im, zm, mask_im), axis=2)
    plt.imshow(mask_im, alpha=alpha)
    plt.grid(0)


def show_detection(image=None, contours=None, coordinates=None, boxes=None, scores=None, masks=None,
                   figsize=None, label_stack=None, class_name='score', contour_line_width=2, contour_linestyle='-',
                   fill=.2, cmap=...):
    if figsize is not None:
        plt.figure(None, figsize)
    plt.grid(0)
    if image is not None:
        imshow(image, **({} if cmap is ... else dict(cmap=cmap)))
    if contours is not None:
        for c_i, c_ in enumerate(contours):
            kw = {}
            if fill:
                f, = plt.fill(c_[:, 0], c_[:, 1], alpha=fill)
                kw['color'] = f.get_facecolor()[:3]
            c_ = np.concatenate((c_, c_[:1]), 0)  # close contour
            plt.plot(c_[:, 0], c_[:, 1], linestyle=contour_linestyle, linewidth=contour_line_width, **kw)
            if scores is not None and boxes is None:
                c_min = np.argmin(c_[:, 1])
                if isinstance(class_name, str):
                    cn = class_name
                else:
                    cn = class_name[c_i]
                plot_text(cn, c_[c_min, 0], c_[c_min, 1], scores[c_i])
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
        for box_i, box in enumerate(boxes):
            plot_box(*box, score=scores[box_i] if scores is not None else scores, class_name=class_name)


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
