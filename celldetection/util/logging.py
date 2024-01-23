import numpy as np
from typing import Union, Optional, Dict, List
from ..visualization.images import figure2img

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj


@register
def log_figure(
        logger,
        tag: str,
        figure,
        global_step: Optional[int] = None,
        close: Optional[bool] = True,
        walltime: Optional[float] = None
):
    if hasattr(logger, 'add_image'):
        if isinstance(figure, list):
            # Note that tensorboard cannot handle RGBA in NCHW, hence A channel must be removed.
            logger.add_image(tag, figure2img(figure, close=close, transpose=True, remove_alpha=True),
                             global_step, walltime, dataformats='NCHW')
        else:
            logger.add_image(tag, figure2img(figure, close=close, transpose=True, remove_alpha=False), global_step,
                             walltime, dataformats='CHW')
