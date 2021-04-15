import numpy as np
import inspect
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from torch import Tensor
from torchvision.models.utils import load_state_dict_from_url


class Dict(dict):
    __getattr__ = dict.__getitem__  # alternative: dict.get if KeyError is not desired
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super().__init__(kwargs)


def lookup_nn(item: str, *a, src=nn, call=True, inplace=True, **kw):
    """

    Examples:
        ```
        >>> lookup_nn('batchnorm2d', 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn(torch.nn.BatchNorm2d, 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('batchnorm2d', num_features=32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('tanh')
            Tanh()
        >>> lookup_nn('tanh', call=False)
            torch.nn.modules.activation.Tanh
        >>> lookup_nn('relu')
            ReLU(inplace=True)
        >>> lookup_nn('relu', inplace=False)
            ReLU()
        ```

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        **kw:

    Returns:
        Looked up item.
    """
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        v = next((getattr(src, i) for i in dir(src) if i.lower() == l_item))
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kw)
    return v


def reduce_loss_dict(losses: dict, divisor):
    return sum((i for i in losses.values() if i is not None)) / divisor


def add_to_loss_dict(d: dict, key: str, loss: torch.Tensor, weight=None):
    dk = d[key]
    if weight is not None:
        loss = loss * weight
    d[key] = loss if dk is None else dk + loss


def to_device(batch: Union[list, tuple, dict, Tensor], device):
    if isinstance(batch, Tensor):
        batch = batch.to(device)
    elif isinstance(batch, dict):
        batch = {k: to_device(b, device) for k, b in batch.items()}
    elif isinstance(batch, (list, tuple)):
        batch = type(batch)([to_device(b, device) for b in batch])
    return batch


def asnumpy(v):
    if v is None:
        return v
    elif isinstance(v, torch.Tensor):
        if str(v.device) != 'cpu':
            v = v.cpu()
        return v.data.numpy()
    elif isinstance(v, (np.ndarray, int, float, bool, np.float, np.int, np.bool)):
        return v
    elif isinstance(v, (tuple, list)):
        return [asnumpy(val) for val in v]
    elif isinstance(v, dict):
        r = dict()
        for k, val in v.items():
            r[k] = asnumpy(val)
        return r
    else:
        raise ValueError(f'Type not supported: {type(v)}')


def fetch_model(name):
    return load_state_dict_from_url(f'https://celldetection.org/torch/models/{name}.pt')

