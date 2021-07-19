import numpy as np
import inspect
import torch
import torch.nn as nn
from typing import Union, List, Tuple, Any, Dict as TDict
from torch import Tensor
from torchvision.models.utils import load_state_dict_from_url
import hashlib
import json

__all__ = ['Dict', 'lookup_nn', 'reduce_loss_dict', 'to_device', 'asnumpy', 'fetch_model', 'random_code_name',
           'dict_hash', 'fetch_image', 'random_seed', 'tweak_module_']


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


def fetch_model(name, map_location=None, **kwargs):
    """ Fetch model.

    Loads model or state dict from url.

    Args:
        name: Model name hosted on `celldetection.org` or url. Urls must start with 'http'.
        map_location: A function, `torch.device`, string or a dict specifying how to remap storage locations.
        **kwargs: From the doc of `torch.models.utils.load_state_dict_from_url`.

    """
    url = name if name.startswith('http') else f'https://celldetection.org/torch/models/{name}.pt'
    return load_state_dict_from_url(url, map_location=map_location, **kwargs)


def random_code_name(chars=4) -> str:
    """Random code name.

    Generates random code names that are somewhat pronounceable and memorable.

    Examples:
        ```
        >>> from celldetection import util
        >>> util.random_code_name()
        kolo
        >>> util.random_code_name(6)
        lotexo
        ```

    Args:
        chars: Number of characters.

    Returns:
        String
    """
    a, b = [i for i in 'aeiou'], [i for i in 'tskyrhzjgqmxlvnfcpwbd']
    return ''.join([np.random.choice(b if j % 2 == 0 else a) for j in range(chars)])


def dict_hash(dictionary: TDict[str, Any]) -> str:
    """MD5 hash of a dictionary.

    References:
        https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html

    Args:
        dictionary: A dictionary.

    Returns:
        Md5 hash of the dictionary as string.
    """
    dhash = hashlib.md5()
    dhash.update(json.dumps(dictionary, sort_keys=True).encode())
    return dhash.hexdigest()


def fetch_image(url, numpy=True):
    """Fetch image from URL.

    Args:
        url: URL
        numpy: Whether to convert PIL Image to numpy array.

    Returns:
        PIL Image or numpy array.
    """
    import requests
    from PIL import Image
    img = Image.open(requests.get(url, stream=True).raw)
    return np.asarray(img) if numpy else img


def random_seed(seed, backends=False, deterministic_torch=True):
    """Set random seed.

    References:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed.
        backends: Whether to also adapt backends. If set True cuDNN's benchmark feature is disabled. This
            causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
            Also the selected algorithm is set to run deterministically.
        deterministic_torch: Whether to set PyTorch operations to behave deterministically.

    """
    from torch import manual_seed
    from torch.backends import cudnn
    import random
    random.seed(seed)
    manual_seed(seed)
    np.random.seed(seed)
    if backends:
        cudnn.deterministic = True
        cudnn.benchmark = False
    if deterministic_torch and 'use_deterministic_algorithms' in dir(torch):
        torch.use_deterministic_algorithms(True)


def tweak_module_(module: nn.Module, class_or_tuple, must_exist=True, **kwargs):
    """Tweak module.

    Sets attributes for all modules that are instances of given `class_or_tuple`.

    Examples:
        ```
        >>> import celldetection as cd, torch.nn as nn
        >>> model = cd.models.ResNet18(in_channels=3)
        >>> cd.util.tweak_module_(model, nn.BatchNorm2d, momentum=0.05)  # sets momentum to 0.05
        ```

    Notes:
        This is an in-place operation.

    Args:
        module: PyTorch `Module`.
        class_or_tuple: All instances of given `class_or_tuple` are to be tweaked.
        must_exist: If `True` an AttributeError is raised if keywords do not exist.
        **kwargs: Attributes to be tweaked: `<attribute_name>=<value>`.
    """
    for mod in module.modules():
        if isinstance(mod, class_or_tuple):
            for k, v in kwargs.items():
                if must_exist:
                    getattr(mod, k)
                setattr(mod, k, v)
