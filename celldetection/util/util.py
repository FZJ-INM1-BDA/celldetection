import numpy as np
import inspect
import torch
import torch.nn as nn
from typing import Union, List, Tuple, Any, Dict as TDict, Iterator
from torch import Tensor
from torch.hub import load_state_dict_from_url
import hashlib
import json
from tqdm import tqdm
from os.path import join
from os import makedirs
import pynvml as nv
from cv2 import getGaussianKernel

__all__ = ['Dict', 'lookup_nn', 'reduce_loss_dict', 'to_device', 'asnumpy', 'fetch_model', 'random_code_name',
           'dict_hash', 'fetch_image', 'random_seed', 'tweak_module_', 'add_to_loss_dict',
           'random_code_name_dir', 'get_device', 'num_params', 'count_submodules', 'train_epoch', 'Bytes', 'Percent',
           'GpuStats', 'trainable_params', 'frozen_params', 'Tiling', 'load_image', 'gaussian_kernel',
           'replace_module_']


class Dict(dict):
    __getattr__ = dict.__getitem__  # alternative: dict.get if KeyError is not desired
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        """Dictionary.

        Just a ``dict`` that treats values like attributes.

        Examples:
            >>> import celldetection as cd
            >>> d = cd.Dict(my_value=42)
            >>> d.my_value
            42
            >>> d.my_value += 1
            >>> d.my_value
            43

        Args:
            **kwargs:
        """
        super().__init__(kwargs)


def lookup_nn(item: str, *a, src=None, call=True, inplace=True, **kw):
    """

    Examples:
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

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    src = src or nn
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        v = next((getattr(src, i) for i in dir(src) if i.lower() == l_item))
    elif isinstance(item, nn.Module):
        return item
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v


def reduce_loss_dict(losses: dict, divisor):
    return sum((i for i in losses.values() if i is not None)) / divisor


def add_to_loss_dict(d: dict, key: str, loss: torch.Tensor, weight=None):
    dk = d[key]
    torch.nan_to_num_(loss, 0., 0., 0.)
    if weight is not None:
        loss = loss * weight
    d[key] = loss if dk is None else dk + loss


def to_device(batch: Union[list, tuple, dict, Tensor], device):
    """To device.

    Move Tensors to device.
    Input can be Tensor, tuple of Tensors, list of Tensors or a dictionary of Tensors.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        batch: Input.
        device: Device.

    Returns:
        Input with Tensors moved to device.
    """
    if isinstance(batch, Tensor):
        batch = batch.to(device)
    elif isinstance(batch, dict):
        batch = {k: to_device(b, device) for k, b in batch.items()}
    elif isinstance(batch, (list, tuple)):
        batch = type(batch)([to_device(b, device) for b in batch])
    return batch


def asnumpy(v):
    """As numpy.

    Converts all Tensors to numpy arrays.

    Notes:
        - Works recursively.
        - The following input items are not altered: Numpy array, int, float, bool, str

    Args:
        v: Tensor or list/tuple/dict of Tensors.

    Returns:
        Input with Tensors converted to numpy arrays.
    """
    if v is None:
        return v
    elif isinstance(v, torch.Tensor):
        if str(v.device) != 'cpu':
            v = v.cpu()
        return v.data.numpy()
    elif isinstance(v, (np.ndarray, int, float, bool, np.float, np.int, np.bool, str)):
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
    """Fetch model from URL.

    Loads model or state dict from URL.

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
        >>> import celldetection as cd
        >>> cd.random_code_name()
        kolo
        >>> cd.random_code_name(6)
        lotexo

    Args:
        chars: Number of characters.

    Returns:
        String.
    """
    a, b = [i for i in 'aeiou'], [i for i in 'tskyrhzjgqmxlvnfcpwbd']
    return ''.join([np.random.choice(b if j % 2 == 0 else a) for j in range(chars)])


def random_code_name_dir(directory='./out', chars=6):
    """Random code name directory.

    Creates random code name and creates a subdirectory with said name under `directory`.
    Code names that are already taken (subdirectory already exists) are not reused.

    Args:
        directory: Root directory.
        chars: Number of characters for the code name.

    Returns:
        Tuple of code name and created directory.
    """
    try:
        code_name = random_code_name(chars=chars)
        out_dir = join(directory, code_name)
        makedirs(out_dir)
    except FileExistsError:
        return random_code_name_dir(directory)
    return code_name, out_dir


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

    Download an image from URL and convert it to a numpy array or PIL Image.

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


def load_image(name, method='imageio') -> np.ndarray:
    """Load image.

    Load image from URL or from filename via ``imageio`` or ``pytiff``.

    Args:
        name: URL (must start with ``http``) or filename.
        method: Method to use for filenames.

    Returns:
        Image.
    """
    if name.startswith('http'):
        img = fetch_image(name)
    elif method == 'imageio':
        from imageio import imread
        img = imread(name)
    elif method == 'pytiff':
        from pytiff import Tiff
        with Tiff(name, 'r') as t:
            img = t[:]
    else:
        raise ValueError(f'Could not load {name} with method {method}. Also note that URLs should start with "http".')
    return img


def random_seed(seed, backends=False, deterministic_torch=True):
    """Set random seed.

    Set random seed to ``random``, ``np.random``, ``torch.backends.cudnn`` and ``torch.manual_seed``.
    Also advise torch to use deterministic algorithms.

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


def train_epoch(model, train_loader, device, optimizer, desc=None, scaler=None, scheduler=None, gpu_stats=False):
    """Basic train function.

    Notes:
        - Model should return dictionary: {'loss': Tensor[], ...}
        - Batch from `train_loader` should be a dictionary: {'inputs': Tensor[...], ...}
        - Model must be callable: `model(batch['inputs'], targets=batch)`

    Args:
        model: Model.
        train_loader: Data loader.
        device: Device.
        optimizer: Optimizer.
        desc: Description, appears in progress print.
        scaler: Gradient scaler. If set PyTorch's autocast feature is used.
        scheduler: Scheduler. Step called after epoch.
        gpu_stats: Whether to print GPU stats.
    """
    from torch.cuda.amp import autocast
    model.train()
    tq = tqdm(train_loader, desc=desc)
    gpu_st = None
    if gpu_stats:
        gpu_st = GpuStats()
    for batch_idx, batch in enumerate(tq):
        batch: dict = to_device(batch, device)
        optimizer.zero_grad()
        with autocast(scaler is not None):
            outputs: dict = model(batch['inputs'], targets=batch)
        loss = outputs['loss']
        info = [] if desc is None else [desc]
        if gpu_st is not None:
            info.append(str(gpu_st))
        losses = outputs.get('losses')
        if losses is not None and isinstance(losses, dict):
            info.append('losses(' + ', '.join(
                [(f'{k}: %g' % np.round(asnumpy(v), 3)) for k, v in losses.items() if v is not None]) + ')')
        info.append('loss %g' % np.round(asnumpy(loss), 3))
        tq.desc = ' - '.join(info)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    if scheduler is not None:
        scheduler.step()


def tweak_module_(module: nn.Module, class_or_tuple, must_exist=True, **kwargs):
    """Tweak module.

    Sets attributes for all modules that are instances of given `class_or_tuple`.

    Examples:
        >>> import celldetection as cd, torch.nn as nn
        >>> model = cd.models.ResNet18(in_channels=3)
        >>> cd.tweak_module_(model, nn.BatchNorm2d, momentum=0.05)  # sets momentum to 0.05

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


def replace_module_(module: nn.Module, class_or_tuple, substitute: nn.Module, recursive=True, **kwargs):
    """Replace module.

    Replace all occurrences of `class_or_tuple` in `module` with `substitute`.

    Examples:
        >>> # Replace all ReLU activations with LeakyReLU
        ... replace_module_(network, nn.ReLU, nn.LeakyReLU)

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes that are to be replaced.
        substitute: Substitute class or object.
        recursive: Whether to replace modules recursively.
        **kwargs: Keyword arguments passed to substitute constructor if it is a class.

    """
    for k, mod in module._modules.items():
        if isinstance(mod, class_or_tuple):
            if type(substitute) == type:
                module._modules[k] = substitute(**kwargs)
            else:
                module._modules[k] = substitute
        elif isinstance(mod, nn.Module) and recursive:
            replace_module_(mod, substitute=substitute, class_or_tuple=class_or_tuple, recursive=recursive, **kwargs)


def get_device(module: Union[nn.Module, Tensor, torch.device]):
    """Get device.

    Get device from Module.

    Args:
        module: Module. If ``module`` is a string or ``torch.device`` already, it is returned as is.

    Returns:
        Device.
    """
    if isinstance(module, torch.device):
        return module
    elif isinstance(module, str):
        return module
    elif hasattr(module, 'device'):
        return module.device
    p: nn.parameter.Parameter = next(module.parameters())
    return p.device


def _params(module: nn.Module, trainable=None, recurse=True) -> Iterator[nn.Parameter]:
    return (p for p in module.parameters(recurse=recurse) if (trainable is None or p.requires_grad == trainable))


def trainable_params(module: nn.Module, recurse=True) -> Iterator[nn.Parameter]:
    """Trainable parameters.

    Retrieve all trainable parameters.

    Args:
        module: Module.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Module parameters.
    """
    return _params(module, True, recurse=recurse)


def frozen_params(module: nn.Module, recurse=True) -> Iterator[nn.Parameter]:
    """Frozen parameters.

    Retrieve all frozen parameters.

    Args:
        module: Module.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Module parameters.
    """
    return _params(module, False, recurse=recurse)


def num_params(module: nn.Module, trainable=None, recurse=True) -> int:
    """Number of parameters.

    Count the number of parameters.

    Args:
        module: Module
        trainable: Optionally filter for trainable or frozen parameters.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in _params(module, trainable, recurse=recurse))


def count_submodules(module: nn.Module, class_or_tuple) -> int:
    """Count submodules.

    Count the number of submodules of the specified type(-es).

    Examples:
        >>> count_submodules(cd.models.U22(1, 0), nn.Conv2d)
        22

    Args:
        module: Module.
        class_or_tuple: All instances of given `class_or_tuple` are to be counted.

    Returns:
        Number of submodules.
    """
    return np.sum([1 for m in module.modules() if isinstance(m, class_or_tuple)])


def ensure_num_tuple(v, num=2, msg=''):
    if isinstance(v, (int, float)):
        v = (v,) * num
    elif isinstance(v, (list, tuple)):
        pass
    else:
        raise ValueError(msg)
    return v


def gaussian_kernel(kernel_size, sigma=-1) -> np.ndarray:
    """Get Gaussian kernel.

    Constructs and returns a Gaussian kernel.

    Args:
        kernel_size: Kernel size as int or tuple. It should be odd and positive.
        sigma: Gaussian standard deviation as float or tuple. If it is non-positive, it is computed from kernel_size as
            ``sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8``.

    Returns:
        Gaussian Kernel.
    """
    kernel_size = ensure_num_tuple(kernel_size, msg='kernel_size must be int, tuple or list.')
    sigma = ensure_num_tuple(sigma, msg='sigma must be int, tuple or list.')
    return getGaussianKernel(kernel_size[0], sigma[0]) @ getGaussianKernel(kernel_size[1], sigma[1]).T


class Bytes(int):
    """Bytes.

    Printable integer that represents Bytes.

    """
    UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', 'BB']

    def __str__(self):
        n = np.log2(int(self)) if self > 0 else 0
        s = None
        for i, tag in enumerate(self.UNITS):
            if n < (i + 1) * 10 or i == len(self.UNITS) - 1:
                s = str(np.round(float(self) / (2 ** (10 * i)), 2)) + tag
                break
        return s

    __repr__ = __str__


class Percent(float):
    """Percent.

    Printable float that represents percentage.

    """
    def __str__(self):
        return '%g%%' % np.round(self, 2)

    __repr__ = __str__


class GpuStats:
    def __init__(self, delimiter=', '):
        """GPU Statistics.

        Simple interface to print live GPU statistics from ``pynvml``.

        Examples:
            >>> import celldetection as cd
            >>> stat = cd.GpuStats()  # initialize once
            >>> print(stat)  # print current statistics
            gpu0(free: 22.55GB, used: 21.94GB, util: 93%), gpu1(free: 1.03GB, used: 43.46GB, util: 98%)

        Args:
            delimiter: Delimiter used for printing.
        """
        try:
            nv.nvmlInit()
            self.num = nv.nvmlDeviceGetCount()
        except:
            self.num = 0
        self.delimiter = delimiter

    def __len__(self):
        return self.num

    def __getitem__(self, item: int):
        if item >= len(self):
            raise IndexError
        h = nv.nvmlDeviceGetHandleByIndex(item)
        idx = nv.nvmlDeviceGetIndex(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        uti = nv.nvmlDeviceGetUtilizationRates(h)
        return idx, dict(
            free=Bytes(mem.free),
            used=Bytes(mem.used),
            util=Percent(uti.gpu)
        )

    def __str__(self):
        deli = self.delimiter
        return deli.join([f'gpu{i}({deli.join([f"{k}: {v}" for k, v in stat.items()])})' for i, stat in self])

    __repr__ = __str__


class Tiling:
    def __init__(self, tile_size: tuple, context_shape: tuple, overlap=0):
        self.overlap = overlap
        self.tile_size = tuple(tile_size)
        self.context_size = context_shape[:len(self.tile_size)]
        self.num_tiles_per_dim = np.ceil(np.array(self.context_size) / np.array(self.tile_size)).astype('int')
        self.num_tiles = np.prod(self.num_tiles_per_dim)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        tile_index = np.unravel_index(item, shape=self.num_tiles_per_dim)
        start = tile_index * np.array(self.tile_size)
        stop = np.minimum(start + self.tile_size, self.context_size)
        start_wo = np.maximum(start - self.overlap, 0)
        stop_wo = np.minimum(stop + self.overlap, self.context_size)
        start_ex = start - start_wo
        stop_ex = start - start_wo + stop - start
        return dict(
            start=start,
            stop=stop,
            slices=tuple([slice(a, b) for a, b in zip(start, stop)]),
            slices_with_overlap=tuple([slice(a, b) for a, b in zip(start_wo, stop_wo)]),
            slices_to_remove_overlap=tuple([slice(a, b) for a, b in zip(start_ex, stop_ex)]),
            start_ex=start_ex,
            stop_ex=stop_ex,
            start_with_overlap=start_wo,
            stop_with_overlap=stop_wo,
            num_tiles=self.num_tiles,
            num_tiles_per_dim=self.num_tiles_per_dim
        )
