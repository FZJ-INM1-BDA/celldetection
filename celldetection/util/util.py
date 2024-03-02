import numpy as np
import inspect
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Tuple, Any, Dict as TDict, Iterator, Type, Callable, Iterable, Sequence
from torch import Tensor
from torch.hub import load_state_dict_from_url
import hashlib
import json
from tqdm import tqdm
from os.path import join, isfile, splitext
from os import makedirs
import pynvml as nv
from cv2 import getGaussianKernel
import h5py
from collections import OrderedDict
import re
import sys
from itertools import product
from inspect import currentframe, signature
from shutil import copy2
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
from glob import glob

__all__ = ['Dict', 'lookup_nn', 'reduce_loss_dict', 'tensor_to', 'to_device', 'asnumpy', 'fetch_model',
           'random_code_name', 'dict_hash', 'fetch_image', 'random_seed', 'tweak_module_', 'add_to_loss_dict',
           'random_code_name_dir', 'get_device', 'num_params', 'count_submodules', 'train_epoch', 'Bytes', 'Percent',
           'GpuStats', 'trainable_params', 'frozen_params', 'Tiling', 'load_image', 'gaussian_kernel',
           'iter_submodules', 'replace_module_', 'wrap_module_', 'spectral_norm_', 'to_h5', 'to_tiff',
           'to_json', 'from_json', 'exponential_moving_average_', 'weight_norm_', 'inject_extra_repr_',
           'ensure_num_tuple', 'get_nd_conv', 'get_nd_linear', 'get_nd_dropout', 'get_nd_max_pool', 'get_nd_batchnorm',
           'get_warmup_factor', 'print_to_file', 'NormProxy', 'num_bytes', 'from_h5', 'update_dict_',
           'get_tiling_slices', 'get_nn', 'copy_script', 'hash_file', 'append_hash_to_filename', 'save_fetchable_model',
           'load_model', 'freeze_', 'unfreeze_', 'freeze_submodules_', 'unfreeze_submodules_',
           'image_to_base64', 'base64_to_image', 'model2dict', 'dict2model', 'is_ipython', 'grouped_glob',
           'tweak_attribute_', 'to_batched_h5', 'compare_file_hashes', 'import_file', 'load_imagej_rois',
           'glob_h5_split', 'say_goodbye', 'parse_url_params', 'save_requirements', 'get_installed_packages',
           'resolve_model', 'is_package_installed', 'has_argument', 'dict_to_json_string', 'resolve_pretrained']


def copy_script(dst, no_script_okay=True, frame=None, verbose=False):
    """Copy current script.

    Copies the script from where this function is called to ``dst``.
    By default, nothing happens if this function is not called from within a script.

    Args:
        dst: Copy destination. Filename or folder.
        no_script_okay: If ``False`` raise ``FileNotFoundError`` if no script is found.
        frame: Context frame.
        verbose: Whether to print source and destination when copying.

    """
    if frame is None:
        current_frame = currentframe()
        if current_frame:
            frame = current_frame.f_back
    if frame is None:
        raise ValueError('Invalid frame.')

    src = frame.f_globals.get('__file__')
    if src is None:
        if not no_script_okay:
            raise FileNotFoundError('Could not find current script.')
        return

    if verbose:
        print(f'Copy `{src}` to `{dst}`.')
    copy2(src, dst)


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


def replace_ndim(s: Union[str, type, Callable], dim: int, allowed_dims=(1, 2, 3)):
    """Replace ndim.

    Replaces dimension statement of ``string``or ``type``.

    Notes:
        - Dimensions are expected to be at the end of the type name.
        - If there is no dimension statement, nothing is changed.

    Examples:
        >>> replace_ndim('BatchNorm2d', 3)
        'BatchNorm3d'
        >>> replace_ndim(nn.BatchNorm2d, 3)
        torch.nn.modules.batchnorm.BatchNorm3d
        >>> replace_ndim(nn.GroupNorm, 3)
        torch.nn.modules.normalization.GroupNorm
        >>> replace_ndim(F.conv2d, 3)
        <function torch._VariableFunctionsClass.conv3d>

    Args:
        s: String or type.
        dim: Desired dimension.
        allowed_dims: Allowed dimensions to look for.

    Returns:
        Input with replaced dimension.
    """
    if isinstance(s, str) and dim in allowed_dims:
        return re.sub(f"[1-3]d$", f'{int(dim)}d', s)
    elif isinstance(s, type) or callable(s):
        return getattr(sys.modules[s.__module__], replace_ndim(s.__name__, dim))
    return s


def lookup_nn(item: str, *a, src=None, call=True, inplace=True, nd=None, **kw):
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
        >>> # Dict notation to contain all keyword arguments for calling in `item`. Always called once.
        ... lookup_nn(dict(relu=dict(inplace=True)), call=False)
            ReLU(inplace=True)
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, call=False)
            NormProxy(GroupNorm, kwargs={'num_groups': 32})
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, 32, call=True)
            GroupNorm(32, 32, eps=1e-05, affine=True)

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        nd: If set, replace dimension statement (e.g. '2d' in nn.Conv2d) with ``nd``.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    if src is None:
        from .. import models
        src = (nn, models)
    if isinstance(item, tuple):
        if len(item) == 1:
            item, = item
        elif len(item) == 2:
            item, _kw = item
            kw.update(_kw)
        else:
            raise ValueError('Allowed formats for item: (item,) or (item, kwargs).')
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        if nd is not None:
            l_item = replace_ndim(l_item, nd)
        if not isinstance(src, (list, tuple)):
            src = src,
        v = None
        for src_ in src:
            try:
                v = next((getattr(src_, i) for i in dir(src_) if i.lower() == l_item))
            except StopIteration:
                continue
            break
        if v is None:
            raise ValueError(f'Could not find `{item}` in {src}.')
    elif isinstance(item, nn.Module):
        return item
    elif isinstance(item, dict):
        assert len(item) == 1
        key, = item
        val = item[key]
        assert isinstance(val, dict)
        cls = lookup_nn(key, src=src, call=False, inplace=inplace, nd=nd)
        if issubclass(cls, nn.modules.loss._WeightedLoss):  # allows weight to be passed as lists (common use case)
            if 'weight' in val and not isinstance(val['weight'], Tensor):
                val['weight'] = torch.as_tensor(val['weight'])
        v = cls(**val)
    elif isinstance(item, type) and nd is not None:
        v = replace_ndim(item, nd)
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v


def get_nn(item: Union[str, 'nn.Module', Type['nn.Module']], src=None, nd=None, call_if_type=False):
    ret = lookup_nn(item, src=src, nd=nd, call=False)
    if call_if_type and type(ret) is type:
        ret = ret()
    return ret


class NormProxy:
    def __init__(self, norm, **kwargs):
        """Norm Proxy.

        Examples:
            >>> GroupNorm = NormProxy('groupnorm', num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> GroupNorm = NormProxy(nn.GroupNorm, num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> BatchNorm2d = NormProxy('batchnorm2d', momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)
            >>> BatchNorm2d = NormProxy(nn.BatchNorm2d, momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)

        Args:
            norm: Norm class or name.
            **kwargs: Keyword arguments.
        """
        self.norm = norm
        self.kwargs = kwargs

    def __call__(self, num_channels):
        Norm = lookup_nn(self.norm, call=False)
        kwargs = dict(self.kwargs)
        args = inspect.getfullargspec(Norm).args
        if 'num_features' in args:
            kwargs['num_features'] = num_channels
        elif 'num_channels' in args:
            kwargs['num_channels'] = num_channels
        return Norm(**kwargs)

    def __repr__(self):
        return f'NormProxy({self.norm}, kwargs={self.kwargs})'

    __str__ = __repr__


def reduce_loss_dict(losses: dict, divisor, ignore_prefix='_'):
    return sum((i for k, i in losses.items() if (i is not None and not k.startswith(ignore_prefix)))) / divisor


def add_to_loss_dict(d: dict, key: str, loss: torch.Tensor, weight=None):
    if loss is None:
        return
    dk = d.get(key, None)
    torch.nan_to_num_(loss, 0., 0., 0.)
    if weight is not None:
        loss = loss * weight
    d[key] = loss if dk is None else dk + loss


def tensor_to(inputs: Union[list, tuple, dict, Tensor], *args, **kwargs):
    """Tensor to device/dtype/other.

    Recursively calls ``tensor.to(*args, **kwargs)`` for all ``Tensors`` in ``inputs``.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        inputs: Tensor, list, tuple or dict. Non-Tensor objects are ignored. Tensors are substituted by result of
            ``tensor.to(*args, **kwargs)`` call.
        *args: Arguments. See docstring of ``torch.Tensor.to``.
        **kwargs: Keyword arguments. See docstring of ``torch.Tensor.to``.

    Returns:
        Inputs with Tensors replaced by ``tensor.to(*args, **kwargs)``.
    """
    if isinstance(inputs, Tensor):
        inputs = inputs.to(*args, **kwargs)
    elif isinstance(inputs, (dict, OrderedDict)):
        inputs = {k: tensor_to(b, *args, **kwargs) for k, b in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inputs = type(inputs)([tensor_to(b, *args, **kwargs) for b in inputs])
    return inputs


def to_device(batch: Union[list, tuple, dict, Tensor], device):
    """To device.

    Move Tensors to device.
    Input can be Tensor, tuple of Tensors, list of Tensors or a dictionary of Tensors.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        batch: Tensor, list, tuple or dict. Non-Tensor objects are ignored. Tensors are moved to ``device``.
        device: Device.

    Returns:
        Input with Tensors moved to device.
    """
    return tensor_to(batch, device)


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
    elif isinstance(v, (np.ndarray, int, float, bool, str)):
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


def dict2model(conf, **kwargs):
    from .. import models

    if len(conf) == 1:  # alternative format: {'class_name': kwargs}
        key, = conf.keys()
        if key != 'model':
            m = getattr(models, key, None)
            if m is not None:
                return m(**conf[key])

    # Format: {'model': class_name, 'kwargs': kwargs}
    kw = {**conf.get('kwargs', conf.get('kw', {})), **kwargs}
    m = getattr(models, conf['model'])(*conf.get('args', conf.get('a', ())), **kw)
    return m


def _load_cd_format(m, pretrained=True, **kwargs):
    assert isinstance(m, dict) and 'cd.models' in m.keys()
    state_dict = m['state_dict']
    conf = m['cd.models']
    m = dict2model(conf, **kwargs)
    if pretrained:
        m.load_state_dict(state_dict, strict=kwargs.get('pretrained_strict', True))
    return m


def load_model(filename, map_location=None, **kwargs):
    assert isfile(filename), f'Could not find file: {filename}'
    m = torch.load(filename, map_location=map_location, **kwargs.pop('load_kwargs', {}))
    if isinstance(m, dict) and 'cd.models' in m.keys():
        return _load_cd_format(m, **kwargs)
    return m


def fetch_model(name, map_location=None, **kwargs):
    """Fetch model from URL.

    Loads model or state dict from URL.

    Args:
        name: Model name hosted on `celldetection.org` or url. Urls must start with 'http'.
        map_location: A function, `torch.device`, string or a dict specifying how to remap storage locations.
        **kwargs: From the doc of `torch.models.utils.load_state_dict_from_url`.

    """
    load_state_dict_kwargs = kwargs.pop('load_state_dict_kwargs', {})
    if name.startswith('cd://'):
        name = name[len('cd://'):]
    if not name.startswith('http'):
        if splitext(name)[1] not in ('.pt', '.pth', 'ckpt'):
            name = name + '.pt'
        url = f'https://celldetection.org/torch/models/{name}'
        load_state_dict_kwargs['check_hash'] = load_state_dict_kwargs.get('check_hash', True)
    else:
        url = name
    # url = name if name.startswith('http') else f'https://celldetection.org/torch/models/{name}.pt'
    m = load_state_dict_from_url(url, map_location=map_location, **load_state_dict_kwargs)
    if isinstance(m, dict) and 'cd.models' in m.keys():
        m = _load_cd_format(m, **kwargs)
    return m


def append_hash_to_filename(filename, num=None, ext=True):
    from os import rename
    prefix = filename
    postfix = ''
    if ext:
        sp = prefix.split('.')
        prefix = '.'.join(sp[:-1])
        postfix = f'.{sp[-1]}'
    sha256 = hash_file(filename)
    if num is not None:
        sha256 = sha256[:num]
    dst = prefix + f'-{sha256}' + postfix
    rename(filename, dst)


def model2dict(model: 'nn.Module'):
    return dict(
        model=model.__class__.__name__,
        kwargs=dict(model.hparams),
    )


def save_fetchable_model(model: 'nn.Module', filename, append_hash=16, **kwargs):
    from ..__meta__ import __version__
    if not len(splitext(filename)[1]):
        filename += '.pt'
    model.eval()
    model = model.to('cpu')
    torch.save({
        'cd.__version__': __version__,
        'cd.models': model2dict(model),
        'state_dict': model.state_dict(),
        **kwargs
    }, filename)
    if append_hash:
        if append_hash is True:
            append_hash = None
        append_hash_to_filename(filename, num=append_hash)


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


def random_code_name_dir(directory='./out', chars=6, comm=None, root_rank=0):
    """Random code name directory.

    Creates random code name and creates a subdirectory with said name under `directory`.
    Code names that are already taken (subdirectory already exists) are not reused.

    Args:
        directory: Root directory.
        chars: Number of characters for the code name.
        comm: MPI Comm. If provided, code name and directory is automatically broadcasted to all ranks of `comm`.
        root_rank: Root rank. Only the root rank creates code name and directory.

    Returns:
        Tuple of code name and created directory.
    """
    rank = code_name = out_dir = None
    if comm is not None:
        rank = comm.Get_rank()
    if rank is None or rank == root_rank:
        try:
            code_name = random_code_name(chars=chars)
            out_dir = join(directory, code_name)
            makedirs(out_dir)
        except FileExistsError:
            return random_code_name_dir(directory, chars=chars)
    if rank is not None:
        code_name, out_dir = comm.bcast((code_name, out_dir), root=root_rank)
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
    torch.cuda.manual_seed(seed)
    if backends:
        cudnn.deterministic = True
        cudnn.benchmark = False
    if deterministic_torch and 'use_deterministic_algorithms' in dir(torch):
        torch.use_deterministic_algorithms(True)


def train_epoch(model, train_loader, device, optimizer, desc=None, scaler=None, scheduler=None, gpu_stats=False,
                progress=True):
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
        progress: Show progress.
    """
    from torch.cuda.amp import autocast
    model.train()
    tq = tqdm(train_loader, desc=desc) if progress else train_loader
    gpu_st = None
    if gpu_stats:
        gpu_st = GpuStats()
    for batch_idx, batch in enumerate(tq):
        batch: dict = to_device(batch, device)
        optimizer.zero_grad()
        with autocast(scaler is not None):
            outputs: dict = model(batch['inputs'], targets=batch)
        loss = outputs['loss']
        if progress:
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


def iter_submodules(module: nn.Module, class_or_tuple, recursive=True):
    for k, mod in module._modules.items():
        if isinstance(mod, class_or_tuple):
            yield module._modules, k, mod
        if isinstance(mod, nn.Module) and recursive:
            yield from iter_submodules(mod, class_or_tuple, recursive=recursive)


def tweak_module_(module: nn.Module, class_or_tuple, must_exist=True, recursive=True, **kwargs):
    """Tweak module.

    Set attributes for all modules that are instances of given `class_or_tuple`.

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
        recursive: Whether to search for modules recursively.
        **kwargs: Attributes to be tweaked: `<attribute_name>=<value>`.
    """
    for handle, key, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        for k, v in kwargs.items():
            if must_exist:
                getattr(mod, k)
            setattr(mod, k, v)


def tweak_attribute_(module: nn.Module, require_existence=True, **kwargs):
    """Tweak attribute.

    Allows to change attributes of a module.

    Args:
        module: Module.
        **kwargs: Key value pairs. Keys specify the attribute (e.g. `submodule.attribute_name`) and value
            the respective new value.

    """
    for k, v in kwargs.items():
        x = module
        sp = k.split('.')
        for k_ in sp[:-1]:
            x = getattr(x, k_)
        if require_existence:
            assert hasattr(x, sp[-1]), f'Could not find {sp[-1]} attribute in {x}.'
        setattr(x, sp[-1], v)


def replace_module_(module: nn.Module, class_or_tuple, substitute: Union[Type[nn.Module], nn.Module], recursive=True,
                    inherit_attr: Union[list, str, dict] = None, **kwargs):
    """Replace module.

    Replace all occurrences of `class_or_tuple` in `module` with `substitute`.

    Examples:
        >>> # Replace all ReLU activations with LeakyReLU
        ... cd.replace_module_(network, nn.ReLU, nn.LeakyReLU)

        >>> # Replace all BatchNorm layers with InstanceNorm and inherit `num_features` attribute
        ... cd.replace_module_(network, nn.BatchNorm2d, nn.InstanceNorm2d, inherit_attr=['num_features'])

        >>> # Replace all BatchNorm layers with GroupNorm and inherit `num_features` attribute
        ... cd.replace_module_(network, nn.BatchNorm2d, nn.GroupNorm, num_groups=32,
        ...                    inherit_attr={'num_channels': 'num_features'})

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes that are to be replaced.
        substitute: Substitute class or object.
        recursive: Whether to replace modules recursively.
        inherit_attr: Attributes to be inherited. String, list or dict of attribute names.
            Attribute values are retrieved from replaced module and passed to substitute constructor.
            Formats:
            ``'attr_name'``,
            ``['attr_name0', 'attr_name1', ...]``,
            ``{'substitute_kw0': 'attr_name0', ...}``
        **kwargs: Keyword arguments passed to substitute constructor if it is a class.
    """
    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if isinstance(substitute, nn.Module):
            handle[k] = substitute
        else:
            kw = {}
            if isinstance(inherit_attr, str):
                inherit_attr = [inherit_attr]
            if isinstance(inherit_attr, list):
                kw = {k: mod.__dict__[k] for k in inherit_attr}
            elif isinstance(inherit_attr, dict):
                kw = {k: mod.__dict__[v] for k, v in inherit_attr.items()}
            handle[k] = substitute(**kwargs, **kw)


def inject_extra_repr_(module, name, fn):
    """Inject extra representation.

    Injects additional ``extra_repr`` function to ``module``.
    This can be helpful to indicate presence of hooks.

    Note:
        This is an inplace operation.

    Notes:
        - This op may impair pickling.

    Args:
        module: Module.
        name: Name of the injected function (only used to avoid duplicate injection).
        fn: Callback function.

    """

    def extra_repr(self=module):
        vals = [self.extra_repr_orig()] + list(f(self) for f in self.extra_repr_funcs.values())
        return ', '.join([v for v in vals if v])

    if not hasattr(module, 'extra_repr_orig'):
        module.extra_repr_orig = module.extra_repr
        module.extra_repr_funcs = {}
        module.extra_repr = extra_repr
    module.extra_repr_funcs[name] = fn


def wrap_module_(module: nn.Module, class_or_tuple, wrapper, recursive=True, **kwargs):
    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        handle[k] = wrapper(handle[k], **kwargs)


def spectral_norm_(module, class_or_tuple=nn.Conv2d, recursive=True, name='weight', add_repr=False, **kwargs):
    """Spectral normalization.

    Applies spectral normalization to parameters of all occurrences of ``class_or_tuple`` in the given module.

    Note:
        This is an inplace operation.

    References:
        - https://arxiv.org/pdf/1802.05957.pdf

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes whose parameters are to be normalized.
        recursive: Whether to search for modules recursively.
        name: Name of weight parameter.
        add_repr: Whether to indicate use of spectral norm in a module's representation.
            Note that this may impair pickling.
        **kwargs: Additional keyword arguments for ``torch.nn.utils.spectral_norm``.
    """

    def extra_repr(self):
        if 'torch.nn.utils.spectral_norm.SpectralNorm' in str(list(self._forward_pre_hooks.values())):
            return 'spectral_norm=True'

    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if mod._parameters.get(name) is not None:
            handle[k] = nn.utils.spectral_norm(handle[k], name=name, **kwargs)
            if add_repr:
                inject_extra_repr_(handle[k], 'spectral_norm', extra_repr)


def weight_norm_(module, class_or_tuple=nn.Conv2d, recursive=True, name='weight', add_repr=False, **kwargs):
    """Weight normalization.

    Applies weight normalization to parameters of all occurrences of ``class_or_tuple`` in the given module.

    Note:
        This is an inplace operation.

    References:
        - https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes whose parameters are to be normalized.
        recursive: Whether to search for modules recursively.
        name: Name of weight parameter.
        add_repr: Whether to indicate use of weight norm in a module's representation.
            Note that this may impair pickling.
        **kwargs: Additional keyword arguments for ``torch.nn.utils.weight_norm``.
    """

    def extra_repr(self):
        if 'torch.nn.utils.weight_norm.WeightNorm' in str(list(self._forward_pre_hooks.values())):
            return 'weight_norm=True'

    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if mod._parameters.get(name) is not None:
            handle[k] = nn.utils.weight_norm(handle[k], name=name, **kwargs)
            if add_repr:
                inject_extra_repr_(handle[k], 'weight_norm', extra_repr)


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
    return int(np.sum([1 for m in module.modules() if isinstance(m, class_or_tuple)]))


def ensure_num_tuple(v, num=2, msg=''):
    if isinstance(v, (int, float)):
        v = (v,) * num
    elif isinstance(v, (list, tuple)):
        pass
    else:
        raise ValueError(msg)
    return v


def gaussian_kernel(kernel_size, sigma=-1, nd=2) -> np.ndarray:
    """Get Gaussian kernel.

    Constructs and returns a Gaussian kernel.

    Args:
        kernel_size: Kernel size as int or tuple. It should be odd and positive.
        sigma: Gaussian standard deviation as float or tuple. If it is non-positive, it is computed from kernel_size as
            ``sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8``.
        nd: Number of kernel dimensions.

    Returns:
        Gaussian Kernel.
    """
    kernel_sizes = ensure_num_tuple(kernel_size, num=nd, msg='kernel_size must be int, tuple or list.')
    sigmas = ensure_num_tuple(sigma, num=nd, msg='sigma must be int, tuple or list.')
    y = None
    for k, s in zip(kernel_sizes, sigmas):
        y_ = getGaussianKernel(k, s)[:, 0]
        if y is None:
            y = y_
        else:
            y = y[..., :, None] * y_[..., None, :]
    return y


class Bytes(int):
    """Bytes.

    Printable integer that represents Bytes.

    """
    UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB', 'BiB']

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
        except Exception:
            self.num = 0
        self.delimiter = delimiter

    def __len__(self):
        return self.num

    @staticmethod
    def _is_mig_enabled(h):
        # https://docs.nvidia.com/deploy/nvml-api/group__nvmlMultiInstanceGPU.html#group__nvmlMultiInstanceGPU_1g1fa0acf9076404ef28c3c6976eb96f2b
        try:
            mode, pending_mode = nv.nvmlDeviceGetMigMode(h)
            is_mig_enabled = mode == nv.NVML_DEVICE_MIG_ENABLE
        except nv.NVMLError:  # if device doesn't support MIG mode
            is_mig_enabled = False
        return is_mig_enabled

    @staticmethod
    def _get_mig_info(h, idx):
        # https://docs.nvidia.com/deploy/nvml-api/group__nvmlMultiInstanceGPU.html#group__nvmlMultiInstanceGPU_1g15e07cc6230a2d90c5bc85de85261ef7
        mig_device_count = nv.nvmlDeviceGetMaxMigDeviceCount(h)
        mig_infos = []
        for i in range(mig_device_count):
            try:
                # Get MIG device handle for the given index under its parent NVML device
                mig_h = nv.nvmlDeviceGetMigDeviceHandleByIndex(h, i)

                # Per-instance information can be queried by using specific MIG device handles
                mem = nv.nvmlDeviceGetMemoryInfo(mig_h)

                # Utilization rates are not available for MIG devices
                # https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g540824faa6cef45500e0d1dc2f50b321

                mig_infos.append((f'{idx}-{i}', {
                    'free': Bytes(mem.free),
                    'used': Bytes(mem.used)
                }))
            except nv.NVMLError:
                continue
        return mig_infos

    @staticmethod
    def _get_nonmig_info(h, idx):
        info = None
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            uti = nv.nvmlDeviceGetUtilizationRates(h)
            info = idx, dict(
                free=Bytes(mem.free),
                used=Bytes(mem.used),
                util=Percent(uti.gpu)
            )
        except nv.NVMLError:
            pass
        return info

    def __getitem__(self, item: int):
        if item >= len(self):
            raise IndexError
        h = nv.nvmlDeviceGetHandleByIndex(item)
        idx = nv.nvmlDeviceGetIndex(h)
        return (self._get_mig_info if self._is_mig_enabled(h) else self._get_nonmig_info)(h, idx)

    def dict(self, byte_lvl=3, prefix='gpu'):
        d = {}
        for r in self:
            if not isinstance(r, list):
                r = [r]  # wrap non-mig info
            for i, stat in r:
                for k, v in stat.items():
                    if isinstance(v, Bytes):
                        v = np.round(float(v) / (2 ** (10 * byte_lvl)), 2)
                    d[f'{prefix}{i}-{k}'] = float(v)
        return d

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


def get_tiling_slices(
        size: Sequence[int],
        crop_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]],
        return_overlaps=False
) -> Union[
    Tuple[Iterable[slice], Tuple[int]],
    Tuple[Iterable[slice], Iterable[Tuple[int]], Tuple[int]]
]:
    """Get tiling slices.

    Args:
        size: Reference size as tuple.
        crop_size: Crop size.
        strides: Strides.
        return_overlaps: Whether to return overlaps.

    Returns:
        Iterable[slice], Tuple[int]:
            Iterator of tiling slices (each slice defining a tile),
            Number of tiles per dimension as tuple.
        Iterable[slice], Iterable[Tuple[int]], Tuple[int]:
            Iterator of tiling slices (each slice defining a tile),
            Iterator of overlaps (overlaps with adjacent tiles for each tile),
            Number of tiles per dimension as tuple.
    """
    assert isinstance(size, (tuple, list))
    crop_size = ensure_num_tuple(crop_size, len(size))
    strides = ensure_num_tuple(strides, len(size))
    slices, shape, overlaps = [], [], []
    for axis in range(len(size)):
        if crop_size[axis] >= size[axis]:
            tl = [size[axis]]
        else:
            tl = range(crop_size[axis],
                       1 + crop_size[axis] + (int(np.ceil((size[axis] - crop_size[axis]) / strides[axis]))) *
                       strides[axis],
                       strides[axis])
        stops = np.minimum(tl, size[axis])
        starts = np.maximum(0, stops - crop_size[axis])
        overlaps_start = np.concatenate((starts[:1], stops[:-1])) - starts
        axis_slices, axis_overlaps = [], []
        for a, b, *ov in zip(starts, stops, overlaps_start, np.concatenate((overlaps_start[1:], [0]))):
            axis_slices.append(slice(a, b))
            axis_overlaps.append(ov)
        slices.append(axis_slices), shape.append(len(starts)), overlaps.append(axis_overlaps)
    slices = product(*slices)
    if return_overlaps:
        return slices, product(*overlaps), shape
    return slices, shape


def to_h5(filename, mode='w', chunks=None, compression=None, overwrite=False, driver=None,
          create_dataset_kw: dict = None, attributes: dict = None, **kwargs):
    """To hdf5 file.

    Write data to hdf5 file.

    Args:
        filename: File name.
        mode: Mode.
        chunks: Chunks setting for created datasets. Chunk shape, or True to enable auto-chunking.
            Can be dictionary, if each dataset needs a different chunking.
            Individual chunks can be integer. Then each dimension is chunked to that integer
            or the dimension, whichever is smaller.
        compression: Compression setting for created datasets. Legal values are 'gzip', 'szip', 'lzf'. If an integer
            in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        overwrite: Whether to overwrite existing dataset. If False, attempt to replace the contents of the existing
            dataset, without creating a new dataset.
        driver: Hdf5 driver.
        create_dataset_kw: Additional keyword arguments for ``h5py.File().create_dataset``.
        attributes: Attributes. Format: `dict(dataset_name=dict(attribute0=value0))`.
            Note that only specific attributes are supported by h5py (e.g. not None).
        **kwargs: Data as ``{dataset_name: data}``.
    """
    attributes = {} if attributes is None else attributes
    create_dataset_kw = {} if create_dataset_kw is None else create_dataset_kw
    with h5py.File(filename, mode, **({} if driver is None else dict(driver=driver))) as h:
        for k, v in kwargs.items():
            chunks_ = chunks[k] if isinstance(chunks, dict) else chunks
            if isinstance(chunks_, int) and v.ndim > 1:
                chunks_ = tuple(np.minimum((256,) * v.ndim, v.shape))
            exists = k in h
            if overwrite and exists:
                del h[k]
            if exists:
                ds = h[k]
                ds[:] = v
            else:
                ds = h.create_dataset(k, data=v, compression=compression, chunks=chunks_, **create_dataset_kw)
            attrs = attributes.get(k)
            if attrs:
                assert isinstance(attrs, dict)
                ds.attrs.update(attrs)


def to_batched_h5(filename, index, batch_size=256, mode='a', chunks=None, compression=None, overwrite=False,
                  driver=None, create_dataset_kw: dict = None, file_digits=6, item_digits=6, **kwargs):
    """To batched hdf5 file.

    Write data to batched hdf5 file.
    When called for multiple data inputs (with different indices), this function creates possibly multiple hdf5 files,
    each containing up to `batch_size` items.
    Each call creates exactly one item (or item group).

    Content is assigned to a `batch_id` and an `item_id`, based on `index` and `batch_size`.
    The filename is changed from `filename.h5` to `filename_000001.h5`, to include the `batch_id`.
    Dataset keys are changed from `key` to `key_000001`, to include the `item_id`.

    Args:
        filename: File name.
        index: Batch index (int).
        batch_size: Batch size (int).
        mode: Mode.
        chunks: Chunks setting for created datasets. Chunk shape, or True to enable auto-chunking.
            Can be dictionary, if each dataset needs a different chunking.
            Individual chunks can be integer. Then each dimension is chunked to that integer
            or the dimension, whichever is smaller.
        compression: Compression setting for created datasets. Legal values are 'gzip', 'szip', 'lzf'. If an integer
            in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        overwrite: Whether to overwrite existing dataset. If False, attempt to replace the contents of the existing
            dataset, without creating a new dataset.
        driver: Hdf5 driver.
        create_dataset_kw: Additional keyword arguments for ``h5py.File().create_dataset``.
        file_digits: Number of digits to display batch index.
        item_digits: Number of digits to display item index.
        **kwargs: Data as ``{dataset_name: data}``.

    """
    batch_id = index // batch_size
    item_id = index % batch_size
    pre, ext = splitext(filename)
    if isinstance(chunks, dict):
        chunks = {f'{k}_%0{item_digits}d' % item_id: v for k, v in chunks.items()}
    to_h5(
        filename=f'{pre}_%0{file_digits}d{ext}' % batch_id,
        mode=mode,
        chunks=chunks,
        compression=compression,
        overwrite=overwrite,
        driver=driver,
        create_dataset_kw=create_dataset_kw,
        **{f'{k}_%0{item_digits}d' % item_id: v for k, v in kwargs.items()}
    )


def from_h5(filename, *keys, file_kwargs=None, **keys_slices):
    """From h5.

    Reads data from hdf5 file.

    Args:
        filename: Filename.
        *keys: Keys to read.
        file_kwargs: File keyword arguments.
        **keys_slices: Keys with indices or slices. E.g. `from_h5('file.h5', 'key0', key=slice(0, 42))`.

    Returns:
        Data from hdf5 file. As tuple if multiple keys are provided.
    """
    with h5py.File(filename, 'r', **(file_kwargs or {})) as h:
        if len(keys) == 0 and len(keys_slices) == 0:
            print('Available keys:', list(h.keys()), flush=True)
        res = tuple(h[k][:] for k in keys) + tuple(h[k][v] for k, v in keys_slices.items())
    if len(res) == 1:
        res, = res
    return res


def to_tiff(filename, image, mode='w', method='tile', bigtiff=True):
    """To tiff file.

    Write ``image`` to tiff file using ``pytiff``.
    By default, the tiff is tiled, s.t. crops can be read from disk without loading the entire image into memory first.

    Notes:
        - ``pytiff`` must be installed to use this function.

    References:
        https://pytiff.readthedocs.io/en/master/quickstart.html

    Args:
        filename: File name.
        image: Image.
        mode: Mode.
        method: Method. Either ``'tile'`` or ``'scanline'``.
        bigtiff: Whether to use bigtiff format.

    """
    try:
        from pytiff import Tiff
    except ModuleNotFoundError:
        raise ModuleNotFoundError('To use the to_tiff function pytiff must be installed.\n'
                                  'See: https://pytiff.readthedocs.io/en/master/quickstart.html')
    with Tiff(filename, mode, bigtiff=bigtiff) as handle:
        handle.write(image, method=method)


def exponential_moving_average_(module_avg, module, alpha=.999, alpha_non_trainable=0., buffers=True):
    """Exponential moving average.

    Update the variables of ``module_avg`` to be slightly closer to ``module``.

    References:
        - https://arxiv.org/pdf/1710.10196.pdf
        - https://arxiv.org/pdf/2006.07733.pdf

    Notes:
        - Whether a parameter is trainable or not is checked on ``module``
        - ``module_avg`` can be on different device and entirely frozen

    Args:
        module_avg: Average module. The parameters of this model are to be updated.
        module: Other Module.
        alpha: Fraction of trainable parameters of ``module_avg``; (1 - alpha) is fraction of trainable
            parameters of ``module``.
        alpha_non_trainable: Same as ``alpha``, but for non-trainable parameters.
        buffers: Whether to copy buffers from ``module`` to ``module_avg``.
    """
    device = get_device(module_avg)
    with torch.no_grad():
        for avg, new in zip(_params(module_avg), _params(module)):
            a = alpha if new.requires_grad else alpha_non_trainable
            avg.data.mul_(a).add_(new.data.to(device), alpha=1 - a)
    if buffers:
        for avg, new in zip(module_avg.buffers(), module.buffers()):
            avg.copy_(new)


def to_json(filename, obj, mode='w'):
    """To JSON.

    Dump ``obj`` to JSON file with name ``filename``.

    Args:
        filename: File name.
        obj: Object.
        mode: File mode.
    """
    with open(filename, mode) as fp:
        json.dump(obj, fp)


def from_json(filename):
    """From JSON.

    Load object from JSON file with name ``filename``.

    Args:
        filename: File name.
    """
    with open(filename, 'r') as fp:
        v = json.load(fp)
    return v


def get_nd_conv(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Conv%dd' % dim)


def get_nd_max_pool(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'MaxPool%dd' % dim)


def get_nd_batchnorm(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'BatchNorm%dd' % dim)


def get_nd_dropout(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Dropout%dd' % dim)


def get_nd_linear(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return ['', 'bi', 'tri'][dim - 1] + 'linear'


def get_warmup_factor(step, steps=1000, factor=0.001, method='linear'):
    if step >= steps:
        return 1.
    if method == 'constant':
        return factor
    elif method == 'linear':
        a = step / steps
        return factor * (1 - a) + a
    raise ValueError(f'Unknown method: {method}')


def print_to_file(*args, filename, mode='w', **kwargs):
    with open(filename, mode=mode) as f:
        print(*args, file=f, **kwargs)


def num_bytes(x: Union[np.ndarray, Tensor]):
    """Num Bytes.

    Returns the size in bytes of the given ndarray or Tensor.

    Args:
        x: Array or Tensor.

    Returns:
        Bytes
    """
    if isinstance(x, np.ndarray):
        bts = x.itemsize * x.size
    elif isinstance(x, Tensor):
        bts = x.numel() * x.element_size()
    else:
        raise ValueError(f'Could not handle type: {type(x)}')
    return Bytes(bts)


def update_dict_(dst, src, override=False, keys: Union[List[str], Tuple[str]] = None):
    for k, v in src.items():
        if keys is not None and k not in keys:
            continue
        if override or k not in dst:
            dst[k] = v


def freeze_(module: "nn.Module", recurse=True):
    """Freeze.

    Freezes a module by setting `param.requires_grad=False` and calling `module.eval()`.

    Args:
        module: Module.
        recurse: Whether to freeze parameters of this layer and submodules or only parameters that are direct members
            of this module.
    """
    for param in module.parameters(recurse=recurse):
        param.requires_grad = False
    module.eval()


def unfreeze_(module: "nn.Module", recurse=True):
    """Unfreeze.

    Unfreezes a module by setting `param.requires_grad=True` and calling `module.train()`.

    Args:
        module: Module.
        recurse: Whether to unfreeze parameters of this layer and submodules or only parameters that are direct members
            of this module.
    """
    for param in module.parameters(recurse=recurse):
        param.requires_grad = False
    module.train()


def freeze_submodules_(module: "nn.Module", *names, recurse=True):
    """Freeze specific submodules.

    Freezes submodules by setting `param.requires_grad=False` and calling `submodule.eval()`.

    Args:
        module: Module.
        names: Names of submodules.
        recurse: Whether to freeze parameters of specified modules and their respective submodules or only parameters
            that are direct members of the specified submodules.
    """
    assert len(names), 'Specify at least one submodule by name.'
    if len(names) == 1 and isinstance(names[0], (tuple, list)):
        names, = names
    for n in names:
        freeze_(module.get_submodule(n), recurse=recurse)


def unfreeze_submodules_(module: "nn.Module", *names, recurse=True):
    """Unfreeze specific submodules.

    Unfreezes submodules by setting `param.requires_grad=True` and calling `submodule.train()`.

    Args:
        module: Module.
        names: Names of submodules.
        recurse: Whether to unfreeze parameters of specified modules and their respective submodules or only parameters
            that are direct members of the specified submodules.
    """
    assert len(names), 'Specify at least one submodule by name.'
    if len(names) == 1 and isinstance(names[0], (tuple, list)):
        names, = names
    for n in names:
        unfreeze_(module.get_submodule(n), recurse=recurse)


def image_to_base64(img: 'np.ndarray', ext='png', as_url=True, url_template=None):
    """Image to base64.

    Converts image to base64 code.

    Args:
        img: Image as numpy array.
        ext: Image format.
        as_url: Whether to format result as URL.
        url_template: Optional URL template containing `ext` and `code` placeholders.

    Returns:
        Base64 code.
    """
    pi = Image.fromarray(img)
    buff = BytesIO()
    pi.save(buff, format='png')
    code = b64encode(buff.getvalue()).decode('utf-8')
    if as_url:
        if url_template is None:
            url_template = 'data:image/{ext};base64,{code}'
        return url_template.format(ext=ext, code=code)
    return code


def base64_to_image(code, as_numpy=True):
    """Base64 to image.

    Converts base64 code to image.

    Args:
        code: Base 64 code.
        as_numpy: Whether to convert results to numpy instead of `PIL.Image`.

    Returns:
        Image.
    """
    base64_decoded = b64decode(code)
    img = Image.open(BytesIO(base64_decoded))
    if as_numpy:
        return np.array(img)
    return img


def is_ipython() -> bool:
    """Is IPython.

    Checks whether function is called via IPython.

    Returns:
        True/False.
    """
    import builtins
    return getattr(builtins, '__IPYTHON__', False)


def grouped_glob(pathname, group_pattern, *, sort=True, keep_unmatched=True, sub_kwargs=None, substitute='', **kwargs):
    """Grouped glob.

    A glob helper that groups search results by `group_pattern`.

    Args:
        pathname: Glob pattern.
        group_pattern: Group pattern for `re.sub`. Matches are substituted with `substitute` for each filename.
            The resulting string is used as the group handle. All files with the same handle belong to the same group.
        sort: Whether to sort results.
        keep_unmatched: Whether to keep files that do not match the `group_pattern`.
        sub_kwargs: Keyword arguments for `re.sub`.
        substitute: Substitute for creation of the group handle (see `group_pattern`).
        **kwargs: Keyword arguments for glob.

    Returns:
        Dictionary of glob results.
    """
    files = glob(pathname, **kwargs)
    if sort:
        files = sorted(files)

    grouped = {}
    for f in files:
        try:
            next(re.finditer(group_pattern, f))
            key = re.sub(group_pattern, substitute, f, **({} if sub_kwargs is None else sub_kwargs))
        except StopIteration:
            if keep_unmatched:
                key = f
            else:
                continue
        grouped[key] = li = grouped.get(key, [])
        li.append(f)
    return grouped


def hash_file(filename, method='sha256', buffer_size=8192) -> str:
    """Hash file.

    Computes a hash for the given file.

    Args:
        filename: Filename.
        method: Hashing method.
        buffer_size: Buffer size (file is hashed in chunks).

    Returns:
        Hash.
    """
    import hashlib

    hasher = getattr(hashlib, method)()
    with open(filename, 'rb') as file:
        buffer = file.read(buffer_size)
        while buffer:
            hasher.update(buffer)
            buffer = file.read(buffer_size)
    return hasher.hexdigest()


def compare_file_hashes(*filenames, method='sha256') -> bool:
    """Compare file hashes.

    Computes file hashes for provided files.
    Returns True if all hashes are equal, False otherwise.

    Args:
        *filenames: Filenames.
        method: Hashing method.

    Returns:
        True/False.
    """
    assert len(filenames)
    if len(filenames) == 1:
        return True
    reference_hash = hash_file(filenames[0], method=method)
    return all(hash_file(f, method=method) == reference_hash for f in filenames[1:])


def import_file(filename):
    """Import Python file.

    Args:
        filename: Python filename.

    Returns:
        Loaded module.
    """
    from os.path import abspath, basename
    import importlib

    filename = abspath(filename)
    name = splitext(basename(filename))[0]
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_imagej_rois(filename, *keys):
    """Load ImageJ ROIs.

    Args:
        filename: Filename.
        *keys: Specific keys to load from the ROIs.

    Returns:
        (boxes, contours) if no keys provided. (boxes, contours, meta) if keys provided.
    """
    try:
        import roifile as rf
    except ModuleNotFoundError as e:
        print('Please install the `roifile` package to use this function: "pip install roifile" '
              '(https://pypi.org/project/roifile/)')
        raise e

    rois = rf.ImagejRoi.fromfile(filename)
    boxes = []
    contours = []
    meta = {k: [] for k in keys}
    for r in rois:
        y0, y1 = r.top, r.bottom
        x0, x1 = r.left, r.right
        contour = r.integer_coordinates + [x0, y0]
        contours.append(contour)
        boxes.append([x0, y0, x1, y1])

    res = np.array(boxes), contours
    if len(meta):
        return res + (meta,)
    return res


def glob_h5_split(pathname, ext='-r.h5', **kwargs):
    """Glob for split HDF5 files.

    This is a helper function for finding split h5 files via glob.
    Filenames are searched with appendix and returned without appendix, as this is what `h5py` expects.

    Args:
        pathname: Glob pattern.
        ext: Split h5 appendix.
        **kwargs: Keyword arguments for glob.

    Returns:
        Modified glob results. Each filename has its appendix (`ext`) removed.
    """
    return [f[:-len(ext)] for f in glob(pathname=pathname if pathname.endswith(ext) else pathname + ext, **kwargs)]


def say_goodbye():
    a = ('All done,Task complete,Script finalized,Operation successful,All set,Execution concluded,Work finished,Proces'
         's ended,All clear,Routine complete,Job done,Sequence finalized,Task accomplished,Chores completed,Duty conclu'
         'ded,Activity finished,Assignment completed,Undertaking achieved,Procedure closed,Milestone reached')
    b = ('have a magnificent day,enjoy your splendid day,hope your day is extraordinary,have a delightful day,may your '
         'day be as bright as your smile,wishing you a day full of joy,have an awesome day ahead,enjoy your amazing day'
         ',hope your day is as fantastic as you,have a superb day,may your day be filled with happiness,wishing you a d'
         'ay of peace and joy,have a wonderful adventure today,may your day be as productive as you are,have a day as w'
         'onderful as your achievements')
    print(', '.join((np.random.choice(i.split(',')) for i in (a, b))) + '!')


def parse_url_params(url, sep='?', param_sep=';'):
    """Parse url params.

    Examples:
        ```Python
        cd.parse_url_params('file.py?a=42;b=43')
        ('file.py', {'a': 42, 'b': 43))
        ```

    Args:
        url: URL.
        sep: Separator between URL and params.
        param_sep: Separator between params.

    Returns:
        Tuple of URL and parameter dict.
    """
    ar = url.split(sep)
    if len(ar) > 1:
        url = sep.join(url.split(sep)[:-1])
    ar = dict([(tuple(i.split('=')) if '=' in i else (i, True)) for i in ar[-1].split(param_sep)]) if len(
        ar) > 1 else None
    return url, ar


def get_installed_packages(template='{name}=={version}'):
    """Get installed packages.

    Returns a list of all installed packages.

    Returns:
        List of installed packages.
    """
    import importlib.metadata

    return [package for package in sorted([
        template.format(name=dist.metadata['Name'], version=dist.version) for dist in importlib.metadata.distributions()
    ])]


def save_requirements(filename, **kwargs):
    """Save requirements.

    Writes all installed packages with specified versions to `filename`.

    Args:
        filename: Filename.
        **kwargs: Additional keyword arguments for `cd.print_to_file`.
    """
    print_to_file('\n'.join(get_installed_packages()), filename=filename, **kwargs)


def update_model_hparams_(obj, resolve=True, **kwargs):
    assert hasattr(obj, '_set_hparams')
    assert hasattr(obj, '_hparams_initial')
    assert hasattr(obj, '_hparams')
    changes = {}
    for key, value in kwargs.items():
        if isinstance(value, nn.Module):
            if resolve:
                value = model2dict(value)
        changes[key] = value

    if len(changes):
        # Override hparams
        obj._set_hparams(changes)
        obj._hparams_initial = copy.deepcopy(obj._hparams)


def resolve_model(value, src=None, map_location='cpu', check_hash=None, **kwargs) -> 'nn.Module':
    """Resolve model.

    Args:
        value: Model description. Either `nn.Module`, `str` (URL (leading http), filename, hosted model name,
            class name, or dict).
        src: Class name source. (cd.models is default).
        map_location: Map location for loaded models. (cpu is default).
        check_hash: Whether to check hash. Only relevant for downloads. Hash must be postfix in filename.
        **kwargs: Keyword arguments for respective handler.

    Returns:
        Module.
    """
    if isinstance(value, nn.Module):
        return value
    elif isinstance(value, str):
        if src is None:
            from .. import models as src

        # Model name
        if value in dir(src):
            item = getattr(src, value)
            assert issubclass(item, nn.Module), (f'Model variable must describe a torch.nn.Module, '
                                                 f'but found {value, item}')
            return item(**kwargs)

        # URL or hosted model
        elif value.startswith('http') or value.startswith('cd://') or (not isfile(value) and not splitext(value)[1]):
            if check_hash is None:
                check_hash = value.startswith('cd://') or (not isfile(value) and not splitext(value)[1])
            return fetch_model(value, map_location=map_location, check_hash=check_hash, **kwargs)

        # Filename
        elif isfile(value):
            return load_model(value, map_location=map_location, **kwargs)
    elif isinstance(value, dict):
        return dict2model(value, **kwargs)
    else:
        raise ValueError(f'Could not handle type of `value`: {type(value)}')


def resolve_pretrained(pretrained, state_dict_mapper=None, **kwargs):
    if isinstance(pretrained, str):
        if isfile(pretrained):
            state_dict = torch.load(pretrained)
        else:
            state_dict = load_state_dict_from_url(pretrained)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if '.pytorch.org' in pretrained:
            if state_dict_mapper is not None:
                state_dict = state_dict_mapper(state_dict=state_dict, **kwargs)
    else:
        raise ValueError('There is no default set of weights for this model. '
                         'Please specify a URL or filename using the `pretrained` argument.')
    return state_dict


def is_package_installed(name) -> bool:
    """Is package installed.

    Checks if package called `name` is installed by attempting to retrieve its version via `importlib.metadata.version`.

    Args:
        name: Package name.

    Returns:
        Bool.
    """
    import importlib.metadata
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def has_argument(fn, *args, mode='any'):
    sig = signature(fn)
    gen = ((a in sig.parameters) for a in args)
    if mode == 'any':
        return any(gen)
    elif mode == 'all':
        return all(gen)
    raise ValueError(f'Unknown mode: {mode}')


def dict_to_json_string(input_dict):
    serializable_dict = {}
    for k, v in input_dict.items():
        try:
            json.dumps(v)
            serializable_dict[k] = v
        except TypeError:
            pass  # skip
    return json.dumps(serializable_dict)
