from .util import dict_hash, Dict, tweak_module_, lookup_nn, print_to_file
from ..optim import lr_scheduler
import json
from itertools import product
from collections import OrderedDict
from torch.nn import Module
from torch import optim
import inspect
from typing import Callable, Union
import albumentations as A
from os.path import splitext
import yaml

__all__ = ['Config', 'Schedule', 'conf2optimizer', 'conf2scheduler', 'conf2augmentation', 'conf2tweaks_', 'conf2call']


def conf2call(settings: Union[dict, str], origin, **kwargs):
    """Config to call.

    Examples:
        >>> import celldetection as cd
        >>> model = cd.conf2call('ResNet18', cd.models, in_channels=1)
        >>> model = cd.conf2call({'ResNet18': dict(in_channels=1)}, cd.models)

    Args:
        settings: Name or dictionary as ``{name: kwargs}``. Name must be the symbol's name that is to be retrieved
            from ``origin``.
        origin: Origin.
        **kwargs: Additional keyword arguments for the call of retrieved symbol.

    Returns:
        Return value of the call of retrieved symbol.
    """
    assert len(settings) == 1 or isinstance(settings, str)
    if not isinstance(origin, (tuple, list)):
        origin = origin,
    if isinstance(settings, str):
        key = settings
        kw = {}
    else:
        key = next(iter(settings.keys()))
        kw = next(iter(settings.values()))
    try:
        fn = next(iter(getattr(o, key) for o in origin if hasattr(o, key)))
    except StopIteration:
        raise ValueError(f'No such function: {key} in {origin}')
    return fn(**kw, **kwargs)


def conf2optimizer(settings: dict, params):
    """Config to optimizer.

    Examples:
        >>> import celldetection as cd
        >>> module = nn.Conv2d(1, 2, 3)
        >>> optimizer = cd.conf2optimizer({'Adam': dict(lr=.0002, betas=(0.5, 0.999))}, module.parameters())
        ... optimizer
        Adam (
            Parameter Group 0
                amsgrad: False
                betas: (0.5, 0.999)
                eps: 1e-08
                lr: 0.0002
                weight_decay: 0
        )

    Args:
        settings:
        params:

    Returns:

    """
    return conf2call(settings, optim, params=params)


def conf2scheduler(settings: dict, optimizer, origins=None):
    if origins is None:
        origins = (lr_scheduler, optim.lr_scheduler)
    return conf2call(settings, origins, optimizer=optimizer)


def conf2augmentation(settings: dict) -> A.Compose:
    """Config to augmentation.

    Maps settings to composed augmentation workflow using ``albumentations``.

    Examples:
        >>> import celldetection as cd
        >>> cd.conf2augmentation({
        ...     'RandomRotate90': dict(p=.5),
        ...     'Transpose': dict(p=.5),
        ... })
        Compose([
          RandomRotate90(always_apply=False, p=0.5),
          Transpose(always_apply=False, p=0.5),
        ], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})

    Args:
        settings: Settings dictionary as ``{name: kwargs}``.

    Returns:
        ``A.Compose`` object.
    """
    return A.Compose([getattr(A, k)(**v) for k, v in settings.items()])


def conf2tweaks_(settings: dict, module: Module):
    """Config to tweaks.

    Apply tweaks to module.

    Notes:
        - If module does not contain specified objects, nothing happens.

    Examples:
        >>> import celldetection as cd, torch.nn as nn
        >>> model = cd.models.ResNet18(in_channels=3)
        >>> cd.conf2tweaks_({nn.BatchNorm2d: dict(momentum=0.05)}, model)  # sets momentum to 0.05
        >>> cd.conf2tweaks_({'BatchNorm2d': dict(momentum=0.42)}, model)  # sets momentum to 0.42
        >>> cd.conf2tweaks_({'LeakyReLU': dict(negative_slope=0.2)}, model)  # sets negative_slope to 0.2

    Args:
        settings: Settings dictionary as ``{name: kwargs}``.
        module: Module that is to be tweaked.

    """
    for key, kwargs in settings.items():
        tweak_module_(module, lookup_nn(key, call=False), **kwargs)


class Config(Dict):
    def __init__(self, **kwargs):
        """Config.

        Just a ``dict`` with benefits.

        ``Config`` objects treat values as attributes, print nicely, and can be saved and loaded to/from json files.
        The ``hash`` method also offers a unique and compact string representation of the ``Config`` content.

        Examples:
            >>> import celldetection as cd, torch.nn as nn
            >>> conf = cd.Config(optimizer={'Adam': dict(lr=.001)}, epochs=100)
            >>> conf
            Config(
              (optimizer): {'Adam': {'lr': 0.001}}
              (epochs): 100
            )
            >>> conf.to_json('config.json')
            >>> conf.hash()
            'cf647b987ca37eb954d8bd01df01809e'
            >>> conf.epochs = 200
            ... conf.epochs
            200
            >>> module = nn.Conv2d(1, 2, 3)
            >>> optimizer = cd.conf2optimizer(conf.optimizer, module.parameters())
            ... optimizer
            Adam (
                Parameter Group 0
                    amsgrad: False
                    betas: (0.9, 0.999)
                    eps: 1e-08
                    lr: 0.001
                    weight_decay: 0
            )

        Args:
            **kwargs: Items.
        """
        super().__init__(**kwargs)

    def hash(self) -> str:
        return dict_hash(self.to_dict())

    @staticmethod
    def from_json(filename):
        c = Config()
        c.load(filename, backend='json')
        return c

    @staticmethod
    def from_yaml(filename):
        c = Config()
        c.load(filename, backend='yaml')
        return c

    @staticmethod
    def from_file(filename):
        c = Config()
        c.load(filename)
        return c

    @staticmethod
    def from_files(filenames, reverse=True):
        """From files.

        Args:
            filenames: Filenames
            reverse: Whether to reverse filenames. True by default. If reversed, the leftmost elements
                are dominant.

        Returns:
            Config.
        """
        if isinstance(filenames, str):
            return Config.from_file(filenames)
        if reverse:
            filenames = filenames[::-1]
        c = Config.from_file(filenames[0])
        for f in filenames[1:]:
            c.update(Config.from_file(f))
        return c

    def load(self, filename, backend=None):
        ext = splitext(filename)[1]
        if backend == 'yaml' or ext in ('.yml', '.yaml'):
            with open(filename, 'r') as fp:
                config = yaml.safe_load(fp)
        else:
            with open(filename, 'r') as fp:
                config = json.load(fp)
        if config is not None:
            self.update(config)

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in dict(self).items() if
                not k.startswith('_')}

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.to_dict(), fp)

    def to_yaml(self, filename):  # cannot preserve types
        with open(filename, 'w') as fp:
            yaml.safe_dump(self.to_dict(), fp)

    def to_txt(self, filename, mode='w', **kwargs):
        print_to_file(self, filename=filename, mode=mode, **kwargs)

    def __str__(self):
        return repr(self)

    def _get_name(self):
        return 'Config'

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        if '_modules' in self:
            return str(self.to_dict())
        self._modules = self.to_dict()
        r = Module.__repr__(self)
        del self._modules
        return r

    def args(self, fn: Callable):
        """

        Examples:
            >>> conf = cd.Config(a=1, b=2, c=42)
            >>> def f(a, b):
            ...     return a + b
            >>> f(*conf.args(f))
            3

        Args:
            fn:

        Returns:

        """
        r = []
        for k in inspect.signature(fn).parameters.keys():
            if k == 'args' or k == 'kwargs':
                break
            r.append(self[k])
        return r

    def kwargs(self, fn: Callable):
        """

        Examples:
            >>> conf = cd.Config(a=1, b=2, c=42)
            >>> def f(a, b):
            ...     return a + b
            >>> f(**conf.kwargs(f))
            3

        Args:
            fn:

        Returns:

        """
        r = dict()
        for k in inspect.signature(fn).parameters.keys():
            if k == 'args' or k == 'kwargs':
                continue
            v = self.get(k, None)
            if v is not None:
                r[k] = v
        return r

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, d: dict):
        self.update(d)


class Schedule:
    def __init__(self, **kwargs):
        """Schedule.

        Provides an easy interface to the cross product of different configurations.

        Examples:
            >>> s = cd.Schedule(
            ...     lr=(0.001, 0.0005),
            ...     net=('resnet34', 'resnet50'),
            ...     epochs=100
            ... )
            ... len(s)
            4
            >>> s[:]
            [Config(
              (epochs): 100
              (lr): 0.001
              (net): 'resnet34'
            ), Config(
              (epochs): 100
              (lr): 0.001
              (net): 'resnet50'
            ), Config(
              (epochs): 100
              (lr): 0.0005
              (net): 'resnet34'
            ), Config(
              (epochs): 100
              (lr): 0.0005
              (net): 'resnet50'
            )]
            >>> for config in s:
            ...     print(config.lr, config.net, config.epoch)
            0.001 resnet34 100
            0.001 resnet50 100
            0.0005 resnet34 100
            0.0005 resnet50 100

        Args:
            **kwargs: Configurations. Possible item layouts:
                ``<name>: <static setting>``,
                ``<name>: (<option1>, ..., <optionN>)``,
                ``<name>: [<option1>, ..., <optionN>]``,
                ``<name>: {<option1>, ..., <optionN>}``.
        """
        self.values = OrderedDict({})
        self.conditions = []
        self.conditioned_values = []
        self.add(kwargs)
        self._iter_conf = None
        self._iter_i = None

    def get_multiples(self, num=2):
        return {k: v for k, v in self.values.items() if (isinstance(v, (list, tuple, set)) and len(v) >= num)}

    def add(self, d: dict = None, conditions: dict = None, **kwargs):
        """Add setting to schedule.

        Examples:
            >>> schedule = cd.Schedule(model=('resnet18', 'resnet50'), batch_size=8)
            ... schedule.add(batch_size=(16, 32), conditions={'model': 'resnet18'})
            ... schedule[:]
            [Config(
               (batch_size): 16,
               (model): resnet18,
             ),
             Config(
               (batch_size): 32,
               (model): resnet18,
             ),
             Config(
               (batch_size): 8,
               (model): resnet50,
             )]

            >>> schedule = cd.Schedule(model=('resnet18', 'resnet50'))
            ... schedule.add(batch_size=(16, 32), conditions={'model': 'resnet18'})
            ... schedule[:]
            [Config(
               (model): resnet18,
               (batch_size): 16,
             ),
             Config(
               (model): resnet18,
               (batch_size): 32,
             ),
             Config(
               (model): resnet50,
             )]

            >>> schedule = cd.Schedule(model=('resnet18', 'resnet50'), batch_size=(64, 128, 256))
            ... schedule.add(batch_size=(16, 32), conditions={'model': 'resnet50'})
            ... schedule[:]
            [Config(
               (batch_size): 64
               (model): 'resnet18'
             ),
             Config(
               (batch_size): 16
               (model): 'resnet50'
             ),
             Config(
               (batch_size): 32
               (model): 'resnet50'
             ),
             Config(
               (batch_size): 128
               (model): 'resnet18'
             ),
             Config(
               (batch_size): 256
               (model): 'resnet18'
             )]

        Args:
            d: Dictionary of settings.
            conditions: If set, added settings are only applied if conditions are met.
                Note: Conditioned settings replace/override existing settings if conditions are met.
            **kwargs: Configurations. Possible item layouts:
                <name>: <static setting>
                <name>: (<option1>, ..., <optionN>)
                <name>: [<option1>, ..., <optionN>]
                <name>: {<option1>, ..., <optionN>}

        """
        if d is not None:
            if isinstance(d, Schedule):
                d = d.to_dict()
            else:
                assert isinstance(d, dict)
            d.update(kwargs)
            kwargs = d
        if conditions is None:
            dst = self.values
        else:
            self.conditions.append(OrderedDict(conditions))
            dst = OrderedDict()
            self.conditioned_values.append(dst)
        for key, val in kwargs.items():
            if not isinstance(val, (tuple, list, set)):
                val = (val,)
            dst[key] = val

    @staticmethod
    def _product(v):
        keys = list(v.keys())
        keys.sort()
        vals = list(product(*[v[k] for k in keys]))
        return [{k: v for k, v in zip(keys, va)} for va in vals]

    @property
    def product(self):
        initials = finals = self._product(self.values)
        for conditions, conditioned_values in zip(self.conditions, self.conditioned_values):
            finals = []
            for i in initials:
                if all(((i[ck] in conditions[ck]) if isinstance(conditions[ck], tuple) else (conditions[ck] == i[ck])
                        for ck in conditions.keys())):
                    extra = self._product(conditioned_values)
                    for j in extra:
                        extra_i = dict(i)
                        extra_i.update(j)
                        finals.append(extra_i)
                else:
                    finals.append(i)
            initials = finals
        return finals

    @property
    def configs(self):
        return list({c.hash(): c for c in [Config(**p) for p in self.product]}.values())

    def __str__(self):
        return Module.__repr__(Dict(_get_name=lambda: 'Schedule', extra_repr=lambda: '', _modules=dict(self.values)))

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.configs[item]

    def __len__(self):
        return len(self.configs)

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.values, fp)

    def to_yaml(self, filename):  # cannot preserve types
        with open(filename, 'w') as fp:
            yaml.safe_dump(self.to_dict(), fp)

    @staticmethod
    def from_json(filename):
        c = Schedule()
        c.load(filename, backend='json')
        return c

    @staticmethod
    def from_yaml(filename):
        c = Schedule()
        c.load(filename, backend='yaml')
        return c

    @staticmethod
    def from_file(filename):
        c = Schedule()
        c.load(filename)
        return c

    @staticmethod
    def from_files(filenames, reverse=True):
        """From files.

        Args:
            filenames: Filenames
            reverse: Whether to reverse filenames. True by default. If reversed, the leftmost elements
                are dominant.

        Returns:
            Schedule.
        """
        if isinstance(filenames, str):
            return Schedule.from_file(filenames)
        if reverse:
            filenames = filenames[::-1]
        c = Schedule.from_file(filenames[0])
        for f in filenames[1:]:
            c.add(Schedule.from_file(f))
        return c

    def load(self, filename, backend=None):
        ext = splitext(filename)[1]
        if backend == 'yaml' or ext in ('.yml', '.yaml'):
            with open(filename, 'r') as fp:
                self.values = yaml.safe_load(fp)
        else:
            with open(filename, 'r') as fp:
                self.values = json.load(fp)

    def to_dict(self):
        return dict(self.values)

    def to_dict_list(self):
        return [c.to_dict() for c in self]

    def __eq__(self, other):
        assert isinstance(other, Schedule)
        return self.values.__eq__(other.to_dict())

    def __iter__(self):
        self._iter_conf = self.configs
        self._iter_i = 0
        return self

    def __next__(self):
        if self._iter_i < len(self._iter_conf):
            res = self._iter_conf[self._iter_i]
            self._iter_i += 1
            return res
        else:
            raise StopIteration
