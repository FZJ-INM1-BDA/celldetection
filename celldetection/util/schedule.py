from .util import dict_hash, Dict, tweak_module_, lookup_nn
import json
from itertools import product
from collections import OrderedDict
from torch.nn import Module
from torch import optim
from typing import Callable, Union
import albumentations as A


__all__ = ['Config', 'conf2optimizer', 'conf2scheduler', 'conf2augmentation', 'conf2tweaks_', 'conf2call']


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


def conf2scheduler(settings: dict, optimizer):
    return conf2call(settings, optim.lr_scheduler, optimizer=optimizer)


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
        c.load(filename)
        return c

    def load(self, filename):
        with open(filename, 'r') as fp:
            config = json.load(fp)
        self.update(config)

    def to_dict(self) -> dict:
        return {k: v for k, v in dict(self).items() if not k.startswith('_')}

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.to_dict(), fp)

    def __str__(self):
        return repr(self)

    def _get_name(self):
        return 'Config'

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        self._modules = self.to_dict()
        return Module.__repr__(self)

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, d: dict):
        self.update(d)
