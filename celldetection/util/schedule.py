from .util import dict_hash, Dict, tweak_module_, lookup_nn
import json
from itertools import product
from collections import OrderedDict
from torch.nn import Module
from torch import optim
from typing import Callable, Union


__all__ = ['Config', 'conf2optimizer', 'conf2scheduler', 'conf2augmentation', 'conf2tweaks_']


def conf2call(settings: Union[dict, str], origin, **kwargs):
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
    return conf2call(settings, optim, params=params)


def conf2scheduler(settings: dict, optimizer):
    return conf2call(settings, optim.lr_scheduler, optimizer=optimizer)


def conf2augmentation(settings: dict):
    import albumentations as A
    return A.Compose([getattr(A, k)(**v) for k, v in settings.items()])


def conf2tweaks_(settings: dict, module: Module):
    for key, kwargs in settings.items():
        tweak_module_(module, lookup_nn(key, call=False), **kwargs)


class Config(Dict):
    def __init__(self, **kwargs):
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
