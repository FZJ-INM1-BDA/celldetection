from .util import dict_hash, Dict, tweak_module_, lookup_nn
import json
from itertools import product
from collections import OrderedDict
from torch.nn import Module
from torch import optim

__all__ = ['Config', 'Schedule', 'conf2optimizer', 'conf2scheduler', 'conf2augmentation', 'conf2tweaks_']


def conf2optimizer(settings: dict, params):
    assert len(settings) == 1
    return getattr(optim, next(iter(settings.keys())))(params=params, **next(iter(settings.values())))


def conf2scheduler(settings: dict, optimizer):
    assert len(settings) == 1
    return getattr(optim.lr_scheduler, next(iter(settings.keys())))(optimizer=optimizer,
                                                                    **next(iter(settings.values())))


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
