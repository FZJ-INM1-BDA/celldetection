from .util import dict_hash, Dict
import json
from torch.nn import Module

__all__ = ['Config']


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
