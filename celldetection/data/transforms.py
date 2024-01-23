from abc import abstractmethod
from pytorch_lightning.core.mixins import HyperparametersMixin
import numpy as np
import cv2

from .misc import random_crop, normalize_percentile

__all__ = ['Transforms', 'BasicTransforms']


class Transforms(HyperparametersMixin):
    def __init__(self, **kwargs):
        """Transforms.

        Defines all transforms for all stages.
        """
        super().__init__()
        self.save_hyperparameters()

    @abstractmethod
    def static(self, **kwargs):
        # All kwargs from __init__ file available via `self.hparams.setting_name`.
        return kwargs

    def fit(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def validate(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def test(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def predict(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def __call__(self, stage='fit', **kwargs):
        assert stage in ('fit', 'validate', 'predict', 'test')
        return getattr(self, stage)(**kwargs)


class BasicTransforms(Transforms):
    def static(self, crop=False, **kwargs):
        image = kwargs['image']
        labels = kwargs['labels']
        if crop and self.hparams.crop_size:
            image, labels = random_crop((image, labels), size=(self.hparams.crop_size,) * 2)

        if image.dtype != np.uint8:
            image = normalize_percentile(image)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        kwargs['image'] = image / 255
        kwargs['labels'] = labels
        return kwargs

    def fit(self, **kwargs):
        kwargs = self.static(crop=True, **kwargs)
        return kwargs

    def validate(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def test(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs

    def predict(self, **kwargs):
        kwargs = self.static(**kwargs)
        return kwargs
