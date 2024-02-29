import segmentation_models_pytorch as smp
from torch import nn
from collections import OrderedDict
from typing import List, Callable
from pytorch_lightning.core.mixins import HyperparametersMixin

__all__ = ['SmpEncoder']


class ExternBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.check_model_name(model_name)
        assert not model_name.startswith('_')
        self.model_name = model_name

    @staticmethod
    def check_model_name(model_name: str, model_list_callback: Callable[[], List[str]] = None):
        if model_name is None:
            msg = 'Please specify `model_name`.'
            if model_list_callback is not None:
                msg += '\n  - '.join([' These are all available models, some of them may not be supported:'
                                      ] + model_list_callback())
            raise ValueError(msg)

    def _get_name(self):
        return self.model_name.title()


class SmpEncoder(ExternBase, HyperparametersMixin):
    def __init__(self, model_name: str, in_channels: int = 3, depth: int = 5, pretrained=False,
                 output_stride: int = 32, **kwargs):
        """Smp Encoder.

        A wrapper that provides compatibility with "segmentation_models_pytorch (smp)".

        References:
            - smp GitHub: https://github.com/qubvel/segmentation_models.pytorch
            - Encoders: https://smp.readthedocs.io/en/latest/encoders.html
            - Documentation: https://smp.readthedocs.io/en/latest/

        Args:
            model_name: Encoder name. Find available encoders here: https://smp.readthedocs.io/en/latest/encoders.html
            in_channels: Input channels.
            depth: Encoder dpeth.
            pretrained: Encoder weights. Find available weights here: https://smp.readthedocs.io/en/latest/encoders.html
            output_stride: Output stride.
            **kwargs:
        """
        self.check_model_name(model_name, smp.encoders.get_encoder_names)
        super().__init__(model_name)
        self.save_hyperparameters()

        # Map pretrained for consistency
        if pretrained is True:
            pretrained = 'imagenet'  # best guess
        elif pretrained is False:
            pretrained = None
        if 'weights' in kwargs:
            pretrained = kwargs.pop('weights')
        self.module = smp.encoders.get_encoder(self.model_name, in_channels=in_channels, depth=depth,
                                               weights=pretrained, output_stride=output_stride, **kwargs)
        if pretrained:
            self.pretrained_cfg = smp.encoders.get_preprocessing_params(self.model_name, pretrained)
        self._skips = 0
        if self.module.out_channels[0] == in_channels:
            self.out_channels = self.module.out_channels[1:]  # skip input image
            self._skips += 1
        while self.out_channels[0] <= 0:
            self.out_channels = self.out_channels[1:]  # skip dummies
            self._skips += 1
        self.out_strides = tuple(
            [2 ** i for i in range(self._skips, len(self.module.out_channels))])  # derive strides from rank

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        assert isinstance(out, list), type(out)
        out = out[self._skips:]
        out = OrderedDict([(str(k), v) for k, v in zip(range(len(out)), out)])
        return out
