"""
PyTorch Hub models

Examples:
    import torch
    model = torch.hub.load('FZJ-INM1-BDA/celldetection', 'ginoro')
"""

import celldetection as cd
from torch.nn import Module
from torch import device as _device
from typing import Optional, Union


def ginoro(
        pretrained: bool = True,
        pretrained_strict=True,
        device: Optional[Union[_device, str, None]] = None,
        **kwargs
) -> 'Module':
    """
    Ginoro: CPN + UNet + ResNeXt101

    References:
        https://proceedings.mlr.press/v212/upschulte23a/upschulte23a.pdf

    Args:
        pretrained: Whether to load the `state_dict` of a pretrained model.
        pretrained_strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by
            the loaded module’s state_dict() function.
        device: Device to map the model to.
        kwargs: Keyword arguments. Allows to override any setting of the model's constructor.
    """
    return cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', pretrained=pretrained,
                          load_state_dict_kwargs=dict(check_hash=True), map_location=device,
                          pretrained_strict=pretrained_strict, **kwargs)
