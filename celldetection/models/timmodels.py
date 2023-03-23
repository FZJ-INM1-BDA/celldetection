import timm
import torch
from torch import nn
from collections import OrderedDict
from .smp import ExternBase
from ..util.util import get_device
import re
from typing import List, Tuple
from warnings import warn

__all__ = ['TimmEncoder']


def get_channels_strides(model, example_input=None, verbose=True) -> Tuple[List[int], List[int]]:
    """Get channels and strides.

    A helper function to probe the channels and strides of the output of ``model``.

    Examples:
        >>> m = cd.models.TimmEncoder(
        ...     'cspdarknet53',
        ...     return_layers=['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3', 'stages.4'],
        ... )
        >>> out_channels, out_strides = get_channels_strides(m, torch.rand(1, 3, 512, 512), verbose=True)
        layer 0 output shape: torch.Size([1, 32, 512, 512]) , stride: 1
        layer 1 output shape: torch.Size([1, 64, 256, 256]) , stride: 2
        layer 2 output shape: torch.Size([1, 128, 128, 128]) , stride: 4
        layer 3 output shape: torch.Size([1, 256, 64, 64]) , stride: 8
        layer 4 output shape: torch.Size([1, 512, 32, 32]) , stride: 16
        layer 5 output shape: torch.Size([1, 1024, 16, 16]) , stride: 32
        >>> out_strides
        [1, 2, 4, 8, 16, 32]
        >>> out_channels
        [32, 64, 128, 256, 512, 1024]

    Args:
        model: Model.
        example_input: Example input.
        verbose: Whether to print output names, shapes and strides.

    Returns:
        Strides as List[int].
    """
    if example_input is None:
        reference = 512
        example_input = torch.rand(1, 3, *(reference,) * 2).to(get_device(model))
    else:
        reference = example_input.shape[-1]
    training = model.training
    model.eval()
    with torch.no_grad():
        out = model(example_input)
    model.train(training)

    strides = []
    channels = []
    if isinstance(out, dict):
        for k, v in out.items():
            strides.append(reference // v.shape[-1])
            channels.append(v.shape[1])
            if verbose:
                print('layer', k, 'output shape:', v.shape, ', stride:', strides[-1])
    elif isinstance(out, torch.Tensor):
        assert out.ndim == example_input.ndim, f'Module output has different number of dimensions: ' \
                                               f'{out.shape, example_input.shape}. ' \
                                               f'Consider updating ``return_layers`` in __init__ call.'
        strides.append(reference // out.shape[-1])
        channels.append(out.shape[1])
    else:
        raise ValueError('Cannot handle output.')
    return channels, strides


def get_names(model, *pattern) -> List[str]:
    """Get names.

    A helper function to discover module names for intermediate feature extraction.
    A list of module names can be passed to the constructor of this class (``return_layers``)
    to convert the timm model to a custom feature extractor.

    Examples:
        >>> m = cd.models.TimmEncoder('convnext_nano')
        >>> get_names(m, 'stem$', 'stages.[0-9]$')
        ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']

    Args:
        model: Model.
        *pattern: Name patterns.
    """
    return [n for n, _ in model.module.named_modules() if
            len(pattern) == 0 or any(list(bool(len(re.findall(p, n))) for p in pattern))]


class TimmEncoder(ExternBase):
    def __init__(
            self,
            model_name: str,
            in_channels: int = 3,
            return_layers: List[str] = None,
            out_channels: List[str] = None,
            out_strides: List[str] = None,
            pretrained: bool = False,
            pretrained_cfg=None,
            keep_names: bool = False,
            output_stride: int = None,
            depth: int = None,
            convert_lists: bool = True,
            **kwargs
    ):
        """Timm Encoder.

        A wrapper that provides compatibility with "PyTorch Image Models (timm)".

        Notes:
            It is possible to customize TimmEncoder`s return_layers to return specific layer outputs.
            In that case ``out_channels`` and ``out_strides`` must be provided.
            To find appropriate settings the helper functions ``get_names`` and ``get_channels_strides`` can be used.

        References:
            - timm GitHub: https://github.com/huggingface/pytorch-image-models
            - timm Documentation: https://huggingface.co/docs/timm/index

        Examples:
            >>> import timm, celldetection as cd
            >>> timm.list_models('*darknet*')  # discover models
            ['cs3darknet_focus_l',
             'cs3darknet_focus_m',
             'cs3darknet_focus_s',
             'cs3darknet_focus_x',
             'cs3darknet_l',
             'cs3darknet_m',
             'cs3darknet_s',
             'cs3darknet_x',
             'cs3sedarknet_l',
             'cs3sedarknet_x',
             'cs3sedarknet_xdw',
             'cspdarknet53',
             'darknet17',
             'darknet21',
             'darknet53',
             'darknetaa53',
             'sedarknet21']
             >>> encoder = cd.models.TimmEncoder('darknet21')
             >>> encoder.out_channels
             ... [32, 64, 128, 256, 512, 1024]
             >>> encoder.out_strides
             ... [1, 2, 4, 8, 16, 32]
             >>> output: Dict[str, Tensor] = encoder(torch.rand(1, 3, 512, 512))
             >>> for key, layer_output in output.items():
             ...    print(key, layer_output.shape)
             0 torch.Size([1, 32, 512, 512])
             1 torch.Size([1, 64, 256, 256])
             2 torch.Size([1, 128, 128, 128])
             3 torch.Size([1, 256, 64, 64])
             4 torch.Size([1, 512, 32, 32])
             5 torch.Size([1, 1024, 16, 16])

        Args:
            model_name: Name of model to instantiate.
            return_layers: List of layer names used for intermediate feature retrieval.
            out_channels: List of output channels per return layer.
            out_strides: List of output strides per return layer.
            pretrained: Whether to load pretrained ImageNet-1k weights.
            pretrained_cfg: External pretrained_cfg for model.
            keep_names: Whether to keep layer names for model output. If ``False``, names are replaced
                with enumeration for consistency.
            output_stride: Some models support different output stride (e.g. 16 instead of 32).
                This is achieved by using ``stride=1`` with dilation instead of downsampling with strides.
            depth: Custom encoder depth. If return_layers provided, this acts as number of return layers.
            convert_lists: Whether to return output lists to dictionaries for consistency.
            **kwargs: Keyword arguments for ``timm.create_model`` call.
        """
        super().__init__(model_name=model_name)
        assert depth is None or depth > 0
        depth_ = slice(None) if depth is None else slice(None, depth)
        if output_stride is not None:
            kwargs['output_stride'] = output_stride

        # Without custom return layers, use default layers from timm
        if return_layers is None:
            if depth is not None:
                kwargs['out_indices'] = tuple(range(depth))
            try:
                self.module: nn.Module = timm.create_model(
                    model_name=self.model_name,
                    in_chans=in_channels,
                    pretrained=pretrained,
                    pretrained_cfg=pretrained_cfg,
                    features_only=kwargs.get('features_only', True),
                    **kwargs
                )
            except RuntimeError as e:
                raise ValueError('This model does not support automatic feature extraction. '
                                 'To retrieve features nonetheless, use `return_layers` to specify a list of names of '
                                 'layers whose output is to be returned.\n\n' + str(e))
            self.pretrained_cfg = self.module.__dict__.get('pretrained_cfg', {})
            self.return_layers = [i['module'] for i in self.module.feature_info[depth_]]
            if out_channels is None:
                out_channels = [i['num_chs'] for i in self.module.feature_info[depth_]]
            if out_strides is None:
                out_strides = [i['reduction'] for i in self.module.feature_info[depth_]]

        # Custom return layers
        else:
            from torchvision.models.feature_extraction import create_feature_extractor
            self.module: nn.Module = timm.create_model(
                model_name=self.model_name, pretrained=pretrained,
                in_chans=in_channels, pretrained_cfg=pretrained_cfg, **kwargs
            )
            self.pretrained_cfg = self.module.__dict__.get('pretrained_cfg', {})
            if out_channels is None:
                warn('For custom return_layers setting out_channels must be specified.')
            if out_strides is None:
                warn('For custom return_layers setting out_strides must be specified.')
            self.return_layers = return_layers[depth_]
            self.module = create_feature_extractor(self.module, self.return_layers)
        self.out_channels = out_channels[depth_] if out_channels is not None else None
        self.out_strides = out_strides[depth_] if out_strides is not None else None
        self.keep_names = keep_names
        self.convert_lists = convert_lists

    def forward(self, x):
        out = self.module(x)  # assuming ordered
        if self.convert_lists and isinstance(out, list):
            out = OrderedDict([(str(k), v) for k, v in enumerate(out)])
        if isinstance(out, dict):
            if not self.keep_names:
                out = OrderedDict([(str(k), v) for k, (_, v) in enumerate(out.items())])
        return out
