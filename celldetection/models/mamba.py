try:
    from mamba_ssm import Mamba
except ModuleNotFoundError:
    Mamba = False


from torch.cuda.amp import autocast
import torch
import torch.nn as nn

__all__ = ['MambaLayer']


class MambaLayer(nn.Module):
    def __init__(self, channels, d_state=16, kernel_size=4, expand=2, nd=None):
        """Mamba Layer.

        References:
            - https://arxiv.org/abs/2401.04722
            - https://github.com/bowang-lab/U-Mamba
            - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

        Examples:
            ```python
            model = cd.models.ResNet50(3, secondary_block=MambaLayer).cuda()
            ```

        Note:
            Requires model to run on Cuda!

        Args:
            channels: Input channels. Equal to output channels.
            d_state: SSM state expansion factor.
            kernel_size: Kernel size of 1D convolution.
            expand: Block expansion factor. (inner_channels=expand*channels)
            nd: Spatial dimension. Just here for compatibility.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        assert Mamba, ('The Python package `mamba_ssm` must be installed to use this package. '
                       'Since the installation may not work on every system it is not a mandatory '
                       'requirement for `celldetection`. '
                       'Please follow the installation instructions here: https://github.com/state-spaces/mamba.')
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=kernel_size, expand=expand)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        n, c, *spatial = x.shape
        x_ = x.flatten(2).transpose(1, 2)  # Tensor[n, h*w(*d), c]
        x_ = self.norm(x_)
        x_ = self.mamba(x_)
        x = x_.transpose(1, 2).reshape(n, c, *spatial)  # Tensor[n, c, h, w(, d)]
        return x
