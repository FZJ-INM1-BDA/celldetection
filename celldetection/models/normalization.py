import torch.nn as nn
from ..ops.normalization import pixel_norm

__all__ = ['PixelNorm']


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        """Pixel normalization.

        References:
            - https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf ("Local Repose Norm")
            - https://arxiv.org/pdf/1710.10196.pdf

        Args:
            dim: Dimension to normalize.
            eps: Epsilon.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return pixel_norm(x, dim=self.dim, eps=self.eps)
