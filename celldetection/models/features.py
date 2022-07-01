import torch
import torch.nn as nn
import numpy as np
from ..models.filters import EdgeFilter2d, GaussianFilter2d
from ..ops.features import texture_filter

__all__ = ['MultiscaleBasicFeatures']


class MultiscaleBasicFeatures(nn.Module):
    def __init__(
            self,
            in_channels,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=.5,
            sigma_max=16,
            num_sigma=None,
            padding_mode='zeros',
            trainable=False
    ):
        """Multiscale Basic Features.

        Notes:
            - Multiscale aspect is implemented using Gaussian blur over different scales.

        References:
            - https://github.com/scikit-image/scikit-image/blob/91f26e9c0a00522137ddb638fa8362ec01d2d7b2/skimage/feature/_basic_features.py#L101

        Args:
            in_channels: Number of input channels.
            intensity: Whether to include intensities to feature set.
            edges: Whether to include results of edge filters to feature set.
            texture: Whether to include eigenvalues of the Hessian matrix after Gaussian blurring to the feature set.
            sigma_min: Smallest sigma for the Gaussian blur kernel.
            sigma_max: Largest sigma for the Gaussian blur kernel.
            num_sigma: Number of different scales.
            padding_mode: One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``.
            trainable: Whether the kernels should be trainable.
        """
        super().__init__()
        self.intensity = intensity
        self.edges = edges
        self.texture = texture
        self.num_sigma = num_sigma or int(np.log2(sigma_max) - np.log2(sigma_min) + 1)
        self.sigmas = np.logspace(np.log2(sigma_min), np.log2(sigma_max), num=self.num_sigma, base=2, endpoint=True)
        self.gaussians = nn.ModuleList()
        for sigma in self.sigmas:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            g = GaussianFilter2d(in_channels, kernel_size, sigma=sigma, padding=kernel_size // 2,
                                 padding_mode=padding_mode, trainable=trainable)
            self.gaussians.add_module(f"sigma{round(sigma, 4)}".replace('.', '_'), g)
        if self.edges:
            self.edge_filter = EdgeFilter2d(in_channels, padding=1, padding_mode=padding_mode, magnitude=True,
                                            trainable=trainable)

    def forward(self, x):
        results = []
        for gaussian in self.gaussians:
            filtered = gaussian(x)
            if self.intensity:
                results.append(filtered)
            if self.edges:
                results.append(self.edge_filter(filtered))
            if self.texture:
                results.append(texture_filter(filtered))
        return torch.concat(results, 1)
