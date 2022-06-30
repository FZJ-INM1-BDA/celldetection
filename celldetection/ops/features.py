import torch
from itertools import combinations_with_replacement

__all__ = ['texture_filter']


def symmetric_image(images: list):
    nd = images[0].ndim - 2
    # Array[..., nd, nd] required e.g. for eigvalsh
    sym = torch.zeros(images[0].shape + (nd,) * 2, dtype=images[0].dtype, device=images[0].device)
    for idx, (row, col) in enumerate(combinations_with_replacement(range(nd), 2)):
        sym[..., row, col] = sym[..., col, row] = images[idx]
    return sym


def symmetric_compute_eigenvalues(images: list):
    # List[Tensor[..., n, n]]
    matrices = symmetric_image(images)
    eigs = torch.flip(torch.linalg.eigvalsh(matrices), (-1,))  # eigenvals in decending order
    eigs = torch.permute(eigs, (0, eigs.ndim - 1,) + tuple(range(1, eigs.ndim - 1)))
    return eigs


def texture_filter(gaussian_filtered, reshape=True):
    """Texture filter.

    References:
        - https://github.com/scikit-image/scikit-image/blob/0e28f6397a475db3f6755f03674c75ce02142bc3/skimage/feature/_basic_features.py#L10

    Args:
        gaussian_filtered: Tensor[n, c, h, w]. Typically a batch of gaussian filtered images.
        reshape: Whether to reshape result from Tensor[n, d, c, h, w] to Tensor[n, d * c, h, w] before returning it.

    Returns:
        Tensor[n, d, c, h, w] or Tensor[n, d * c, h, w].
    """
    f_gf = torch.gradient(gaussian_filtered, dim=list(range(2, gaussian_filtered.ndim)))
    axes = combinations_with_replacement(range(2, gaussian_filtered.ndim), 2)
    elems = [torch.gradient(f_gf[ax0 - 2], dim=ax1)[0] for ax0, ax1 in axes]
    eigvals = symmetric_compute_eigenvalues(elems)
    if reshape:
        n, d, c, h, w = eigvals.shape
        eigvals = eigvals.reshape(n, d * c, h, w)
    return eigvals
