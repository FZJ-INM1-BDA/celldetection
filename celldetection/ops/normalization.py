import torch

__all__ = ['pixel_norm']


def pixel_norm(x, dim=1, eps=1e-8):
    """Pixel normalization.

    References:
        - https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf ("Local Repose Norm")
        - https://arxiv.org/pdf/1710.10196.pdf

    Args:
        x: Input Tensor.
        dim: Dimension to normalize.
        eps: Epsilon.

    Returns:
        Normalized Tensor.
    """
    return x * torch.rsqrt(torch.mean(torch.square(x), dim=dim, keepdim=True) + eps)
