import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ['reduce_loss', 'log_margin_loss', 'margin_loss']


def reduce_loss(x: Tensor, reduction: str):
    """Reduce loss.

    Reduces Tensor according to ``reduction``.

    Args:
        x: Input.
        reduction: Reduction method. Must be a symbol of ``torch``.

    Returns:
        Reduced Tensor.
    """
    if reduction == 'none':
        return x
    fn = getattr(torch, reduction, None)
    if fn is None:
        raise ValueError
    return fn(x)


def log_margin_loss(inputs: Tensor, targets: Tensor, m_pos=.9, m_neg=None, exponent=1, reduction='mean', eps=1e-6):
    if m_neg is None:
        m_neg = 1 - m_pos

    pos = torch.pow(F.relu_(torch.log(m_pos / (inputs + eps))), exponent)
    neg = torch.pow(F.relu_(torch.log((1 - m_neg) / (1 - inputs + eps))), exponent)
    loss = targets * pos + (1 - targets) * neg
    return reduce_loss(loss, reduction)


def margin_loss(inputs: Tensor, targets: Tensor, m_pos=.9, m_neg=None, exponent=2, reduction='mean'):
    if m_neg is None:
        m_neg = 1 - m_pos

    pos = torch.pow(F.relu_(m_pos - inputs), exponent)
    neg = torch.pow(F.relu_(inputs - m_neg), exponent)
    loss = targets * pos + (1 - targets) * neg
    return reduce_loss(loss, reduction)
