import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torchvision.ops.boxes as bx
from .boxes import pairwise_box_iou, pairwise_generalized_box_iou

__all__ = ['reduce_loss', 'log_margin_loss', 'margin_loss', 'r1_regularization', 'iou_loss']


def reduce_loss(x: Tensor, reduction: str, **kwargs):
    """Reduce loss.

    Reduces Tensor according to ``reduction``.

    Args:
        x: Input.
        reduction: Reduction method. Must be a symbol of ``torch``.
        **kwargs: Additional keyword arguments.

    Returns:
        Reduced Tensor.
    """
    if reduction == 'none':
        return x
    fn = getattr(torch, reduction, None)
    if fn is None:
        raise ValueError(f'Unknown reduction: {reduction}')
    return fn(x, **kwargs)


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


def r1_regularization(logits, inputs, gamma=1., reduction='sum'):
    r"""R1 regularization.

    A gradient penalty regularization.
    This regularization may for example be applied to a discriminator with real data:

    .. math::

        R_1(\psi) &:= \frac{\gamma}{2} \mathbb E_{ p_{\mathcal D}(x)} \left[\|\nabla D_\psi(x)\|^2\right]

    References:
        - https://arxiv.org/pdf/1801.04406.pdf (Eq. 9)
        - https://arxiv.org/pdf/1705.09367.pdf
        - https://arxiv.org/pdf/1711.09404.pdf

    Examples:
        >>> real.requires_grad_(True)
        ... real_logits = discriminator(real)
        ... loss_d_real = F.softplus(-real_logits)
        ... loss_d_r1 = r1_regularization(real_logits, real)
        ... loss_d_real = (loss_d_r1 + loss_d_real).mean()
        ... loss_d_real.backward()
        ... real.requires_grad_(False)

    Args:
        logits: Logits.
        inputs: Inputs.
        gamma: Gamma.
        reduction: How to reduce all non-batch dimensions. E.g. ``'sum'`` or ``'mean'``.

    Returns:
        Penalty Tensor[n].
    """
    grads = torch.autograd.grad(logits.sum(), inputs=inputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = reduce_loss(grads.square(), reduction, dim=list(range(1, grads.ndim)))
    return penalty * (gamma * .5)


def iou_loss(boxes, boxes_targets, reduction='mean', generalized=True, method='linear', min_size=None):
    if min_size is not None:  # eliminates invalid boxes
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets = (c[keep] for c in (boxes, boxes_targets))

    if generalized:
        iou = pairwise_generalized_box_iou(boxes, boxes_targets)  # Tensor[n]
    else:
        iou = pairwise_box_iou(boxes, boxes_targets)  # Tensor[n]

    if method == 'log':
        if generalized:
            iou = iou * .5 + .5
        loss = -torch.log(iou + 1e-8)
    elif method == 'linear':
        loss = 1 - iou
    else:
        raise ValueError

    loss = reduce_loss(loss, reduction=reduction)
    return loss


def box_npll_loss(uncertainty, boxes, boxes_targets, factor=10., sigmoid=False, epsilon=1e-8, reduction='mean',
                  min_size=None):
    """NPLL.

    References:
        https://arxiv.org/abs/2006.15607

    Args:
        uncertainty: Tensor[n, 4].
        boxes: Tensor[n, 4].
        boxes_targets: Tensor[n, 4].
        sigmoid: Whether to apply the ``sigmoid`` function to ``uncertainty``.
        factor: Uncertainty factor.
        epsilon: Epsilon.
        reduction: Loss reduction.
        min_size: Minimum box size. May be used to remove degenerate boxes.

    Returns:
        Loss.
    """
    if min_size is not None:  # eliminates invalid boxes
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets, uncertainty = (c[keep] for c in (boxes, boxes_targets, uncertainty))
    delta_sq = torch.square((torch.sigmoid(uncertainty) if sigmoid else uncertainty) * factor)
    a = torch.square(boxes - boxes_targets) / (2 * delta_sq + epsilon)
    b = 0.5 * torch.log(delta_sq + epsilon)
    iou = pairwise_box_iou(boxes, boxes_targets)  # Tensor[n]
    loss = iou * ((a + b).sum(dim=1) + 2 * np.log(2 * np.pi))
    loss = reduce_loss(loss, reduction=reduction)
    return loss
