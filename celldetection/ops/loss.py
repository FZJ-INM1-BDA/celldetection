import torch
import torch.nn.functional as F
from torch import Tensor
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
