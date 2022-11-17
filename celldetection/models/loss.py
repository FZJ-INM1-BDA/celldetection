from torch.nn.modules.loss import _Loss
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
from ..ops.loss import iou_loss

__all__ = ['SigmoidFocalLoss', 'IoULoss']


class _FocalLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', alpha=.5, gamma=2) -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.gamma = gamma


class SigmoidFocalLoss(_FocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)


class IoULoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, generalized=True, method='linear', min_size=None, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.generalized = generalized
        self.method = method
        self.min_size = min_size

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return iou_loss(input, target, self.reduction, generalized=self.generalized, method=self.method,
                        min_size=self.min_size)

    def extra_repr(self) -> str:
        return f"generalized={self.generalized}, method='{self.method}'"