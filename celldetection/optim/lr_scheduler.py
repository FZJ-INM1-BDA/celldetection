from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim import Optimizer
from typing import Union, Callable, List
import warnings

__all__ = ['WarmUp']


def linear_schedule(step, steps):
    return 1. if step > steps else min(step / steps, 1.)


class WarmUp(MultiplicativeLR):
    def __init__(
            self,
            optimizer: Optimizer,
            steps: int,
            lr_lambda: Union[Callable[[int, int], float], List[Callable[[int, int], float]]] = None,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        """

        Applies a scaling factor to learning rate for steps ``1`` to ``steps``.
        Applies no changes otherwise.
        ``WarmUp`` can be chained or used sequentally with other schedulers.

        Notes:
            - WarmUp sets learning rates based on captured initial learning rates (``base_lrs``).
            - Changes from other schedulers applied before WarmUp will be ovewritten.
            - Chaining ``WarmUp`` is equivalent to SequentialLR if other schedule dynamically manipulates
                optimizer.param_groups without relying on ``base_lrs``.

        Examples:
            >>> # First WarmUp, then other schedule
            >>> from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR
            >>> from celldetection.optim import WarmUp
            >>> warmup_staps = 512
            >>> scheduler = SequentialLR(optimizer, [
            ...     WarmUp(optimizer, warmup_staps),  # warmup for 512 steps
            ...     CosineAnnealingLR(optimizer, T_max=512, eta_min=0.00001),  # after 512 steps switch to cosine ann.
            ... ], milestones=[warmup_staps])

            >>> # Chaining WarmUp and other schedule
            >>> from torch.optim.lr_scheduler import ChainedScheduler, StepLR
            >>> scheduler = ChainedScheduler([
            ...     StepLR(optimizer, 5, gamma=.99),
            ...     WarmUp(optimizer, 512),  # overwrites changes of previous scheduler during warmup
            ... ])

        Args:
            optimizer: Optimizer.
            steps: Number of warmup steps.
            lr_lambda: A function which computes a multiplicative factor given an integer parameter epoch, or a list
                of such functions, one for each group in optimizer.param_groups.
            last_epoch: Last epoch.
            verbose: If ``True``, prints a message to stdout for each update. Default: ``False``.
        """
        if lr_lambda is None:
            lr_lambda = linear_schedule
        self.steps = steps
        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch <= self.steps:
            return [lr * lmbda(self.last_epoch, self.steps) for lmbda, lr in zip(self.lr_lambdas, self.base_lrs)]
        return [group['lr'] for group in self.optimizer.param_groups]
