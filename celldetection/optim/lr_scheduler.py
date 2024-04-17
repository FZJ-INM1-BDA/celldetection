from torch.optim.lr_scheduler import MultiplicativeLR, SequentialLR as _SequentialLR, \
    ReduceLROnPlateau as _ReduceLROnPlateau
from torch.optim import Optimizer
from typing import Union, Callable, List
import warnings
from bisect import bisect_right

from ..util.util import has_argument

__all__ = ['WarmUp', 'SequentialLR', 'ReduceLROnPlateau']


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
        """WarmUp.

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
            >>> warmup_steps = 512
            >>> scheduler = SequentialLR(optimizer, [
            ...     WarmUp(optimizer, warmup_steps),  # warmup for 512 steps
            ...     CosineAnnealingLR(optimizer, T_max=512, eta_min=0.00001),  # after 512 steps switch to cosine ann.
            ... ], milestones=[warmup_steps])

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


class SequentialLR(_SequentialLR):

    def step(self, metrics=None):  # fixes TypeError caused by use of metric
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:

            if metrics is None:
                scheduler.step()
            else:
                # Some schedulers require metrics, others do not. If not handled here, it'll raise a TypeError
                if has_argument(scheduler.step, 'metric', 'metrics', mode='any'):
                    scheduler.step(metrics)
                else:
                    scheduler.step()

        self._last_lr = scheduler.get_last_lr()


class ReduceLROnPlateau(_ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, warmup=1, verbose="deprecated"):
        """
        Initializes the ReduceLROnPlateau object. This scheduler decreases the learning rate
        when a metric has stopped improving, which is commonly used to fine-tune a model in
        machine learning.

        Notes:
            - Adds the warmup option to PyTorch's ``ReduceLROnPlateau``.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str): One of `min` or `max`. In `min` mode, the learning rate will be reduced
                        when the quantity monitored has stopped decreasing; in `max` mode, it will
                        be reduced when the quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be reduced. `new_lr = lr * factor`.
                            Default: 0.1.
            patience (int): Number of epochs with no improvement after which learning rate will be
                            reduced. Default: 10.
            threshold (float): Threshold for measuring the new optimum, to only focus on significant
                               changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * (1 + threshold)
                                  in 'max' mode or best * (1 - threshold) in `min` mode. In `abs` mode,
                                  dynamic_threshold = best + threshold in `max` mode or best - threshold in
                                  `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
                            Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A lower bound on the learning rate of
                                    all param groups or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller
                         than eps, the update is ignored. Default: 1e-8.
            warmup (int): Number of epochs to wait before initially starting normal operation. Default: 1.
            verbose (str): Deprecated argument. Not used. Default: "deprecated".
        """
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                         threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                         min_lr=min_lr, eps=eps, verbose=verbose)
        self.warmup_counter = int(warmup)  # ignores bad epochs for the first number of `warmup` steps

    def get_last_lr(self):  # required by PyTorch functions to be implemented right here
        return self._last_lr

    def step(self, metrics, epoch=None):
        best_ = None
        if self.warmup_counter:
            self.warmup_counter -= 1
            best_ = self.best

        res = super().step(metrics, epoch)
        if best_ is not None:
            self.best = best_
            self.num_bad_epochs = 0  # ignore any bad epochs in warmup
        return res
