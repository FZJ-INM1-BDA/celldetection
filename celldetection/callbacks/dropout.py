from pytorch_lightning import Trainer, LightningModule, Callback
from torch.nn.modules.dropout import _DropoutNd
from typing import Any

__all__ = ['StepDropout']


class StepDropout(Callback):
    def __init__(
            self,
            step_size,
            base_drop_rate,
            gamma=0.,
            update_interval='epoch',
            log=True,
            log_name='drop_rate',
            ascending=False,
            **kwargs
    ):
        """Step Dropout.

        A simple Dropout Scheduler.

        References:
            - https://arxiv.org/abs/2303.01500

        Examples:
            >>> from pytorch_lightning import Trainer
            >>> # Early Dropout (drop rate from .1 to 0 after 50 epochs)
            >>> trainer = Trainer(callbacks=[StepDropout(50, base_drop_rate=.1, gamma=0.)])

            >>> # Late Dropout (drop rate from 0 to .1 after 50 epochs)
            >>> trainer = Trainer(callbacks=[StepDropout(50, base_drop_rate=.1, gamma=0., ascending=True)])

        Args:
            step_size: Period of drop rate decay.
            base_drop_rate: Base drop rate.
            gamma: Multiplicative factor of drop rate decay. Default: 0. to replicate "Early Dropout".
            update_interval: One of ``('step', 'epoch')``.
            log: Whether to log drop rates using ``module.log(log_name, drop_rate)``.
            log_name: Name for logging.
            logger: If ``True`` logs to the logger.
            ascending: If ``True`` drop rate decays from right to left, i.e. it starts at ``0`` and
                ascends towards ``base_drop_rate``. Using ``ascending=True, gamma=0.`` replicates "Late Dropout".
            **kwargs: Keyword arguments for ``module.log``.
        """
        super().__init__()
        self.step_size = step_size
        self.gamma = gamma
        self.base_drop_rate = base_drop_rate
        assert update_interval in ('epoch', 'step')
        self.update_interval = update_interval
        self.last_rate = -1
        self.log = log
        self.log_name = log_name
        self.log_kwargs = {**dict(
            on_step=self.update_interval == 'step',
            on_epoch=self.update_interval == 'epoch',
        ), **kwargs}
        self.ascending = ascending

    def update_drop_rate(self, pl_module: "LightningModule", drop_rate: float):
        self.last_rate = drop_rate
        for mod in pl_module.modules():
            if isinstance(mod, _DropoutNd):
                mod.p = drop_rate
        if self.log:
            pl_module.log(self.log_name, drop_rate, **self.log_kwargs)

    @staticmethod
    def get_rate(base, gamma, step, step_size, ascending):
        return base * (ascending + (-1 if ascending else 1) * gamma ** (step // step_size))

    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if self.update_interval == 'epoch':
            self.update_drop_rate(pl_module, self.get_rate(
                self.base_drop_rate, self.gamma, trainer.current_epoch, self.step_size, self.ascending))

    def on_train_batch_start(self, trainer: "Trainer", pl_module: "LightningModule", batch: Any,
                             batch_idx: int) -> None:
        if self.update_interval == 'step':
            self.update_drop_rate(pl_module, self.get_rate(
                self.base_drop_rate, self.gamma, trainer.global_step, self.step_size, self.ascending))
