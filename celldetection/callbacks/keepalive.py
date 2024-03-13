import threading
import time
import os
import warnings
from typing import Any
import pytorch_lightning as pl

__all__ = ['KeepAlive']


def terminate_program(log_fn=None):
    from ..mpi.mpi import has_mpi, MPI
    import signal

    log_fn = log_fn or print

    if 'SLURM_JOB_ID' in os.environ:
        try:
            from subprocess import Popen, PIPE
            cmd = ['scancel', os.environ['SLURM_JOB_ID']]
            log_fn(' '.join(cmd), flush=True)
            process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            stdout, stderr = process.communicate()
            log_fn(f'scancel stdout, stderr:\n{stdout.decode()}\n\n{stderr.decode()}')
        except Exception as e:
            log_fn(f'Failed terminating the SLURM job: {e}')

    if has_mpi():
        try:
            log_fn('MPI.COMM_WORLD.Abort()', flush=True)
            MPI.COMM_WORLD.Abort()  # May not work if comm is down
        except Exception as e:
            log_fn(f'Failed terminating the MPI job: {e}')

    log_fn('Sending SIGTERM signal to the current process...')
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(20)

    log_fn('Forcefully exiting...')
    os._exit(1)


class KeepAlive(pl.Callback):
    def __init__(self, timeout=60 * 15, interval=None, start=False, verbose=True):  # timeout in seconds
        super().__init__()
        self.timeout = timeout
        self.interval = timeout / 4 if interval is None else interval
        self.last_signal = time.time()
        self.monitor_thread = None
        self.running = False
        self.verbose = verbose
        if start:
            self.start()

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)

    def start(self):
        self.log("Starting KeepAlive messages")

        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor)
            self.monitor_thread.daemon = True  # exits when main thread exits
            self.monitor_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.monitor_thread.join()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.local_rank == 0:
            self.start()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.local_rank == 0:
            self.stop()

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int,
            dataset_idx: int = 0
    ) -> None:
        self.keep_alive_signal()

    def on_terminate(self):
        terminate_program(log_fn=self.log)

    def _monitor(self):
        while self.running:
            time.sleep(self.interval)  # Check every second
            if time.time() - self.last_signal > self.timeout:
                msg = "Keepalive signal timeout. Terminating the program."
                warnings.warn(msg)
                self.log(msg)
                self.on_terminate()

    def keep_alive_signal(self):
        self.last_signal = time.time()

    on_validate_start = on_test_start = on_predict_start = on_train_start
    on_validate_end = on_test_end = on_predict_end = on_train_end
    on_predict_batch_start = on_validation_batch_start = on_test_batch_start = on_train_batch_start
