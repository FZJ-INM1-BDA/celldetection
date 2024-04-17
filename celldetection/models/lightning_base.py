import json
import traceback
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from typing import Any, Optional, Dict, List, Union, Sequence
from torch import Tensor, nn
from ..optim.lr_scheduler import SequentialLR
from itertools import chain, product
from torch.distributed import is_available, all_gather_object, get_world_size, is_initialized
from collections import OrderedDict, ChainMap
from ..visualization.images import show_detection, imshow_row
from ..visualization.cmaps import label_cmap
from ..util.util import GpuStats, to_device, asnumpy, update_model_hparams_, resolve_model, has_argument
from ..util.logging import log_figure
from ..data.misc import channels_first2channels_last
from ..util.schedule import conf2scheduler, conf2optimizer
from ..data.instance_eval import LabelMatcherList
from ..optim import WarmUp

__all__ = ['LitBase', 'resolve_rank_factor', 'GPU_STATS']

GPU_STATS = GpuStats() if torch.cuda.is_available() else None
STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]


def resolve_rank_factor(factor, world_size):
    if isinstance(factor, str):
        if factor == 'sqrt':
            factor = lambda x: np.sqrt(x)
        elif factor == 'linear':
            factor = lambda x: x
        else:
            raise ValueError(f'Unknown factor: {factor}')
    if callable(factor):
        return factor(world_size)
    return factor


def merge_dictionaries(dict_list: List[Dict[str, Dict[str, float]]]):
    result = {}
    for d in dict_list:
        for key, value in d.items():
            result.setdefault(key, {}).update(value)
    return result


class LitBase(pl.LightningModule):  # requires pl.LightningModule
    def __init__(
            self,
            model: Union[str, dict, nn.Module],
            losses_prog_bar=True,
            optimizer=None,
            scheduler=None,
            scheduler_conf=None,
            warmup_steps=512,
            lr_scale='sqrt',
            weight_decay_scale=None,
            **kwargs
    ):
        """

        Note:
            In a distributed environment it is assumed that the DistributedSampler is being used for evaluation.
             This ensures that each rank only computes a subset of the validation data.

        Args:
            model:
            losses_prog_bar:
            optimizer:
            scheduler:
            scheduler_conf:
            warmup_steps:
            lr_scale:
            weight_decay_scale:
            **kwargs:
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        update_model_hparams_(self, resolve=kwargs.pop('resolve_model_hparam', True), model=model)
        self.model = self.build_model(model, **kwargs)

        self.inputs_key = 'inputs'
        self.targets_key = 'targets'

        self.nd = kwargs.get('nd', getattr(self.model, 'nd', 2))  # if no information on nd is provided, assume nd=2
        self.val_iou_thresholds = kwargs.get('val_iou_thresholds', (.5, .6, .7, .8, .9))
        self.test_iou_thresholds = kwargs.get('test_iou_thresholds', (.5, .6, .7, .8, .9))
        self.val_calibration = kwargs.get('val_calibration', True)
        self.val_best_by = kwargs.get('val_best_by', 'f1_np')

        self.val_hparams = dict(
            **(kwargs.get('val_hparams') or {})
        )
        self.test_hparams = dict(
            **(kwargs.get('test_hparams') or {})
        )

        self.losses_prog_bar = losses_prog_bar
        self.gpu_stats = kwargs.pop('gpu_stats', True)
        self.warmup_steps = warmup_steps
        self.loss_alpha = .99
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_conf = scheduler_conf
        self._lr_scale = lr_scale
        self._weight_decay_scale = weight_decay_scale
        self.figsize = kwargs.get('figsize', (32, 18))
        self._log_figures = kwargs.get('log_figures', True)
        self._eval_zero_division = kwargs.get('eval_zero_division', 0.)
        self._validation_outputs = None
        self._test_outputs = None
        self._predict_outputs = None
        self._val_mean_keys = (
            'f1',

            'f1_np',
            'jaccard_np',
            'fowlkes_mallows_np',
            'recall',
            'precision',

            'avg_recall',
            'avg_precision',
            'avg_f1',
            'avg_jaccard',
            'avg_fowlkes_mallows',
        )
        self._val_sum_keys = ('true_positives', 'false_negatives', 'false_positives')
        self._test_mean_keys = tuple(self._val_mean_keys)
        self._test_sum_keys = tuple(self._val_sum_keys)
        self.max_imsize = kwargs.get('max_imsize', 2048)
        self.item_record = {}
        self._running_avg = {}
        self.calibrate_by = kwargs.get('calibrate_by')  # allows to set specific val dataset (by name) for calibration

    def _iter_loggers(self, experiment=True):
        for logger in self.loggers:
            if experiment and hasattr(logger, 'experiment'):
                logger = logger.experiment
            if logger is not None:
                yield logger

    @staticmethod
    def build_model(model: str, src=None, **kwargs):
        return resolve_model(model, src=src, **kwargs)

    def log_label_figures(self, tag, inputs, labels, close=True):
        for logger in self._iter_loggers():
            if hasattr(logger, 'add_image'):
                if self.nd == 2:
                    try:
                        figures = []
                        for i in range(len(inputs)):
                            img = channels_first2channels_last(asnumpy(inputs[i]))
                            labels_ = asnumpy(labels[i])
                            if not isinstance(labels_, (tuple, list)):
                                labels_ = labels_,
                            labels_ = [label_cmap(i, reduce_axis=2 if self.nd == 2 else None) for i in labels_]
                            imshow_row(img, *labels_, figsize=self.figsize)
                            figures.append(plt.gcf())
                        log_figure(logger, tag=tag, figure=figures, global_step=self.global_step, close=close)
                        plt.close('all')  # should already be done above
                    except Exception as e:
                        print('Exception during label logging', e)
                        traceback.print_exc()
                elif self.nd == 3:
                    # todo: 3d
                    warnings.warn('3d logging not available yet')
                elif self.nd == 1:
                    # todo: 3d
                    warnings.warn('1d logging not available yet')

    def log_contour_figures(self, tag, inputs, contours, close=True, logger=None, global_step=None):
        if global_step is None:
            global_step = self.global_step
        loggers = self._iter_loggers() if logger is None else (logger,)
        for logger in loggers:
            if hasattr(logger, 'add_image'):
                figures = []
                for i in range(len(inputs)):
                    img = channels_first2channels_last(asnumpy(inputs[i]))
                    cons = asnumpy(contours[i])
                    imshow_row(img, img, figsize=self.figsize)
                    show_detection(contours=cons)
                    figures.append(plt.gcf())
                log_figure(logger, tag=tag, figure=figures, global_step=global_step, close=close)
                plt.close('all')  # should already be done above

    def log_batch(self: 'pl.LightningModule', batch: dict, stage: str, keys=('inputs', 'labels'), global_step=None):
        if global_step is None:
            global_step = self.global_step
        for logger in self._iter_loggers():
            if hasattr(logger, 'add_images'):
                for k in keys:
                    try:
                        v = batch[k]
                    except KeyError:
                        warnings.warn(f'Could not find {k} in batch ({batch.keys()} during logging. '
                                      f'Skipping this item for now.')
                        continue
                    if v.ndim == 3:
                        v = v[:, None]
                    logger.add_images(f'{stage}/{k}', v, global_step)

    def on_predict_epoch_end(self) -> None:
        outputs = self._predict_outputs
        return outputs

    def on_predict_epoch_start(self) -> None:
        self._predict_outputs = []  # todo: also consider dict

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Note: Remove everything that is not required for postprocessing from batch (e.g. via batch.pop) for speedup.
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if batch is None:
            return ...  # signal omission (avoiding warning that None would produce)
        if isinstance(batch, Tensor):
            y = super().predict_step(batch, batch_idx, dataloader_idx)
            return y
        assert isinstance(batch, dict)

        out = self._predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        out = OrderedDict(**batch, **out)
        if self.inputs_key in out:
            del out[self.inputs_key]

        self._predict_outputs.append(to_device(out, 'cpu'))
        return out

    def on_test_epoch_start(self) -> None:
        self._test_outputs = {}

    def on_validation_epoch_start(self) -> None:
        self._validation_outputs = {}

    def _training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        # Expected to return outputs dict: {'loss': loss, 'losses': {'loss0': 0., 'loss1': 0.}}
        # Example:
        # inputs = batch.pop(self.inputs_key)
        # outputs: dict = self.model(inputs, targets=batch)
        # return outputs
        raise NotImplementedError

    def training_item_record(self, batch: dict, losses: dict, **kwargs):
        """Training item record.

        Allows to track certain batch statistics during training.
        Called from ``training_step``.

        Args:
            batch: Batch dict.
            losses: Loss dict.
        """
        indices = asnumpy(batch['indices'])
        dataset_indices = batch.get('dataset_indices')
        for j, idx in enumerate(indices):
            li = self.item_record[idx] = self.item_record.get(idx, [])
            li.append(dict(
                dataset_index=None if dataset_indices is None else dataset_indices[j],
                batch_loss=asnumpy(losses['loss']),
            ))

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        if (self.trainer.global_step % 512) == 0 or (self.trainer.current_epoch == 0 and batch_idx % 25 == 0):
            # logger = None if self.logger is None else self.logger.experiment
            self.log_batch(dict(batch) if isinstance(batch, dict) else batch, 'train', global_step=self.global_step)

        # outputs: dict = self.model(inputs, targets=batch)
        outputs: dict = self._training_step(batch=batch, batch_idx=batch_idx)
        log_d = {}
        if self.gpu_stats and GPU_STATS is not None and self.trainer.local_rank == 0:
            log_d.update(GPU_STATS.dict(prefix=f'gpus/node{self.trainer.node_rank}/gpu'))
        losses = outputs.get('losses')
        loss = losses['loss'] = losses.get('loss', outputs['loss'])

        with torch.no_grad():
            alpha = self.loss_alpha
            self._running_avg['loss'] = self._running_avg.get('loss', loss) * alpha + (1. - alpha) * loss

        self.training_item_record(batch, losses)
        self.trainer.strategy.barrier()

        # Log
        self.log_losses(losses)
        if len(self._running_avg):
            self.log_losses(self._running_avg, prefix='ema')
        if log_d:
            self.log_dict(log_d, prog_bar=self.losses_prog_bar, logger=True, on_step=True, sync_dist=False)

        return outputs

    def training_step_end(self, training_step_outputs):
        return training_step_outputs['loss']

    def on_train_epoch_start(self) -> None:
        self.item_record = {}  # reset

    def log_item_record(self, item_record):
        # Log
        for logger in self._iter_loggers():
            if hasattr(logger, 'add_histogram') and self.trainer.global_rank == 0:
                indices = np.array(list(chain.from_iterable([[k] * len(v) for k, v in item_record.items()])))
                dataset_indices = np.array(
                    [asnumpy(d['dataset_index']) for k, val in item_record.items() for d in val])
                logger.add_histogram('sampler/indices', indices, self.global_step)
                if None not in dataset_indices:
                    logger.add_histogram('sampler/dataset_indices', dataset_indices, self.global_step)

    def gather_item_records(self):
        # Gather all records
        world_size = get_world_size() if is_initialized() and is_available() else 1
        if world_size > 1:
            o = ([None] * get_world_size())
            all_gather_object(obj=self.item_record, object_list=o)  # give every rank access to results
        else:
            o = [self.item_record]
        item_record = {}
        for o_ in o:
            for i, v in o_.items():
                assert v is not None
                li = item_record[i] = item_record.get(i, [])
                li += v
        return item_record

    def update_sampler_weights(self, item_record=None):
        """Update sampler weights.

        Optionally updates sampler weights using the current data source (only if the source supports this).

        Expects training data source (e.g. DataModule) to have a method ``update_sampler_weights``.
        If data source implements ``data_source.live_sampler_weights['fit']`` a histogram of it is logged.
        Only implemented for `fit`, as other stages should not require sampler weights.

        Notes:
            For this sampler weight update to have an effect, the data loaders need to be rebuilt.
            This can be done by passing `reload_dataloaders_every_n_epochs` to the Trainer.
        """
        # Update data sampling weights based on training loss
        if hasattr(self.trainer, 'datamodule'):  # get data module or source instance
            dm = self.trainer.datamodule
        else:
            dm = self.trainer.fit_loop._data_source.instance

        if hasattr(dm, 'update_sampler_weights'):  # check if data source has update_sampler_weights method
            if item_record is None:
                item_record = self.gather_item_records()

            # Update sampler weights (assuming data source respects the updated weights on its own)
            dm.update_sampler_weights('fit', item_record)

            if hasattr(dm, 'live_sampler_weights') and dm.live_sampler_weights.get('fit') is not None:
                weights = dm.live_sampler_weights['fit']
                for logger in self._iter_loggers():
                    if hasattr(logger, 'add_histogram') and self.trainer.global_rank == 0:
                        logger.add_histogram('sampler/live_weights', weights, self.global_step)
        else:
            warnings.warn('Data source does not offer `update_sampler_weights` method. '
                          'Hence, adaptive sampling is not possible.')

    def on_train_epoch_end(self: 'pl.LightningModule') -> None:
        item_record = self.gather_item_records()
        self.log_item_record(item_record=item_record)
        self.update_sampler_weights(item_record=item_record)

    def log_losses(self: 'pl.LightningModule', losses: dict, prefix='losses'):
        if losses is not None and isinstance(losses, dict):
            self.log_dict(
                {f'{prefix}/{k}': (torch.zeros((), device=self.device) if v is None else v) for k, v in losses.items()},
                prog_bar=self.losses_prog_bar,
                logger=True, on_step=True, sync_dist=self.hparams.get('sync_loss', True))
        else:
            warnings.warn(f'Could not log losses (type={type(losses)}). Note that this may lead to deadlocks in '
                          f'distributed training.')

    def configure_optimizers(self):
        if isinstance(self._optimizer, (dict, str, type(None))):
            optimizer = dict(AdamW=dict()) if self._optimizer is None else self._optimizer
            optimizer = conf2optimizer(optimizer, filter(lambda p: p.requires_grad, self.parameters()))
        else:
            optimizer = self._optimizer

        if self._lr_scale is not None:
            lr_mul = resolve_rank_factor(self._lr_scale, self.trainer.world_size)
            for gri, gr in enumerate(optimizer.param_groups):
                if self.global_rank == 0:
                    print('Update learning rate from', gr['lr'], end='')
                gr['lr'] *= lr_mul
                if self.global_rank == 0:
                    print(' to', gr['lr'], f'(group {gri})', flush=True)
        if self._weight_decay_scale is not None:
            wd_mul = resolve_rank_factor(self._weight_decay_scale, self.trainer.world_size)
            for gri, gr in enumerate(optimizer.param_groups):
                if 'weight_decay' in gr:
                    if self.global_rank == 0:
                        print('Update weight decay from', gr['weight_decay'], end='')
                    gr['weight_decay'] *= wd_mul
                    if self.global_rank == 0:
                        print(' to', gr['weight_decay'], f'(group {gri})', flush=True)

        if self._scheduler is None:
            return optimizer

        if isinstance(self._scheduler, (dict, str)):
            scheduler = conf2scheduler(self._scheduler, optimizer)
        else:
            scheduler = self._scheduler

        if self.warmup_steps:
            warmup = WarmUp(optimizer, self.warmup_steps)

            if scheduler is None:
                scheduler = warmup
            else:
                scheduler = SequentialLR(optimizer, [warmup, scheduler], milestones=[self.warmup_steps])

        scheduler = {
            **dict(
                interval='step',
                frequency=1,
                scheduler=scheduler,
                strict=True,
                name=None,
            ),
            **(self._scheduler_conf or {})
        }

        return [optimizer], [scheduler]

    def get_loader_name(self: 'pl.LightningModule', stage, dataloader_idx=0):
        if 'val' in stage:
            loaders = self.trainer.val_dataloaders
        elif 'test' in stage:
            loaders = self.trainer.test_dataloaders
        else:
            raise ValueError(stage)
        ds_name = dataloader_idx
        if isinstance(loaders, dict):
            ds_name = next(v for k, v in enumerate(loaders.keys()) if k == dataloader_idx)
        else:
            ds_name = getattr(loaders, 'name', ds_name)
        return ds_name

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        ds_name = self.get_loader_name('test', dataloader_idx)
        outputs = self.evaluation_step(batch, batch_idx, f'test-{ds_name}', hparams=self.test_hparams)
        vo = self._test_outputs[ds_name] = self._test_outputs.get(ds_name, [])
        vo.append(to_device(outputs, 'cpu'))

    def on_test_epoch_end(self) -> None:
        for ds_name, outputs in self._test_outputs.items():
            self.evaluation_epoch_end(outputs, f'test-{ds_name}', iou_thresholds=self.test_iou_thresholds,
                                      sum_keys=self._test_sum_keys, mean_keys=self._test_mean_keys, calibrate=False)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        ds_name = self.get_loader_name('val', dataloader_idx)
        outputs = self.evaluation_step(batch, batch_idx, f'val-{ds_name}', hparams=self.val_hparams)
        vo = self._validation_outputs[ds_name] = self._validation_outputs.get(ds_name, [])
        vo.append(to_device(outputs, 'cpu'))
        return outputs

    def on_validation_epoch_end(self) -> None:
        assert self._validation_outputs is not None
        calibrate = self.val_calibration
        validation_results = []
        for i, (ds_name, outputs) in enumerate(self._validation_outputs.items()):
            res = self.evaluation_epoch_end(
                outputs, f'val-{ds_name}', iou_thresholds=self.val_iou_thresholds,
                sum_keys=self._val_sum_keys, mean_keys=self._val_mean_keys,
                calibrate=calibrate and self.calibrate_by is not None and self.calibrate_by == ds_name,
            )
            if res is not None:
                validation_results.append(res)

        # Reduce results over datasets and perform calibration
        num = len(validation_results)
        if num:
            best = None
            best_key = None
            best_hparams: dict = {}
            best_by = self.val_best_by
            results_per_setting = {}
            for hparams_key in validation_results[0].keys():  # assuming all have same keys
                primary_avg = {}
                secondary_avg = {}
                for res in validation_results:
                    primary, secondary = res[hparams_key]
                    for k, v in primary.items():  # Average primary
                        if v is None or k not in self._val_mean_keys:  # only mean keys
                            continue
                        primary_avg[k] = primary_avg.get(k, 0.) + v / num
                    for k, v in secondary.items():  # Average secondary
                        if v is None or not any(k_ in k for k_ in self._val_mean_keys):  # only mean keys
                            continue
                        secondary_avg[k] = secondary_avg.get(k, 0.) + v / num

                # Check if hparam settings are best
                score = primary_avg[best_by]
                if best is None or score > best:
                    if self.trainer.is_global_zero:
                        print(f'New best ({hparams_key}):', score)
                    best_key = hparams_key
                    best_hparams = json.loads(hparams_key)
                    best = score
                results_per_setting[hparams_key] = primary_avg, secondary_avg

            self._update_best_hparams(
                results_per_setting=results_per_setting,
                best_key=best_key,
                best_hparams=best_hparams,
                calibrate=calibrate and self.calibrate_by is None,
                prefix='val',
                iou_thresholds=self.val_iou_thresholds,
                best=best,
                best_by=best_by
            )

    def _evaluation_step(self, batch: dict, batch_idx: int, prefix: str, hparams_key, inputs, indices, matches,
                         log_step: bool):
        raise NotImplementedError

    def evaluation_step(self, batch: dict, batch_idx: int, prefix: str, hparams: dict):
        inputs = batch[self.inputs_key]
        indices = asnumpy(batch['indices'])

        hks = sorted(hparams.keys())
        hvs = [hparams[k] for k in hks]
        hp_iter = list(product(*hvs)) if len(hvs) else [None]

        rank = self.trainer.global_rank
        ranks = self.trainer.world_size

        prev = [getattr(self.model, k) for k in hks]
        prev_key = json.dumps({i: j for i, j in zip(hks, prev)})
        matches = {}
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        try:
            logged = False
            for hv_idx, hv in enumerate(hp_iter):
                hparams_ = dict()  # track params

                if hv is not None:
                    for k, v in zip(hks, hv):
                        if self.trainer.is_global_zero:
                            print(f'Set model.{k}={v} during {prefix}, rank({rank}/{ranks}), indices({indices})')
                        assert hasattr(self.model, k), (f'Could not find attribute in model: {k}. '
                                                        f'Check provided hparams for {prefix}: {hks}')
                        setattr(self.model, k, v)
                    hparams_.update({i: j for i, j in zip(hks, hv)})  # make sure altered params are tracked
                hparams_key = json.dumps(hparams_)  # 1.56 µs
                log_step = (prev_key == hparams_key) or (
                        (hv_idx + 1) >= len(hp_iter) and not logged)  # log for previous hp or last idx
                logged = logged or log_step
                self._evaluation_step(inputs=inputs, batch=batch, batch_idx=batch_idx, prefix=prefix,
                                      hparams_key=hparams_key, indices=indices, matches=matches, log_step=log_step)
        finally:
            self.model.clear_cache()
            if len(prev):  # restore previous settings
                for k, v in zip(hks, prev):
                    setattr(self.model, k, v)

        return matches  # dict(hparams_key: dict(idx: label_matcher))

    def _process_evaluation_epoch_outputs(self, outputs, prefix, iou_thresholds, mean_keys, sum_keys,
                                          calibrate=False):
        """Process Evaluation Epoch Outputs.

        This method aggregates results from distributed evaluation across different IoU (Intersection over Union)
        thresholds. It computes mean and sum aggregates of specified keys, logs results, and updates the best
        hyperparameter setting based on a predefined metric.

        Args:
            outputs (dict or ChainMap): A dictionary or ChainMap containing the evaluation outputs for each
                hyperparameter setting, where keys are strings representing hyperparameter configurations and
                values are dicts of results indexed by sample indices.
            prefix (str): A string prefix to prepend to logging keys when recording metrics.
            iou_thresholds (list of float): A list of IoU thresholds to evaluate against.
            mean_keys (list of str): Keys for which the mean should be computed across the outputs.
            sum_keys (list of str): Keys for which the sum should be computed across the outputs.
            calibrate (bool, optional): If True, calibrates the model based on the best hyperparameters found.
                Default is False.

        Returns:
            dict: A dictionary mapping each hyperparameter setting (as JSON strings) to a tuple containing two dicts:
                - A primary dictionary with aggregated results for mean and sum calculations.
                - A secondary dictionary with detailed results for each IoU threshold.

        Raises:
            AssertionError: If `outputs` is not a dictionary or a ChainMap.
        """
        assert isinstance(outputs, (dict, ChainMap)), f'Expected type dict but got {type(outputs)}'

        rank = self.trainer.global_rank
        ranks = self.trainer.world_size
        best = None
        best_key = None
        best_hparams = None
        best_by = self.val_best_by
        if best_by not in mean_keys:
            mean_keys = mean_keys + type(mean_keys)([best_by])
        results_per_setting = {}
        for hparams_key, outputs_ in outputs.items():
            hparams: dict = json.loads(hparams_key)

            # Select outputs according to DistributedSampler
            output_selection = [val for idx, val in outputs_.items() if (idx % ranks) == rank]

            # Construct distributed LabelMatcherList (each rank processes only its own results)
            res = LabelMatcherList(output_selection, rank=self.trainer.global_rank,
                                   num_ranks=self.trainer.world_size, cache=True, device=self.device)

            # Gather results for different IoU thresholds
            primary = {k: [] for k in mean_keys + sum_keys}
            for hk, hv in hparams.items():
                primary[hk] = hv

            # Log total number of examples (distributed) for transparency
            primary['num_examples'] = res.length

            # Log mean_keys and sum_keys (distributed)
            secondary = {}
            for res.iou_thresh in iou_thresholds:
                for k in mean_keys + sum_keys:
                    v = getattr(res, k)
                    primary[k].append(v)
                    secondary[f'{k}_{int(res.iou_thresh * 100)}'] = v

            # Reduce results (across different IoU thresholds) for primary log
            for k in mean_keys:
                primary[k] = np.mean(primary[k])
            for k in sum_keys:
                primary[k] = np.sum(primary[k])

            # Check if hparam settings are best
            score = primary[best_by]
            if best is None or score > best:
                best_key = hparams_key
                best_hparams = hparams
                best = score
            results_per_setting[hparams_key] = primary, secondary

        self._update_best_hparams(
            results_per_setting=results_per_setting,
            best_key=best_key,
            best_hparams=best_hparams,
            calibrate=calibrate,
            prefix=prefix,
            iou_thresholds=iou_thresholds,
            best=best,
            best_by=best_by
        )

        return results_per_setting

    def _update_best_hparams(self, results_per_setting, best_key, best_hparams, calibrate, prefix, iou_thresholds,
                             best=None, best_by=None):
        """
        Updates the model's hyperparameters to the best found configuration based on validation metrics,
        logs results, and handles optional hyperparameter calibration.

        Args:
            results_per_setting (dict): A dictionary where keys are hyperparameter settings and values are tuples of primary and secondary results.
            best_key (tuple): The key to the best performing hyperparameters from `results_per_setting`.
            best_hparams (dict): A dictionary of the best hyperparameter values.
            calibrate (bool): A flag to determine if the model's hyperparameters should be calibrated.
            prefix (str): A prefix used for logging to distinguish between different stages or types of validation.
            iou_thresholds (list): List of IoU (Intersection over Union) thresholds used for evaluation.
            best (float): Best result.
            best_by (str): Metric of best result.
        """
        if best_key is not None:
            # Get results for best setting
            primary, secondary = results_per_setting[best_key]

            # Optionally calibrate model to best hyperparameters
            if calibrate:
                for hk, hv in best_hparams.items():
                    if self.trainer.is_global_zero:
                        print(f'Setting model.{hk}={hv}, since it yields the best {prefix} results: '
                              f'{best_by}={best}')
                    assert hasattr(self.model, hk), (f'Could not find attribute in model: {hk}. '
                                                     f'Check provided hparams for {prefix}')
                    setattr(self.model, hk, hv)

            # Prepare and log primary metrics
            log_d = {f'{prefix}/{k}': float(v) for k, v in primary.items()}
            if self.trainer.is_global_zero:
                formatted = '\n  - '.join(
                    f'{k}:' + (str(v) if isinstance(v, int) else '%0.2f' % v) for k, v in log_d.items()
                )
                print(f'\nValidation ({prefix}, IoU thresholds: {iou_thresholds}):\n  -', formatted, flush=True)
            self.log_dict(log_d, logger=True, sync_dist=True, reduce_fx='max')  # ranks are assumed to have same results

            # Prepare and log primary metrics
            secondary_log = {f'{prefix}_detail/{k}': float(v) for k, v in secondary.items()}
            self.log_dict(secondary_log, logger=True, sync_dist=True, reduce_fx='max')
        else:
            warnings.warn(f'Best key was None during {prefix}.')

    def evaluation_epoch_end(self, outputs, prefix, iou_thresholds, mean_keys, sum_keys, calibrate=False) -> (
            Union)[None, Dict]:
        """

        Note:
            `outputs` is a nested sequence of dictionaries, mapping data_index to data_item. Use of dictionary prevents
            double calculations of examples during evaluation. Note that double calculations are the default
            behaviour in PyTorch's DistributedSampler and example omission the only provided
            alternative (drop_last). Both would give wrong evaluation results.
            The use of dictionaries prevents this, yielding correct evaluation results.

        Args:
            outputs:
            prefix:
            iou_thresholds:
            mean_keys:
            sum_keys:
            calibrate: Whether to update internal hyperparameters, based on evaluation results.

        Returns:

        """

        # Assume rank only processes indices where (idx % ranks) == rank is True (since this is the distributed
        # sampler rule). Then you can locally check which results to process and which to discard.
        # Index is used as key in outputs. Only put relevant outputs in LabelMatcherList.

        if isinstance(outputs, list):
            if outputs is not None:
                while len(outputs) and isinstance(outputs[0], list):
                    outputs = list(chain.from_iterable(outputs))
                assert isinstance(outputs[0], dict), f'Expected type dict but got {type(outputs[0])}'
                outputs = merge_dictionaries(outputs)
        else:
            raise ValueError(type(outputs))

        if outputs is not None:
            return self._process_evaluation_epoch_outputs(outputs, prefix=prefix, iou_thresholds=iou_thresholds,
                                                          mean_keys=mean_keys, sum_keys=sum_keys, calibrate=calibrate)

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            # Some schedulers require metrics, others do not. If not handled here, it'll raise a TypeError
            # Note that lots of scheduler compositions cannot receive metrics and need patching to allow forwarding
            if has_argument(scheduler.step, 'metric', 'metrics', mode='any'):
                scheduler.step(metric)
            else:
                scheduler.step()

    def forward(
            self,
            inputs: Tensor,
            targets: Optional[Dict[str, Tensor]] = None,
            max_imsize: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        max_imsize = self.max_imsize if max_imsize is None else max_imsize
        if max_imsize and max(inputs.shape[2:]) > max_imsize and not self.training:
            return self.forward_tiled(inputs, targets=targets, **kwargs)

        device = self.device
        if inputs.device != device:
            inputs = inputs.to(device)

        return self.model(inputs, targets=targets, **kwargs)

    def forward_tiled(
            self,
            inputs: Tensor,
            crop_size: Union[int, Sequence[int]] = 768,
            stride: Union[int, Sequence[int]] = 384,
            **kwargs
    ):
        raise NotImplementedError
