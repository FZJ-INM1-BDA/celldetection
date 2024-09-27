import argparse
import json
import traceback
import tifffile
import torch
import torch.nn as nn
from glob import glob
from os.path import isfile, isdir, join, basename, splitext, sep
from os import makedirs
import celldetection as cd
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import ImageFile
import cv2
from torch.distributed import is_available, get_world_size, is_initialized, get_rank, gather_object
import albumentations.augmentations.functional as F
from typing import Union, List, Optional, Dict, Any
from warnings import warn


def dict_collate_fn(batch, check_padding=True, img_min_ndim=2) -> Union[OrderedDict, None]:
    results = OrderedDict({})
    ref = None
    for ref in batch:
        if ref is not None:
            break
    if ref is None:
        return None

    for k in ref.keys():
        items = [b[k] for b in batch]
        if isinstance(ref[k], (list, tuple, dict)):
            results[k] = items
        else:
            image_like = len(items) and isinstance(items[0], (torch.Tensor, np.ndarray)) and items[
                0].ndim >= img_min_ndim
            if check_padding and image_like:
                results[k] = cd.data.padding_stack(*items, axis=0)
            else:
                if isinstance(items[0], torch.Tensor):
                    results[k] = torch.stack(items, dim=0)
                else:
                    results[k] = np.stack(items, axis=0)
            if image_like:
                results[k] = cd.to_tensor(results[k], transpose=True, spatial_dims=2, has_batch=True)
    return results


class TileLoader:
    def __init__(self, img, mask=None, point_mask=None, point_mask_exclusive=False, transforms=None, reps=1,
                 crop_size=(768, 768), strides=(384, 384)):
        """

        Notes:
            - if mask is used, batch_size may be smaller, as items may be dropped

        Args:
            img: Array[h, w, ...] or Tensor[..., h, w].
            mask: Always as Array[h, w, ...]
            point_mask
            transforms:
            reps:
            crop_size:
            strides:
        """
        if isinstance(img, torch.Tensor):
            size = img.shape[-len(crop_size):]
            self.slice_prefix = (...,)
        else:
            size = img.shape[:len(crop_size)]
            self.slice_prefix = ()
        self.crop_size = crop_size
        self.slices, self.overlaps, self.num_slices_per_axis = cd.get_tiling_slices(size, crop_size, strides,
                                                                                    return_overlaps=True)
        self.slices, self.overlaps = list(self.slices), list(self.overlaps)
        self.reps = reps
        self.img = img
        self.transforms = transforms
        self.mask = mask
        self.point_mask = point_mask
        self.point_mask_exclusive = point_mask_exclusive

    def __len__(self):
        return len(self.slices) * self.reps

    def __getitem__(self, item):
        slice_idx = item // self.reps
        rep_idx = item % self.reps
        slices = self.slices[slice_idx]

        scores_lower_bound = scores_upper_bound = None
        if self.mask is not None:
            mask_crop = self.mask[slices]
            if not np.any(mask_crop):
                return None
            if mask_crop.ndim == 2:
                mask_crop = mask_crop[..., None]
            scores_upper_bound = mask_crop.astype('float32')

        if self.point_mask is not None:
            point_mask_crop = self.point_mask[slices]
            if not np.any(point_mask_crop):
                return None
            if point_mask_crop.ndim == 2:
                point_mask_crop = point_mask_crop[..., None]

            scores_lower_bound = np.clip(point_mask_crop, 0., 1.)
            if self.point_mask_exclusive:
                scores_upper_bound = scores_lower_bound

        crop = self.img[self.slice_prefix + slices]
        meta = None
        if self.transforms is not None:
            if self.mask is not None or self.point_mask is not None:
                raise NotImplementedError('Use of masks and transforms not supported yet.')
            crop, meta = self.transforms(crop, rep_idx)
        h_start, w_start = [s.start for s in slices]
        return dict(
            inputs=crop,
            slice_idx=slice_idx,
            rep_idx=rep_idx,
            slices=slices,
            overlaps=torch.as_tensor(self.overlaps[slice_idx]),
            offsets=torch.as_tensor([w_start, h_start]),
            transforms=meta,
            **{k: v for k, v in dict(scores_upper_bound=scores_upper_bound,
                                     scores_lower_bound=scores_lower_bound).items() if v is not None}
        )


def apply_keep_indices_(items: dict, keep, ignore_keys=None):
    # Applies keep indices to all Tensors, except keys listed in ignore_keys
    for k, v in items.items():
        if v is None:
            continue
        is_tensor = None
        if ignore_keys is not None and k in ignore_keys:
            continue
        for i, v_ in enumerate(v):
            if is_tensor is None:
                is_tensor = isinstance(v_, torch.Tensor)
            if not is_tensor:
                break
            keep_ = keep[i]
            v[i] = v_[keep_]


def apply_keep_indices_flat_(items: dict, keep, ignore_keys=None):
    # Applies keep indices to all Tensors, except keys listed in ignore_keys
    for k, v in items.items():
        if (ignore_keys is not None and k in ignore_keys) or not isinstance(v, torch.Tensor):
            continue
        items[k] = v[keep]


def concat_results_(coll, new):
    for k, v in new.items():
        is_tensor = None
        if v is None:
            continue
        for v_ in v:
            if is_tensor is None:
                is_tensor = isinstance(v_, torch.Tensor)
            if not is_tensor:
                break
            coll[k] = torch.cat((coll.get(k, v_[:0]), v_))


def oom_safe_concat_results_flat_(coll, new, target_device=None, fallback_device='cpu'):
    for k, v in new.items():
        if not isinstance(v, torch.Tensor):
            continue

        # OOM safe concatenation
        def on_oom():
            nonlocal target_device, coll
            warn(f'Not enough memory. Moving data to {fallback_device} in order to continue.')
            target_device = fallback_device
            for k__, v__ in coll.items():
                coll[k__] = cd.to_device(v__, target_device)

        v_ = None
        oom = cd.OomCatcher(2, callback=on_oom)
        while oom:
            if target_device is not None:
                v = v.to(target_device)
            with oom:
                v_ = torch.cat((coll.get(k, v[:0]), v))
        assert v_ is not None
        coll[k] = v_
    return target_device


def preprocess(img, gamma=1., contrast=1., brightness=0., percentile=None):
    # TODO: Add more options
    if percentile is not None:
        img = cd.data.normalize_percentile(img, percentile)
    if img.itemsize > 1:
        warn('Performing implicit percentile normalization, since input is not uint8.')
        img = cd.data.normalize_percentile(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if gamma != 1.:
        img = F.gamma_transform(img, gamma)
    if contrast != 1.:
        img = F.brightness_contrast_adjust(img, alpha=contrast, beta=brightness)
    return img


def resolve_model(model_name, model_parameters, verbose=True, **kwargs):
    if isinstance(model_name, nn.Module):
        # Is module already
        model = model_name
    elif callable(model_name):
        # Callback
        model = model_name(map_location='cpu')
    else:
        if model_name.endswith('.ckpt'):
            if len(kwargs):
                warn(f'Cannot use kwargs when loading Lightning Checkpoints. Ignoring the following: {kwargs}')
            # Lightning checkpoint
            model = cd.models.LitCpn.load_from_checkpoint(model_name, map_location='cpu')
        else:
            model = cd.load_model(model_name, map_location='cpu', **kwargs)
    if not isinstance(model, cd.models.LitCpn):
        if verbose:
            print('Wrap model with lightning', end='')
        model = cd.models.LitCpn(model, **kwargs)
    model.model.max_imsize = None
    model.eval()
    model.requires_grad_(False)
    if model_parameters is not None:
        for k, v in model_parameters.items():
            if hasattr(model.model, k):
                setattr(model.model, k, type(getattr(model.model, k))(v))
            else:
                warn(f'Could not find attribute {k} in model {model_name}. '
                     f'Hence, the setting was not changed!')
    return model


def oom_safe_gather_dict(local_dict: Dict[str, torch.Tensor], dst=0, fallback_device='cpu',
                         device: torch.device = None) -> List[Dict[str, torch.Tensor]]:
    rank, ranks = cd.get_rank(True)
    result = [{} for _ in range(ranks)] if rank == dst else None
    target_device = None
    for k, v in local_dict.items():
        assert isinstance(v, torch.Tensor), f'Expected type Tensor, but found {type(v)}'

        # Gather sizes
        sizes = [torch.empty(1, dtype=int, device=device) for _ in range(ranks)] if rank == dst else None
        size = torch.empty(1, dtype=int, device=device).fill_(v.shape[0])
        torch.distributed.gather(size, sizes, dst=dst)

        # Custom gather Tensors
        vs = None
        recv_tensor = None
        if rank == dst:
            vs = []
            ds = tuple(v.shape[1:])
            for src in range(ranks):
                recv_size = tuple(sizes[src].cpu().data.numpy()) + ds
                if src == dst:
                    recv_tensor = v.to(device)
                    if target_device is None:  # if not target device, send to where everything else is
                        def on_oom():
                            nonlocal recv_tensor, target_device
                            target_device = fallback_device
                            recv_tensor = v.to(fallback_device)

                        oom = cd.OomCatcher(2, callback=on_oom)
                        while oom:
                            with oom:
                                recv_tensor = recv_tensor.to(device)
                else:
                    # Create OOM safe recv Tensor
                    def on_oom():
                        nonlocal target_device, result, vs
                        warn(f'Not enough memory on {device}. Moving data to {fallback_device} in order to continue.')
                        target_device = fallback_device
                        result = cd.to_device(result, target_device)
                        vs = cd.to_device(vs, target_device)

                    oom = cd.OomCatcher(2, callback=on_oom)
                    while oom:
                        with oom:
                            recv_tensor = torch.empty(recv_size, dtype=v.dtype, device=device)

                    torch.distributed.recv(recv_tensor, src=src)  # todo: receive unordered
                if target_device is not None:  # move to other device right away
                    recv_tensor = recv_tensor.to(target_device)
                vs.append(recv_tensor)
        else:
            # Send tensor to destination rank
            torch.distributed.send(v.to(device), dst=dst)

        if rank == dst:
            # Rearrange to output format
            for r in range(ranks):
                result[r][k] = vs[r]
    return result


def apply_model(img, models, trainer, mask=None, point_mask=None, crop_size=(768, 768), strides=(384, 384), reps=1,
                transforms=None, model_kwargs_list=None,
                batch_size=1, num_workers=0, pin_memory=False, border_removal=4, min_vote=1, stitching_rule='nms',
                gamma=1., contrast=1., brightness=0., percentile=None, model_parameters=None,
                point_mask_exclusive=False, verbose=True, **kwargs):
    assert len(models) >= 1, 'Please specify at least one model.'
    assert min_vote >= 1, f'Min vote smaller than minimum: {min_vote}'
    assert len(models) >= min_vote, f'Min vote greater than number of models: {min_vote}'

    if not isinstance(crop_size, (tuple, list)):
        crop_size = (crop_size,) * 2
    elif len(crop_size) == 1:
        crop_size *= 2
    if not isinstance(strides, (tuple, list)):
        strides = (strides,) * 2
    elif len(strides) == 1:
        strides *= 2
    img = preprocess(img, gamma=gamma, contrast=contrast, brightness=brightness, percentile=percentile)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):  # todo: should depend on model
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x = img  # uint8 converted in lightning_base from now on
    if x.dtype.kind == 'f':
        x = x.astype(np.float32)

    tile_loader = TileLoader(x, mask=mask, point_mask=point_mask, crop_size=crop_size, strides=strides, reps=reps,
                             transforms=transforms, point_mask_exclusive=point_mask_exclusive)
    data_loader = DataLoader(
        tile_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dict_collate_fn,
        shuffle=False,
        persistent_workers=False,
        pin_memory=pin_memory,
        **({'prefetch_factor': kwargs.pop('prefetch_factor', 2)} if num_workers else {})
    )

    results = {}
    h_tiles, w_tiles = tile_loader.num_slices_per_axis
    nms_thresh = None
    target_device = None  # if it is necessary to move data to another device (e.g. CPU), this variable will be set
    for model_name, model_kwargs in zip(models, model_kwargs_list):
        model = resolve_model(model_name, model_parameters, verbose=verbose, **model_kwargs)
        nms_thresh = kwargs.get('nms_thresh', model.model.nms_thresh)

        y = trainer.predict(model, data_loader)

        is_dist = is_available() and is_initialized()
        rank, ranks = cd.get_rank(return_world_size=True)

        pre_results = {}
        for y_idx, y_ in enumerate(y):

            if y_ is None or y_ is ...:  # skip
                continue

            # Iterate batch
            keeps = []
            for n in range(len(y_['contours'])):
                # Determine window position
                h_i, w_i = np.unravel_index(y_['slice_idx'][n], tile_loader.num_slices_per_axis)

                # Remove partial contours
                top, bottom = h_i > 0, h_i < (h_tiles - 1)
                right, left = w_i < (w_tiles - 1), w_i > 0
                keep = cd.ops.cpn.remove_border_contours(y_['contours'][n], tile_loader.crop_size[:2],
                                                         border_removal,
                                                         top=top, right=right, bottom=bottom, left=left,
                                                         offsets=-y_['offsets'][n])

                if stitching_rule != 'nms':
                    keep = cd.ops.filter_contours_by_stitching_rule(y_['contours'][n], tile_loader.crop_size[:2],
                                                                    y_['overlaps'][n], rule=stitching_rule,
                                                                    offsets=-y_['offsets'][n]) & keep

                keeps.append(keep)
            apply_keep_indices_(y_, keeps, ['offsets', 'overlaps'])
            concat_results_(pre_results, y_)

        if is_dist:
            # oom_safe_gather_dict may move everything to cpu if necessary
            pre_results = oom_safe_gather_dict(pre_results, dst=0, device=trainer.strategy.root_device)
            assert (rank == 0) or (pre_results is None)

        if (is_dist and rank == 0) or not is_dist:
            results_ = {}
            if isinstance(pre_results, dict):
                pre_results = pre_results,
            for r_idx, r in enumerate(pre_results):
                assert isinstance(r, dict)
                # oom_safe_concat_results_flat_ may move everything to cpu if necessary
                target_device = oom_safe_concat_results_flat_(results_, r, target_device=target_device)

            # Remove duplicates from tiling
            if 'nms' in stitching_rule.split(','):
                keep = torch.ops.torchvision.nms(results_['boxes'], results_['scores'], nms_thresh)
                apply_keep_indices_flat_(results_, keep, ['offsets', 'overlaps'])

            # Concat all batch items to flat results
            target_device = oom_safe_concat_results_flat_(results, results_, target_device=target_device)

    if 'offsets' in results:
        del results['offsets']
    if 'overlaps' in results:
        del results['overlaps']

    # Remove duplicates from multi model
    if len(models) > 1 and len(results):
        if min_vote > 1:
            keep, votes = cd.ops.filter_by_box_voting(results['boxes'], nms_thresh, min_vote, return_votes=True)
            results['votes'] = votes
            apply_keep_indices_flat_(results, keep)

        # NOTE: nms_thresh inherited from last model, unless specified in kwargs
        keep = torch.ops.torchvision.nms(results['boxes'], results['scores'], nms_thresh)
        apply_keep_indices_flat_(results, keep)

    return results


def cpn_inference(
        inputs: Union[str, List[str], 'np.ndarray'],
        models: Union[str, List[str], 'nn.Module'],
        outputs: Union[str, List[str]] = 'outputs',
        inputs_method: str = 'imageio',
        inputs_dataset: str = 'image',
        masks: Optional[List[str]] = None,
        point_masks: Optional[List[str]] = None,
        point_mask_exclusive: bool = False,
        masks_dataset: str = 'mask',
        point_masks_dataset: str = 'point_mask',
        devices: Union[str, int, List[int]] = 'auto',
        accelerator: str = 'auto',
        strategy: str = 'auto',
        precision: str = '32-true',
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        batch_size: int = 1,
        tile_size: Union[int, List[int]] = 1024,
        stride: Union[int, List[int]] = 768,
        border_removal: int = 4,
        stitching_rule: str = 'nms',
        min_vote: int = 1,
        labels: bool = False,
        flat_labels_: bool = False,
        demo_figure: bool = False,
        overlay: bool = False,
        truncated_images: bool = False,
        properties: Optional[List[str]] = None,
        spacing: float = 1.0,
        separator: str = '-',
        gamma: float = 1.0,
        contrast: float = 1.0,
        brightness: float = 0.0,
        percentile: Optional[List[float]] = None,
        model_parameters: str = '',
        verbose: bool = True,
        skip_existing: bool = False,
        model_kwargs: Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]] = None,
        group_level: str = 'job',
        continue_on_exception: bool = False,
        return_results: bool = False,
        num_nodes: Union[str, int] = 'auto',
):
    """
    Process contour proposals for instance segmentation using specified parameters.

    Args:
        inputs (list[str]): Inputs. Either filename, name pattern (glob), or URL (leading http:// or https://).
        models (list[str]): Model. Either filename, name pattern (glob), URL (leading http:// or https://), or hosted model name.
        outputs (str): Output path. Default is 'outputs'.
        inputs_method (str): Method used for loading non-hdf5 inputs. Default is 'imageio'.
        inputs_dataset (str): Dataset name for hdf5 inputs. Default is 'image'.
        masks (list[str], optional): Masks. Either filename, name pattern (glob), or URL (leading http:// or https://).
        point_masks (list[str], optional): Point masks. Either filename, name pattern (glob), or URL (leading http:// or https://).
        point_mask_exclusive (bool): If set, the points in `point_masks` are the only objects to be segmented. Default is False.
        masks_dataset (str): Dataset name for hdf5 mask inputs. Default is 'mask'.
        point_masks_dataset (str): Dataset name for hdf5 point mask inputs. Default is 'point_mask'.
        devices (str): Devices. Specifies the devices for model inference.
            'auto': Auto-select GPUs, falls back to CPU if GPUs unavailable.
            Integer: Number of GPUs to use (e.g., 1 for a single GPU).
            List of Integers: Specific GPU IDs (e.g., [0, 2]).
            '-1': Use all available GPUs.
            '0': Use CPU only. Default is 'auto'.
        accelerator (str): Accelerator. Defines the hardware accelerator for training.
            'auto': Auto-selects best accelerator (GPU/TPU) based on environment.
            'gpu': Uses GPU for training (requires CUDA).
            'tpu': Utilizes Tensor Processing Units.
            'cpu': Forces use of CPU.
            'ipu': Uses Graphcore's Intelligence Processing Units. Default is 'auto'.
        strategy (str): Strategy for distributed execution.
            'auto': Auto-selects best strategy based on accelerator.
            'dp': Data Parallel - splits batches across GPUs.
            'ddp': Distributed Data Parallel - each GPU runs model copy.
            'ddp2': Like 'ddp' but replicates model on each GPU.
            'horovod': Utilizes Horovod distributed training framework.
            'tpu_spawn': For training on TPUs.
            'ddp_spawn': 'ddp' variant that spawns processes.. Default is 'auto'.
        precision (str): Precision. One of (64, 64-true, 32, 32-true, 16, 16-mixed, bf16, bf16-mixed). Default is '32-true'.
        num_workers (int): Number of workers. Default is 0.
        prefetch_factor (int): Number of batches loaded in advance by each worker. Default is 2.
        pin_memory (list, optional): If set, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
        batch_size (int): How many samples per batch to load. Default is 1.
        tile_size (int): Tile/window size for sliding window processing. Default is 1024.
        stride (int): Stride for sliding window processing. Default is 768.
        border_removal (int): Number of border pixels for the removal of partial objects during tiled inference. Default is 4.
        stitching_rule (str): Stitching rule to use for collating results from sliding window processing. Default is 'nms'.
        min_vote (int): Required smallest vote count for a detected object to be accepted. Default is 1.
        labels (bool): Whether to convert contours to label image. Default is False.
        flat_labels_ (bool): Whether to use labels without channels. Default is False.
        demo_figure (bool): Whether to write a demo figure to disk. Note: Intended for smaller images! Default is False.
        overlay (bool): Whether to write transparent label image to disk. This is mainly useful as an overlay for visual inspection. Default is False.
        truncated_images (bool): Whether to support truncated images. Default is False.
        properties (list, optional): Region properties.
        spacing (float): The pixel spacing. Relevant for pixel-based region properties. Default is 1.0.
        separator (str): Separator string for region properties that are written to multiple columns. Default is '-'.
        gamma (float): Gamma value for gamma transform. Default is 1.0.
        contrast (float): Factor for contrast adjustment. Default is 1.0.
        brightness (float): Factor for brightness adjustment. Default is 0.0.
        percentile (list[float], optional): Percentile norm. Performs min-max normalization with specified percentiles. Default is None.
        model_parameters (str): Model parameters. Pass as string in "key=value,key1=value1" format. Default is ''.
        verbose (bool): Verbosity toggle.
        skip_existing (bool): Whether to inputs with existing output files.
        model_kwargs (str, dict, list[str], list[dict]): Model kwargs. If passed as string, JSON format is expected.
        group_level (str): Processing group level. One of `("job", "node", "rank")`, indicating the scope of processing
            groups that jointly process the same inputs. `"rank"` indicates for example that each input is processed
            by just one rank. Note that each rank is assumed to only have access to a single device, which can be
            ensured for example via `CUDA_VISIBLE_DEVICES` for GPUs.
        continue_on_exception (bool): If ``True``, try to continue processing when certain Exceptions are raised.
            Only works for selected stages (e.g. loading of an input file).
        return_results (bool): Whether to return results. Should be False when used for long sequences of large inputs,
            as collection of large results can lead to OOM exception.
        num_nodes (int): Number of nodes. Default is 'auto'.
    """

    args = dict(locals())

    if isinstance(devices, str) and devices.isnumeric():
        devices = int(devices)

    # Group level
    assert group_level in ('node', 'job', 'rank'), '`group_level` must be one of "node", "job", "rank"'
    comm, mpi_rank, mpi_ranks = cd.mpi.get_comm(return_ranks=True)
    mpi_local_rank = mpi_local_ranks = None
    if group_level != 'job':
        assert cd.mpi.has_mpi(), f'To use `group_level={group_level}` MPI must be available.'
        if group_level == 'node':
            raise NotImplementedError(f'`group_level={group_level}` is not yet available.')
        local_comm, mpi_local_rank, mpi_local_ranks = cd.mpi.get_local_comm(comm, return_ranks=True)

        if group_level == 'rank':
            # Check strategy
            if strategy != 'auto':
                warn(f'Strategy is being set to `"auto"` to comply with `group_level={group_level}`. '
                     f'It was initially set to {strategy}.')
            strategy = 'auto'

            # Check devices
            if isinstance(devices, int) and devices != 1:
                warn(f'Devices is being set to `1` to comply with `group_level={group_level}`. '
                     f'It was initially set to {devices}.')
            devices = 1
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                warn(f'Group level was set `group_level={group_level}`, but found multiple devices.\n'
                     'By default each rank will only use one device in this mode.\n'
                     'To ensure that each rank has its own dedicated device please change visibility settings, e.g. '
                     'via `CUDA_VISIBLE_DEVICES`.')

    if truncated_images:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def resolve_inputs_(collection, x, tag='inputs'):
        if isinstance(x, np.ndarray) or x.startswith('http://') or x.startswith('https://') or isfile(x):
            collection.append(x)
        else:
            input_files = sorted(glob(x))
            assert len(input_files), f'Could not find {tag}: {x}'
            collection += input_files

    if inputs is not None and not isinstance(inputs, (tuple, list)):
        inputs = [inputs]
    if masks is not None and not isinstance(masks, (tuple, list)):
        masks = [masks]
    if models is not None and not isinstance(models, (tuple, list)):
        models = [models]
    if not isinstance(model_kwargs, (tuple, list)):
        model_kwargs = [model_kwargs] * len(models)
    else:
        assert model_kwargs is None or len(models) == len(model_kwargs), ('Please provide one keyword argument '
                                                                          'dict per model.')

    # Prepare input args
    input_list = []
    mask_list = []
    point_mask_list = []
    for idx, i in enumerate(inputs):
        resolve_inputs_(input_list, i, 'inputs')
        if masks:
            resolve_inputs_(mask_list, masks[idx], 'masks')
            assert len(input_list) == len(mask_list), ('Expecting same number of inputs and masks, but found '
                                                       f'{len(input_list)} inputs and {len(mask_list)} masks.')
        else:
            mask_list = None
        if point_masks:
            resolve_inputs_(point_mask_list, point_masks[idx], 'point_masks')
            assert len(input_list) == len(point_mask_list), ('Expecting same number of inputs and masks, but found '
                                                             f'{len(input_list)} inputs and {len(point_mask_list)} '
                                                             f'masks.')
        else:
            point_mask_list = None

    # Prepare model args
    model_list = []
    model_kwargs_list = []
    for m, m_kwargs in zip(models, model_kwargs):
        if m_kwargs is None:
            m_kwargs = dict()
        elif isinstance(m_kwargs, str):
            m_kwargs = json.loads(m_kwargs)
        else:
            assert isinstance(m_kwargs, dict), 'Please provide `model_kwargs` as a dictionary ' \
                                               '(JSON string of dictionary also supported).'

        if isinstance(m, nn.Module):
            model_list.append(m)
            model_kwargs_list.append(m_kwargs)
        else:
            assert isinstance(m, str)
            if m.startswith('http://') or m.startswith('https://') or m.startswith('cd://') or (
                    not isfile(m) and not splitext(m)[1]):
                # Either URL (leading http(s)) or hosted model (leading cd or just no file extension as a fallback)
                model_list.append(lambda _m=m, _mkw=m_kwargs, **kwargs: cd.fetch_model(_m, **kwargs, **_mkw))
                model_kwargs_list.append(dict())
            else:
                files = sorted(glob(m))
                if len(files) == 0 and sep not in m and '.' not in m:
                    files = [lambda _m=m, **kwargs: cd.fetch_model(_m, **kwargs)]  # fallback: try cd-hosted
                assert len(files), f'Could not find models: {m}'
                model_list += files
                model_kwargs_list += [m_kwargs] * len(files)

    # Prepare model parameters
    model_parameters = [i.strip().split('=') for i in model_parameters.split(',') if len(i.strip())]
    model_parameters = {k: v for k, v in model_parameters}
    if verbose and model_parameters is not None and len(model_parameters):
        print('Changing the following model parameters:', model_parameters)

    # Num nodes
    if num_nodes == 'auto':
        num_nodes = cd.get_num_nodes()

    if verbose:
        print('Summary:\n ', '\n  '.join([
            f'Number of inputs: {len(input_list)}',
            f'Number of models: {len(model_list)}',
            f'Output path: {outputs}' + ' (newly created)' * (not isdir(outputs)),
            f'Workers: {num_workers}',
            f'Devices: {devices}',
            f'Cuda available: {torch.cuda.is_available()}',
            f'Cuda device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}',
            f'Accelerator: {accelerator}',
            f'Strategy: {strategy}',
            f'Num nodes: {num_nodes}',
        ]))

    # Load model
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        precision=precision
    )

    makedirs(outputs, exist_ok=True)

    def load_inputs(x, dataset_name, method, tag, idx, ext_checks=('.h5',)):
        if isinstance(x, np.ndarray):
            dst = join(outputs, f'ndarray_{idx}' + '{ext}')
            image = x
        else:
            prefix, ext = splitext(basename(x))
            dst = join(outputs, prefix + '{ext}')

            if skip_existing:
                if any(isfile(dst.format(ext=ext)) for ext in ext_checks):
                    raise FileExistsError

            if x.startswith('http://') or x.startswith('https://'):
                image = cd.fetch_image(x)
            elif ext in ('.h5', '.hdf5'):
                assert inputs_dataset is not None, ('Please specify the dataset name for hdf5 inputs via '
                                                    f'--{tag}_dataset <name>')
                if verbose:
                    print('Read from h5:', dataset_name)
                try:
                    image = cd.from_h5(x, dataset_name)
                except KeyError as e:
                    print(str(e), f'Please specify the dataset name for hdf5 inputs via --{tag}_dataset <name>')
                    raise e
            else:
                image = cd.load_image(x, method=method)
        return image, dst

    output_list = []
    for src_idx, src in enumerate(input_list):
        # Group level: Make sure inputs are assigned correctly
        if group_level == 'rank':
            if (src_idx % mpi_ranks) != mpi_rank:
                continue
        elif group_level == 'node':
            if (src_idx % mpi_local_ranks) != mpi_local_rank:
                continue

        print(f'Next input: {src_idx} (rank {mpi_rank}/{mpi_ranks})', src)

        # Load inputs
        try:
            img, dst = load_inputs(src, inputs_dataset, inputs_method, 'inputs', idx=src_idx)
        except FileExistsError:
            if verbose:
                print('Skipping input, because output exists already:', src)
            continue
        except Exception as e:
            if continue_on_exception:
                # assuming that all ranks fail to load input
                warn(f"An exception occurred: {e}\nTraceback:\n{traceback.format_exc()}")
                if cd.mpi.has_mpi():
                    comm.barrier()
                continue
            else:
                raise e

        dst_h5 = dst.format(ext='.h5')

        if isinstance(src, np.ndarray):
            inputs_tup = 'ndarray',
        else:
            inputs_tup = src,
        if mask_list is None:
            mask = None
        else:
            mask_src = mask_list[src_idx]
            inputs_tup += mask_src,
            mask, _ = load_inputs(mask_src, masks_dataset, inputs_method, 'masks', idx=src_idx)
        if point_mask_list is None:
            point_mask = None
        else:
            point_mask_src = point_mask_list[src_idx]
            inputs_tup += point_mask_src,
            point_mask, _ = load_inputs(point_mask_src, point_masks_dataset, inputs_method, 'masks', idx=src_idx)

        if verbose:
            print(inputs_tup[0] if len(inputs_tup) == 1 else inputs_tup, '-->', dst.format(ext='.*'), flush=True)

        # Resolve model now if it's just one
        if len(model_list) == 1:
            model_list[0] = resolve_model(model_list[0], model_parameters, verbose=verbose, **model_kwargs_list[0])

        y = cd.asnumpy(apply_model(
            img, model_list, trainer,
            mask=mask,
            point_mask=point_mask,
            crop_size=tile_size,
            strides=stride,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            border_removal=border_removal,
            batch_size=batch_size,
            min_vote=min_vote,
            stitching_rule=stitching_rule,
            gamma=gamma,
            contrast=contrast,
            brightness=brightness,
            percentile=percentile,
            model_parameters=model_parameters,
            model_kwargs_list=model_kwargs_list,
            point_mask_exclusive=point_mask_exclusive,
            verbose=verbose
        ))

        is_dist = is_available() and is_initialized()
        output = cd.asnumpy(y)
        if return_results:
            output_list.append(output)
        out_files = dict()
        if (is_dist and get_rank() == 0) or not is_dist:
            props = properties
            do_props = props is not None and len(props)
            do_labels = do_props or labels or flat_labels_

            labels_ = flat_labels_ = None
            if do_labels:
                if 'contours' in y:
                    labels_ = cd.data.contours2labels(y['contours'], img.shape[:2])
                else:
                    labels_ = np.zeros(tuple(img.shape[:2]) + (1,), dtype='uint8')
                if labels:
                    y['labels'] = output['labels'] = labels_
            if flat_labels_:
                flat_labels_ = cd.data.resolve_label_channels(labels_)
                if flat_labels_:
                    y['flat_labels'] = output['flat_labels'] = flat_labels_

            out_files['h5'] = dst_h5
            cd.to_h5(dst_h5, **output,  # json since None values in attrs are not supported
                     attributes=dict(contours=dict(args=cd.dict_to_json_string(args))))
            if do_props:  # TODO: Use mask in properties (writing out labels)
                if flat_labels_:
                    assert flat_labels_ is not None
                    tab = cd.data.labels2property_table(flat_labels_, props, spacing=spacing,
                                                        separator=separator)
                    output['properties_flat'] = tab
                    out_files['properties_flat'] = dst_flat_csv = dst.format(ext='_flat.csv')
                    tab.to_csv(dst_flat_csv)
                if labels or not flat_labels_:
                    assert labels_ is not None
                    tab = cd.data.labels2property_table(labels_, props, spacing=spacing, separator=separator)
                    output['properties'] = tab
                    out_files['properties'] = dst_csv = dst.format(ext='.csv')
                    tab.to_csv(dst_csv)

            if overlay:
                if do_labels:
                    assert labels_ is not None or flat_labels_ is not None
                    label_vis = cd.label_cmap(flat_labels_ if labels_ is None else labels_, ubyte=True)
                else:
                    label_vis = cd.data.contours2overlay(y.get('contours'), img.shape[:2])
                dst_ove_tif = dst.format(ext='_overlay.tif')
                tifffile.imwrite(dst_ove_tif, label_vis, compression='ZLIB', bigtiff=label_vis.size > (2 ** 28))
                output['overlay'] = label_vis
                out_files['overlay'] = dst_ove_tif

            if demo_figure:
                from matplotlib import pyplot as plt
                cd.imshow_row(img, img, figsize=(30, 15), titles=('input', 'contours'))
                if 'contours' in y:
                    cd.plot_contours(y['contours'])
                if 'boxes' in y:
                    cd.plot_boxes(y['boxes'])
                if 'locations' in y:
                    loc = cd.asnumpy(y['locations'])
                    plt.scatter(loc[:, 0], loc[:, 1], marker='x')
                out_files['demo_figure'] = dst_demo = dst.format(ext='_demo.png')
                cd.save_fig(dst_demo)
        if len(out_files):
            output['files'] = out_files

    if cd.mpi.has_mpi():
        comm.barrier()

    if return_results:
        return output_list


def main():
    from inspect import signature

    par = signature(cpn_inference).parameters

    def d(name):
        return par[name].default

    parser = argparse.ArgumentParser('Contour Proposal Networks for Instance Segmentation')

    parser.add_argument('-i', '--inputs', nargs='+', type=str,
                        help='Inputs. Either filename, name pattern (glob), or URL (leading http:// or https://).')
    parser.add_argument('-o', '--outputs', default='outputs', type=str, help='output path')
    parser.add_argument('--inputs_method', default=d('inputs_method'),
                        help='Method used for loading non-hdf5 inputs.')
    parser.add_argument('--inputs_dataset', default=d('inputs_dataset'),
                        help='Dataset name for hdf5 inputs.')
    parser.add_argument('-m', '--models', nargs='+',
                        help='Model. Either filename, name pattern (glob), URL (leading http:// or https://), or '
                             'hosted model name (leading cd://). '
                             'Example: `--models \'cd://ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c\'`')
    parser.add_argument('--model_kwargs', nargs='+',
                        help='Model kwargs in JSON format. '
                             'Example: `--model_kwargs \'{"augment": true}\'')
    parser.add_argument('--masks', default=d('masks'), nargs='+', type=str,
                        help='Masks. Either filename, name pattern (glob), or URL (leading http:// or https://). '
                             'A mask determines where the model searches for objects. Regions with values <= 0'
                             'are ignored. Hence, objects will only be found where the mask is positive. '
                             'Masks are linked to inputs by order. If masks are used, all inputs must have one.')
    parser.add_argument('--point_masks', default=d('point_masks'), nargs='+', type=str,
                        help='Point masks. Either filename, name pattern (glob), or URL (leading http:// or https://). '
                             'A point mask is a mask image with positive values at an object`s location. '
                             'The model aims to convert points to contours. '
                             'Masks are linked to inputs by order. If masks are used, all inputs must have one.')
    parser.add_argument('--point_mask_exclusive', action='store_true',
                        help='If set, the points in `point_masks` (if provided) are the only objects to be segmented. '
                             'Otherwise (default), the points in `point_masks` are considered non-exclusive, meaning '
                             'other objects are detected and segmented in addition. '
                             'Note that this option overrides `masks`.')
    parser.add_argument('--masks_dataset', default=d('masks_dataset'), help='Dataset name for hdf5 inputs.')
    parser.add_argument('--point_masks_dataset', default=d('point_masks_dataset'), help='Dataset name for hdf5 inputs.')
    parser.add_argument('--devices', default=d('devices'), type=str, help='Devices.')
    parser.add_argument('--accelerator', default=d('accelerator'), type=str, help='Accelerator.')
    parser.add_argument('--strategy', default=d('strategy'), type=str, help='Strategy.')
    parser.add_argument('--precision', default='32-true', type=str,
                        help='Precision. One of (64, 64-true, 32, 32-true, 16, 16-mixed, bf16, bf16-mixed)')
    parser.add_argument('--num_workers', default=d('num_workers'), type=int, help='Number of workers.')
    parser.add_argument('--prefetch_factor', default=d('prefetch_factor'), type=int,
                        help='Number of batches loaded in advance by each worker.')
    parser.add_argument('--pin_memory', action='store_true',
                        help='If set, the data loader will copy Tensors into device/CUDA '
                             'pinned memory before returning them.')
    parser.add_argument('--batch_size', default=d('batch_size'), type=int, help='How many samples per batch to load.')
    parser.add_argument('--tile_size', default=d('tile_size'), nargs='+', type=int,
                        help='Tile/window size for sliding window processing.')
    parser.add_argument('--stride', default=d('stride'), nargs='+', type=int,
                        help='Stride for sliding window processing.')
    parser.add_argument('--border_removal', default=d('border_removal'), type=int,
                        help='Number of border pixels for the removal of '
                             'partial objects during tiled inference.')
    parser.add_argument('--stitching_rule', default=d('stitching_rule'), type=str,
                        help='Stitching rule to use for collating results from sliding window processing.')
    parser.add_argument('--min_vote', default=d('min_vote'), type=int,
                        help='Required smallest vote count for a detected object to be accepted. '
                             'Only used for ensembles. Minimum vote count is 1, maximum the number of '
                             'models that are part of the ensemble.')
    parser.add_argument('--labels', action='store_true', help='Whether to convert contours to label image.')
    parser.add_argument('--flat_labels', action='store_true',
                        help='Whether to use labels without channels.')
    parser.add_argument('--demo_figure', action='store_true',
                        help='Whether to write a demo figure to disk. Note: Intended for smaller images!')
    parser.add_argument('--overlay', action='store_true',
                        help='Whether to write transparent label image to disk. '
                             'This is mainly useful as an overlay for visual inspection.')

    parser.add_argument('--truncated_images', action='store_true',
                        help='Whether to support truncated images.')
    parser.add_argument('-p', '--properties', nargs='*', help='Region properties')
    parser.add_argument('--spacing', default=d('spacing'), type=float,
                        help='The pixel spacing. Relevant for pixel-based region properties.')
    parser.add_argument('--separator', default=d('separator'), type=str,
                        help='Separator string for region properties that are written to multiple columns. '
                             'Default is "-" as in bbox-0, bbox-1, bbox-2, bbox-4.')

    parser.add_argument('--group_level', type=str, default='job',
                        help='Processing group level. One of `("job", "node", "rank")`, indicating the scope of '
                             'processing groups that jointly process the same inputs. `"rank"` indicates for example '
                             'that each input is processed by just one rank.')

    parser.add_argument('--gamma', default=d('gamma'), type=float, help='Gamma value for gamma transform.')
    parser.add_argument('--contrast', default=d('contrast'), type=float, help='Factor for contrast adjustment.')
    parser.add_argument('--brightness', default=d('brightness'), type=float, help='Factor for brightness adjustment.')
    parser.add_argument('--percentile', default=d('percentile'), nargs='+', type=float,
                        help='Percentile norm. Performs min-max normalization with specified percentiles.'
                             'Specify either two values `(min, max)` or just `max` interpreted as '
                             '(1 - max, max).')
    parser.add_argument('--model_parameters', default=d('model_parameters'), type=str,
                        help='Model parameters. Pass as string in "key=value,key1=value1" format')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Whether to skip existing files. ')
    parser.add_argument('--continue_on_exception', action='store_true',
                        help='Whether to continue if an exception occurs. ')

    args, unknown = parser.parse_known_args()

    assert args.inputs, ('Please provide inputs to the script! Example to add all tif files of the '
                         'images folder: --inputs \'images/*.tif\'')
    assert args.models, ('Please provide models to the script! Example to add celldetection model: '
                         '`--model \'cd://ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c\'`')

    cpn_inference(
        inputs=args.inputs,
        models=args.models,
        outputs=args.outputs,
        inputs_method=args.inputs_method,
        inputs_dataset=args.inputs_dataset,
        masks=args.masks,
        point_masks=args.point_masks,
        point_mask_exclusive=args.point_mask_exclusive,
        masks_dataset=args.masks_dataset,
        point_masks_dataset=args.point_masks_dataset,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        stride=args.stride,
        border_removal=args.border_removal,
        stitching_rule=args.stitching_rule,
        min_vote=args.min_vote,
        labels=args.labels,
        flat_labels_=args.flat_labels,
        demo_figure=args.demo_figure,
        overlay=args.overlay,
        truncated_images=args.truncated_images,
        properties=args.properties,
        spacing=args.spacing,
        separator=args.separator,
        gamma=args.gamma,
        contrast=args.contrast,
        brightness=args.brightness,
        percentile=args.percentile,
        model_parameters=args.model_parameters,
        skip_existing=args.skip_existing,
        model_kwargs=args.model_kwargs,
        group_level=args.group_level,
        return_results=False,
        continue_on_exception=args.continue_on_exception
    )

    if not (is_available() and is_initialized()) or get_rank() == 0:  # because why not
        cd.say_goodbye()


if __name__ == "__main__":
    main()
