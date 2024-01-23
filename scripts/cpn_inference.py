import argparse
import json

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
from torch.distributed import is_available, all_gather_object, get_world_size, is_initialized, get_rank
from itertools import chain
import albumentations.augmentations.functional as F
from typing import Union
from warnings import warn
from skimage import img_as_float, img_as_ubyte




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
                    results[k] = torch.stack(items, axis=0)
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
        for v_ in v:
            if is_tensor is None:
                is_tensor = isinstance(v_, torch.Tensor)
            if not is_tensor:
                break
            coll[k] = torch.cat((coll.get(k, v_[:0]), v_))


def concat_results_flat_(coll, new):
    for k, v in new.items():
        if not isinstance(v, torch.Tensor):
            continue
        coll[k] = torch.cat((coll.get(k, v[:0]), v))


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


def resolve_model(model_name, model_parameters):
    if isinstance(model_name, nn.Module):
        # Is module already
        model = model_name
    elif callable(model_name):
        # Callback
        model = model_name(map_location='cpu')
    else:
        if model_name.endswith('.ckpt'):
            # Lightning checkpoint
            model = cd.models.LitCpn.load_from_checkpoint(model_name, map_location='cpu')
        else:
            print("LOAD MODEL VIA cd.load_model", model_name, flush=True)
            model = cd.load_model(model_name, map_location='cpu')
    if not isinstance(model, cd.models.LitCpn):
        print('Wrap model with lightning', end='')
        model = cd.models.LitCpn(model)
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


def apply_model(img, models, trainer, mask=None, point_mask=None, crop_size=(768, 768), strides=(384, 384), reps=1,
                transforms=None,
                batch_size=1, num_workers=0, pin_memory=False, border_removal=6, min_vote=1, stitching_rule='nms',
                gamma=1., contrast=1., brightness=0., percentile=None, model_parameters=None, **kwargs):
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
    x = img.astype('float32') / 255

    tile_loader = TileLoader(x, mask=mask, point_mask=point_mask, crop_size=crop_size, strides=strides, reps=reps,
                             transforms=transforms)
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
    for model_name in models:
        model = resolve_model(model_name, model_parameters)
        nms_thresh = kwargs.get('nms_thresh', model.model.nms_thresh)

        y = trainer.predict(model, data_loader)

        is_dist = is_available() and is_initialized()
        if is_dist:
            o = ([None] * get_world_size())
            all_gather_object(obj=y, object_list=o)  # give every rank access to results
            y = list(chain.from_iterable(o))

        if (is_dist and get_rank() == 0) or not is_dist:

            results_ = {}
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
                concat_results_(results_, y_)

            # Remove duplicates from tiling
            if 'nms' in stitching_rule.split(','):
                keep = torch.ops.torchvision.nms(results_['boxes'], results_['scores'], nms_thresh)
                apply_keep_indices_flat_(results_, keep, ['offsets', 'overlaps'])

            # Concat all batch items to flat results
            concat_results_flat_(results, results_)

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


def main():
    parser = argparse.ArgumentParser('Contour Proposal Networks for Instance Segmentation')
    parser.add_argument('-i', '--inputs', nargs='+', type=str,
                        help='Inputs. Either filename, name pattern (glob), or URL (leading http:// or https://).')
    parser.add_argument('-o', '--outputs', default='outputs', type=str, help='output path')
    parser.add_argument('--inputs_method', default='imageio',
                        help='Method used for loading non-hdf5 inputs.')
    parser.add_argument('--inputs_dataset', default='image', help='Dataset name for hdf5 inputs.')
    parser.add_argument('-m', '--models', nargs='+',
                        help='Model. Either filename, name pattern (glob), URL (leading http:// or https://), or '
                             'hosted model name (leading cd://). '
                             'Example: `--model \'cd://ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c\'`')
    parser.add_argument('--masks', default=None, nargs='+', type=str,
                        help='Masks. Either filename, name pattern (glob), or URL (leading http:// or https://). '
                             'A mask determines where the model searches for objects. Regions with values <= 0'
                             'are ignored. Hence, objects will only be found where the mask is positive. '
                             'Masks are linked to inputs by order. If masks are used, all inputs must have one.')
    parser.add_argument('--point_masks', default=None, nargs='+', type=str,
                        help='Point masks. Either filename, name pattern (glob), or URL (leading http:// or https://). '
                             'A point mask is a mask image with positive values at an object`s location. '
                             'The model aims to convert points to contours. '
                             'Masks are linked to inputs by order. If masks are used, all inputs must have one.')
    parser.add_argument('--point_mask_exclusive', action='store_true',
                        help='If set, the points in `point_masks` (if provided) are the only objects to be segmented. '
                             'Otherwise (default), the points in `point_masks` are considered non-exclusive, meaning '
                             'other objects are detected and segmented in addition. '
                             'Note that this option overrides `masks`.')
    parser.add_argument('--masks_dataset', default='mask', help='Dataset name for hdf5 inputs.')
    parser.add_argument('--point_masks_dataset', default='point_mask', help='Dataset name for hdf5 inputs.')
    parser.add_argument('--devices', default='auto', type=str, help='Devices.')
    parser.add_argument('--accelerator', default='auto', type=str, help='Accelerator.')
    parser.add_argument('--strategy', default='auto', type=str, help='Strategy.')
    parser.add_argument('--precision', default='32-true', type=str,
                        help='Precision. One of (64, 64-true, 32, 32-true, 16, 16-mixed, bf16, bf16-mixed)')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers.')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Number of batches loaded in advance by each worker.')
    parser.add_argument('--pin_memory', nargs='*',
                        help='If set, the data loader will copy Tensors into device/CUDA '
                             'pinned memory before returning them.')
    parser.add_argument('--batch_size', default=1, type=int, help='How many samples per batch to load.')
    parser.add_argument('--tile_size', default=1024, nargs='+', type=int,
                        help='Tile/window size for sliding window processing.')
    parser.add_argument('--stride', default=768, nargs='+', type=int,
                        help='Stride for sliding window processing.')
    parser.add_argument('--border_removal', default=4, type=int,
                        help='Number of border pixels for the removal of '
                             'partial objects during tiled inference.')
    parser.add_argument('--stitching_rule', default='nms', type=str,
                        help='Stitching rule to use for collating results from sliding window processing.')
    parser.add_argument('--min_vote', default=1, type=int,
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
    parser.add_argument('--spacing', default=1., type=float,
                        help='The pixel spacing. Relevant for pixel-based region properties.')
    parser.add_argument('--separator', default='-', type=str,
                        help='Separator string for region properties that are written to multiple columns. '
                             'Default is "-" as in bbox-0, bbox-1, bbox-2, bbox-4.')

    parser.add_argument('--gamma', default=1., type=float, help='Gamma value for gamma transform.')
    parser.add_argument('--contrast', default=1., type=float, help='Factor for contrast adjustment.')
    parser.add_argument('--brightness', default=0., type=float, help='Factor for brightness adjustment.')
    parser.add_argument('--percentile', default=None, nargs='+', type=float,
                        help='Percentile norm. Performs min-max normalization with specified percentiles.'
                             'Specify either two values `(min, max)` or just `max` interpreted as '
                             '(1 - max, max).')
    parser.add_argument('--model_parameters', default='', type=str,
                        help='Model parameters. Pass as string in "key=value,key1=value1" format')

    args, unknown = parser.parse_known_args()

    assert args.inputs, ('Please provide inputs to the script! Example to add all tif files of the '
                         'images folder: --inputs \'images/*.tif\'')
    assert args.models, ('Please provide models to the script! Example to add celldetection model: '
                         '`--model \'cd://ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c\'`')

    if args.truncated_images:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    outputs = args.outputs

    def resolve_inputs_(collection, name, tag='inputs'):
        if name.startswith('http://') or name.startswith('https://') or isfile(name):
            collection.append(name)
        else:
            input_files = sorted(glob(name))
            assert len(input_files), f'Could not find {tag}: {name}'
            collection += input_files

    # Prepare input args
    inputs = []
    masks = []
    point_masks = []
    for idx, i in enumerate(args.inputs):
        resolve_inputs_(inputs, i, 'inputs')
        if args.masks:
            resolve_inputs_(masks, args.masks[idx], 'masks')
            assert len(inputs) == len(masks), ('Expecting same number of inputs and masks, but found '
                                               f'{len(inputs)} inputs and {len(masks)} masks.')
        else:
            masks = None
        if args.point_masks:
            resolve_inputs_(point_masks, args.point_masks[idx], 'point_masks')
            assert len(inputs) == len(point_masks), ('Expecting same number of inputs and masks, but found '
                                                     f'{len(inputs)} inputs and {len(point_masks)} masks.')
        else:
            point_masks = None

    # Prepare model args
    models = []
    for m in args.models:
        if m.startswith('http://') or m.startswith('https://') or m.startswith('cd://') or (
                not isfile(m) and not splitext(m)[1]):
            # Either URL (leading http(s)) or hosted model (leading cd or just no file extension as a fallback)
            models.append(lambda _m=m, **kwargs: cd.fetch_model(_m, **kwargs))
        else:
            files = sorted(glob(m))
            if len(files) == 0 and sep not in m and '.' not in m:
                files = [lambda _m=m, **kwargs: cd.fetch_model(_m, **kwargs)]  # fallback: try cd-hosted
            assert len(files), f'Could not find models: {m}'
            models += files

    # Prepare model parameters
    model_parameters = [i.strip().split('=') for i in args.model_parameters.split(',') if len(i.strip())]
    model_parameters = {k: v for k, v in model_parameters}
    if model_parameters is not None and len(model_parameters):
        print('Changing the following model parameters:', model_parameters)

    devices = args.devices
    if devices.isnumeric():
        devices = int()

    print('Args:', args, f'\nUnknown args: {unknown}' * bool(len(unknown)))
    print('Summary:\n ', '\n  '.join([
        f'Number of inputs: {len(inputs)}',
        f'Number of models: {len(models)}',
        f'Output path: {outputs}' + ' (newly created)' * (not isdir(outputs)),
        f'Workers: {args.num_workers}',
        f'Devices: {devices}',
        f'Strategy: {args.strategy}',
    ]))

    # Load model
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision
    )

    makedirs(outputs, exist_ok=True)

    def load_inputs(name, dataset_name, method, tag):
        prefix, ext = splitext(basename(name))
        dst = join(outputs, prefix + '{ext}')
        if name.startswith('http://') or name.startswith('https://'):
            image = cd.fetch_image(name)
        elif ext in ('.h5', '.hdf5'):
            assert args.inputs_dataset is not None, ('Please specify the dataset name for hdf5 inputs via '
                                                     f'--{tag}_dataset <name>')
            print('Read from h5:', dataset_name)
            try:
                image = cd.from_h5(name, dataset_name)
            except KeyError as e:
                print(str(e), f'Please specify the dataset name for hdf5 inputs via --{tag}_dataset <name>')
                raise e
        else:
            image = cd.load_image(name, method=method)
        return image, dst

    for src_idx, src in enumerate(inputs):
        img, dst = load_inputs(src, args.inputs_dataset, args.inputs_method, 'inputs')
        inputs_tup = src,
        if masks is None:
            mask = None
        else:
            mask_src = masks[src_idx]
            inputs_tup += mask_src,
            mask, _ = load_inputs(mask_src, args.masks_dataset, args.inputs_method, 'masks')
        if point_masks is None:
            point_mask = None
        else:
            point_mask_src = point_masks[src_idx]
            inputs_tup += point_mask_src,
            point_mask, _ = load_inputs(point_mask_src, args.point_masks_dataset, args.inputs_method, 'masks')

        print(inputs_tup[0] if len(inputs_tup) == 1 else inputs_tup, '-->', dst.format(ext='.*'), flush=True)

        if len(models) == 1:
            models[0] = resolve_model(models[0], model_parameters)

        y = cd.asnumpy(apply_model(
            img, models, trainer,
            mask=mask,
            point_mask=point_mask,
            crop_size=args.tile_size,
            strides=args.stride,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            border_removal=args.border_removal,
            min_vote=args.min_vote,
            stitching_rule=args.stitching_rule,
            gamma=args.gamma,
            contrast=args.contrast,
            brightness=args.brightness,
            percentile=args.percentile,
            model_parameters=model_parameters
        ))
        is_dist = is_available() and is_initialized()
        if (is_dist and get_rank() == 0) or not is_dist:
            props = args.properties
            do_props = props is not None and len(props)
            do_labels = do_props or args.labels or args.flat_labels or args.overlay

            labels = flat_labels = None
            if do_labels:
                labels = cd.data.contours2labels(y['contours'], img.shape[:2])
                if args.labels:
                    y['labels'] = labels
            if args.flat_labels:
                flat_labels = cd.data.resolve_label_channels(labels)
                if args.flat_labels:
                    y['flat_labels'] = flat_labels
            cd.to_h5(dst.format(ext='.h5'), **cd.asnumpy(y),  # json since None values in attrs are not supported
                     attributes=dict(contours=dict(args=json.dumps(vars(args)))))
            if do_props:  # TODO: Maybe use mask in properties (writing out labels)
                if args.flat_labels:
                    assert flat_labels is not None
                    cd.data.labels2property_table(flat_labels, props, spacing=args.spacing,
                                                  separator=args.separator).to_csv(dst.format(ext='_flat.csv'))
                if args.labels or not args.flat_labels:
                    assert labels is not None
                    cd.data.labels2property_table(labels, props, spacing=args.spacing, separator=args.separator).to_csv(
                        dst.format(ext='.csv'))

            if args.overlay:
                assert labels is not None or flat_labels is not None
                label_vis = img_as_ubyte(cd.label_cmap(flat_labels if labels is None else labels))
                tifffile.imwrite(dst.format(ext='_overlay.tif'), label_vis, compression='ZLIB')

            if args.demo_figure:
                from matplotlib import pyplot as plt
                cd.imshow_row(img, img, figsize=(30, 15), titles=('input', 'contours'))
                cd.plot_contours(y['contours'])
                cd.plot_boxes(y['boxes'])
                loc = cd.asnumpy(y['locations'])
                plt.scatter(loc[:, 0], loc[:, 1], marker='x')
                cd.save_fig(dst.format(ext='_demo.png'))

    if get_rank() == 0:  # because why not
        cd.say_goodbye()


if __name__ == "__main__":
    main()
