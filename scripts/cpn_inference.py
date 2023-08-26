import argparse
import torch
from glob import glob
from os.path import isfile, isdir, join, basename, splitext
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


def dict_collate_fn(batch, check_padding=True, img_min_ndim=2) -> OrderedDict:
    results = OrderedDict({})
    ref = batch[0]
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
    def __init__(self, img, mask=None, transforms=None, reps=1, crop_size=(768, 768), strides=(384, 384)):
        """

        Notes:
            - if mask is used, batch_size may be smaller, as items may be dropped

        Args:
            img: Array[h, w, ...] or Tensor[..., h, w].
            mask: Always as Array[h, w, ...]
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

    def __len__(self):
        return len(self.slices) * self.reps

    def __getitem__(self, item):
        slice_idx = item // self.reps
        rep_idx = item % self.reps
        slices = self.slices[slice_idx]
        if self.mask is not None:
            mask_crop = self.mask[slices]
            if not np.any(mask_crop):
                return None
        crop = self.img[self.slice_prefix + slices]
        meta = None
        if self.transforms is not None:
            crop, meta = self.transforms(crop, rep_idx)
        h_start, w_start = [s.start for s in slices]
        return dict(
            inputs=crop,
            slice_idx=slice_idx,
            rep_idx=rep_idx,
            slices=slices,
            overlaps=torch.as_tensor(self.overlaps[slice_idx]),
            offsets=torch.as_tensor([w_start, h_start]),
            transforms=meta
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


def apply_model(img, models, trainer, crop_size=(768, 768), strides=(384, 384), reps=1, transforms=None,
                batch_size=1, num_workers=0, pin_memory=False, border_removal=6, min_vote=1, stitching_rule='nms',
                **kwargs):
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
    mask = None
    # TODO: Add more options
    if img.itemsize > 1:
        img = cd.data.normalize_percentile(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x = img.astype('float32') / 255

    tile_loader = TileLoader(x, mask=mask, crop_size=crop_size, strides=strides, reps=reps, transforms=transforms)
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
        if model_name.endswith('.ckpt'):
            model = cd.models.LitCpn.load_from_checkpoint(model_name, map_location='cpu')
        else:
            model = cd.load_model(model_name, map_location='cpu')
        model.eval()
        model.requires_grad_(False)
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
    parser.add_argument('-i', '--inputs', default='/inputs', type=str,
                        help='inputs path or filename')
    parser.add_argument('-o', '--outputs', default='/outputs', type=str, help='output path')
    parser.add_argument('--inputs_glob', default='*.*', type=str, help='Should `inputs` specify a directory, this'
                                                                       'can be used to filter for specific files '
                                                                       'in that directory.')
    parser.add_argument('--inputs_method', default='imageio', help='Method used for loading non-hdf5 inputs.')
    parser.add_argument('--inputs_dataset', default=None, help='Dataset name for hdf5 inputs.')
    parser.add_argument('-m', '--models', default='/models', help='Model directory or filename.')
    parser.add_argument('--devices', default='auto', type=str, help='Devices.')
    parser.add_argument('--accelerator', default='auto', type=str, help='Accelerator.')
    parser.add_argument('--strategy', default='auto', type=str, help='Strategy.')
    parser.add_argument('--precision', default='32-true', type=str,
                        help='Precision. One of (64, 64-true, 32, 32-true, 16, 16-mixed, bf16, bf16-mixed)')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers.')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Number of batches loaded in advance by each worker.')
    parser.add_argument('--pin_memory', nargs='*', help='If set, the data loader will copy Tensors into device/CUDA '
                                                        'pinned memory before returning them.')
    parser.add_argument('--batch_size', default=1, type=int, help='How many samples per batch to load.')
    parser.add_argument('--tile_size', default=1024, nargs='+', type=int,
                        help='Tile/window size for sliding window processing.')
    parser.add_argument('--stride', default=768, nargs='+', type=int, help='Stride for sliding window processing.')
    parser.add_argument('--border_removal', default=4, type=int, help='Number of border pixels for the removal of '
                                                                      'partial objects during tiled inference.')
    parser.add_argument('--stitching_rule', default='nms', type=str, help='Stitching rule to use for collating results '
                                                                          'from sliding window processing.')
    parser.add_argument('--min_vote', default=1, type=int,
                        help='Required smallest vote count for a detected object to be accepted. '
                             'Only used for ensembles. Minimum vote count is 1, maximum the number of '
                             'models that are part of the ensemble.')
    parser.add_argument('--labels', action='store_true', help='Whether to convert contours to label image.')
    parser.add_argument('--flat_labels', action='store_true', help='Whether to use labels without channels.')
    parser.add_argument('--demo_figure', action='store_true', help='Whether to write a demo figure to disk. '
                                                                   'Note: Intended for smaller images!')
    parser.add_argument('--truncated_images', action='store_true', help='Whether to support truncated images.')
    parser.add_argument('-p', '--properties', nargs='*', help='Region properties')
    parser.add_argument('--spacing', default=1., type=float, help='The pixel spacing. Relevant for pixel-based '
                                                                  'region properties.')
    parser.add_argument('--separator', default='-', type=str,
                        help='Separator string for region properties that are written to multiple columns. '
                             'Default is "-" as in bbox-0, bbox-1, bbox-2, bbox-4.')
    args, unknown = parser.parse_known_args()

    if args.truncated_images:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    if isdir(args.inputs):
        inputs = sorted(glob(join(args.inputs, args.inputs_glob)))
    elif isfile(args.inputs):
        inputs = [args.inputs]
    else:
        raise ValueError(args.inputs)

    outputs = args.outputs
    makedirs(outputs, exist_ok=True)
    if isdir(args.models):
        models = sorted(glob(join(args.models, '*.pt'))) + sorted(glob(join(args.models, '*.ckpt')))
    else:
        models = sorted(glob(args.models))

    devices = args.devices
    if devices.isnumeric():
        devices = int()

    print('Args:', args, '\nUnknown args:', unknown)
    print('Summary:\n ', '\n  '.join([
        f'Number of inputs: {len(inputs)}',
        f'Number of models: {len(models)}',
        f'Output path: {outputs}',
        f'Workers: {args.num_workers}',
        f'Devices: {devices}',
    ]))

    # Load model
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision
    )

    for src in inputs:
        prefix, ext = splitext(basename(src))
        dst = join(outputs, prefix + '{ext}')
        print(src, '-->', dst.format(ext='.*'), flush=True)
        if ext in ('.h5', '.hdf5'):
            assert args.inputs_dataset is not None, 'Please specify the dataset name for hdf5 inputs via --dataset <name>'
            print('Read from h5:', args.inputs_dataset)
            img = cd.from_h5(src, args.inputs_dataset)
        else:
            img = cd.load_image(src, method=args.inputs_method)
        y = cd.asnumpy(apply_model(
            img, models, trainer,
            crop_size=args.tile_size,
            strides=args.stride,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            border_removal=args.border_removal,
            min_vote=args.min_vote,
            stitching_rule=args.stitching_rule
        ))
        is_dist = is_available() and is_initialized()
        if (is_dist and get_rank() == 0) or not is_dist:
            props = args.properties
            do_props = props is not None and len(props)
            do_labels = do_props or args.labels or args.flat_labels

            labels = flat_labels = None
            if do_labels:
                labels = cd.data.contours2labels(y['contours'], img.shape[:2])
                if args.labels:
                    y['labels'] = labels
            if args.flat_labels:
                flat_labels = cd.data.resolve_label_channels(labels)
                if args.flat_labels:
                    y['flat_labels'] = flat_labels
            cd.to_h5(dst.format(ext='.h5'), **cd.asnumpy(y))
            if do_props:
                if args.flat_labels:
                    assert flat_labels is not None
                    cd.data.labels2property_table(flat_labels, props, spacing=args.spacing,
                                                  separator=args.separator).to_csv(dst.format(ext='_flat.csv'))
                if args.labels or not args.flat_labels:
                    assert labels is not None
                    cd.data.labels2property_table(labels, props, spacing=args.spacing, separator=args.separator).to_csv(
                        dst.format(ext='.csv'))

            if args.demo_figure:
                from matplotlib import pyplot as plt
                cd.imshow_row(img, img, figsize=(30, 15), titles=('input', 'contours'))
                cd.plot_contours(y['contours'])
                cd.plot_boxes(y['boxes'])
                loc = cd.asnumpy(y['locations'])
                plt.scatter(loc[:, 0], loc[:, 1], marker='x')
                cd.save_fig(dst.format(ext='_demo.png'))


if __name__ == "__main__":
    main()
