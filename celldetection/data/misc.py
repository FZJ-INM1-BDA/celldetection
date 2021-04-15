import numpy as np
import torch
from collections import OrderedDict


def transpose_spatial(inputs, spatial_dims=2, has_batch=False):
    if spatial_dims == 0:
        return inputs
    return np.transpose(inputs, ([0] * has_batch) + list(range(spatial_dims + has_batch, inputs.ndim)) + list(
        range(has_batch, spatial_dims + has_batch)))


def to_tensor(inputs, spatial_dims=2, has_batch=False):
    return torch.as_tensor(transpose_spatial(inputs, spatial_dims=spatial_dims, has_batch=has_batch))


def universal_dict_collate_fn(batch):
    results = OrderedDict({})
    ref = batch[0]
    for k in ref.keys():
        if isinstance(ref[k], (list, tuple)):
            max_dim = np.max([b[k][0].shape[0] for b in batch])
            results[k] = np.stack(
                [np.pad(b[k][0], ((0, max_dim - b[k][0].shape[0]),) + ((0, 0),) * (b[k][0].ndim - 1)) for b in batch],
                axis=0)
            results[k] = to_tensor(results[k], 0, True)
        else:
            results[k] = np.stack([b[k] for b in batch], axis=0)
            results[k] = to_tensor(results[k], 2, True)
    return results

