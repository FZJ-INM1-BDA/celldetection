import torch
from torch import Tensor
import torchvision.ops.boxes as bx
from typing import Tuple, List
import numpy as np

__all__ = ['nms', 'contours2boxes', 'pairwise_box_iou', 'pairwise_generalized_box_iou', 'filter_by_box_voting']


@torch.compile(dynamic=True)
def nms(boxes, scores, thresh=.5) -> torch.Tensor:
    """Non-maximum suppression.

    Perform non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    Notes:
        - Use ``torchvision.ops.boxes.nms`` if possible; This is just a "pure-python" alternative
        - ``cd.ops.boxes.nms`` for 8270 boxes: 13.9 ms ± 131 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        - ``tv.ops.boxes.nms`` for 8270 boxes: 1.84 ms ± 4.91 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        - ``cd.ops.boxes.nms`` for 179 boxes: 265 µs ± 1.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        - ``tv.ops.boxes.nms`` for 179 boxes: 103 µs ± 2.61 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    Args:
        boxes: Boxes. Tensor[num_boxes, 4] in (x0, y0, x1, y1) format.
        scores: Scores. Tensor[num_boxes].
        thresh: Threshold. Discards all overlapping boxes with ``IoU > thresh``.

    Returns:
        Keep indices. Tensor[num_keep].
    """
    indices = torch.argsort(scores, descending=True)
    boxes = boxes[indices]
    mask = bx.box_iou(boxes, boxes) > thresh
    vals, idx = torch.max(mask, 1)
    torch.logical_not(vals, out=vals)
    torch.logical_or(vals, idx >= torch.arange(mask.shape[0], device=mask.device), out=vals)
    return indices[vals]


@torch.compile(dynamic=True)
def get_iou_voting(boxes: Tensor, thresh: float):
    iou = bx.box_iou(boxes, boxes)
    iou *= iou > thresh  # consistent with nms thresh
    votes = iou.sum(-1)
    return votes


def filter_by_box_voting(boxes, thresh, min_vote, return_votes: bool = False):
    """Filter by box voting.

    Filter boxes by popular vote. A box receives a vote if it has an IoU larger than `thresh` with another box.
    Each box also votes for itself, hence, the smallest possible vote is 1.

    Args:
        boxes: Boxes.
        thresh: IoU threshold for two boxes to be considered redundant, counting as a vote for both boxes.
        min_vote: Minimum voting for a box to be accepted. A vote is the sum of IoUs of a box compared to all `boxes`,
            including itself. Hence, the smallest possible vote is 1.
        return_votes: Whether to return voting results.

    Returns:
        Keep indices and optionally voting results.
    """
    keep_indices = torch.arange(len(boxes), device=boxes.device, dtype=torch.int)
    votes = get_iou_voting(boxes, thresh)
    votes_mask = votes >= min_vote
    keep_indices = keep_indices[votes_mask]
    if return_votes:
        return keep_indices, votes[votes_mask]
    return keep_indices


def contours2boxes(contours, axis=-2):
    """Contours to boxes.

    Converts contours to bounding boxes in (x0, y0, x1, y1) format.

    Args:
        contours: Contours as Tensor[(..., )num_points, 2]
        axis: The ``num_points`` axis.

    Returns:

    """
    return torch.cat((contours.min(axis).values, contours.max(axis).values), axis + (axis < 0))


# implementation adapted from torchvision.ops.boxes._box_inter_union
def _pairwise_box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = bx.box_area(boxes1)
    area2 = bx.box_area(boxes2)
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = bx._upcast(rb - lt).clamp(min=0)
    intersection = torch.prod(wh, dim=1)
    union = area1 + area2 - intersection
    return intersection, union


def pairwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    return torch.abs(inter / union)


# implementation adapted from torchvision.ops.boxes.generalized_box_iou
def pairwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    iou = inter / union
    lti = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    whi = bx._upcast(rbi - lti).clamp(min=0)
    areai = torch.prod(whi, dim=1)
    return iou - (areai - union) / areai
