import torch
import torchvision as tv


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
    mask = tv.ops.boxes.box_iou(boxes, boxes) > thresh
    vals, idx = torch.max(mask, 1)
    torch.logical_not(vals, out=vals)
    torch.logical_or(vals, idx >= torch.arange(mask.shape[0], device=mask.device), out=vals)
    return indices[vals]
