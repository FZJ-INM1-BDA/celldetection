from itertools import product, chain
from typing import Union
from warnings import warn
import numpy as np
import torch


def get_pos_labels(v):
    labels = np.unique(v)
    return labels[labels > 0]


def vec2matches(v):
    a_vec, b_vec = v
    return list(set(product(get_pos_labels(a_vec), get_pos_labels(b_vec))))


def intersection_mask(a, b):
    return ((a > 0).any(2).astype('uint8') + (b > 0).any(2).astype('uint8')) > 1


def matching_labels(a, b):
    ac = (a > 0).sum(-1)
    bc = (b > 0).sum(-1)
    maximum = max(ac.max(), bc.max())
    intersect_one = (ac == 1) & (bc == 1)
    matches = np.stack((a[intersect_one].max(-1), b[intersect_one].max(-1)), 1)
    if maximum > 1:  # prefilter (may still show no overlap, due to self-overlap)
        intersect_mul = ((ac > 1) & (bc > 0)) | ((bc > 1) & (ac > 0))
        a_ = a[intersect_mul]
        if len(a_):  # actual filter
            new = np.array(list(chain.from_iterable(map(vec2matches, zip(a_, b[intersect_mul])))))
            try:
                matches = np.concatenate((
                    matches, new
                ))
            except Exception as e:
                print(matches.shape, new.shape, list(chain.from_iterable(map(vec2matches, zip(a_, b[intersect_mul])))),
                      '\n', e, flush=True)
                raise e
    matches, counts = np.unique(matches, axis=0, return_counts=True)
    return matches, counts


def labels2counts(a):
    count_dict = {}
    uni, cnt = np.unique(a, return_counts=True)
    for u, c in zip(uni, cnt):
        if u == 0:
            continue
        count_dict[u] = c
    return count_dict


def labels_exist(func):
    def func_wrapper(self, *a, **k):
        try:
            self.matches
        except AttributeError:
            raise ValueError('No labels found. Add labels before retrieving results.')
        return func(self, *a, **k)

    return func_wrapper


def _f1_np(v, epsilon=1e-12):
    tp = v.true_positives
    fn = v.false_negatives
    fp = v.false_positives
    return (2 * tp) / (2 * tp + fn + fp + epsilon)


def _jaccard_np(v, epsilon=1e-12):
    tp = v.true_positives
    fn = v.false_negatives
    fp = v.false_positives
    return tp / (tp + fn + fp + epsilon)


def _fowlkes_mallows_np(v, epsilon=1e-12):
    tp = v.true_positives
    fn = v.false_negatives
    fp = v.false_positives
    return tp / np.sqrt((tp + fp) * (tp + fn) + epsilon)


def _precision(v, epsilon=1e-12):
    tp = v.true_positives
    fp = v.false_positives
    return tp / (tp + fp + epsilon)


def _recall(v, epsilon=1e-12):
    tp = v.true_positives
    fn = v.false_negatives
    return tp / (tp + fn + epsilon)


class LabelMatcher:
    """Evaluation of a label image with a target label image.

    Simple interface to evaluate a label image with a target label image with different metrics and IOU thresholds.

    The IOU threshold is the minimum IOU that two objects must have to be counted as a match.
    Each target object can be matched with at most one inferred object and vice versa.
    """

    def __init__(self, inputs=None, targets=None, iou_thresh=None, zero_division='warn', epsilon=1e-12):
        """Initialize LabelMatcher object.

        Args:
            inputs: Input labels. Array[height, width, channels].
            targets: Target labels. Array[height, width, channels].
            iou_thresh: IOU threshold.
            zero_division: One of `('warn', 0, 1)`. Sets the default return value for ZeroDivisionErrors.
                The default `'warn'` will show a warning and return `0`.
                For example: If there are no true positives and no false positives, `precision` will return
                the value of `zero_division` and optionally show a warning.
        """
        self._iou_thresh = 0. if iou_thresh is None else iou_thresh
        self._sel = None
        self.ious, self.unions, self.input_labels = (None,) * 3
        self.target_labels, self.matches, self.intersections, self.input_counts, self.target_counts = (None,) * 5
        self.zero_division = zero_division if isinstance(zero_division, int) else 0
        self.zero_division_warn = zero_division == 'warn'
        self.epsilon = epsilon
        if inputs is not None and targets is not None:
            self.update(inputs, targets, iou_thresh)

    def update(self, inputs, targets, iou_thresh=None):
        inputs = inputs[:, :, None] if inputs.ndim == 2 else inputs
        targets = targets[:, :, None] if targets.ndim == 2 else targets
        self.input_labels = get_pos_labels(inputs)
        self.target_labels = get_pos_labels(targets)
        self.matches, self.intersections = matching_labels(inputs, targets)
        self.input_counts = labels2counts(inputs)  # total num pixels of a label
        self.target_counts = labels2counts(targets)  # total num pixels of a label
        self.unions = np.array(
            [self.input_counts[i] + self.target_counts[j] for (i, j) in self.matches]) - self.intersections

        self.ious = self.intersections / self.unions
        assert np.all(self.intersections <= self.unions)
        self.iou_thresh = self.iou_thresh if iou_thresh is None else iou_thresh  # also calls filter_and_threshold

    @labels_exist
    def filter_and_threshold(self):
        matches = self.matches
        ious = self.ious
        iou_thresh = self.iou_thresh
        indices = np.argsort(ious)[::-1]  # from largest iou to smallest
        self._sel = ious >= iou_thresh
        for i, index in enumerate(indices):
            if not self._sel[index]:
                continue
            iou = ious[index]
            self._sel[index] = iou_pass = iou >= iou_thresh
            if not iou_pass or i + 1 >= len(indices):
                continue
            indices_ = indices[i + 1:]
            mat_match = (matches[index:index + 1] == matches[indices_]).any(-1)
            self._sel[indices_[mat_match]] = False

    @property
    def iou_thresh(self):
        return self._iou_thresh

    @iou_thresh.setter
    def iou_thresh(self, v):
        assert self.ious is not None
        self._iou_thresh = v
        self.filter_and_threshold()

    @property
    @labels_exist
    def false_positive_labels(self):
        a = set(self.input_labels)
        b = set(self.matches[:, 0][self._sel]) if len(self.matches) > 0 else set()
        result = a - b
        if len(result) == 0:
            assert a == b
        return result

    @property
    @labels_exist
    def false_positives(self):
        return len(self.false_positive_labels)

    @property
    @labels_exist
    def false_negative_labels(self):
        a = set(self.target_labels)
        b = set(self.matches[:, 1][self._sel]) if len(self.matches) > 0 else set()
        result = a - b
        if len(result) == 0:
            assert a == b
        return result

    @property
    @labels_exist
    def false_negatives(self):
        return len(self.false_negative_labels)

    @property
    @labels_exist
    def true_positive_labels(self):
        return set(self.matches[:, 0][self._sel]) if len(self.matches) > 0 else set()

    @property
    @labels_exist
    def true_positives(self):
        return len(self.true_positive_labels)

    def _zero_div(self, name):
        if self.zero_division_warn:
            warn(f'ZeroDivisionError in {name} calculation. '
                 f'Assuming {self.zero_division} as result.')
        return self.zero_division

    @property
    @labels_exist
    def precision(self):
        try:
            return _precision(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('precision')

    @property
    @labels_exist
    def recall(self):
        try:
            return _recall(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('recall')

    @property
    @labels_exist
    def f1(self):
        pr = self.precision
        rc = self.recall
        try:
            return (2 * pr * rc) / (pr + rc + self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('f1')

    @property
    @labels_exist
    def jaccard(self):
        try:
            return _jaccard_np(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('jaccard')

    @property
    @labels_exist
    def fowlkes_mallows(self):
        try:
            return _fowlkes_mallows_np(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('fowlkes_mallows_np')


class LabelMatcherList(list):
    def __init__(self, *args, epsilon=1e-12, rank=None, num_ranks=None, device=None, cache=False, **kwargs):
        """Label Matcher List.

        Simple interface to get averaged results from a list of `LabelMatcher` objects.

        Note:
            Distributed use assumes, that each example is shown exactly once. Duplicates are not removed.
            Check your sampler accordingly. Also make sure that each rank calls the same methods in the same order.

        Examples:
            >>> lml = LabelMatcherList([
            ...     LabelMatcher(pred_labels_0, target_labels0),
            ...     LabelMatcher(pred_labels_1, target_labels1),
            ... ])
            >>> lml.iou_thresh = 0.5  # set iou_thresh for all LabelMatcher objects
            >>> print('Average F1 score for iou threshold 0.5:', lml.avg_f1)
            Average F1 score for iou threshold 0.5: 0.92

            >>> # Testing different IOU thresholds:
            >>> for lml.iou_thresh in (.5, .75):
            ...     print('thresh:', lml.iou_thresh, '\t f1:', lml.avg_f1)
            thresh: 0.5 	 f1: 0.92
            thresh: 0.75 	 f1: 0.91

        Args:
            *args:
            epsilon:
            rank: Rank (e.g. `trainer.global_rank). Allows for distributed communication.
                If not passed, results will only be computed locally. If passed results are synced across all ranks.
            num_ranks: Number of ranks (e.g. `trainer.world_size`). Allows for distributed communication.
                If not passed, results will only be computed locally. If passed results are synced across all ranks.
            cache: Whether to cache aggregated results. Currently only for distributed environments.
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon
        if rank is not None or num_ranks is not None:
            assert None not in (rank, num_ranks), 'Please provide both `ranks` and `num_ranks`.'
        self.rank = rank
        self.num_ranks = num_ranks
        self.device = device
        self.cache = cache
        self._cache = {}
        self._iou_thresh = None

    @property
    def distributed(self):
        return self.rank is not None and self.num_ranks is not None and self.num_ranks > 1

    def append(self, __object):
        self.clear_cache()
        return super().append(__object)

    def __add__(self, other):
        self.clear_cache()
        return super().__add__(other)

    def __iadd__(self, other):
        self.clear_cache()
        return super().__iadd__(other)

    def extend(self, __iterable):
        self.clear_cache()
        return super().extend(__iterable)

    def clear(self):
        self.clear_cache()
        return super().clear()

    def copy(self):
        self.clear_cache()
        return super().copy()

    def __setitem__(self, key, value):
        self.clear_cache()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self.clear_cache()
        return super().__delitem__(key)

    def pop(self, *args, **kwargs):
        self.clear_cache()
        return super().pop(*args, **kwargs)

    def insert(self, __index, __object):
        self.clear_cache()
        return super().insert(__index, __object)

    def clear_cache(self):
        self._cache = {}

    @property
    def iou_thresh(self):
        """Gets local unique IOU thresholds from all items, if there is only one unique threshold, it is returned."""
        # todo distributed
        if super().__len__():
            iou_thresholds = np.unique([s.iou_thresh for s in self])
            if len(iou_thresholds) == 1:
                iou_thresholds, = iou_thresholds
            return iou_thresholds
        return self._iou_thresh  # fallback

    @iou_thresh.setter
    def iou_thresh(self, v):
        """Set IOU threshold for all items.

        The IOU threshold is the minimum IOU that two objects must have to be counted as a match.
        The higher the threshold, the closer the inferred objects must be to the targeted objects to be counted as
        true positives.
        """
        if self.distributed:
            # Note that the code below is just a fail-save, hence gather is used to reduce cost

            # Convert v to a tensor and move it to the appropriate device
            v_tensor = torch.tensor([v], device=self.device)

            # Gather all v_tensor values to rank 0
            gathered_v_tensors = [torch.zeros_like(v_tensor) for _ in range(self.num_ranks)]
            torch.distributed.gather(v_tensor, gather_list=gathered_v_tensors if self.rank == 0 else None,
                                     dst=0)

            # On rank 0, check if all values are equal
            if self.rank == 0 and len(gathered_v_tensors):
                gathered_v_tensors = torch.concatenate(gathered_v_tensors).ravel()
                if not torch.allclose(gathered_v_tensors[:1], gathered_v_tensors):
                    raise ValueError(f"IoU threshold is not equal across all ranks: {gathered_v_tensors}")

        self._cache = {}
        self._iou_thresh = v  # fallback
        for s in self:
            s.iou_thresh = v

    @property
    def length(self) -> int:
        local_count = super().__len__()
        if self.distributed:
            if self.cache:
                res = self._cache.get('length', None)
                if res is not None:
                    return res
            count = torch.tensor([local_count], device=self.device)

            # Perform the sum-reduce operation across all ranks
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

            res = count.item()
            if self.cache:
                self._cache['length'] = res
            return res
        return local_count

    def _avg_x(self, x) -> float:
        attributes = [getattr(m, x) for m in self]

        # Handle the case where the attributes list is empty
        if not attributes:
            local_sum = 0.
            local_count = 0.
        else:
            local_sum = np.sum(attributes)
            local_count = len(attributes)

        # Check if the training is distributed
        if self.distributed:

            if self.cache:
                res = self._cache.get(f'_avg_{x}', None)
                if res is not None:
                    return res

            # Combine local sum and count into a single tensor
            local_sum_count_tensor = torch.tensor([local_sum, local_count], dtype=torch.float32, device=self.device)

            # Perform a single reduce operation for both sum and count
            torch.distributed.all_reduce(local_sum_count_tensor, op=torch.distributed.ReduceOp.SUM)  # inplace

            total_sum, total_count = local_sum_count_tensor.tolist()
            res = total_sum / total_count if total_count != 0 else 0
            if self.cache:
                self._cache[f'_avg_{x}'] = res
            return res

        return local_sum / local_count if local_count != 0 else 0

    def _sum_x(self, x) -> Union[int, float]:
        # Calculate the local sum
        local_sum = np.sum([getattr(m, x) for m in self])

        # Check if the training is distributed (world_size > 1)
        if self.distributed:
            if self.cache:
                res = self._cache.get(f'_sum_{x}', None)
                if res is not None:
                    return res
            # Convert the numpy value to a torch tensor and move it to the appropriate device
            local_sum_tensor = torch.tensor(local_sum, dtype=torch.float32, device=self.device)

            # Perform the sum-reduce operation across all ranks
            torch.distributed.all_reduce(local_sum_tensor, op=torch.distributed.ReduceOp.SUM)

            res = local_sum_tensor.item()
            if self.cache:
                self._cache[f'_sum_{x}'] = res
            return res
        else:
            # If not distributed, return the local sum
            return local_sum

    @property
    def false_positives(self):
        return self._sum_x('false_positives')

    @property
    def false_negatives(self):
        return self._sum_x('false_negatives')

    @property
    def true_positives(self):
        return self._sum_x('true_positives')

    @property
    def f1(self):
        """F1 score from average recall and precision."""
        recall = self.avg_recall
        precision = self.avg_precision
        try:
            return (2 * recall * precision) / (recall + precision + self.epsilon)
        except ZeroDivisionError:
            warn('ZeroDivisionError in f1 calculation.')
            return 0

    @property
    def f1_np(self):
        """F1 score from negatives and positives."""
        try:
            return _f1_np(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('f1_np')

    @property
    def jaccard_np(self):
        try:
            return _jaccard_np(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('jaccard_np')

    @property
    def fowlkes_mallows_np(self):
        try:
            return _fowlkes_mallows_np(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('fowlkes_mallows_np')

    @property
    def avg_f1(self):
        """Average F1 score."""
        return self._avg_x('f1')

    @property
    def avg_jaccard(self):
        """Average Jaccard index."""
        return self._avg_x('jaccard')

    @property
    def avg_fowlkes_mallows(self):
        return self._avg_x('fowlkes_mallows')

    @property
    def avg_recall(self):
        """Average recall."""
        return self._avg_x('recall')

    @property
    def avg_precision(self):
        """Average precision."""
        return self._avg_x('precision')

    @property
    def precision(self):
        try:
            return _precision(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('precision')

    @property
    def recall(self):
        try:
            return _recall(self, epsilon=self.epsilon)
        except ZeroDivisionError:
            return self._zero_div('recall')
