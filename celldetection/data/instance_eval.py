from itertools import product, chain
from warnings import warn
import numpy as np


def get_pos_labels(v):
    labels = np.unique(v)
    return labels[labels > 0]


def vec2matches(v):
    a_vec, b_vec = v
    return list(set(product(get_pos_labels(a_vec), get_pos_labels(b_vec))))


def intersection_mask(a, b):
    return ((a > 0).any(2).astype('uint8') + (b > 0).any(2).astype('uint8')) > 1


def matching_labels(a, b):
    inter_mask = intersection_mask(a, b)
    matches = list(chain.from_iterable(map(vec2matches, zip(a[inter_mask], b[inter_mask]))))
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


class LabelMatcher:
    """Evaluation of a label image with a target label image.

    Simple interface to evaluate a label image with a target label image with different metrics and IOU thresholds.

    The IOU threshold is the minimum IOU that two objects must have to be counted as a match.
    Each target object can be matched with at most one inferred object and vice versa.
    """

    def __init__(self, inputs=None, targets=None, iou_thresh=None, zero_division='warn'):
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
            a, b = matches[index]
            for index_ in indices[i + 1:]:
                if self._sel[index_]:
                    u, v = matches[index_]
                    if a == u or b == v:
                        self._sel[index_] = False

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
        return set(set(self.matches[:, 0][self._sel])) if len(self.matches) > 0 else set()

    @property
    @labels_exist
    def true_positives(self):
        return len(self.true_positive_labels)

    def _zero_div(self, name):
        if self.zero_division_warn:
            warn(f'ZeroDivisionError in {name} calculation.')
        return self.zero_division

    @property
    @labels_exist
    def precision(self):
        tp = self.true_positives
        fp = self.false_positives
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return self._zero_div('precision')

    @property
    @labels_exist
    def recall(self):
        tp = self.true_positives
        fn = self.false_negatives
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return self._zero_div('recall')

    @property
    @labels_exist
    def f1(self):
        pr = self.precision
        rc = self.recall
        try:
            return (2 * pr * rc) / (pr + rc)
        except ZeroDivisionError:
            return self._zero_div('f1')

    @property
    @labels_exist
    def ap(self):
        tp = self.true_positives
        fn = self.false_negatives
        fp = self.false_positives
        try:
            return tp / (tp + fn + fp)
        except ZeroDivisionError:
            return self._zero_div('ap')


class LabelMatcherList(list):
    """Label Matcher List.

    Simple interface to get averaged results from a list of `LabelMatcher` objects.

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

    """

    @property
    def iou_thresh(self):
        """Gets unique IOU thresholds from all items, if there is only one unique threshold, it is returned."""
        iou_thresholds = np.unique([s.iou_thresh for s in self])
        if len(iou_thresholds) == 1:
            iou_thresholds, = iou_thresholds
        return iou_thresholds

    @iou_thresh.setter
    def iou_thresh(self, v):
        """Set IOU threshold for all items.

        The IOU threshold is the minimum IOU that two objects must have to be counted as a match.
        The higher the threshold, the closer the inferred objects must be to the targeted objects to be counted as
        true positives.
        """
        for s in self:
            s.iou_thresh = v

    def _avg_x(self, x):
        return np.mean([getattr(m, x) for m in self])

    @property
    def avg_f1(self):
        """Average F1 score."""
        return self._avg_x('f1')

    @property
    def f1(self):
        """F1 score from average recall and precision."""
        recall = self.avg_recall
        precision = self.avg_precision
        try:
            return (2 * recall * precision) / (recall + precision)
        except ZeroDivisionError:
            warn('ZeroDivisionError in f1 calculation.')
            return 0

    @property
    def avg_ap(self):
        """Average AP."""
        return self._avg_x('ap')

    @property
    def avg_recall(self):
        """Average recall."""
        return self._avg_x('recall')

    @property
    def avg_precision(self):
        """Average precision."""
        return self._avg_x('precision')
