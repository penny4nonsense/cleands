"""
Decision-tree–style classifiers (recursive partitioning and random forest).

This module implements a univariate-split recursive partitioning classifier
that chooses splits by maximizing validation accuracy and uses a likelihood
ratio test (multinomial) to control splitting with a significance threshold.
It also provides a simple random-forest–style variant via feature subsampling.

Classes:
    recursive_partitioning_classifier:
        Greedy tree that recursively splits features to maximize accuracy,
        with optional class weights and a split-significance test.

Factory Aliases:
    random_forest_classifier:
        Partially-applied constructor for randomized feature selection at
        each node (√p columns), i.e., random-forest–style splits.
    RecursivePartitioningClassifier:
        Wrapper for constructing a `recursive_partitioning_classifier`
        via `ClassificationModel`.
    RandomForestClassifier:
        Wrapper for constructing the random-forest–style variant via
        `ClassificationModel`.

Typical usage example:
    >>> from cleands.Classification.tree import RecursivePartitioningClassifier
    >>> model = RecursivePartitioningClassifier(x, y, max_level=3)
    >>> model.tidy; model.glance  # via classification_model mixins
"""

from ..Distribution import multinomial
from ..base import *
from ..utils import *
from functools import partial
import numpy as np
from typing import Optional


class recursive_partitioning_classifier(classification_model):
    """Recursive partitioning (decision tree) classifier.

    Builds a binary tree by recursively selecting a single feature and a split
    (threshold or binary split) that maximizes classification accuracy on the
    current node. Splits are accepted only if a multinomial likelihood-ratio
    test is sufficiently significant (controlled by `sign_level`), unless
    `max_level` stops recursion earlier. Optionally subsamples features
    (`random_x=True`) by taking √p columns at each node (random-forest style).

    Attributes:
        _col_indx (np.ndarray): Column indices used at this fit (subsampled
            when `random_x=True`), applied to both training and prediction.
        max_level (Optional[int]): Maximum tree depth (None for data-driven).
        sign_level (float): Significance level used to gate new splits.
        _level (str): String encoding of the path from the root (e.g., 'LLR').
        weights (Optional[np.ndarray]): Optional per-sample weights.
        _split_variable (float|int): Index of the chosen split feature (w.r.t.
            the possibly subsampled columns), or `np.nan` if terminal.
        _split_value (float): Threshold for the chosen feature, or `np.nan`
            for binary splits.
        _p_value (float): p-value from the split-significance test at the node.
        _left, _right (recursive_partitioning_classifier|None): Children nodes.
        _terminal_prediction (np.ndarray): Class probability vector at a leaf
            (estimated by a multinomial model on node samples).
    """

    def __init__(self, x, y, sign_level=0.95, max_level=None, random_x=False, level='', classes: Optional[int] = None, weights: Optional[np.ndarray] = None):
        """Fit a recursive partitioning classifier at the current node.

        Args:
            x (np.ndarray): Feature matrix (n_obs, n_feat). May be subsampled
                in columns if `random_x=True`.
            y (np.ndarray): Integer class labels (n_obs,).
            sign_level (float, optional): Split significance (1 - α) gate.
                Higher values demand stronger evidence to split. Defaults to 0.95.
            max_level (int, optional): Maximum depth (including current level).
                If None, depth determined by significance criterion. Defaults to None.
            random_x (bool, optional): If True, consider only √p random columns
                at this node (random-forest behavior). Defaults to False.
            level (str, optional): Path label from the root (for printing). Defaults to ''.
            classes (int, optional): Override number of classes (K). Defaults to inferred.
            weights (np.ndarray, optional): Optional per-sample weights used for
                terminal node probabilities and split testing. Defaults to None.
        """
        self._col_indx = np.random.permutation(x.shape[1])[ :int(np.round(np.sqrt(x.shape[1])))] if random_x else np.arange(x.shape[1])
        x = x[:, self._col_indx]
        super(recursive_partitioning_classifier, self).__init__(x, y)
        if classes != None: self.n_classes = classes
        self.max_level, self.sign_level, self._level, self.weights = max_level, sign_level, level, weights
        self._split_variable, self._split_value, self._p_value, self._left, self._right = np.nan, np.nan, np.nan, None, None

        # Leaf prediction: multinomial probabilities at this node
        self._terminal_prediction = multinomial(y, w_x=weights, classes=self.n_classes).params

        # Stop if we've reached the maximum depth
        if max_level != None and len(level) + 1 == max_level: return

        # Evaluate candidate splits for each feature
        outp = np.array([self.__get_acc(self.x[:, i]) for i in range(self.n_feat)])
        if np.isnan(outp[:, 0]).all(): return

        # Choose split with best accuracy
        self._split_variable = np.nanargmax(outp[:, 0])
        self._split_value = outp[self._split_variable, 1]

        # Compute split p-value; gate by significance if depth is not fixed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._p_value = self._get_p_value()
        if max_level == None and self._p_value > (1 - sign_level) / 2 ** (2 * len(level) + 1): return

        # Recurse left/right
        left_indx = x[:, self._split_variable] == 0 if np.isnan(self._split_value) else x[:, self._split_variable] <= self._split_value
        self._left = recursive_partitioning_classifier(
            x[left_indx, :], y[left_indx], sign_level=sign_level, max_level=max_level, random_x=False,
            level=level + 'L', classes=self.n_classes, weights=weights[left_indx] if weights is not None else None
        )
        self._right = recursive_partitioning_classifier(
            x[~left_indx, :], y[~left_indx], sign_level=sign_level, max_level=max_level, random_x=False,
            level=level + 'R', classes=self.n_classes, weights=weights[~left_indx] if weights is not None else None
        )

    # MODIFY WEIGHTS HERE
    def __get_acc(self, var, binary=False):
        """Score a candidate split on a single variable by accuracy.

        For a binary (0/1) variable, compare the majority class on the two
        partitions. For continuous variables, search 10 evenly spaced thresholds
        between min and max and choose the best.

        Args:
            var (np.ndarray): Candidate split feature (n_obs,).
            binary (bool, optional): Treat `var` as already-binary. Defaults to False.

        Returns:
            tuple[float, float]:
                - Best achievable accuracy at this node for this variable.
                - Corresponding threshold (np.nan if the split is binary).

            If no valid split exists, returns (np.nan, np.nan).
        """
        if var.var() == 0: return (np.nan, np.nan)
        if binary or np.logical_or(var == 0, var == 1).all():
            retp = [self.y[var == 0], self.y[var == 1]]
            if any([item.shape[0] == 0 for item in retp]): return np.nan, np.nan
            outp = [itemfreq(item).argmax() for item in retp]
            retp = [(retp[i] == outp[i]).sum() for i in range(2)]
            return (sum(retp) / self.n_obs, np.nan)
        splits = np.linspace(var.min(), var.max(), 10)
        acc = np.array([self.__get_acc(var > split, True)[0] for split in splits])
        if np.isnan(acc).all(): return np.full(2, np.nan)
        indx = np.nanargmax(acc)
        return acc[indx], splits[indx]

    def _get_p_value(self):
        """Compute p-value for the chosen split using a multinomial LRT.

        Constructs a two-sample test on the label distributions on either
        side of the split, comparing against a single pooled multinomial.

        Returns:
            float: p-value of the likelihood-ratio test.
        """
        xvar = (self.x[:, self._split_variable] == 1) if np.isnan(self._split_value) else (self.x[:, self._split_variable] > self._split_value)
        try:
            null = multinomial(self.y, w_x=self.weights)
            if self.weights == None:
                alt = two_sample(x=self.y[xvar], y=self.y[~xvar], model_type=multinomial)
            else:
                alt = two_sample(x=self.y[xvar], y=self.y[~xvar], w_x=self.weights[xvar], w_y=self.weights[~xvar], model_type=multinomial)
            p_value = likelihood_ratio_test(null, alt)
        except:
            return 1
        return p_value['p.value']

    def __str__(self):
        """String representation of the tree (multi-line, one node per line).

        Shows path level, sample size, chosen split (if any), node accuracy,
        and split p-value. Leaf nodes report terminal prediction probabilities.

        Returns:
            str: Human-readable summary of the recursive tree.
        """
        if self._left == None: return f'Level:{self._level} Accuracy:{self.accuracy}; n.obs:{self.n_obs}; Classification:{self._terminal_prediction}; Probability:{self.y.mean()}; p.value:{self._p_value}\n'
        return f'Level:{self._level} n.obs:{self.n_obs}; Variable:{self._col_indx[self._split_variable]}; Split:{self._split_value}; Accuracy:{self.accuracy}; p.value:{self._p_value}\n' + str(self._left) + str(self._right)

    def predict_proba(self, newx, fitted=False):
        """Predict class probabilities for new samples by tree traversal.

        Args:
            newx (np.ndarray): Feature matrix (n_new, n_feat).
            fitted (bool, optional): If True, assumes `newx` is already aligned
                with the column subset `_col_indx`. If False, applies the same
                feature selection used at fit time. Defaults to False.

        Returns:
            np.ndarray: Class probabilities of shape (n_new, n_classes).
        """
        outp = np.array([self._terminal_prediction] * newx.shape[0])
        if self._left == None: return outp
        if not fitted: newx = newx[:, self._col_indx]
        left_indx = newx[:, self._split_variable] == 0 if np.isnan(self._split_value) else newx[:, self._split_variable] <= self._split_value
        outp[left_indx,:], outp[~left_indx,:] = self._left.predict_proba(newx[left_indx, :]).reshape(outp[left_indx,:].shape), self._right.predict_proba(newx[~left_indx, :]).reshape(outp[~left_indx,:].shape)
        return outp

    def classify(self, target, fitted=False):
        """Predict hard class labels (argmax over probabilities).

        Args:
            target (np.ndarray): Feature matrix (n_new, n_feat).
            fitted (bool, optional): See `predict_proba` for column handling.

        Returns:
            np.ndarray: Integer class labels of shape (n_new,).
        """
        return self.predict_proba(target, fitted).argmax(1)

    @property
    def fitted(self):
        """Return hard labels for the training data (in-sample classification).

        Returns:
            np.ndarray: Integer class labels of shape (n_obs,).
        """
        return self.classify(self.x, fitted=True)


class RecursivePartitioningClassifier(ClassificationModel):
    """Convenience wrapper for recursive partitioning classification.

    Fits a decision tree classifier by recursively splitting the feature
    space to maximize classification accuracy. Provides a formula/DataFrame
    interface for the :class:`recursive_partitioning_classifier`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`recursive_partitioning_classifier`.

    Example:
        >>> model = RecursivePartitioningClassifier.from_formula("y ~ x1 + x2", data=df, max_level=3)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = recursive_partitioning_classifier
