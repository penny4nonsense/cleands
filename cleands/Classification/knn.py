r"""k-Nearest Neighbors (kNN) classification models.

This module provides:

- A standard kNN classifier that estimates class probabilities from the label
  frequencies of the k nearest training points.
- A cross-validated kNN classifier that selects ``k`` by maximizing average
  accuracy across K folds.

Both classes integrate with the ``cleands`` classification framework and
expose the usual classification API (for example, ``predict_proba``,
``accuracy``).

"""

from functools import partial
from typing import Optional
import numpy as np
import pandas as pd  # noqa: F401  (kept if you want DataFrame inputs)

from ..Prediction import k_nearest_neighbors_regressor
from ..base import *
from ..utils import *


class k_nearest_neighbors_classifier(classification_model, k_nearest_neighbors_regressor):
    r"""k-Nearest Neighbors (kNN) classifier.

    Combines the classification interface with the kNN neighbor search
    implemented for regression. Class probabilities are computed as the
    empirical frequency of labels among the ``k`` nearest training samples.

    Attributes:
        k (int): Number of neighbors used for prediction.
        norms_train (np.ndarray): Precomputed squared norms of training rows
            for fast distance computation.

    """

    def __init__(self, x: np.ndarray, y: np.ndarray, k: int = 1) -> None:
        r"""Initialize a basic kNN classifier.

        Args:
            x (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (np.ndarray): Integer labels of shape ``(n_samples,)``.
            k (int, optional): Number of neighbors. Defaults to ``1``.

        """
        super(k_nearest_neighbors_classifier, self).__init__(x, y)
        self.k = k
        self.norms_train = (x ** 2).sum(1).reshape(-1, 1)

    def predict_proba(self, target: np.ndarray) -> np.ndarray:
        r"""Predict class probabilities for new samples.

        Uses Euclidean distance in the original feature space. For each
        sample, the class probabilities are the label frequencies among
        the ``k`` nearest neighbors.

        Args:
            target (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted probabilities of shape ``(n_samples, n_classes)``.

        """
        nearest_neighbors = self.neighbors(target, self.k)
        pred_vals = self.y[nearest_neighbors]
        return itemprob(pred_vals, 1)


class k_nearest_neighbors_cross_validation_classifier(k_nearest_neighbors_classifier):
    r"""kNN classifier with cross-validated ``k``.

    Selects the number of neighbors ``k`` by K-fold cross-validation,
    maximizing mean accuracy over the validation folds, and then fits
    a final kNN classifier using the selected ``k``.

    Attributes:
        k (int): Selected number of neighbors after cross-validation.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, k_max: int = 25, folds: int = 5,
                 seed: Optional[int] = None) -> None:
        r"""Initialize and select ``k`` via cross-validation.

        Runs K-fold cross-validation over ``k = 1..k_max``, selects the ``k`` with
        the highest mean validation accuracy, then initializes the parent kNN
        classifier with the chosen ``k``.

        Args:
            x (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (np.ndarray): Integer class labels of shape ``(n_samples,)``.
            k_max (int, optional): Maximum neighbors to consider. Defaults to ``25``.
            folds (int, optional): Number of CV folds. Defaults to ``5``.
            seed (int | None, optional): Random seed for reproducibility. Defaults to ``None``.

        """
        n = x.shape[0]
        outp = []
        deck = np.arange(n)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(deck)
        for i in range(folds):
            test = deck[int(i * n / folds):int((i + 1) * n / folds)]
            train_lower = deck[:int(i * n / folds)]
            train_upper = deck[int((i + 1) * n / folds):]
            train = np.concatenate((train_lower, train_upper))
            nearest_neighbors = k_nearest_neighbors_classifier(
                x[train], y[train], k_max + 1
            ).neighbors(x[test], k_max + 1)
            outp += [[
                (y[test] == y[train][nearest_neighbors[:, :k]].mean(1)).mean()
                for k in range(1, k_max + 1)
            ]]
        acc = np.array(outp).mean(0)
        k = int(acc.argmax() + 1)
        super(k_nearest_neighbors_cross_validation_classifier, self).__init__(x, y, k)


class kNearestNeighborsClassifier(ClassificationModel):
    """Convenience wrapper for k-nearest neighbors classification.

    Provides a formula/DataFrame interface for the
    :class:`k_nearest_neighbors_classifier`, which predicts class labels
    based on the majority vote among the nearest neighbors.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`k_nearest_neighbors_classifier`.

    Example:
        >>> model = kNearestNeighborsClassifier.from_formula("y ~ x1 + x2", data=df, k=5)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = k_nearest_neighbors_classifier


class kNearestNeighborsCrossValidationClassifier(ClassificationModel):
    """Convenience wrapper for cross-validated k-nearest neighbors classification.

    Selects the optimal number of neighbors via k-fold cross-validation
    and provides a formula/DataFrame interface for the resulting
    :class:`k_nearest_neighbors_cross_validation_classifier`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`k_nearest_neighbors_cross_validation_classifier`.

    Example:
        >>> model = kNearestNeighborsCrossValidationClassifier.from_formula("y ~ x1 + x2", data=df)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = k_nearest_neighbors_cross_validation_classifier
