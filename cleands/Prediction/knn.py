"""
k-Nearest Neighbors (kNN) regression models.

This module implements kNN regression using squared Euclidean distance.
Predictions are made by averaging the target values of the k nearest training
samples. A cross-validation variant is also included to automatically select
the optimal number of neighbors.

Classes:
    k_nearest_neighbors_regressor:
        Basic kNN regressor that predicts by averaging the outcomes of the
        k closest training samples.

    k_nearest_neighbors_cross_validation_regressor:
        Extension of kNN regressor that selects the best k (up to k_max)
        using k-fold cross-validation to minimize mean squared prediction error.

Notes:
    - Distance computation uses precomputed squared L2 norms for efficiency.
    - The cross-validation model defaults to 5 folds and evaluates k from 1
      up to `k_max` (default 25).
"""


import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type
from abc import ABC, abstractmethod
from functools import partial

from ..base import prediction_model, SupervisedModel, PredictionModel
from ..utils import *


class k_nearest_neighbors_regressor(prediction_model):
    """k-Nearest Neighbors (kNN) regressor.

    Predicts values by averaging the outcomes of the `k` closest points
    (neighbors) in the training set, using squared Euclidean distance.

    Attributes:
        k (int): Number of neighbors to use.
        norms_train (np.ndarray): Precomputed squared L2 norms of training samples.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, k: int = 1) -> None:
        """Initialize the kNN regressor.

        Args:
            x (np.ndarray): Training feature matrix of shape (n_obs, n_features).
            y (np.ndarray): Training target vector of shape (n_obs,).
            k (int, optional): Number of nearest neighbors to use. Defaults to 1.
        """
        super(k_nearest_neighbors_regressor, self).__init__(x, y)
        self.k = k
        self.norms_train = (x ** 2).sum(1).reshape(-1, 1)

    def neighbors(self, target: np.ndarray, k: int) -> np.ndarray:
        """Find the k nearest neighbors of each target sample.

        Args:
            target (np.ndarray): Query feature matrix of shape (m, n_features).
            k (int): Number of neighbors to return.

        Returns:
            np.ndarray: Indices of nearest neighbors with shape (m, k).
        """
        norms_test = (target ** 2).sum(1).reshape(-1, 1)
        distance_matrix = self.norms_train - 2 * self.x @ target.T + norms_test.T
        nearest_neighbors = distance_matrix.argsort(0)[:k, :].T
        return nearest_neighbors

    def predict(self, target: np.ndarray) -> np.ndarray:
        """Predict by averaging target values of nearest neighbors.

        Args:
            target (np.ndarray): Query feature matrix of shape (m, n_features).

        Returns:
            np.ndarray: Predicted values of shape (m,).
        """
        nearest_neighbors = self.neighbors(target, self.k)
        return self.y[nearest_neighbors].mean(1)


class k_nearest_neighbors_cross_validation_regressor(k_nearest_neighbors_regressor):
    """Cross-validated kNN regressor.

    Automatically selects the best value of `k` (1 ≤ k ≤ k_max) using
    k-fold cross-validation to minimize mean squared prediction error (MSPE).

    Inherits from:
        k_nearest_neighbors_regressor

    Attributes:
        k (int): Optimal number of neighbors chosen by cross-validation.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 k_max: int = 25, folds: int = 5, seed: Optional[int] = None) -> None:
        """Initialize and select the optimal k using cross-validation.

        Args:
            x (np.ndarray): Training feature matrix of shape (n_obs, n_features).
            y (np.ndarray): Training target vector of shape (n_obs,).
            k_max (int, optional): Maximum number of neighbors to evaluate. Defaults to 25.
            folds (int, optional): Number of CV folds. Defaults to 5.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
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

            nearest_neighbors = k_nearest_neighbors_regressor(
                x[train], y[train], k_max + 1
            ).neighbors(x[test], k_max + 1)

            outp += [[
                ((y[test] - y[train][nearest_neighbors[:, :k]].mean(1)) ** 2).mean()
                for k in range(1, k_max + 1)
            ]]

        mspe = np.array(outp).mean(0)
        k = mspe.argmin() + 1
        super(k_nearest_neighbors_cross_validation_regressor, self).__init__(x, y, k)


class kNearestNeighborsRegressor(PredictionModel):
    """Convenience wrapper for k-nearest neighbors regression.

    Provides a formula/DataFrame interface for the
    :class:`k_nearest_neighbors_regressor`, which predicts continuous
    outcomes by averaging the responses of the nearest neighbors.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`k_nearest_neighbors_regressor`.

    Example:
        >>> model = kNearestNeighborsRegressor.from_formula("y ~ x1 + x2", data=df, k=5)
        >>> model.predict(df[["x1", "x2"]])
    """

    MODEL_TYPE = k_nearest_neighbors_regressor


class kNearestNeighborsCrossValidationRegressor(PredictionModel):
    """Convenience wrapper for cross-validated k-nearest neighbors regression.

    Selects the optimal number of neighbors via k-fold cross-validation
    and provides a formula/DataFrame interface for the resulting model.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`k_nearest_neighbors_cross_validation_regressor`.

    Example:
        >>> model = kNearestNeighborsCrossValidationRegressor.from_formula("y ~ x1 + x2", data=df)
        >>> model.predict(df[["x1", "x2"]])
    """

    MODEL_TYPE = k_nearest_neighbors_cross_validation_regressor
