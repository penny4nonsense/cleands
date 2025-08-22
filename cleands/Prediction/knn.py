import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type
from abc import ABC, abstractmethod

from ..base import prediction_model, SupervisedModel
from ..utils import *

class k_nearest_neighbors_regressor(prediction_model):
    def __init__(self, x: np.array, y: np.array, k: int = 1) -> None:
        super(k_nearest_neighbors_regressor, self).__init__(x, y)
        self.k = k
        self.norms_train = (x ** 2).sum(1).reshape(-1, 1)

    def neighbors(self, target: np.array, k: int) -> np.array:
        norms_test = (target ** 2).sum(1).reshape(-1, 1)
        distance_matrix = self.norms_train - 2 * self.x @ target.T + norms_test.T
        nearest_neighbors = distance_matrix.argsort(0)[:k, :].T
        return nearest_neighbors

    def predict(self, target: np.array) -> np.array:
        nearest_neighbors = self.neighbors(target, self.k)
        return self.y[nearest_neighbors].mean(1)


class k_nearest_neighbors_cross_validation_regressor(k_nearest_neighbors_regressor):
    def __init__(self, x: np.array, y: np.array, k_max: int = 25, folds: int = 5, seed: Optional[int] = None) -> None:
        n = x.shape[0]
        outp = []
        deck = np.arange(n)
        if seed is not None: np.random.seed(seed)
        np.random.shuffle(deck)
        for i in range(folds):
            test = deck[int(i * n / folds):int((i + 1) * n / folds)]
            train_lower = deck[:int(i * n / folds)]
            train_upper = deck[int((i + 1) * n / folds):]
            train = np.concatenate((train_lower, train_upper))
            nearest_neighbors = k_nearest_neighbors_regressor(x[train], y[train], k_max + 1).neighbors(x[test],
                                                                                                       k_max + 1)
            outp += [
                [((y[test] - y[train][nearest_neighbors[:, :k]].mean(1)) ** 2).mean() for k in range(1, k_max + 1)]]
        mspe = np.array(outp).mean(0)
        k = mspe.argmin() + 1
        super(k_nearest_neighbors_cross_validation_regressor, self).__init__(x, y, k)

