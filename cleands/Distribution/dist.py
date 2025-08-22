import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


from ..base import parametric_distribution_model
from ..utils import *

@dataclass(eq=False, order=False)
class two_sample(parametric_distribution_model):
    x: np.ndarray
    y: np.ndarray
    model_type: Type[parametric_distribution_model]
    w_x: Optional[np.ndarray] = None
    w_y: Optional[np.ndarray] = None
    x_model: parametric_distribution_model = field(init=False)
    y_model: parametric_distribution_model = field(init=False)
    params: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x_model = self.model_type(self.x, self.w_x) if self.w_x is not None else self.model_type(self.x)
        self.y_model = self.model_type(self.y, self.w_y) if self.w_y is not None else self.model_type(self.x)
        self.params = bind(self.x_model.params, self.y_model.params)

    def pdf(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError()

    def cdf(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError()

    @property
    def log_likelihood(self) -> float: return self.x_model.log_likelihood + self.y_model.log_likelihood

    @property
    def null_likelihood(self) -> float: return self.x_model.null_likelihood + self.y_model.null_likelihood

class multinomial(parametric_distribution_model):
    def __init__(self, x: np.ndarray, w_x: Optional[np.ndarray] = None, classes: Optional[int] = None) -> None:
        super().__init__(x.reshape(-1, 1))
        self.n_classes = x.max() + 1 if classes == None else classes
        self.bins = self.n_obs*itemfreq(x*w_x) if w_x is not None else itemfreq(x, classes=self.n_classes)
        self.params = self.bins / self.n_obs
        self.w_x = w_x

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return sp.stats.multinomial.pmf(x, n=1, p=self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return sp.stats.multinomial.cdf(x, n=1, p=self.params)

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> np.ndarray:
        return self.likelihood_helper(target, self.params)

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> np.ndarray:
        return self.likelihood_helper(target, np.full(self.n_classes, self.n_classes/self.n_obs))

    def likelihood_helper(self,target : np.ndarray, probs: np.ndarray) -> np.ndarray:
        bins = itemprob(target, classes=self.n_classes)
        return sp.special.gammaln(target.sum() + 1) - sp.special.gammaln(bins + 1).sum() - np.log(bins / probs).sum()

class normal(parametric_distribution_model):
    def __init__(self, x: np.ndarray, w_x: Optional[np.ndarray] = None) -> None:
        super().__init__(x.reshape(-1, 1))
        if w_x is not None:
            mean = np.average(self.x, weights=w_x)
            std = np.sqrt(np.average((self.x - self.mean) ** 2, weights=w_x))
        else:
            mean = self.x.mean()
            std = self.x.std(ddof=0)
        self.params = np.array([mean, std])
        self.w_x = w_x

    def pdf(self, target: np.ndarray) -> np.ndarray:
        return sp.stats.norm.pdf(target, loc=self.params[0], scale=self.params[1])

    def cdf(self, target: np.ndarray) -> np.ndarray:
        return sp.stats.norm.cdf(target, loc=self.params[0], scale=self.params[1])

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> float:
        return self.likelihood_helper(target, loc=self.params[0], scale=self.params[1])

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> float:
        null_mean = target.mean()
        null_std = target.std(ddof=0)
        return self.likelihood_helper(target, null_mean, null_std)

    def likelihood_helper(self, target: np.ndarray, mu: float, sigma: float) -> float:
        sigma = max(sigma, 1e-12)
        return sp.stats.norm(loc=mu, scale=sigma).logpdf(target.flatten()).sum()


class uniform(parametric_distribution_model):
    def __init__(self, x: np.ndarray) -> None:
        super().__init__(x.reshape(-1, 1))
        self.params = np.array([self.x.min(), self.x.max()])
        if self.params[1] <= self.params[0]:
            raise ValueError("Uniform distribution requires upper > lower.")

    def pdf(self, target: np.ndarray) -> np.ndarray:
        return sp.stats.uniform.pdf(target, loc=self.params[0], scale=self.params[1]-self.params[0])

    def cdf(self, target: np.ndarray) -> np.ndarray:
        return sp.stats.uniform.cdf(target, loc=self.params[0], scale=self.params[1]-self.params[0])

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> float:
        return self.likelihood_helper(target, self.params[0], self.params[1])

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> float:
        null_lower = target.min()
        null_upper = target.max()
        return self.likelihood_helper(target, null_lower, null_upper)

    def likelihood_helper(self, target: np.ndarray, lower: float, upper: float) -> float:
        if upper <= lower:
            raise ValueError("Uniform distribution requires upper > lower.")
        target = target.flatten()
        in_bounds = (target >= lower) & (target <= upper)
        return -np.inf if not np.all(in_bounds) else -len(target) * np.log(upper - lower)
