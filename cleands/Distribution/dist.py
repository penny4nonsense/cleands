"""
dist.py
-------

Implements parametric probability distribution models and a two-sample wrapper.

Classes:
    two_sample:
        Generic wrapper to fit and compare the same distribution type on two samples.

    multinomial:
        Multinomial distribution model for categorical data.

    normal:
        Normal (Gaussian) distribution model with optional weighting.

    uniform:
        Uniform distribution model.

Notes:
    - Each distribution extends `parametric_distribution_model` from `base`.
    - Models expose `.params` (fitted parameters), `.pdf`, `.cdf`,
      and log-likelihood utilities.
"""

import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..base import parametric_distribution_model
from ..utils import *

class multinomial(parametric_distribution_model):
    """Multinomial distribution model.

    Args:
        x (np.ndarray): Discrete class labels (integers 0..C-1).
        w_x (np.ndarray, optional): Weights. Defaults to None.
        classes (int, optional): Number of classes. Defaults to max(x)+1.

    Attributes:
        n_classes (int): Number of classes.
        bins (np.ndarray): Bin counts.
        params (np.ndarray): Class probabilities.
    """

    def __init__(self, x: np.ndarray, w_x: Optional[np.ndarray] = None, classes: Optional[int] = None) -> None:
        super().__init__(x.reshape(-1, 1))
        self.n_classes = x.max() + 1 if classes is None else classes
        self.bins = self.n_obs * itemfreq(x * w_x) if w_x is not None else itemfreq(x, classes=self.n_classes)
        self.params = self.bins / self.n_obs
        self.w_x = w_x

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Multinomial probability mass function."""
        return sp.stats.multinomial.pmf(x, n=1, p=self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Multinomial cumulative distribution function."""
        return sp.stats.multinomial.cdf(x, n=1, p=self.params)

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> np.ndarray:
        """Log-likelihood for out-of-sample targets under fitted params."""
        return self.likelihood_helper(target, self.params)

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> np.ndarray:
        """Log-likelihood for out-of-sample targets under uniform null model."""
        return self.likelihood_helper(target, np.full(self.n_classes, self.n_classes / self.n_obs))

    def likelihood_helper(self, target: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Helper for computing multinomial log-likelihoods."""
        bins = itemprob(target, classes=self.n_classes)
        return sp.special.gammaln(target.sum() + 1) - sp.special.gammaln(bins + 1).sum() - np.log(bins / probs).sum()


class normal(parametric_distribution_model):
    """Normal (Gaussian) distribution model.

    Args:
        x (np.ndarray): Sample data.
        w_x (np.ndarray, optional): Weights for weighted mean/std. Defaults to None.

    Attributes:
        params (np.ndarray): [mean, std].
    """

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
        """Normal probability density function."""
        return sp.stats.norm.pdf(target, loc=self.params[0], scale=self.params[1])

    def cdf(self, target: np.ndarray) -> np.ndarray:
        """Normal cumulative distribution function."""
        return sp.stats.norm.cdf(target, loc=self.params[0], scale=self.params[1])

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> float:
        """Log-likelihood of target under fitted parameters."""
        return self.likelihood_helper(target, loc=self.params[0], scale=self.params[1])

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> float:
        """Log-likelihood under null model (target mean/std)."""
        null_mean = target.mean()
        null_std = target.std(ddof=0)
        return self.likelihood_helper(target, null_mean, null_std)

    def likelihood_helper(self, target: np.ndarray, mu: float, sigma: float) -> float:
        """Helper for normal log-likelihood computation."""
        sigma = max(sigma, 1e-12)
        return sp.stats.norm(loc=mu, scale=sigma).logpdf(target.flatten()).sum()


class uniform(parametric_distribution_model):
    """Uniform distribution model.

    Args:
        x (np.ndarray): Sample data.

    Attributes:
        params (np.ndarray): [lower, upper].
    """

    def __init__(self, x: np.ndarray) -> None:
        super().__init__(x.reshape(-1, 1))
        self.params = np.array([self.x.min(), self.x.max()])
        if self.params[1] <= self.params[0]:
            raise ValueError("Uniform distribution requires upper > lower.")

    def pdf(self, target: np.ndarray) -> np.ndarray:
        """Uniform probability density function."""
        return sp.stats.uniform.pdf(target, loc=self.params[0], scale=self.params[1] - self.params[0])

    def cdf(self, target: np.ndarray) -> np.ndarray:
        """Uniform cumulative distribution function."""
        return sp.stats.uniform.cdf(target, loc=self.params[0], scale=self.params[1] - self.params[0])

    def out_of_sample_log_likelihood(self, target: np.ndarray) -> float:
        """Log-likelihood of target under fitted parameters."""
        return self.likelihood_helper(target, self.params[0], self.params[1])

    def out_of_sample_null_likelihood(self, target: np.ndarray) -> float:
        """Log-likelihood under null model (target min/max)."""
        null_lower = target.min()
        null_upper = target.max()
        return self.likelihood_helper(target, null_lower, null_upper)

    def likelihood_helper(self, target: np.ndarray, lower: float, upper: float) -> float:
        """Helper for uniform log-likelihood computation."""
        if upper <= lower:
            raise ValueError("Uniform distribution requires upper > lower.")
        target = target.flatten()
        in_bounds = (target >= lower) & (target <= upper)
        return -np.inf if not np.all(in_bounds) else -len(target) * np.log(upper - lower)
