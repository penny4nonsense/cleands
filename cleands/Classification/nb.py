"""
Naive Bayes classifiers (Gaussian and Multinomial).

This module provides a small hierarchy for Naive Bayes:

- `naive_bayes`: abstract base implementing common plumbing:
  priors (optionally weighted), log-prior caching, input checks,
  and `predict_proba` via log-sum-exp stabilization.
- `gaussian_naive_bayes`: continuous features modeled as independent
  univariate Gaussians per class with optional ridge variance `reg`.
- `multinomial_naive_bayes`: count/nonnegative features with Laplace
  smoothing (`alpha`) for per-class feature probabilities.

Factory Aliases:
    GaussianNaiveBayes:
        Wrapper for constructing `gaussian_naive_bayes` via `ClassificationModel`.
    MultinomialNaiveBayes:
        Wrapper for constructing `multinomial_naive_bayes` via `ClassificationModel`.

Typical usage example:
    >>> from cleands.Classification.nb import GaussianNaiveBayes
    >>> model = GaussianNaiveBayes(X, y)
    >>> model.tidy; model.glance
"""

import numpy as np
from ..base import classification_model, ClassificationModel
from ..utils import itemprob, hstack
from typing import Optional
from functools import partial


class naive_bayes(classification_model):
    """Abstract Naive Bayes classifier.

    Handles class priors (with optional sample weights), caches log-priors,
    validates inputs, and provides stable `predict_proba`. Subclasses must
    implement `log_likelihood(target)` returning class-wise log-likelihoods.

    Attributes:
        priors (np.ndarray): Class priors of shape (K,), normalized to sum to 1.
        _log_priors (np.ndarray): Log(priors + ε) cached for numerical stability.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Naive Bayes base model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_obs, n_feat).
            y (np.ndarray): Integer class labels of shape (n_obs,).
            priors (np.ndarray, optional): Class prior probabilities of shape (K,).
                If None, estimated from `y` (optionally with `sample_weight`).
            sample_weight (np.ndarray, optional): Nonnegative weights for samples,
                shape (n_obs,). Used when estimating priors if provided.

        Raises:
            ValueError: If `sample_weight` shape is invalid or has nonpositive sum;
                if provided `priors` has invalid shape or nonpositive sum.
        """
        super(naive_bayes, self).__init__(X, y)

        # --- fit priors (optionally weighted) ---
        if priors is None:
            if sample_weight is None:
                self.priors = itemprob(self.y)  # (K,)
            else:
                w = np.asarray(sample_weight, dtype=float).reshape(-1)
                if w.shape[0] != self.n_obs:
                    raise ValueError("sample_weight must have shape (n_obs,).")
                pri = np.array([w[self.y == k].sum() for k in range(self.n_classes)], dtype=float)
                s = pri.sum()
                if s <= 0:
                    raise ValueError("sum of sample weights must be > 0 for prior estimation.")
                self.priors = pri / s
        else:
            self.priors = np.asarray(priors, dtype=float).reshape(-1)
            if self.priors.shape[0] != self.n_classes:
                raise ValueError("priors must have length equal to number of classes.")
            s = self.priors.sum()
            if s <= 0:
                raise ValueError("priors must sum to a positive value.")
            self.priors = self.priors / s

        self._log_priors = np.log(self.priors + 1e-300)  # safety epsilon

    # --- subclass contract ---
    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        """Compute per-class log-likelihoods log p(x | y=k) for each row.

        Args:
            target (np.ndarray): Feature matrix of shape (n, p) or (p,).

        Returns:
            np.ndarray: Log-likelihoods of shape (n, K).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement log_likelihood(target).")

    # --- common helpers ---
    def _check_target(self, target: np.ndarray) -> np.ndarray:
        """Validate and coerce prediction input to 2D float array.

        Args:
            target (np.ndarray): Input data of shape (n, p) or (p,).

        Returns:
            np.ndarray: Array of shape (n, p), dtype float.

        Raises:
            ValueError: If number of features does not match fitted model.
        """
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        if target.shape[1] != self.n_feat:
            raise ValueError(f"target must have shape (?, {self.n_feat})")
        return target

    def predict_proba(self, target: np.ndarray) -> np.ndarray:
        """Predict class posterior probabilities.

        Implements stabilized softmax over log posterior:
            log p(y=k | x) ∝ log p(x | y=k) + log π_k

        Args:
            target (np.ndarray): Feature matrix of shape (n, p) or (p,).

        Returns:
            np.ndarray: Posterior probabilities of shape (n, K),
                rows summing to 1.
        """
        target = self._check_target(target)
        log_lik = self.log_likelihood(target)  # (n, K)
        log_post = log_lik + self._log_priors  # add log-priors
        m = np.max(log_post, axis=1, keepdims=True)
        stabilized = np.exp(log_post - m)
        probs = stabilized / stabilized.sum(axis=1, keepdims=True)
        return probs


class gaussian_naive_bayes(naive_bayes):
    r"""Gaussian Naive Bayes classifier.

    Assumes conditional independence across features with per-class Gaussian
    likelihoods.

    Args:
        X (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
        y (np.ndarray): Integer class labels of shape ``(n_samples,)``.
        priors (np.ndarray | None, optional): Class prior probabilities. If
            ``None``, uses empirical class frequencies. Defaults to ``None``.
        var_smoothing (float, optional): Small additive term to variances for
            numerical stability. Defaults to ``1e-9``.

    Attributes:
        means (np.ndarray): Per-class feature means, shape ``(n_classes, n_features)``.
        variances (np.ndarray): Per-class feature variances, same shape as ``means``.
        priors (np.ndarray): Class prior probabilities, shape ``(n_classes,)``.

    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        reg: float = 1e-9,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit per-class univariate Gaussian parameters.

        Args:
            X (np.ndarray): Feature matrix (n_obs, n_feat).
            y (np.ndarray): Integer class labels (n_obs,).
            priors (np.ndarray, optional): Class priors (K,), normalized if given.
            reg (float, optional): Nonnegative ridge added to variances. Defaults to 1e-9.
            sample_weight (np.ndarray, optional): Nonnegative weights (n_obs,).

        Notes:
            Variances are estimated as (weighted) MLE and clipped by adding `reg`.
        """
        self.reg = float(reg)
        super(gaussian_naive_bayes, self).__init__(X, y, priors=priors, sample_weight=sample_weight)

        # --- fit per-class means/vars ---
        self.means = np.zeros((self.n_classes, self.n_feat), dtype=float)
        self.vars  = np.zeros((self.n_classes, self.n_feat), dtype=float)

        if sample_weight is None:
            for k in range(self.n_classes):
                Xk = self.x[self.y == k]
                if Xk.shape[0] == 0:
                    continue
                self.means[k, :] = Xk.mean(axis=0)
                self.vars[k,  :] = Xk.var(axis=0) + self.reg
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            for k in range(self.n_classes):
                mask = (self.y == k)
                Xk = self.x[mask]
                wk = w[mask]
                if Xk.shape[0] == 0 or wk.sum() == 0:
                    continue
                mk = (wk.reshape(-1, 1) * Xk).sum(axis=0) / wk.sum()
                diff = Xk - mk
                vk = (wk.reshape(-1, 1) * (diff ** 2)).sum(axis=0) / wk.sum()
                self.means[k, :] = mk
                self.vars[k,  :] = vk + self.reg

        # Precompute constants for log-density
        self._log_const = -0.5 * (np.log(2.0 * np.pi) + np.log(self.vars))  # (K, p)

    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        """Compute log p(x | y=k) under the Gaussian NB model.

        Uses vectorized per-class, per-feature Gaussian log-densities with
        precomputed constants for efficiency.

        Args:
            target (np.ndarray): Feature matrix (n, p) or (p,).

        Returns:
            np.ndarray: Log-likelihoods of shape (n, K).
        """
        target = self._check_target(target)
        n = target.shape[0]
        K = self.n_classes
        out = np.empty((n, K), dtype=float)
        for c in range(K):
            diff = target - self.means[c, :]
            term = self._log_const[c, :] - 0.5 * (diff ** 2) / self.vars[c, :]
            out[:, c] = term.sum(axis=1)
        return out


class multinomial_naive_bayes(naive_bayes):
    """Multinomial Naive Bayes classifier.

    Suitable for count-like, sparse, or term-frequency features (nonnegative).
    Uses per-class multinomial parameters with Laplace smoothing.

    Attributes:
        alpha (float): Laplace smoothing parameter (≥ 0).
        feature_counts (np.ndarray): Per-class feature totals, shape (K, p).
        feature_probs (np.ndarray): Per-class feature probabilities, shape (K, p).
        _log_feature_probs (np.ndarray): Cached log probabilities, shape (K, p).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit per-class multinomial parameters with Laplace smoothing.

        Args:
            X (np.ndarray): Nonnegative feature matrix (n_obs, n_feat).
            y (np.ndarray): Integer class labels (n_obs,).
            priors (np.ndarray, optional): Class priors (K,), normalized if given.
            alpha (float, optional): Laplace smoothing parameter (≥0). Defaults to 1.0.
            sample_weight (np.ndarray, optional): Nonnegative weights (n_obs,).

        Raises:
            ValueError: If `X` contains negative entries or `alpha` < 0;
                if `sample_weight` shape is invalid.
        """
        if np.any(np.asarray(X) < 0):
            raise ValueError("Multinomial NB requires nonnegative features.")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        self.alpha = float(alpha)

        super(multinomial_naive_bayes, self).__init__(X, y, priors=priors, sample_weight=sample_weight)

        # --- accumulate (possibly weighted) feature counts per class ---
        self.feature_counts = np.zeros((self.n_classes, self.n_feat), dtype=float)

        if sample_weight is None:
            for k in range(self.n_classes):
                Xk = self.x[self.y == k]
                if Xk.size == 0:
                    continue
                self.feature_counts[k, :] = Xk.sum(axis=0)
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != self.n_obs:
                raise ValueError("sample_weight must have shape (n_obs,).")
            for k in range(self.n_classes):
                mask = (self.y == k)
                if not np.any(mask):
                    continue
                Xk = self.x[mask]
                wk = w[mask].reshape(-1, 1)
                self.feature_counts[k, :] = (wk * Xk).sum(axis=0)

        # --- Laplace smoothing and normalization per class ---
        smoothed = self.feature_counts + self.alpha
        class_totals = smoothed.sum(axis=1, keepdims=True)  # (K, 1)
        class_totals[class_totals == 0.0] = 1.0  # guard (empty class, alpha==0)
        self.feature_probs = smoothed / class_totals  # (K, p)
        self._log_feature_probs = np.log(self.feature_probs)  # cache

    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        """Compute log p(x | y=k) under the Multinomial NB model.

        Treats each sample row as a bag of feature counts (or TF-like weights).
        Log-likelihood is proportional to x · log θ_k (constant terms cancel).

        Args:
            target (np.ndarray): Nonnegative features, shape (n, p) or (p,).

        Returns:
            np.ndarray: Log-likelihoods of shape (n, K).

        Raises:
            ValueError: If `target` has negative entries.
        """
        target = self._check_target(target)
        if np.any(target < 0):
            raise ValueError("Multinomial NB requires nonnegative features.")
        # (n, p) @ (p,) per class -> (n, K)
        n = target.shape[0]
        K = self.n_classes
        out = np.empty((n, K), dtype=float)
        for c in range(K):
            out[:, c] = target @ self._log_feature_probs[c, :].T
        return out


class GaussianNaiveBayes(ClassificationModel):
    """Convenience wrapper for Gaussian Naive Bayes classification.

    Assumes features are conditionally independent given the class
    label and normally distributed within each class. Provides a
    formula/DataFrame interface for the
    :class:`gaussian_naive_bayes`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`gaussian_naive_bayes`.

    Example:
        >>> model = GaussianNaiveBayes.from_formula("y ~ x1 + x2", data=df)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = gaussian_naive_bayes


class MultinomialNaiveBayes(ClassificationModel):
    """Convenience wrapper for Multinomial Naive Bayes classification.

    Assumes features are conditionally independent given the class
    label and follow a multinomial distribution. Often used for
    discrete count data such as text classification. Provides a
    formula/DataFrame interface for the
    :class:`multinomial_naive_bayes`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`multinomial_naive_bayes`.

    Example:
        >>> model = MultinomialNaiveBayes.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.classify(df[["x1", "x2", "x3"]])
        >>> model.predict_proba(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = multinomial_naive_bayes
