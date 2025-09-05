"""
Linear and Quadratic Discriminant Analysis (LDA/QDA) classifiers.

This module implements classical discriminant analysis methods:

- `linear_discriminant_analysis` (LDA): assumes shared covariance across classes.
  Provides a low-dimensional discriminant projection and class posterior
  probabilities under a Gaussian generative model with equal covariance.
- `quadratic_discriminant_analysis` (QDA): allows class-specific covariance
  matrices with optional ridge-type regularization for numerical stability.

Utility:
    `_quad_form_rows(X, A)` efficiently computes row-wise quadratic forms
    xᵢᵀ A xᵢ using `numpy.einsum`, which is used by QDA.

Factory Aliases:
    LinearDiscriminantAnalysis:  Wrapper for `linear_discriminant_analysis`
        via `ClassificationModel`.
    QuadraticDiscriminantAnalysis:  Wrapper for `quadratic_discriminant_analysis`
        via `ClassificationModel`.

Typical usage example:

    >>> from cleands.Classification.lda import LinearDiscriminantAnalysis
    >>> model = LinearDiscriminantAnalysis(x, y)
    >>> model.tidy
    >>> model.glance
"""

from ..base import *
from ..utils import *
from functools import partial
import numpy as np
from typing import Optional


def _quad_form_rows(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compute row-wise quadratic forms xᵢᵀ A xᵢ.

    Uses `numpy.einsum` for speed and numerical stability.

    Args:
        X (np.ndarray): Matrix of shape (n_samples, n_features) containing row
            vectors xᵢ.
        A (np.ndarray): Square matrix of shape (n_features, n_features).

    Returns:
        np.ndarray: Vector of length n_samples with values xᵢᵀ A xᵢ.
    """
    # einsum is fast and stable for this pattern
    return np.einsum("ij,jk,ik->i", X, A, X)


class linear_discriminant_analysis(classification_model):
    """Linear Discriminant Analysis (LDA) classifier.

    Fits class means and a pooled within-class covariance (shared across
    classes) to derive a linear discriminant subspace and compute class
    posterior probabilities under a Gaussian generative model.

    Attributes:
        mean_vectors (list[np.ndarray]): Per-class mean vectors of shape
            (n_features, 1).
        priors (np.ndarray): Class prior probabilities of shape (n_classes,).
        Sigma_within (np.ndarray): Pooled within-class covariance matrix of
            shape (n_features, n_features).
        overall_mean (np.ndarray): Overall mean vector of shape (n_features, 1).
        Sigma_between (np.ndarray): Between-class scatter matrix of shape
            (n_features, n_features).
        eigenvalues (np.ndarray): Eigenvalues from generalized eigenproblem
            `inv(S_w) S_b`, sorted descending.
        eigenvectors (np.ndarray): Top `n_classes - 1` eigenvectors forming the
            discriminant projection matrix of shape (n_features, n_classes-1).
    """

    def __init__(self, x: np.array, y: np.array) -> None:
        """Initialize and fit the LDA model.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).
            y (np.ndarray): Integer class labels (n_samples,).
        """
        super(linear_discriminant_analysis, self).__init__(x, y)
        self.mean_vectors = [x[y == i].mean(0).reshape(-1, 1) for i in range(self.n_classes)]
        self.priors = itemprob(y)
        self.Sigma_within = np.zeros((self.n_feat, self.n_feat))
        for i, mean in enumerate(self.mean_vectors):
            for row in x[y == i]:
                row = row.reshape(-1, 1)
                self.Sigma_within += (row - mean) @ (row - mean).T
        self.overall_mean = x.mean(0).reshape(-1, 1)
        self.Sigma_between = np.zeros((self.n_feat, self.n_feat))
        for i, mean in enumerate(self.mean_vectors):
            n_class = x[y == i, :].shape[0]
            self.Sigma_between += n_class * (mean - self.overall_mean) @ (mean - self.overall_mean).T
        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.linalg.solve(self.Sigma_within, self.Sigma_between))
        self.eigenvalues = np.flip(self.eigenvalues)
        self.eigenvectors = np.fliplr(self.eigenvectors)[:, :self.n_classes - 1]
        self.eigenvectors = self.eigenvectors

    def discriminant(self, target: np.array) -> np.array:
        """Project data into the discriminant space.

        Args:
            target (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Discriminant scores of shape (n_samples, n_classes-1).
        """
        return target @ self.eigenvectors

    def predict_proba(self, target: np.array) -> np.array:
        """Compute posterior probabilities for each class.

        Implements the LDA log-posterior up to a constant and returns
        softmax-normalized probabilities.

        Args:
            target (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        outp = np.zeros((target.shape[0], self.n_classes))
        Sw_inv = np.linalg.inv(self.Sigma_within)
        for i, mean_vec in enumerate(self.mean_vectors):
            for j, x in enumerate(target):
                tmp = x.reshape(-1, 1) - mean_vec
                outp[j, i] = -0.5 * tmp.T @ Sw_inv @ tmp + np.log(self.priors[i])
        outp -= outp.max(1).reshape(-1, 1)
        outp = np.exp(outp)
        outp /= outp.sum(1).reshape(-1, 1)
        return outp


class quadratic_discriminant_analysis(classification_model):
    """Quadratic Discriminant Analysis (QDA) classifier.

    QDA models each class with its own Gaussian distribution:
    x | y = k ~ N(μ_k, Σ_k). Predictions use the quadratic log-density
    with class-specific covariance, allowing non-linear decision boundaries.

    Attributes:
        classes (np.ndarray): Sorted unique class labels of shape (n_classes,).
        n_classes (int): Number of classes.
        priors (np.ndarray): Class prior probabilities of shape (n_classes,).
        means (np.ndarray): Per-class mean vectors, shape (n_classes, n_features).
        covs (np.ndarray): Per-class covariance matrices,
            shape (n_classes, n_features, n_features).
        inv_covs (np.ndarray): Inverses of covariance matrices, same shape as `covs`.
        log_dets (np.ndarray): Log-determinants of covariance matrices, shape (n_classes,).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        reg: float = 1e-6,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize and fit the QDA model.

        Uses (optionally weighted) MLEs for class means and covariances with
        ridge-type regularization to ensure positive definiteness.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).
            y (np.ndarray): Integer class labels (n_samples,).
            priors (np.ndarray, optional): Class prior probabilities of shape
                (n_classes,). If None, estimated from sample weights. Defaults to None.
            reg (float, optional): Nonnegative regularization added to each
                class covariance (λ I) for numerical stability. Defaults to 1e-6.
            sample_weight (np.ndarray, optional): Nonnegative weights of shape
                (n_samples,). If None, uses uniform weights. Defaults to None.

        Raises:
            ValueError: On invalid `sample_weight` shape or priors shape/values.
            np.linalg.LinAlgError: If a class covariance is not PD even after
                regularization.
        """
        super().__init__(x, y)

        # Validate sample weights
        if sample_weight is None:
            w = np.ones(self.n_obs, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.ndim != 1 or w.shape[0] != self.n_obs:
                raise ValueError("sample_weight must be shape (n_obs,).")
            if np.sum(w) <= 0:
                raise ValueError("sample_weight must sum to a positive value.")

        # Classes and encoded indices
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes = classes                    # store labels
        self.n_classes = classes.size             # <-- keep this an INT
        k_classes = self.n_classes

        # Priors
        if priors is None:
            cw = np.bincount(y_idx, weights=w, minlength=k_classes).astype(float)
            priors = cw / cw.sum()
        else:
            priors = np.asarray(priors, dtype=float)
            if priors.shape != (k_classes,):
                raise ValueError(f"priors must be shape ({k_classes},)")
            if np.any(priors < 0) or priors.sum() <= 0:
                raise ValueError("priors must be nonnegative and sum to a positive value.")
            priors = priors / priors.sum()
        self.priors = priors

        # Means, covariances, inverses, log determinants
        self.means = np.zeros((k_classes, self.n_feat))
        self.covs = np.zeros((k_classes, self.n_feat, self.n_feat))
        self.inv_covs = np.zeros_like(self.covs)
        self.log_dets = np.zeros(k_classes)

        for k in range(k_classes):
            mask = (y_idx == k)
            wk = w[mask]
            Xk = x[mask]
            W = wk.sum()
            if W <= 0:
                raise ValueError(f"Class {classes[k]} has zero total weight.")

            mu = np.average(Xk, axis=0, weights=wk)
            self.means[k] = mu

            Xc = Xk - mu
            cov = (Xc.T * wk) @ Xc / W  # MLE covariance
            if reg > 0:
                cov = cov + reg * np.eye(self.n_feat)

            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + max(reg, 1e-8) * np.eye(self.n_feat)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError(
                        f"Covariance for class {classes[k]} is not PD even after regularization."
                    )

            self.covs[k] = cov
            self.inv_covs[k] = np.linalg.inv(cov)
            self.log_dets[k] = logdet

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Compute unnormalized class scores (log-posterior up to a constant).

        For each class k:
            score_k(x) = log π_k - 0.5 * log|Σ_k| - 0.5 * (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k)

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Scores of shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if x.shape[1] != self.n_feat:
            raise ValueError(f"X has {x.shape[1]} features; expected {self.n_feat}")

        k_classes = self.n_classes  # int
        scores = np.zeros((x.shape[0], k_classes))
        for k in range(k_classes):
            mu = self.means[k]
            inv_cov = self.inv_covs[k]
            quad = _quad_form_rows(x - mu, inv_cov)
            scores[:, k] = np.log(self.priors[k]) - 0.5 * self.log_dets[k] - 0.5 * quad
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Compute class posterior probabilities via softmax over scores.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        scores = self.decision_function(x)
        m = scores.max(axis=1, keepdims=True)
        P = np.exp(scores - m)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def _check_is_fitted(self) -> None:
        """Ensure the model has been fitted before scoring/predicting.

        Raises:
            AttributeError: If any required fitted attributes are missing.
        """
        attrs = ("means", "covs", "inv_covs", "log_dets", "priors")
        missing = [a for a in attrs if not hasattr(self, a)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} is not fitted. Missing: {', '.join(missing)}"
            )


class LinearDiscriminantAnalysis(ClassificationModel):
    """Convenience wrapper for Linear Discriminant Analysis (LDA).

    LDA projects data into a lower-dimensional space that maximizes
    class separability, assuming normally distributed features with
    equal covariance matrices. Provides a formula/DataFrame interface
    for the :class:`linear_discriminant_analysis`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`linear_discriminant_analysis`.

    Example:
        >>> model = LinearDiscriminantAnalysis.from_formula("y ~ x1 + x2", data=df)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = linear_discriminant_analysis


class QuadraticDiscriminantAnalysis(ClassificationModel):
    """Convenience wrapper for Quadratic Discriminant Analysis (QDA).

    QDA is similar to LDA but allows each class to have its own
    covariance matrix, resulting in quadratic rather than linear
    decision boundaries. Provides a formula/DataFrame interface
    for the :class:`quadratic_discriminant_analysis`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`quadratic_discriminant_analysis`.

    Example:
        >>> model = QuadraticDiscriminantAnalysis.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.classify(df[["x1", "x2", "x3"]])
        >>> model.predict_proba(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = quadratic_discriminant_analysis
