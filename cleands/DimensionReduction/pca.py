"""
pca.py
------

Implements dimension reduction models including Principal Components Analysis (PCA)
and Canonical Correlation Analysis (CCA).

Classes:
    principal_components_analysis:
        Standard PCA for unsupervised dimension reduction, with centering and scaling options.
    canonical_correlation_analysis:
        Supervised dimension reduction that extracts correlated projections between two datasets.

Functions:
    select_k:
        Eigenvalue ratio test to automatically select the number of components.

Notes:
    - PCA is fit via Singular Value Decomposition (SVD).
    - CCA is fit via whitening of covariance matrices and SVD of the cross-covariance.
"""

from ..utils import *
from ..base import dimension_reduction_model, supervised_dimension_reduction_model, DimensionReductionModel
import numpy as np
from functools import partial


class principal_components_analysis(dimension_reduction_model):
    """Principal Components Analysis (PCA).

    Reduces dimensionality by projecting data onto top `k` principal components
    that maximize variance. Provides explained variance statistics.

    Args:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).
        k (int): Number of components to extract. Must be in [1, min(n_obs, n_feat)].
        center (bool, optional): Whether to mean-center the data. Defaults to True.
        scale (bool, optional): Whether to scale variables to unit variance. Defaults to False.

    Attributes:
        mean (np.ndarray): Feature means (if centered).
        scale (np.ndarray): Feature scales (if scaled).
        singular_values (np.ndarray): Singular values from SVD.
        rotation (np.ndarray): Principal component loadings (p, k).
        components (np.ndarray): Projected data (scores), shape (n, k).
        explained_variance (np.ndarray): Variance explained by each component.
        explained_variance_ratio (np.ndarray): Proportion of variance explained.
        k (int): Number of components retained.
    """

    def __init__(self, x: np.ndarray, k: int, *, center: bool = True, scale: bool = False) -> None:
        super().__init__(x)

        if not (1 <= k <= min(self.n_obs, self.n_feat)):
            raise ValueError("k must be between 1 and min(n_obs, n_feat)")

        # training stats
        self.mean = x.mean(0) if center else np.zeros(self.n_feat, dtype=float)
        std = x.std(0, ddof=1)
        if scale:
            self.scale = np.where(std == 0.0, 1.0, std)  # avoid div-by-zero
        else:
            self.scale = np.ones(self.n_feat, dtype=float)

        x_cs = (x - self.mean) / self.scale

        # SVD decomposition
        U, S, Vt = np.linalg.svd(x_cs, full_matrices=False)
        q = S.shape[0]
        self.singular_values = S
        self.rotation = Vt.T[:, :k]
        self.components = (U * S)[:, :k]
        self.k = k

        # variance bookkeeping
        denom = max(self.n_obs - 1, 1)
        ev = (S**2) / denom
        self.explained_variance = ev
        total_var = x_cs.var(axis=0, ddof=1).sum()
        self.explained_variance_ratio = ev / total_var if total_var > 0 else np.zeros_like(ev)

    def reduce(self, target: np.ndarray) -> np.ndarray:
        """Project new data into the PCA space.

        Args:
            target (np.ndarray): New data of shape (m, n_features).

        Returns:
            np.ndarray: Reduced representation of shape (m, k).
        """
        if target.ndim != 2 or target.shape[1] != self.rotation.shape[0]:
            raise ValueError(f"target must be shape (m, {self.rotation.shape[0]})")
        x_cs = (target - self.mean) / self.scale
        return x_cs @ self.rotation


def select_k(eigs: np.ndarray, k_max: int | None = None, include_zero: bool = True, allow_zero: bool = False) -> int:
    """Eigenvalue ratio test for choosing number of components.

    Uses the Ahn-Horenstein ratio-based test (2013) for selecting k. Optionally
    includes a "zeroth" eigenvalue based on the average.

    Args:
        eigs (np.ndarray): Eigenvalues or variances in descending order.
        k_max (int, optional): Maximum number of components to consider. Defaults to all.
        include_zero (bool, optional): Include artificial λ₀ for ratio. Defaults to True.
        allow_zero (bool, optional): Allow selection of k=0. Defaults to False.

    Returns:
        int: Selected number of components.
    """
    lam = np.asarray(eigs, dtype=float)
    lam = lam[~np.isnan(lam)]
    if lam.size < 2:
        return 0 if allow_zero else 1

    lam = np.sort(lam)[::-1]
    if k_max is None:
        k_max = lam.size
    k_max = max(2, min(k_max, lam.size))

    lam_use = lam[:k_max]
    if include_zero:
        lam0 = lam.sum() / k_max
        lam_aug = np.concatenate(([lam0], lam_use))
        ratios = lam_aug[:-1] / np.maximum(lam_aug[1:], 1e-15)
        k_hat = int(np.argmax(ratios))
        if not allow_zero:
            k_hat = max(1, k_hat)
        return k_hat
    else:
        ratios = lam_use[:-1] / np.maximum(lam_use[1:], 1e-15)
        return int(np.argmax(ratios) + 1)


class canonical_correlation_analysis(supervised_dimension_reduction_model):
    """Canonical Correlation Analysis (CCA).

    Finds linear projections of X and Y that maximize their correlation.
    Useful for studying relationships between two multivariate datasets.

    Args:
        X (np.ndarray): Predictor data matrix of shape (n, p).
        Y (np.ndarray): Response data matrix of shape (n, q).
        k (int): Number of canonical variates to compute.
        center (bool, optional): Center data. Defaults to True.
        scale (bool, optional): Scale data to unit variance. Defaults to False.
        reg (float, optional): Regularization term added to covariance diagonals. Defaults to 1e-6.

    Attributes:
        mean_x (np.ndarray): Mean of X features.
        mean_y (np.ndarray): Mean of Y features.
        scale_x (np.ndarray): Scaling factors for X.
        scale_y (np.ndarray): Scaling factors for Y.
        canonical_correlations (np.ndarray): Canonical correlations.
        rotation_x (np.ndarray): Canonical directions for X (p, k).
        rotation_y (np.ndarray): Canonical directions for Y (q, k).
        components_x (np.ndarray): Canonical variates for X (n, k).
        components_y (np.ndarray): Canonical variates for Y (n, k).
        k (int): Number of canonical variates retained.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        k: int,
        *,
        center: bool = True,
        scale: bool = False,
        reg: float = 1e-6,
    ) -> None:
        # initialize supervised base
        super().__init__(X, Y)

        n, p = X.shape
        n2, q = Y.shape
        if n != n2:
            raise ValueError("X and Y must have the same number of rows.")
        k_max = min(p, q, max(n - 1, 1))
        if not (1 <= k <= k_max):
            raise ValueError(f"k must be in [1, {k_max}] given shapes X{X.shape}, Y{Y.shape}.")

        # training stats
        self.mean_x = X.mean(0) if center else np.zeros(p, dtype=float)
        self.mean_y = Y.mean(0) if center else np.zeros(q, dtype=float)

        std_x = X.std(0, ddof=1)
        std_y = Y.std(0, ddof=1)
        self.scale_x = np.where(std_x == 0.0, 1.0, std_x) if scale else np.ones(p, dtype=float)
        self.scale_y = np.where(std_y == 0.0, 1.0, std_y) if scale else np.ones(q, dtype=float)

        Xc = (X - self.mean_x) / self.scale_x
        Yc = (Y - self.mean_y) / self.scale_y

        # sample covariance blocks (ddof=1)
        d = max(n - 1, 1)
        Sxx = (Xc.T @ Xc) / d
        Syy = (Yc.T @ Yc) / d
        Sxy = (Xc.T @ Yc) / d

        # symmetrize then eigendecompose
        Sxx = (Sxx + Sxx.T) * 0.5
        Syy = (Syy + Syy.T) * 0.5

        evalx, evecx = np.linalg.eigh(Sxx)
        evaly, evecy = np.linalg.eigh(Syy)

        tiny = np.finfo(float).tiny
        evalx = np.clip(evalx + reg, tiny, None)
        evaly = np.clip(evaly + reg, tiny, None)

        inv_sqrt_x = evecx @ (np.diag(1.0 / np.sqrt(evalx)) @ evecx.T)
        inv_sqrt_y = evecy @ (np.diag(1.0 / np.sqrt(evaly)) @ evecy.T)
        inv_sqrt_x = (inv_sqrt_x + inv_sqrt_x.T) * 0.5
        inv_sqrt_y = (inv_sqrt_y + inv_sqrt_y.T) * 0.5

        # whitened cross-covariance
        M = inv_sqrt_x @ Sxy @ inv_sqrt_y

        # SVD on whitened matrix
        U, Sigma, Vt = np.linalg.svd(M, full_matrices=False)

        self.canonical_correlations = Sigma[:k]

        # canonical directions
        A = inv_sqrt_x @ U[:, :k]
        B = inv_sqrt_y @ Vt.T[:, :k]

        # normalize to unit variance
        AtS = A.T @ Sxx
        Aj_var = np.sqrt(np.maximum(np.diag(AtS @ A), tiny))
        A = A / Aj_var.reshape(1, -1)
        BtS = B.T @ Syy
        Bj_var = np.sqrt(np.maximum(np.diag(BtS @ B), tiny))
        B = B / Bj_var.reshape(1, -1)

        self.rotation_x = A
        self.rotation_y = B
        self.k = k

        # canonical variates
        self.components_x = Xc @ self.rotation_x
        self.components_y = Yc @ self.rotation_y

    def reduce_X(self, x_new: np.ndarray) -> np.ndarray:
        """Project new X data into the canonical variate space.

        Args:
            x_new (np.ndarray): New X data, shape (m, p).

        Returns:
            np.ndarray: Canonical variates of shape (m, k).
        """
        if x_new.ndim != 2 or x_new.shape[1] != self.rotation_x.shape[0]:
            raise ValueError(f"x_new must be shape (m, {self.rotation_x.shape[0]}).")
        xc = (x_new - self.mean_x) / self.scale_x
        return xc @ self.rotation_x

    def reduce_Y(self, y_new: np.ndarray) -> np.ndarray:
        """Project new Y data into the canonical variate space.

        Args:
            y_new (np.ndarray): New Y data, shape (m, q).

        Returns:
            np.ndarray: Canonical variates of shape (m, k).
        """
        if y_new.ndim != 2 or y_new.shape[1] != self.rotation_y.shape[0]:
            raise ValueError(f"y_new must be shape (m, {self.rotation_y.shape[0]}).")
        yc = (y_new - self.mean_y) / self.scale_y
        return yc @ self.rotation_y


class PrincipalComponentsAnalysis(DimensionReductionModel):
    """Convenience wrapper for principal components analysis (PCA).

    PCA reduces the dimensionality of data by projecting it onto a set of
    orthogonal components that capture the maximum variance. This wrapper
    provides a formula/DataFrame interface for the
    :class:`principal_components_analysis`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`principal_components_analysis`.

    Example:
        >>> model = PrincipalComponentsAnalysis.from_formula("~ x1 + x2 + x3", data=df)
        >>> Z = model.reduce(df[["x1", "x2", "x3"]])   # reduced components
        >>> model.glance  # variance explained and diagnostics
    """

    MODEL_TYPE = principal_components_analysis
