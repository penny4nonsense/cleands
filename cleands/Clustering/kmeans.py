"""
k-means clustering models.

This module implements a basic version of k-means clustering (`simple_k_means`)
and a multi-start variant (`k_means`). It also includes helper functions to
evaluate the optimal number of clusters using the total within-group sum of
squares (TWSS) and the "elbow method".

Classes:
    simple_k_means:
        Basic k-means clustering algorithm with iterative centroid updates.

    k_means:
        Extension of `simple_k_means` that runs multiple random initializations
        (n_start) and selects the solution with the lowest TWSS.

Functions:
    total_within_group_sum_of_squares_for_different_k:
        Computes TWSS across different k values (1..k_max).

    select_k:
        Heuristic to choose optimal k using the elbow method based on TWSS ratios.
"""

from ..utils import *
from ..base import clustering_model, ClusteringModel
import numpy as np
from functools import partial


class simple_k_means(clustering_model):
    """Basic k-means clustering.

    Iteratively assigns points to the nearest cluster centroid and updates
    centroids until convergence or the maximum number of iterations is reached.

    Args:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int, optional): Maximum iterations. Defaults to 100.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.

    Attributes:
        n_clusters (int): Number of clusters.
        iters (int): Number of iterations performed.
        _means (np.ndarray): Cluster centroids of shape (k, n_features).
    """

    def __init__(self, x, k, max_iters=100, seed=None):
        super(simple_k_means, self).__init__(x)
        self.n_clusters = k
        if seed is not None:
            np.random.seed(seed)
        outp = np.random.randint(k, size=(x.shape[0],))
        inpt = np.zeros((x.shape[0],))
        means = self._calc_means(outp)
        for j in range(max_iters):
            if (inpt == outp).all():
                break
            inpt = outp.copy()
            means_new = self._calc_means(inpt)
            means = self._replace_means(means, means_new)
            outp = simple_k_means._get_groups(x, means)
        self.iters = j
        self._means = means

    @staticmethod
    def _get_groups(x, means):
        """Assign points to the nearest centroid.

        Args:
            x (np.ndarray): Data matrix of shape (n_samples, n_features).
            means (np.ndarray): Centroid matrix of shape (k, n_features).

        Returns:
            np.ndarray: Cluster assignments of shape (n_samples,).
        """
        outp = [x - means[i, :] for i in range(means.shape[0])]
        outp = [(item ** 2).sum(1) for item in outp]
        return np.array(outp).argmin(0)

    def _replace_means(self, means, means_new):
        """Replace centroids with new ones if valid.

        Args:
            means (np.ndarray): Current centroids.
            means_new (np.ndarray): Newly computed centroids.

        Returns:
            np.ndarray: Updated centroid matrix.
        """
        for i in range(self.n_clusters):
            if not np.isnan(means_new[i, :]).all():
                means[i, :] = means_new[i, :]
        return means

    def cluster(self, newx):
        """Cluster new data based on learned centroids.

        Args:
            newx (np.ndarray): New data matrix of shape (m, n_features).

        Returns:
            np.ndarray: Cluster assignments of shape (m,).
        """
        return simple_k_means._get_groups(newx, self._means)


class k_means(simple_k_means):
    """Multi-start k-means clustering.

    Runs multiple random initializations (``n_start``) of ``simple_k_means`` and
    selects the model with the lowest total within-group sum of squares (TWSS).

    Args:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int, optional): Maximum iterations for each run. Defaults to 100.
        seed (int | None, optional): Random seed. Defaults to None.
        n_start (int, optional): Number of random initializations. Defaults to 10.

    Attributes:
        n_clusters (int): Number of clusters.
        iters (int): Iterations used by the best model.
        means (np.ndarray): Centroids of the best solution.
    """

    def __init__(self, x, k, max_iters=100, seed=None, n_start=10):
        if seed is not None:
            np.random.seed(seed)
        clustering_model.__init__(self, x)
        outp = {}
        for i in range(n_start):
            model = simple_k_means(x, k=k, max_iters=max_iters, seed=None)
            outp[model.total_within_group_sum_of_squares] = model
        model = outp[min(outp.keys())]
        self.iters = model.iters
        self.means = model.means
        self.n_clusters = k


def total_within_group_sum_of_squares_for_different_k(x: np.ndarray, k_max: int = 10, *args, **kwargs):
    """Compute TWSS across different values of ``k``.

    Args:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).
        k_max (int, optional): Maximum number of clusters to evaluate. Defaults to 10.
        ``*args``, ``**kwargs``: Passed to ``k_means``.

    Returns:
        np.ndarray: TWSS values indexed by ``k`` (1..k_max).
    """
    outp = np.full(k_max + 1, np.nan)
    for k in range(1, k_max + 1):
        model = k_means(x, k, *args, **kwargs)
        outp[k] = model.total_within_group_sum_of_squares
    return outp


def select_k(x: np.ndarray, k_max: int = 10, *args, **kwargs):
    """Select the optimal number of clusters using the elbow method.

    Uses the ratio of successive TWSS differences to detect the "elbow point."

    Args:
        x (np.ndarray): Data matrix of shape (n_samples, n_features).
        k_max (int, optional): Maximum number of clusters to test. Defaults to 10.
        ``*args``, ``**kwargs``: Passed to ``k_means``.

    Returns:
        int: Estimated optimal number of clusters.
    """
    twss = total_within_group_sum_of_squares_for_different_k(x, k_max=k_max, *args, **kwargs)
    dwss = -np.diff(twss)
    dwss[0] = np.nansum(dwss) / np.log(k_max)
    ratio = dwss[:-1] / dwss[1:]
    return ratio.argmax() + 1


class kMeans(ClusteringModel):
    """Convenience wrapper for k-means clustering.

    The k-means algorithm partitions observations into a fixed number of
    clusters by minimizing within-cluster sum of squares. This wrapper
    provides a formula/DataFrame interface for the
    :class:`k_means`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`k_means`.

    Example:
        >>> model = kMeans.from_formula("~ x1 + x2", data=df, k=3)
        >>> clusters = model.cluster_from_df(df)
        >>> model.means  # cluster centroids
    """

    MODEL_TYPE = k_means
