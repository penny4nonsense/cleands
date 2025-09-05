"""
Ensemble models for prediction tasks.

This module implements bootstrap aggregating (bagging) methods for regression
and classification, extending base learners such as least squares, logistic
regression, and recursive partitioning (decision trees). It also defines
convenience wrappers for random forest–style ensembles.

Classes:
    bootstrap_model:
        Base container for storing bootstrap resamples of a supervised model.

    bagging_least_squares_regressor:
        Bagging ensemble of least squares regressors. Produces averaged
        parameter estimates and predictions with bootstrap variance estimation.

    bagging_logistic_regressor:
        Bagging ensemble of logistic regressors. Averages coefficients across
        bootstrap samples to improve stability and predictive performance.

    bagging_recursive_partitioning_regressor:
        Bagging ensemble of recursive partitioning regressors (decision trees).
        Supports optional feature subsampling to mimic random forests.
        Selects the bootstrap tree closest to the full-sample fit for final
        structure, while predictions are aggregated across bootstraps.

Functions:
    random_forest_regressor:
        Partial constructor of bagging_recursive_partitioning_regressor with
        feature subsampling enabled (random_x=True), equivalent to a regression
        random forest.

    BaggingLogisticRegressor:
        Partial wrapper that constructs a SupervisedModel around
        bagging_logistic_regressor for formula-notation usage.

    BaggingRecursivePartitioningRegressor:
        Partial wrapper for bagging_recursive_partitioning_regressor.

    RandomForestRegressor:
        Partial wrapper for random_forest_regressor.

Notes:
    - All ensembles rely on the `bootstrap` utility from utils.py, which
      generates bootstrap resamples of the training data.
    - These models are intended as drop-in replacements for their base learners
      but with improved stability through aggregation.
"""


import numpy as np

from .glm import least_squares_regressor, logistic_regressor
from .tree import recursive_partitioning_regressor
from ..utils import *
from ..base import PredictionModel
from functools import partial

class bootstrap_model(ABC):
    """Generic bootstrap wrapper for supervised models.

    Creates and stores a collection of bootstrap-fitted models for a given
    `model_type`, providing a common structure you can extend.

    Attributes:
        model_type (Type[supervised_model]): The underlying model class to fit.
        model (supervised_model): The model fit on the original (non-bootstrapped) data.
        seed (Optional[int]): Seed for reproducibility of bootstrap sampling.
        n_boot (int): Number of bootstrap resamples.
        bootstraps (list[supervised_model]): List of models fit on bootstrap samples.
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 model_type: Type[supervised_model],
                 seed: Optional[int] = None,
                 bootstraps: int = 1000,
                 *args, **kwargs) -> None:
        """Initialize the bootstrap collection.

        Args:
            x (np.ndarray): Training features of shape (n_obs, n_features).
            y (np.ndarray): Training targets of shape (n_obs,) or (n_obs, 1).
            model_type (Type[supervised_model]): Class to bootstrap (must accept (x, y)).
            seed (Optional[int]): Random seed for resampling.
            bootstraps (int): Number of bootstrap draws/models to fit.
            *args: Unused, kept for compatibility.
            **kwargs: Unused, kept for compatibility.
        """
        self.model_type: Type[supervised_model] = model_type
        self.model: supervised_model = model_type(x, y)
        self.seed: Optional[int] = seed
        self.n_boot: int = bootstraps
        self.bootstraps: list[supervised_model] = bootstrap(model_type, x, y, seed=seed, bootstraps=bootstraps)


class bagging_least_squares_regressor(least_squares_regressor):
    """Bagged OLS regressor.

    Trains many OLS models on bootstrap resamples and aggregates:
    - `params` is set to the mean of bootstrap coefficient vectors.
    - Predictions are the mean of bootstrap predictions.
    - `vcov_params` is the bootstrap covariance of coefficients.
    """

    def __init__(self, x, y, seed=None, bootstraps=1000):
        """Fit a bagged OLS model.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training targets.
            seed (Optional[int]): Random seed for bootstrapping.
            bootstraps (int): Number of bootstrap resamples.
        """
        super().__init__(x, y)
        self.seed = seed
        self.n_boot = bootstraps
        self.bootstraps = bootstrap(least_squares_regressor, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)

    def predict(self, newx):
        """Predict by averaging bootstrap model predictions.

        Args:
            newx (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Mean prediction across bootstraps.
        """
        return np.array([item.predict(newx) for item in self.bootstraps]).mean(0)

    @property
    def vcov_params(self):
        """Bootstrap covariance of parameter estimates.

        Returns:
            np.ndarray: (p x p) covariance matrix from bootstrap params.
        """
        return (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot


class bagging_logistic_regressor(logistic_regressor):
    """Bagged logistic regressor.

    Trains many logistic models on bootstrap resamples and aggregates:
    - `params` is the mean of bootstrap coefficient vectors.
    - Predictions are the mean of bootstrap probability predictions.
    - `vcov_params` is the bootstrap covariance of coefficients.
    """

    def __init__(self, x, y, seed=None, bootstraps=1000):
        """Fit a bagged logistic model.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Binary targets (0/1) or probabilities shape-compatible.
            seed (Optional[int]): Random seed for bootstrapping.
            bootstraps (int): Number of bootstrap resamples.
        """
        super().__init__(x, y)
        self.seed = seed
        self.n_boot = bootstraps
        self.bootstraps = bootstrap(logistic_regressor, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)

    def predict(self, target: np.ndarray) -> np.ndarray:
        """Predict by averaging bootstrap probability predictions.

        Args:
            target (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Mean predicted probabilities across bootstraps.
        """
        return np.array([item.predict(target) for item in self.bootstraps]).mean(0)

    @property
    def vcov_params(self):
        """Bootstrap covariance of parameter estimates.

        Returns:
            np.ndarray: (p x p) covariance matrix from bootstrap params.
        """
        return (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot


class bagging_recursive_partitioning_regressor(recursive_partitioning_regressor):
    """Bagged regression trees (recursive partitioning).

    Fits multiple trees on bootstrap samples, then chooses the single tree
    whose fitted values are closest (MSE) to the bagged fit; the chosen tree’s
    learned structure is copied onto `self` for efficient prediction.

    Notes:
        - If `random_x=True` via the `random_forest_regressor` partial, each tree
          samples a subset of features at each split (random subspace), similar
          to Random Forests.
    """

    def __init__(self, x, y, seed=None, bootstraps=1000, sign_level=0.95, max_level=None, random_x=False):
        """Fit bagged recursive partitioning model and select a representative tree.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training targets.
            seed (Optional[int]): Random seed for bootstrapping.
            bootstraps (int): Number of bootstrap resamples (trees).
            sign_level (float): Split significance level used by the tree learner.
            max_level (Optional[int]): Maximum depth (levels) of the tree.
            random_x (bool): Whether to randomize feature selection per split.
        """
        super().__init__(x, y, max_level=1)
        self.seed, self.n_boot = seed, bootstraps
        model = lambda x, y: recursive_partitioning_regressor(
            x, y, sign_level=sign_level, max_level=max_level, random_x=random_x
        )
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)

        # Choose the tree whose in-sample fit best matches the bagged fit (MSE).
        fit = self.fitted
        indx = np.array([np.mean((item.fitted - fit) ** 2) for item in self.bootstraps]).argmin()
        model = self.bootstraps[indx]
        for key, value in vars(model).items():
            setattr(self, key, value)

    def predict(self, newx, fitted: bool = False):
        """Predict by averaging over bootstrap trees (or on original feature order if fitted=True).

        Args:
            newx (np.ndarray): Feature matrix for prediction.
            fitted (bool): If True, `newx` is assumed already aligned to the selected
                tree’s internal column order; otherwise, alignment is handled.

        Returns:
            np.ndarray: Mean prediction across bootstrap trees.
        """
        return np.array([item.predict(newx, fitted) for item in self.bootstraps]).mean(0)


# Convenience constructors / aliases
class random_forest_regressor(bagging_recursive_partitioning_regressor):
    """Random Forest regressor using recursive partitioning trees.

    This class implements a random forest by combining multiple
    recursive partitioning trees with bootstrapping and feature
    sub-sampling. At each split, only a random subset of predictors
    is considered, reducing correlation between trees.

    Args:
        x (np.ndarray): Training feature matrix of shape (n_obs, n_feat).
        y (np.ndarray): Training response vector of shape (n_obs,).
        seed (int, optional): Random seed for reproducibility.
        bootstraps (int, optional): Number of bootstrap resamples. Defaults to 1000.
        sign_level (float, optional): Significance level for splitting. Defaults to 0.95.
        max_level (int, optional): Maximum tree depth. If None, grows until no valid splits remain.

    Inherits:
        bagging_recursive_partitioning_regressor: Provides the ensemble
        logic and base tree structure.
    """


class BaggingLogisticRegressor(PredictionModel):
    """Convenience wrapper for bagging logistic regression.

    This class applies the unified :class:`PredictionModel` interface
    to the :class:`bagging_logistic_regressor`, enabling construction
    from formulas and DataFrames.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`bagging_logistic_regressor`.

    Example:
        >>> model = BaggingLogisticRegressor.from_formula("y ~ x1 + x2", data=df)
        >>> model.predict(df[["x1", "x2"]])
    """

    MODEL_TYPE = bagging_logistic_regressor


class BaggingRecursivePartitioningRegressor(PredictionModel):
    """Convenience wrapper for bagging recursive partitioning regression.

    Provides a formula/DataFrame interface for the
    :class:`bagging_recursive_partitioning_regressor`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`bagging_recursive_partitioning_regressor`.

    Example:
        >>> model = BaggingRecursivePartitioningRegressor.from_formula("y ~ x1 + x2", data=df)
        >>> model.predict(df[["x1", "x2"]])
    """

    MODEL_TYPE = bagging_recursive_partitioning_regressor


class RandomForestRegressor(PredictionModel):
    """Convenience wrapper for random forest regression.

    Provides a formula/DataFrame interface for the
    :class:`random_forest_regressor`, which implements an ensemble of
    recursive partitioning trees with random feature sub-sampling.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`random_forest_regressor`.

    Example:
        >>> model = RandomForestRegressor.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.predict(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = random_forest_regressor
