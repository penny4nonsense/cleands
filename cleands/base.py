"""
Core base classes and abstract interfaces for the CleanDS statistical modeling framework.

This module defines the fundamental building blocks for supervised, unsupervised,
prediction, classification, clustering, and distribution models. It establishes
a consistent API across model families, providing shared properties such as
`fitted`, `residuals`, `mean_squared_error`, `aic`, and `bic`.

Key features:
    • Abstract base classes for supervised and unsupervised models.
    • Prediction and classification models with common evaluation metrics.
    • Clustering model interface with mean/cluster management utilities.
    • Distribution model protocols with PDF and CDF support.
    • Likelihood-based model mixins for deviance, AIC, BIC, and log-likelihood.
    • Variance models for parameter uncertainty (standard errors, confidence intervals).
    • Supervised wrappers (`SupervisedModel`, `PredictionModel`, `ClassificationModel`)
      for integrating with formula notation and tidy outputs.

These base definitions are designed for extensibility:
custom regression, classification, clustering, or distribution models
should inherit from the appropriate abstract class to ensure interoperability
within the CleanDS ecosystem.
"""

import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type, ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .utils import *
from .formula import parse


class learning_model(ABC):
    """Abstract base class for all learning models."""

    def __init__(self, x: np.ndarray) -> None:
        """
        Initialize a learning model.

        Args:
            x (np.ndarray): Input feature matrix of shape (n_obs, n_feat).

        Attributes:
            x (np.ndarray): Stored feature matrix.
            n_obs (int): Number of observations.
            n_feat (int): Number of features.
        """
        self.x = x
        (self.n_obs, self.n_feat) = x.shape


class supervised_model(learning_model, ABC):
    """Base class for supervised learning models with features and labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize a supervised learning model.

        Args:
            x (np.ndarray): Feature matrix of shape (n_obs, n_feat).
            y (np.ndarray): Target vector of shape (n_obs,).
        """
        super().__init__(x)
        self.y = y


class unsupervised_model(learning_model, ABC):
    """Base class for unsupervised learning models."""
    ...


class prediction_model(supervised_model, ABC):
    """Base class for supervised prediction models."""

    @abstractmethod
    def predict(self, target: np.ndarray) -> np.ndarray:
        """
        Predict outcomes for new input data.

        Args:
            target (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        ...

    @property
    def fitted(self) -> np.ndarray:
        """np.ndarray: Predictions for training data (`self.x`)."""
        return self.predict(self.x)

    @property
    def residuals(self) -> np.ndarray:
        """np.ndarray: Difference between observed and fitted values."""
        return self.y - self.fitted

    @property
    def residual_sum_of_squares(self) -> float:
        """float: Residual sum of squares (RSS)."""
        return np.sum(self.residuals ** 2)

    @property
    def mean_squared_error(self) -> float:
        """float: Mean squared error (MSE)."""
        return np.mean(self.residuals ** 2)

    def out_of_sample_mean_squared_error(self, x, y) -> float:
        """
        Compute out-of-sample mean squared error.

        Args:
            x (np.ndarray): Test feature matrix.
            y (np.ndarray): Test target vector.

        Returns:
            float: Out-of-sample MSE.
        """
        return np.mean((y - self.predict(x)) ** 2)

    @property
    def root_mean_squared_error(self) -> float:
        """float: Root mean squared error (RMSE)."""
        return np.sqrt(self.mean_squared_error)

    def out_of_sample_root_mean_squared_error(self, x, y) -> float:
        """
        Compute out-of-sample RMSE.

        Args:
            x (np.ndarray): Test feature matrix.
            y (np.ndarray): Test target vector.

        Returns:
            float: Out-of-sample RMSE.
        """
        return np.sqrt(self.out_of_sample_mean_squared_error(x, y))

    @property
    def r_squared(self) -> float:
        """float: Coefficient of determination (R²)."""
        return 1 - self.residuals.var() / self.y.var()

    @property
    def adjusted_r_squared(self) -> float:
        """float: Adjusted R² that accounts for model complexity."""
        return 1 - (1 - self.r_squared) * (self.n_obs - 1) / self.degrees_of_freedom

    @property
    def degrees_of_freedom(self) -> int:
        """int: Degrees of freedom = n_obs - n_feat."""
        return self.n_obs - self.n_feat

    @property
    def residual_variance(self) -> float:
        """float: Estimated residual variance."""
        return self.residuals.var() * (self.n_obs - 1) / self.degrees_of_freedom


class classification_model(supervised_model, ABC):
    """Base class for supervised classification models."""

    @abstractmethod
    def predict_proba(self, target: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            target (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Class probability estimates.
        """
        ...

    def __init__(self, x, y):
        """
        Initialize a classification model.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        super(classification_model, self).__init__(x, y)
        self._n_classes = y.max() + 1

    @property
    def n_classes(self) -> int:
        """int: Number of classes in the dataset."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value: int):
        """Set the number of classes."""
        self._n_classes = value

    def classify(self, target: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            target (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.predict_proba(target).argmax(1)

    @property
    def fitted(self):
        """np.ndarray: Predicted class labels for training data."""
        return self.classify(self.x)

    @property
    def accuracy(self):
        """float: Training accuracy."""
        return np.mean(self.y == self.fitted)

    def out_of_sample_accuracy(self, x, y):
        """
        Compute out-of-sample accuracy.

        Args:
            x (np.ndarray): Test features.
            y (np.ndarray): Test labels.

        Returns:
            float: Accuracy score.
        """
        return np.mean(y == self.classify(x))

    @property
    def misclassification_probability(self):
        """float: Misclassification probability = 1 - accuracy."""
        return 1 - self.accuracy

    def out_of_sample_misclassification_probability(self, x, y):
        """
        Compute out-of-sample misclassification probability.

        Args:
            x (np.ndarray): Test features.
            y (np.ndarray): Test labels.

        Returns:
            float: Misclassification probability.
        """
        return 1 - self.out_of_sample_accuracy(x, y)

    @property
    def confusion_matrix(self):
        """np.ndarray: Confusion matrix for training data."""
        return one_hot_encode(self.y).T @ one_hot_encode(self.fitted)

    def out_of_sample_confusion_matrix(self, x, y):
        """
        Compute confusion matrix for test data.

        Args:
            x (np.ndarray): Test features.
            y (np.ndarray): Test labels.

        Returns:
            np.ndarray: Confusion matrix.
        """
        return one_hot_encode(y).T @ one_hot_encode(self.classify(x))


class clustering_model(unsupervised_model, ABC):
    """Base class for clustering models."""

    _n_clusters: int
    _means: int

    @abstractmethod
    def cluster(self, target: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels for given data.

        Args:
            target (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Cluster assignments.
        """
        ...

    @property
    def n_clusters(self) -> int:
        """int: Number of clusters."""
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, x: int) -> None:
        self._n_clusters = x

    @property
    def means(self) -> np.ndarray:
        """np.ndarray: Cluster centroids."""
        return self._means

    @means.setter
    def means(self, x: np.ndarray) -> None:
        self._means = x

    @property
    def groups(self) -> np.ndarray:
        """np.ndarray: Cluster assignments for training data."""
        return self.cluster(self.x)

    def _calc_means(self, groups: np.ndarray) -> np.ndarray:
        """
        Compute cluster means from group assignments.

        Args:
            groups (np.ndarray): Cluster assignment array.

        Returns:
            np.ndarray: Cluster centroids.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outp = [self.x[groups == i, :].mean(0) for i in range(self.n_clusters)]
        return np.array(outp)

    @property
    def within_group_sum_of_squares(self) -> np.ndarray:
        """np.ndarray: Within-group sum of squares per cluster."""
        outp = [self.x[self.groups == i, :] - self.means[i, :] for i in range(self.n_clusters)]
        outp = [(item ** 2).sum() for item in outp]
        return np.array(outp)

    @property
    def total_within_group_sum_of_squares(self) -> float:
        """float: Total within-group sum of squares across clusters."""
        return self.within_group_sum_of_squares.sum()


class distribution_model(unsupervised_model, ABC):
    """Base class for unsupervised parametric/nonparametric distributions."""

    def pdf(self, target: np.ndarray) -> np.ndarray:
        """
        Probability density (or mass) function evaluated at `target`.

        Args:
            target (np.ndarray): Points at which to evaluate the pdf/pmf.

        Returns:
            np.ndarray: Density (or probability) values with shape compatible with `target`.
        """
        ...

    def cdf(self, target: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function evaluated at `target`.

        Args:
            target (np.ndarray): Points at which to evaluate the CDF.

        Returns:
            np.ndarray: Cumulative probabilities with shape compatible with `target`.
        """
        ...


class dimension_reduction_model(unsupervised_model, ABC):
    """Base class for unsupervised dimension reduction algorithms (e.g., PCA)."""

    def reduce(self, target: np.ndarray) -> np.ndarray:
        """
        Project `target` into a lower-dimensional space.

        Args:
            target (np.ndarray): Data matrix to reduce, shape (n_obs, n_feat).

        Returns:
            np.ndarray: Reduced representation, shape (n_obs, k) where k <= n_feat.
        """
        ...

    def out_of_sample_mean_squared_error(self, target: np.ndarray) -> np.ndarray:
        """
        Reprojection MSE: squared reconstruction error per element.

        This computes the MSE of projecting `target` onto the learned subspace and
        measuring the residual in the orthogonal complement.

        Args:
            target (np.ndarray): Data matrix to evaluate, shape (n_obs, n_feat).

        Returns:
            np.ndarray: Scalar MSE (float-like) computed as trace(T' M T) / T.size.
        """
        Z = self.reduce(target)
        M = np.eye(target.shape[0])-Z@np.linalg.inv(Z.T@Z)@Z.T
        return np.trace(target.T@M@target)/target.size

    @property
    def mean_squared_error(self) -> np.ndarray:
        """np.ndarray: In-sample mean squared error for `self.x`."""
        return self.out_of_sample_mean_squared_error(self.x)

    def out_of_sample_root_mean_squared_error(self, target: np.ndarray) -> np.ndarray:
        """
        Root mean squared reconstruction error for new data.

        Args:
            target (np.ndarray): Data matrix to evaluate.

        Returns:
            np.ndarray: Scalar RMSE.
        """
        return np.sqrt(self.out_of_sample_mean_squared_error(target))

    @property
    def root_mean_squared_error(self) -> np.ndarray:
        """np.ndarray: In-sample root mean squared error for `self.x`."""
        return self.out_of_sample_root_mean_squared_error(self.x)


class supervised_dimension_reduction_model(supervised_model, ABC):
    """Base class for supervised dimension reduction (e.g., CCA)."""

    def reduce_X(self, x_new: np.ndarray) -> np.ndarray:
        """
        Project `x_new` into the supervised lower-dimensional X-space.

        Args:
            x_new (np.ndarray): Feature matrix, shape (n_obs, n_feat).

        Returns:
            np.ndarray: Reduced X scores, shape (n_obs, kx).
        """
        ...

    def reduce_Y(self, y_new: np.ndarray) -> np.ndarray:
        """
        Project `y_new` (targets) into the supervised lower-dimensional Y-space.

        Args:
            y_new (np.ndarray): Target vector or matrix, shape (n_obs, ...) depending on model.

        Returns:
            np.ndarray: Reduced Y scores, shape (n_obs, ky).
        """
        ...

    def reduce(self, x_new: Optional[np.ndarray] = None, y_new: Optional[np.ndarray] = None):
        """
        Reduce X, Y, or both, depending on provided inputs.

        Exactly one (or both) of `x_new` or `y_new` must be provided.

        Args:
            x_new (Optional[np.ndarray]): Feature matrix to reduce.
            y_new (Optional[np.ndarray]): Target vector/matrix to reduce.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]:
                - If both provided: (X_reduced, Y_reduced).
                - If only `x_new` provided: X_reduced.
                - If only `y_new` provided: Y_reduced.

        Raises:
            ValueError: If neither `x_new` nor `y_new` is provided.
        """
        if x_new is None and y_new is None:
            raise ValueError("Provide at least one of x_new or y_new.")
        if x_new is not None and y_new is not None:
            return self.reduce_X(x_new), self.reduce_Y(y_new)
        return self.reduce_X(x_new) if x_new is not None else self.reduce_Y(y_new)


class likelihood_type(Protocol):
    """Structural protocol for objects exposing likelihood metrics."""

    @property
    def log_likelihood(self) -> float:
        """float: Model log-likelihood."""
        ...

    @property
    def null_likelihood(self) -> float:
        """float: Log-likelihood of the null/reference model."""
        ...


class likelihood_model(ABC):
    """Mixin-like base for models that report likelihood-based criteria."""

    n_feat: int
    n_obs: int

    @property
    @abstractmethod
    def log_likelihood(self) -> float:
        """float: Model log-likelihood under fitted parameters."""
        ...

    @property
    @abstractmethod
    def null_likelihood(self) -> float:
        """float: Log-likelihood of the null/reference model."""
        ...

    @property
    def aic(self) -> float:
        """float: Akaike Information Criterion (smaller is better)."""
        return 2 * self.n_feat - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """float: Bayesian Information Criterion (smaller is better)."""
        return np.log(self.n_obs) * self.n_feat - 2 * self.log_likelihood

    @property
    def deviance(self) -> float:
        """float: Model deviance = 2*LL(model) - 2*LL(null)."""
        return 2 * self.log_likelihood - 2 * self.null_likelihood


class parametric_distribution_model(distribution_model, likelihood_model, ABC):
    """Base class for parametric distributions that expose likelihood metrics."""

    params: np.ndarray
    x: np.ndarray

    @abstractmethod
    def pdf(self, target: np.ndarray) -> np.ndarray:
        """
        Probability density (or mass) evaluated at `target`.

        Args:
            target (np.ndarray): Points for evaluation.

        Returns:
            np.ndarray: Density/probability values.
        """
        ...

    @abstractmethod
    def cdf(self, target: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution evaluated at `target`.

        Args:
            target (np.ndarray): Points for evaluation.

        Returns:
            np.ndarray: Cumulative probabilities.
        """
        ...

    @property
    def log_likelihood(self) -> float:
        """float: In-sample log-likelihood for `self.x`."""
        return self.out_of_sample_log_likelihood(self.x)

    @abstractmethod
    def out_of_sample_log_likelihood(self, target: np.ndarray):
        """
        Log-likelihood evaluated on arbitrary data.

        Args:
            target (np.ndarray): Data on which to evaluate LL.

        Returns:
            float: Log-likelihood value.
        """
        ...

    @property
    def null_likelihood(self) -> float:
        """float: In-sample null log-likelihood for `self.x`."""
        return self.out_of_sample_null_likelihood(self.x)

    @abstractmethod
    def out_of_sample_null_likelihood(self, target: np.ndarray):
        """
        Null-model log-likelihood on arbitrary data.

        Args:
            target (np.ndarray): Data on which to evaluate null LL.

        Returns:
            float: Null log-likelihood value.
        """
        ...

    @property
    def deviance(self) -> np.ndarray:
        """np.ndarray: Deviance for `self.x` = 2*(LL - LL_null)."""
        return 2*self.log_likelihood-2*self.null_likelihood

    def out_of_sample_deviance(self, target: np.ndarray) -> np.ndarray:
        """
        Deviance on arbitrary data.

        Args:
            target (np.ndarray): Data on which to compute deviance.

        Returns:
            np.ndarray: Deviance value(s).
        """
        return 2*self.out_of_sample_log_likelihood(target)-2*self.out_of_sample_null_likelihood(target)


class prediction_likelihood_model(ABC):
    """Base for prediction models that define likelihood via `evaluate_lnL`."""

    y: np.ndarray
    n_obs: int
    n_feat: int

    @abstractmethod
    def evaluate_lnL(self, pred: np.ndarray) -> float:
        """
        Evaluate log-likelihood given predictions `pred`.

        Args:
            pred (np.ndarray): Predicted values or probabilities aligned with `y`.

        Returns:
            float: Log-likelihood value.
        """
        ...

    @property
    @abstractmethod
    def fitted(self) -> np.ndarray:
        """
        Model-fitted predictions on training data.

        Returns:
            np.ndarray: Predictions aligned with `y`.
        """
        ...

    @property
    def log_likelihood(self) -> float:
        """float: Log-likelihood at fitted values."""
        return self.evaluate_lnL(self.fitted)

    @property
    def null_likelihood(self) -> float:
        """float: Log-likelihood of a mean-only/constant (null) predictor."""
        return self.evaluate_lnL(np.full(self.y.shape, self.y.mean()))

    @property
    def aic(self) -> float:
        """float: Akaike Information Criterion."""
        return 2 * self.n_feat - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """float: Bayesian Information Criterion."""
        return np.log(self.n_obs) * self.n_feat - 2 * self.log_likelihood

    @property
    def deviance(self) -> float:
        """float: Deviance = 2*LL(model) - 2*LL(null)."""
        return 2 * self.log_likelihood - 2 * self.null_likelihood


class broom_model(Protocol):
    """Protocol for tidy/glance accessors (broom-like API)."""

    @property
    def tidy(self) -> pd.DataFrame:
        """pd.DataFrame: Per-parameter summary (estimates, SEs, tests, etc.)."""
        ...

    @property
    def glance(self) -> pd.DataFrame:
        """pd.DataFrame: Model-level summary (fit statistics, diagnostics, etc.)."""
        ...


class variance_model(ABC):
    """Mixin for models that expose variance-covariance and inferential stats."""

    _glance: pd.DataFrame

    @abstractmethod
    def vcov_params(self) -> np.ndarray:
        """
        Variance-covariance matrix of parameter estimates.

        Returns:
            np.ndarray: (p x p) covariance matrix for the first `n_feat` parameters.
        """
        ...

    @property
    def std_error(self):
        """np.ndarray: Standard errors for parameters (from `vcov_params`)."""
        return np.sqrt(np.diagonal(self.vcov_params))

    @property
    def t_statistic(self):
        """np.ndarray: t-statistics = params / std_error."""
        return self.params / self.std_error

    @property
    def p_value(self):
        """np.ndarray: Two-sided p-values under Student-t with df = n_obs - n_feat."""
        return 2 * sp.stats.t.cdf(-np.abs(self.t_statistic), df=self.n_obs - self.n_feat)

    def conf_int(self, level=0.95):
        """
        Confidence intervals for parameters.

        Args:
            level (float): Coverage probability (default 0.95).

        Returns:
            np.ndarray: 2 x p array with lower/upper bounds by column.
        """
        spread = -self.std_error * sp.stats.t.ppf((1 - level) / 2, df=self.degrees_of_freedom)
        return vstack(self.params - spread, self.params + spread)

    @property
    def tidy(self):
        """pd.DataFrame: Tidy per-parameter table (no CIs)."""
        return self.tidyci(ci=False)

    def tidyci(self, level=0.95, ci=True):
        """
        Tidy per-parameter table with optional confidence intervals.

        Args:
            level (float): CI level (default 0.95).
            ci (bool): If True, include CI columns.

        Returns:
            pd.DataFrame: Columns include variable, estimate, std.error, t.statistic, p.value,
                and optionally ci.lower, ci.upper.
        """
        n = self.n_feat
        df = [np.arange(n), self.params[:n], self.std_error[:n], self.t_statistic[:n], self.p_value[:n]]
        cols = ['variable', 'estimate', 'std.error', 't.statistic', 'p.value']
        if ci:
            df += [self.conf_int(level)[:, :n]]
            cols += ['ci.lower', 'ci.upper']
        df = pd.DataFrame(np.vstack(df).T, columns=cols)
        return df

    @property
    def glance(self) -> pd.DataFrame:
        """pd.DataFrame: Model-level summary table."""
        return self._glance

    @glance.setter
    def glance(self, x: pd.DataFrame) -> None:
        """Set the glance DataFrame."""
        self._glance = x


class SupervisedModel(ABC):
    """Abstract base class for supervised models constructed from a formula.

    Subclasses must set the class attribute :attr:`MODEL_TYPE` to a concrete
    supervised model implementation. This wrapper handles parsing a formula,
    extracting predictor and response variables, and fitting the underlying
    algorithm.

    Attributes:
        formula (str): Formula string used to specify the model.
        x_vars (list[str]): Names of predictor variables.
        y_var (str): Name of response variable.
        data (pd.DataFrame): Parsed DataFrame containing predictors and response.
        model (supervised_model): Fitted underlying model implementation.
    """

    MODEL_TYPE: ClassVar[Type["supervised_model"]]  # must be provided by subclasses

    def __init__(self, formula: str, data: pd.DataFrame, *args, **kwargs) -> None:
        """Initialize and fit a supervised model.

        Args:
            formula (str): Patsy-like formula string (e.g., ``"y ~ x1 + x2"``).
            data (pd.DataFrame): Input DataFrame containing all variables.
            *args: Positional arguments passed to the underlying model type.
            **kwargs: Keyword arguments passed to the underlying model type.

        Raises:
            TypeError: If the subclass does not define :attr:`MODEL_TYPE`.
        """
        self.formula: str = formula
        x_vars, y_var, _, data_out = parse(formula, data)

        if getattr(self.__class__, "MODEL_TYPE", None) is None:
            raise TypeError(f"{self.__class__.__name__} must define MODEL_TYPE")

        self.x_vars: list[str] = list(x_vars)
        self.y_var: str = y_var
        self.data: pd.DataFrame = data_out

        X = self.data[self.x_vars]
        y = self.data[self.y_var]
        self.model: supervised_model = self.MODEL_TYPE(X, y, *args, **kwargs)

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        """Return a tidy coefficient table with optional confidence intervals.

        Args:
            level (float, default=0.95): Confidence level for intervals.
            ci (bool, default=True): Whether to include confidence intervals.

        Returns:
            pd.DataFrame: Tidy table of parameter estimates. If the model
            outputs a ``variable`` column of matching length, it is replaced
            with the predictor names from :attr:`x_vars`.
        """
        outp = self.model.tidyci(level, ci)
        if "variable" in outp.columns and len(outp["variable"]) == len(self.x_vars):
            outp = outp.copy()
            outp["variable"] = pd.Index(self.x_vars)
        return outp

    @property
    def tidy(self) -> pd.DataFrame:
        """Return a tidy table of parameter estimates without confidence intervals.

        Equivalent to calling :meth:`tidyci` with ``ci=False``.

        Returns:
            pd.DataFrame: Table of parameter estimates.
        """
        return self.tidyci(ci=False)

    @property
    def glance(self) -> pd.DataFrame:
        """Return model-level summary statistics.

        Returns:
            pd.DataFrame: One-row DataFrame of model fit diagnostics
            (e.g., log-likelihood, R², AIC).
        """
        return self.model.glance


class PredictionModel(SupervisedModel, ABC):
    """Concrete interface for supervised prediction models.

    Extends :class:`SupervisedModel` by adding a :meth:`predict` method
    for generating predictions on new data.
    """

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        """Generate predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the predictor
                variables referenced in the original formula.

        Returns:
            pd.Series: Predictions indexed to ``new_data.index`` and
            named after the response variable (:attr:`y_var`).
        """
        _, _, _, new_data_out = parse(self.formula, new_data)
        yhat = self.model.predict(new_data_out[self.x_vars])
        return pd.Series(yhat, index=new_data_out.index, name=self.y_var)


class ClassificationModel(SupervisedModel, ABC):
    """Concrete interface for supervised prediction models.

    Extends :class:`SupervisedModel` by adding a :meth:`classify` method
    for generating classifications on new data.
    """

    def classify(self, new_data: pd.DataFrame) -> pd.Series:
        """Generate classifications on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the predictor
                variables referenced in the original formula.

        Returns:
            pd.Series: Classifications indexed to ``new_data.index`` and
            named after the response variable (:attr:`y_var`).
        """
        _, _, _, new_data_out = parse(self.formula, new_data)
        yhat = self.model.classify(new_data_out[self.x_vars])
        return pd.Series(yhat, index=new_data_out.index, name=self.y_var)

    def predict_proba(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for classes on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the predicted
                points to be assigned to classes.

        Returns:
            pd.DataFrame: Predicted probabilities with shape (n_samples, n_classes).
            Columns are named ``class=0, class=1, ..., class=r`` corresponding to the
            r classes returned by the underlying model.
        """
        _, _, _, new_data_out = parse(self.formula, new_data)
        Z = self.model.predict_proba(new_data_out[self.x_vars])  # shape (n, r)

        # build column names: class=0, class=1, ..., class=r
        n_classes = Z.shape[1]
        col_names = [f"class={i}" for i in range(n_classes)]

        return pd.DataFrame(Z, index=new_data_out.index, columns=col_names)


class UnsupervisedModel(ABC):
    """Abstract base class for unsupervised models constructed from a formula.

    Subclasses must set the class attribute :attr:`MODEL_TYPE` to a concrete
    supervised model implementation. This wrapper handles parsing a formula,
    extracting predictor and response variables, and fitting the underlying
    algorithm.

    Attributes:
        formula (str): Formula string used to specify the model.
        x_vars (list[str]): Names of predictor variables.
        y_var (str): Name of response variable.
        data (pd.DataFrame): Parsed DataFrame containing predictors and response.
        model (supervised_model): Fitted underlying model implementation.
    """

    MODEL_TYPE: ClassVar[Type["unsupervised_model"]]  # must be provided by subclasses

    def __init__(self, formula: str, data: pd.DataFrame, *args, **kwargs) -> None:
        """Initialize and fit a supervised model.

        Args:
            formula (str): Patsy-like formula string (e.g., ``"y ~ x1 + x2"``).
            data (pd.DataFrame): Input DataFrame containing all variables.
            *args: Positional arguments passed to the underlying model type.
            **kwargs: Keyword arguments passed to the underlying model type.

        Raises:
            TypeError: If the subclass does not define :attr:`MODEL_TYPE`.
        """
        self.formula: str = formula
        x_vars, _, _, data_out = parse(formula, data)

        if getattr(self.__class__, "MODEL_TYPE", None) is None:
            raise TypeError(f"{self.__class__.__name__} must define MODEL_TYPE")

        self.x_vars: list[str] = list(x_vars)
        self.data: pd.DataFrame = data_out

        X = self.data[self.x_vars]
        self.model: unsupervised_model = self.MODEL_TYPE(X, *args, **kwargs)

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        """Return a tidy coefficient table with optional confidence intervals.

        Args:
            level (float, default=0.95): Confidence level for intervals.
            ci (bool, default=True): Whether to include confidence intervals.

        Returns:
            pd.DataFrame: Tidy table of parameter estimates. If the model
            outputs a ``variable`` column of matching length, it is replaced
            with the predictor names from :attr:`x_vars`.
        """
        outp = self.model.tidyci(level, ci)
        if "variable" in outp.columns and len(outp["variable"]) == len(self.x_vars):
            outp = outp.copy()
            outp["variable"] = pd.Index(self.x_vars)
        return outp

    @property
    def tidy(self) -> pd.DataFrame:
        """Return a tidy table of parameter estimates without confidence intervals.

        Equivalent to calling :meth:`tidyci` with ``ci=False``.

        Returns:
            pd.DataFrame: Table of parameter estimates.
        """
        return self.tidyci(ci=False)

    @property
    def glance(self) -> pd.DataFrame:
        """Return model-level summary statistics.

        Returns:
            pd.DataFrame: One-row DataFrame of model fit diagnostics
            (e.g., log-likelihood, R², AIC).
        """
        return self.model.glance

class ClusteringModel(SupervisedModel, ABC):
    """Concrete interface for supervised clustering models.

    Extends :class:`SupervisedModel` by adding a :meth:`cluster` method
    for generating predictions on new data.
    """

    def cluster(self, new_data: pd.DataFrame) -> pd.Series:
        """Generate predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the predictor
                variables referenced in the original formula.

        Returns:
            pd.Series: Predictions indexed to ``new_data.index`` and
            named after the response variable (:attr:`y_var`).
        """
        _, _, _, new_data_out = parse(self.formula, new_data)
        yhat = self.model.cluster(new_data_out[self.x_vars])
        return pd.Series(yhat, index=new_data_out.index, name='clusters')


class DimensionReductionModel(SupervisedModel, ABC):
    """Concrete interface for supervised dimension-reduction models.

    Extends :class:`SupervisedModel` by adding a :meth:`reduce` method
    for projecting new data into a lower-dimensional space.
    """

    def reduce(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Project new data into the reduced space.

        Args:
            new_data (pd.DataFrame): DataFrame containing the predictor
                variables referenced in the original formula.

        Returns:
            pd.DataFrame: Reduced representation with shape (n_samples, r).
            Columns are named ``z1, z2, ..., zr`` corresponding to the
            r components returned by the underlying model.
        """
        _, _, _, new_data_out = parse(self.formula, new_data)
        Z = self.model.reduce(new_data_out[self.x_vars])  # shape (n, r)

        # build column names: z1, z2, ..., zr
        n_components = Z.shape[1]
        col_names = [f"z{i+1}" for i in range(n_components)]

        return pd.DataFrame(Z, index=new_data_out.index, columns=col_names)
