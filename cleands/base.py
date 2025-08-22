import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .utils import *


class learning_model(ABC):
    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        (self.n_obs, self.n_feat) = x.shape


class supervised_model(learning_model, ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__(x)
        self.y = y


class unsupervised_model(learning_model, ABC): ...


class prediction_model(supervised_model,ABC):
    @abstractmethod
    def predict(self, target:np.ndarray)->np.ndarray: ...

    @property
    def fitted(self)->np.ndarray: return self.predict(self.x)

    @property
    def residuals(self)->np.ndarray: return self.y - self.fitted

    @property
    def residual_sum_of_squares(self)->float: return np.sum(self.residuals ** 2)

    @property
    def mean_squared_error(self)->float: return np.mean(self.residuals ** 2)

    def out_of_sample_mean_squared_error(self, x, y)->float:
        return np.mean((y - self.predict(x)) ** 2)

    @property
    def root_mean_squared_error(self)->float: return np.sqrt(self.mean_squared_error)

    def out_of_sample_root_mean_squared_error(self, x, y)->float:
        return np.sqrt(self.out_of_sample_mean_squared_error(x, y))

    @property
    def r_squared(self)->float:
        return 1 - self.residuals.var() / self.y.var()

    @property
    def adjusted_r_squared(self)->float:
        return 1 - (1 - self.r_squared) * (self.n_obs - 1) / self.degrees_of_freedom

    @property
    def degrees_of_freedom(self)->int:
        return self.n_obs - self.n_feat

    @property
    def residual_variance(self)->float:
        return self.residuals.var() * (self.n_obs - 1) / self.degrees_of_freedom


class classification_model(supervised_model, ABC):
    @abstractmethod
    def predict_proba(self,target): ...

    def __init__(self,x,y):
        super(classification_model,self).__init__(x,y)
        self._n_classes = y.max()+1

    @property
    def n_classes(self): return self._n_classes

    @n_classes.setter
    def n_classes(self, value:int): self._n_classes = value

    def classify(self,target:np.array): return self.predict_proba(target).argmax(1)

    @property
    def fitted(self): return self.classify(self.x)

    @property
    def accuracy(self): return np.mean(self.y==self.fitted)

    def out_of_sample_accuracy(self,x,y): return np.mean(y==self.classify(x))

    @property
    def misclassification_probability(self): return 1-self.accuracy

    def out_of_sample_misclassification_probability(self,x,y): return 1-self.out_of_sample_accuracy(x,y)

    @property
    def confusion_matrix(self): return one_hot_encode(self.y).T@one_hot_encode(self.fitted)

    def out_of_sample_confusion_matrix(self,x,y): return one_hot_encode(y).T@one_hot_encode(self.classify(x))


class clustering_model(unsupervised_model, ABC):
    n_clusters: int
    means: int

    @abstractmethod
    def cluster(self, target:np.ndarray) -> np.ndarray: ...

    @property
    def n_clusters(self) -> int: return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, x:int) -> None: self._n_clusters = x

    @property
    def means(self) -> np.ndarray: return self._means

    @means.setter
    def means(self, x: np.ndarray) -> None: self._means = x

    @property
    def groups(self) -> np.ndarray: return self.cluster(self.x)

    def _calc_means(self, groups: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outp = [self.x[groups == i, :].mean(0) for i in range(self.n_clusters)]
        return np.array(outp)

    @property
    def within_group_sum_of_squares(self) -> np.ndarray:
        outp = [self.x[self.groups == i, :] - self.means[i, :] for i in range(self.n_clusters)]
        outp = [(item ** 2).sum() for item in outp]
        return np.array(outp)

    @property
    def total_within_group_sum_of_squares(self) -> float:
        return self.within_group_sum_of_squares.sum()


class distribution_model(unsupervised_model, ABC):
    def pdf(self, target: np.ndarray) -> np.ndarray: ...

    def cdf(self, target: np.ndarray) -> np.ndarray: ...


class dimension_reduction_model(unsupervised_model, ABC):
    def reduce(self, target: np.ndarray) -> np.ndarray: ...

    def out_of_sample_mean_squared_error(self, target: np.ndarray) -> np.ndarray:
        Z = self.reduce(target)
        M = np.eye(target.shape[0])-Z@np.linalg.inv(Z.T@Z)@Z.T
        return np.trace(target.T@M@target)/target.size

    @property
    def mean_squared_error(self) -> np.ndarray:
        return self.out_of_sample_mean_squared_error(self.x)

    def out_of_sample_root_mean_squared_error(self, target: np.ndarray) -> np.ndarray:
        return np.sqrt(self.out_of_sample_mean_squared_error(target))

    @property
    def root_mean_squared_error(self) -> np.ndarray:
        return self.out_of_sample_root_mean_squared_error(self.x)

class supervised_dimension_reduction_model(supervised_model, ABC):
    def reduce_X(self, x_new: np.ndarray) -> np.ndarray:
        ...

    def reduce_Y(self, y_new: np.ndarray) -> np.ndarray:
        ...

    def reduce(self, x_new: Optional[np.ndarray] = None, y_new: Optional[np.ndarray] = None):
        if x_new is None and y_new is None:
            raise ValueError("Provide at least one of x_new or y_new.")
        if x_new is not None and y_new is not None:
            return self.reduce_X(x_new), self.reduce_Y(y_new)
        return self.reduce_X(x_new) if x_new is not None else self.reduce_Y(y_new)



class likelihood_type(Protocol):
    @property
    def log_likelihood(self) -> float: ...

    @property
    def null_likelihood(self) -> float: ...


class likelihood_model(ABC):
    n_feat: int
    n_obs: int

    @property
    @abstractmethod
    def log_likelihood(self) -> float: ...

    @property
    @abstractmethod
    def null_likelihood(self) -> float: ...

    @property
    def aic(self) -> float: return 2 * self.n_feat - 2 * self.log_likelihood

    @property
    def bic(self) -> float: return np.log(self.n_obs) * self.n_feat - 2 * self.log_likelihood

    @property
    def deviance(self) -> float: return 2 * self.log_likelihood - 2 * self.null_likelihood


class parametric_distribution_model(distribution_model, likelihood_model, ABC):
    params: np.ndarray
    x: np.ndarray

    @abstractmethod
    def pdf(self, target: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def cdf(self, target: np.ndarray) -> np.ndarray: ...

    @property
    def log_likelihood(self) -> float: return self.out_of_sample_log_likelihood(self.x)

    @abstractmethod
    def out_of_sample_log_likelihood(self, target: np.ndarray): ...

    @property
    def null_likelihood(self) -> float: return self.out_of_sample_null_likelihood(self.x)

    @abstractmethod
    def out_of_sample_null_likelihood(self, target: np.ndarray): ...

    @property
    def deviance(self) -> np.ndarray:
        return 2*self.log_likelihood-2*self.null_likelihood

    def out_of_sample_deviance(self, target: np.ndarray) -> np.ndarray:
        return 2*self.out_of_sample_log_likelihood(target)-2*self.out_of_sample_null_likelihood(target)



class prediction_likelihood_model(ABC):
    y: np.ndarray
    n_obs: int
    n_feat: int

    @abstractmethod
    def evaluate_lnL(self, pred: np.ndarray) -> float: ...

    @property
    @abstractmethod
    def fitted(self) -> np.ndarray: ...

    @property
    def log_likelihood(self) -> float: return self.evaluate_lnL(self.fitted)

    @property
    def null_likelihood(self) -> float: return self.evaluate_lnL(np.full(self.y.shape, self.y.mean()))

    @property
    def aic(self) -> float: return 2 * self.n_feat - 2 * self.log_likelihood

    @property
    def bic(self) -> float: return np.log(self.n_obs) * self.n_feat - 2 * self.log_likelihood

    @property
    def deviance(self) -> float: return 2 * self.log_likelihood - 2 * self.null_likelihood


class broom_model(Protocol):
    @property
    def tidy(self) -> pd.DataFrame: ...

    @property
    def glance(self) -> pd.DataFrame: ...


class variance_model(ABC):
    _glance: pd.DataFrame

    @abstractmethod
    def vcov_params(self) -> np.ndarray: ...

    @property
    def std_error(self): return np.sqrt(np.diagonal(self.vcov_params))

    @property
    def t_statistic(self): return self.params / self.std_error

    @property
    def p_value(self): return 2 * sp.stats.t.cdf(-np.abs(self.t_statistic), df=self.n_obs - self.n_feat)

    def conf_int(self, level=0.95):
        spread = -self.std_error * sp.stats.t.ppf((1 - level) / 2, df=self.degrees_of_freedom)
        return vstack(self.params - spread, self.params + spread)

    @property
    def tidy(self): return self.tidyci(ci=False)

    def tidyci(self, level=0.95, ci=True):
        n = self.n_feat
        df = [np.arange(n), self.params[:n], self.std_error[:n], self.t_statistic[:n], self.p_value[:n]]
        cols = ['variable', 'estimate', 'std.error', 't.statistic', 'p.value']
        if ci:
            df += [self.conf_int(level)[:, :n]]
            cols += ['ci.lower', 'ci.upper']
        df = pd.DataFrame(np.vstack(df).T, columns=cols)
        return df

    @property
    def glance(self) -> pd.DataFrame: return self._glance

    @glance.setter
    def glance(self, x: pd.DataFrame) -> None: self._glance = x


class FormulaNotationMixin:
    """
    Mixin for the SupervisedModel wrapper that enables:
      - SupervisedModel.from_formula("y ~ x1 + x2 + ...", data, model_type, ...)
      - Predicting / classifying directly from DataFrames that match the formula.
    Usage:
      class LM(FormulaNotationMixin, SupervisedModel):
          pass
      m = LM.from_formula("y ~ x1 + x2", df, model_type=least_squares_regressor)
    """
    # ---------- Constructors ----------
    @classmethod
    def from_formula(cls,
                     formula: str,
                     data: pd.DataFrame,
                     model_type: Type[supervised_model],
                     *args, **kwargs) -> "SupervisedModel":
        # parse returns (x_vars, y_var, conditionals, processed_df)
        x_vars, y_var, conditionals, processed = parse(formula, data)

        # construct the standard wrapper with parsed args
        obj = cls(x_vars=x_vars,
                  y_var=y_var,
                  data=processed,
                  model_type=model_type,
                  *args, **kwargs)

        # keep metadata for later use
        obj.formula: str = formula
        obj.conditionals: list[str] = conditionals
        return obj

    # ---------- Helpers ----------
    def _process_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Re-apply the stored formula to a new DataFrame to rebuild derived columns,
        interactions, and the intercept column, then return the processed subset.
        """
        if not hasattr(self, "formula"):
            raise AttributeError("This instance has no stored formula. "
                                 "Construct it with .from_formula().")
        x_vars, y_var, conditionals, processed = parse(self.formula, new_data.copy())
        # Ensure we use the same x_vars order learned at fit time:
        missing = [c for c in self.x_vars if c not in processed.columns]
        if missing:
            raise ValueError(f"New data is missing required columns after parsing: {missing}")
        return processed[self.x_vars]

    # ---------- Convenience Methods (DF in → ndarray out) ----------
    def design_matrix(self, new_data: pd.DataFrame) -> np.ndarray:
        """Return the X matrix for new_data according to the stored formula."""
        return self._process_new_data(new_data).values

    def predict_from_df(self, new_data: pd.DataFrame) -> np.ndarray:
        """Predict using underlying regression-like models (that implement .predict)."""
        if not hasattr(self.model, "predict"):
            raise AttributeError("Underlying model does not support .predict().")
        X = self.design_matrix(new_data)
        return self.model.predict(X)

    def classify_from_df(self, new_data: pd.DataFrame) -> np.ndarray:
        """Classify using underlying classifiers (that implement .classify or .predict_proba)."""
        if hasattr(self.model, "classify"):
            X = self.design_matrix(new_data)
            return self.model.classify(X)
        if hasattr(self.model, "predict_proba"):
            X = self.design_matrix(new_data)
            return self.model.predict_proba(X).argmax(1)
        raise AttributeError("Underlying model does not support classification methods.")


class SupervisedModel(FormulaNotationMixin, ABC):
    def __init__(self, x_vars: list[str], y_var: str, data: pd.DataFrame, model_type: Type[supervised_model], *args, **kwargs) -> None:
        self.model_type: Type[supervised_model] = model_type
        self.model: supervised_model = model_type(data[x_vars], data[y_var], *args, **kwargs)
        self.x_vars: list[str] = x_vars
        self.y_var: str = y_var
        self.data: pd.DataFrame = data

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        outp = self.model.tidyci(level, ci)
        outp['variable'] = pd.Series(self.x_vars)
        return outp

    @property
    def tidy(self) -> pd.DataFrame: return self.tidyci(ci=False)

    @property
    def glance(self) -> pd.DataFrame: return self.model.glance


class PredictionModel(SupervisedModel, ABC):
    def predict(self, target: np.ndarray) -> np.ndarray: return self.model.predict(target)


class ClassificationModel(SupervisedModel, ABC):
    def classify(self, target: np.ndarray) -> np.ndarray: return self.model.classify(target)