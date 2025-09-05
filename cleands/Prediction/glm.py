import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type, Union, Any
from abc import ABC, abstractmethod

from functools import partial

from ..base import prediction_model, PredictionModel, prediction_likelihood_model, variance_model
from ..utils import *


class linear_model(prediction_model, prediction_likelihood_model):
    """Ordinary least squares linear regression.

    Inherits from:
        - prediction_model: supervised regression base.
        - prediction_likelihood_model: provides log-likelihood evaluation.

    Attributes:
        params (np.ndarray): Estimated regression coefficients.
    """

    def __init__(self, x, y, *args, **kwargs):
        """Fit a linear regression model.

        Args:
            x (np.ndarray): Design matrix of shape (n_obs, n_features).
            y (np.ndarray): Response vector.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(linear_model, self).__init__(x, y)
        self.params = self._fit(x, y, *args, **kwargs)

    def _fit(self, x, y, *args, **kwargs):
        """Estimate regression coefficients via normal equations."""
        return np.linalg.solve(x.T @ x, x.T @ y)

    def predict(self, newdata: np.ndarray) -> np.ndarray:
        """Predict responses for new data.

        Args:
            newdata (np.ndarray): Design matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return newdata @ self.params

    def evaluate_lnL(self, pred: np.ndarray) -> float:
        """Evaluate log-likelihood of predictions under Gaussian errors.

        Args:
            pred (np.ndarray): Predicted values.

        Returns:
            float: Log-likelihood value.
        """
        return -self.n_obs / 2 * (np.log(2 * np.pi * (self.y - pred).var()) + 1)


class logistic_regressor(linear_model, variance_model):
    """Logistic regression with Newton–Raphson estimation.

    Provides likelihood-based fit, variance-covariance matrix,
    and pseudo-R² measures (McFadden, Ben-Akiva–Lerman).
    """

    def __init__(self, *args, **kwargs):
        """Fit a logistic regression model using Newton–Raphson."""
        super().__init__(*args, **kwargs)
        self.glance = pd.DataFrame(
            {
                'mcfaddens.r.squared': self.mcfaddens_r_squared,
                'ben.akiva.lerman.r.squared': self.ben_akiva_lerman_r_squared,
                'self.df': self.n_feat,
                'resid.df': self.degrees_of_freedom,
                'aic': self.aic,
                'bic': self.bic,
                'log.likelihood': self.log_likelihood,
                'deviance': self.deviance,
            },
            index=['']
        )

    def _fit(self, x, y):
        """Fit coefficients by Newton–Raphson optimization."""
        params, self.iters = newton(self.gradient, self.hessian, np.zeros(self.n_feat))
        return params

    @property
    def vcov_params(self) -> np.ndarray:
        """Variance-covariance matrix of parameters."""
        H = self.hessian(self.params)
        try:
            return -np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return -np.linalg.pinv(H)

    def evaluate_lnL(self, pred: np.ndarray) -> float:
        """Log-likelihood for Bernoulli outcomes.

        Args:
            pred (np.ndarray): Predicted probabilities.

        Returns:
            float: Log-likelihood value.
        """
        eps = 1e-15
        pred = np.clip(pred, eps, 1 - eps)
        return self.y.T @ np.log(pred) + (1 - self.y).T @ np.log(1 - pred)

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        """Gradient of the log-likelihood."""
        return self.x.T @ (self.y - expit(self.x @ coefs))

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        """Hessian matrix of the log-likelihood."""
        x = self.x.values if isinstance(self.x, (pd.DataFrame, pd.Series)) else self.x
        Fx = expit(x @ coefs)
        inside = np.diagflat(Fx * (1 - Fx))
        return -x.T @ inside @ x

    def predict(self, target: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        return expit(target @ self.params)

    @property
    def mcfaddens_r_squared(self) -> float:
        """McFadden's pseudo-R²."""
        return 1 - self.log_likelihood / self.null_likelihood

    @property
    def ben_akiva_lerman_r_squared(self) -> float:
        """Ben-Akiva–Lerman pseudo-R²."""
        return (self.y.T @ self.fitted + (1 - self.y).T @ (1 - self.fitted)) / self.n_obs

    def marginal_effects(
        self,
        newx: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        average: bool = True,
    ) -> np.ndarray:
        """Compute marginal effects of predictors.

        Args:
            newx (np.ndarray | pd.DataFrame, optional): New design matrix.
                If None, use training data.
            average (bool): If True, return average marginal effects.
                If False, return case-specific effects.

        Returns:
            np.ndarray: Marginal effects.
        """
        X = self.x if newx is None else newx
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if X.ndim == 2 and X.shape[1] != self.n_feat:
            if self.x.ndim == 2 and np.allclose(self.x[:, 0], 1) and X.shape[1] == self.n_feat - 1:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            else:
                raise ValueError(f"newx has shape {X.shape}, but model expects {self.n_feat} features.")
        xb = X @ self.params
        Fx = expit(xb)
        slope = Fx * (1.0 - Fx)
        effects = slope.reshape(-1, 1) * self.params.reshape(1, -1)
        return effects.mean(0) if average else effects


class least_squares_regressor(linear_model, variance_model):
    """Ordinary least squares regression with optional robust SEs."""

    def __init__(self, x, y, white: bool = False, hc: int = 3, *args, **kwargs):
        """Initialize least squares regressor.

        Args:
            x (np.ndarray): Design matrix.
            y (np.ndarray): Response vector.
            white (bool): If True, use heteroskedasticity-consistent SEs.
            hc (int): HC variant (1–5).
        """
        super(least_squares_regressor, self).__init__(x, y, *args, **kwargs)
        self.white = white
        self.hc = hc
        self.glance = pd.DataFrame(
            {
                'r.squared': self.r_squared,
                'adjusted.r.squared': self.adjusted_r_squared,
                'self.df': self.n_feat,
                'resid.df': self.degrees_of_freedom,
                'aic': self.aic,
                'bic': self.bic,
                'log.likelihood': self.log_likelihood,
                'deviance': self.deviance,
                'resid.var': self.residual_variance,
            }, index=['']
        )

    @property
    def vcov_params(self):
        """Variance-covariance matrix of parameters."""
        if self.white:
            return self.__white(self.hc)
        return np.linalg.inv(self.x.T @ self.x) * self.residual_variance

    def __white(self, hc: int) -> np.ndarray:
        """White’s heteroskedasticity-consistent covariance estimator."""
        e = self.residuals.values if isinstance(self.residuals, pd.Series) else self.residuals
        esq = self.__hc_correction(e ** 2, hc)
        meat = np.diagflat(esq)
        bread = np.linalg.inv(self.x.T @ self.x) @ self.x.T
        return bread @ meat @ bread.T

    def __hc_correction(self, esq, hc):
        """Apply HC1–HC5 finite-sample corrections."""
        mx = 1 - np.diagonal(self.x @ np.linalg.solve(self.x.T @ self.x, self.x.T))
        match hc:
            case 1:
                esq *= self.n_obs / (self.n_obs - self.n_feat)
            case 2:
                esq /= mx
            case 3:
                esq /= mx ** 2
            case 4:
                p = int(np.round((1 - mx).sum()))
                delta = 4 * np.ones((self.n_obs, 1))
                delta = hstack(delta, self.n_obs * (1 - mx) / p)
                delta = delta.min(1)
                esq /= np.power(mx, delta)
            case 5:
                p = int(np.round((1 - mx).sum()))
                delta = max(4, self.n_obs * 0.7 * (1 - mx).max() / p) * np.ones((self.n_obs, 1))
                delta = hstack(delta.reshape(-1, 1), self.n_obs * (1 - mx.reshape(-1, 1)) / p)
                delta = delta.min(1) / 2
                esq /= np.power(mx, delta)
        return esq


class poisson_regressor(linear_model):
    """Poisson regression for count data."""

    def __init__(self, x, y, *args, **kwargs):
        """Fit a Poisson regression model."""
        super().__init__(x, y, *args, **kwargs)
        self.glance = pd.DataFrame(self._glance_dict, index=[''])

    def _fit(self, x, y):
        """Fit coefficients by Newton–Raphson optimization."""
        params, self.iters = newton(self.gradient, self.hessian, np.zeros(self.n_feat))
        return params

    @property
    def vcov_params(self) -> np.ndarray:
        """Variance-covariance matrix of parameters."""
        return -np.linalg.inv(self.hessian(self.params))

    def evaluate_lnL(self, pred: np.ndarray) -> float:
        """Log-likelihood for Poisson-distributed outcomes."""
        return (
            self.y.T @ np.log(pred)
            - np.ones((1, self.n_obs)) @ pred
            + np.ones((1, self.n_obs)) @ np.log(sp.special.factorial(self.y))
        )

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        """Gradient of the log-likelihood."""
        return self.x.T @ (self.y - np.exp(self.x @ coefs))

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        """Hessian matrix of the log-likelihood."""
        Fx = np.exp(self.x @ coefs)
        if isinstance(Fx, (pd.DataFrame, pd.Series)):
            Fx = Fx.values
        return -self.x.T @ np.diagflat(Fx) @ self.x

    def predict(self, target: np.ndarray) -> np.ndarray:
        """Predict expected counts for new data."""
        return np.exp(target @ self.params)

    @property
    def _glance_dict(self) -> dict:
        """Model summary statistics for glance output."""
        return {
            'self.df': self.n_feat,
            'resid.df': self.degrees_of_freedom,
            'aic': self.aic,
            'bic': self.bic,
            'log.likelihood': self.log_likelihood,
            'deviance': self.deviance
        }


def backward_stepwise(model: Any,
                      criterion: str = "aic",
                      keep_vars: list[str] = None,
                      min_features: int = 1,
                      verbose: bool = False) -> Dict[str, Any]:
    """Perform backward stepwise feature selection.

    Iteratively removes features to optimize a model fit according to an
    information criterion (e.g., AIC, BIC, MSE).

    Args:
        model: Model object. Either:
            - Raw supervised_model with `.x` and `.y`.
            - SupervisedModel wrapper with `.x_vars`, `.y_var`, `.data`, `.model_type`.
        criterion: Model selection criterion ("aic", "bic", "mse", etc.).
        keep_vars: Variable names that must not be dropped.
        min_features: Minimum number of features to retain.
        verbose: If True, print progress messages.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "model": The best fitted model.
            - "selected_features": List of selected feature names.
            - "history": pd.DataFrame with stepwise history.
    """
    keep_vars = set(keep_vars or [])

    # unwrap
    if hasattr(model, "x_vars"):   # SupervisedModel
        x_vars, y_var, data = model.x_vars.copy(), model.y_var, model.data
        model_type = model.model_type
        def fit(subset):
            X = data[subset].values
            y = data[y_var].values
            return model_type(X, y)
        feature_names = x_vars
    else:  # supervised_model
        x, y = model.x, model.y
        feature_names = [f"x{i}" for i in range(x.shape[1])]
        def fit(subset_idx):
            return type(model)(x[:, subset_idx], y)

    # scoring
    def score(m):
        val = getattr(m, criterion)
        lower_is_better = criterion.lower() in ["aic","bic","mse","misclassification_probability"]
        return val, lower_is_better

    # initialize
    current = list(range(len(feature_names)))
    fitted = model
    best_score, lower = score(fitted)
    history = [{"step":0,"removed":None,"score":best_score,"features":feature_names.copy()}]
    step = 0

    while len(current) > max(min_features, len(keep_vars)):
        trial = []
        for j in current:
            if feature_names[j] in keep_vars:
                continue
            cand = [idx for idx in current if idx != j]
            try:
                cand_model = fit([feature_names[i] for i in cand] if hasattr(model,"x_vars") else cand)
                sc, _ = score(cand_model)
                trial.append((j, sc, cand_model))
            except Exception as e:
                if verbose: print("skip", feature_names[j], e)
                continue
        if not trial: break

        j_best, sc_best, best_model = (min(trial, key=lambda t: t[1]) if lower else max(trial, key=lambda t: t[1]))
        improved = (sc_best < best_score if lower else sc_best > best_score)
        if not improved: break

        step += 1
        if verbose:
            print(f"removed {feature_names[j_best]} -> {criterion}={sc_best}")
        current.remove(j_best)
        best_score, fitted = sc_best, best_model
        history.append({"step":step,"removed":feature_names[j_best],"score":best_score,
                        "features":[feature_names[i] for i in current]})

    return {"model":fitted,
            "selected_features":[feature_names[i] for i in current],
            "history":pd.DataFrame(history)}


def forward_stepwise(
    model: Any,
    criterion: str = "aic",
    keep_vars: Optional[List[str]] = None,
    max_features: Optional[int] = None,
    prefer_intercept: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Perform forward stepwise feature selection.

    Iteratively adds features to optimize a model according to a selection
    criterion (e.g., AIC, BIC, MSE). Supports both raw models and SupervisedModel
    wrappers, with optional intercept preference.

    Args:
        model: Model object. Either:
            - Raw supervised_model with `.x` and `.y`.
            - SupervisedModel wrapper with `.x_vars`, `.y_var`, `.data`, `.model_type`.
        criterion: Model selection criterion ("aic", "bic", "mse", etc.).
        keep_vars: Variables that must always be included.
        max_features: Maximum number of features allowed to be selected.
        prefer_intercept: If True, attempt to start with an intercept term (if detected).
        verbose: If True, print progress messages.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "model": The best fitted model.
            - "selected_features": List of selected feature names.
            - "history": pd.DataFrame with stepwise history.
    """
    keep_vars = set(keep_vars or [])

    # -------------------------
    # Unwrap the provided model
    # -------------------------
    if hasattr(model, "x_vars"):  # SupervisedModel wrapper
        x_vars: List[str] = model.x_vars.copy()
        y_var: str = model.y_var
        data: pd.DataFrame = model.data
        model_type = model.model_type
        feature_names = x_vars

        def fit(subset_vars: List[str]):
            X = data[subset_vars].values
            y = data[y_var].values
            return model_type(X, y)

        # Find intercept by name or const column
        intercept_name = None
        if "(intercept)" in feature_names:
            intercept_name = "(intercept)"
        elif prefer_intercept:
            for name in feature_names:
                col = data[name].values
                if _is_intercept_col(col, name):
                    intercept_name = name
                    break

        all_pool = feature_names.copy()
        start_set_vars: List[str] = []
        if prefer_intercept and intercept_name:
            start_set_vars = [intercept_name]
            keep_vars.add(intercept_name)

        current_vars = sorted(set(start_set_vars) | keep_vars, key=lambda v: feature_names.index(v) if v in feature_names else 10**9)

        # initial fit on the full model we were given (for baseline score)
        full_model_score, lower_is_better = _score_generic(model.model if hasattr(model, "model") else model, criterion)

    else:  # raw supervised_model
        X = model.x
        y = model.y
        p = X.shape[1]
        feature_names = [f"x{i}" for i in range(p)]

        def fit(subset_idx: List[int]):
            if len(subset_idx) == 0:
                # Can't fit a 0-column model; fallback to best single variable start
                raise ValueError("Empty model: no columns to fit.")
            return type(model)(X[:, subset_idx], y)

        # Detect intercept column (constant)
        intercept_idx = None
        if prefer_intercept:
            for j in range(p):
                if _is_intercept_col(X[:, j], feature_names[j]):
                    intercept_idx = j
                    break

        all_pool_idx = list(range(p))
        current_idx: List[int] = []
        if prefer_intercept and intercept_idx is not None:
            current_idx = [intercept_idx]

        # baseline score from supplied model
        full_model_score, lower_is_better = _score_generic(model, criterion)

    # ----------------------------------------
    # Helper to get candidates and do 1-step add
    # ----------------------------------------
    history = []
    step = 0

    def current_feature_list():
        if hasattr(model, "x_vars"):
            return current_vars.copy()
        return [feature_names[i] for i in current_idx]

    # Initialize with starting (possibly intercept-only or keep_vars)
    try:
        if hasattr(model, "x_vars"):
            init = fit(current_vars if current_vars else [])
        else:
            init = fit(current_idx if current_idx else [])
        best_score, _ = _score_generic(init, criterion)
        best_model = init
    except Exception:
        # If intercept-only (or empty) fails, fall back to greedy best single variable
        trial = []
        if hasattr(model, "x_vars"):
            for v in [v for v in all_pool if v not in keep_vars]:
                try:
                    m = fit([v] + sorted(list(keep_vars)))
                    s, _ = _score_generic(m, criterion)
                    trial.append((v, s, m))
                except Exception:
                    continue
            if not trial:
                raise
            v_best, s_best, m_best = (min if lower_is_better else max)(trial, key=lambda t: t[1])
            current_vars = sorted(set([v_best]) | keep_vars, key=lambda v: feature_names.index(v) if v in feature_names else 10**9)
            best_score, best_model = s_best, m_best
        else:
            pool = [j for j in all_pool_idx if j not in (current_idx)]
            trial = []
            for j in pool:
                try:
                    m = fit(sorted(list(set([j]) | set(current_idx))))
                    s, _ = _score_generic(m, criterion)
                    trial.append((j, s, m))
                except Exception:
                    continue
            if not trial:
                raise
            j_best, s_best, m_best = (min if lower_is_better else max)(trial, key=lambda t: t[1])
            current_idx = sorted(set([j_best]) | set(current_idx))
            best_score, best_model = s_best, m_best

    history.append({"step": step, "added": None, "score": best_score, "features": current_feature_list()})

    # Limit size
    target_max = max_features if max_features is not None else (len(feature_names))

    # ------------------------------------------------
    # Greedy forward: add the single best new variable
    # ------------------------------------------------
    while True:
        if hasattr(model, "x_vars"):
            pool = [v for v in all_pool if v not in current_vars]
            if len(current_vars) >= target_max or len(pool) == 0:
                break

            trials = []
            for v in pool:
                try:
                    cand = current_vars + [v]
                    m = fit(cand)
                    sc, _ = _score_generic(m, criterion)
                    trials.append((v, sc, m))
                except Exception as e:
                    if verbose:
                        print(f"skip add {v}: {e}")
                    continue

            if not trials:
                break
            v_best, s_best, m_best = (min if lower_is_better else max)(trials, key=lambda t: t[1])
            improved = (s_best < best_score) if lower_is_better else (s_best > best_score)
            if not improved:
                break

            current_vars.append(v_best)
            best_score, best_model = s_best, m_best
            step += 1
            if verbose:
                print(f"[+] add {v_best} -> {criterion}={best_score:.6g} (k={len(current_vars)})")
            history.append({"step": step, "added": v_best, "score": best_score, "features": current_feature_list()})

        else:
            pool = [j for j in all_pool_idx if j not in current_idx]
            if len(current_idx) >= target_max or len(pool) == 0:
                break

            trials = []
            for j in pool:
                try:
                    cand = current_idx + [j]
                    m = fit(cand)
                    sc, _ = _score_generic(m, criterion)
                    trials.append((j, sc, m))
                except Exception as e:
                    if verbose:
                        print(f"skip add {feature_names[j]}: {e}")
                    continue

            if not trials:
                break
            j_best, s_best, m_best = (min if lower_is_better else max)(trials, key=lambda t: t[1])
            improved = (s_best < best_score) if lower_is_better else (s_best > best_score)
            if not improved:
                break

            current_idx.append(j_best)
            best_score, best_model = s_best, m_best
            step += 1
            if verbose:
                print(f"[+] add {feature_names[j_best]} -> {criterion}={best_score:.6g} (k={len(current_idx)})")
            history.append({"step": step, "added": feature_names[j_best], "score": best_score, "features": current_feature_list()})

    # Wrap result consistently
    if hasattr(model, "x_vars"):
        selected = current_vars
        final_model = best_model
    else:
        selected = [feature_names[i] for i in current_idx]
        final_model = best_model

    return {
        "model": final_model,
        "selected_features": selected,
        "history": pd.DataFrame(history),
    }


def _metric_value(m: Any, metric: str) -> float:
    """Extract a metric value from a model or wrapper.

    Tries, in order:
      1) Direct attribute on the object.
      2) Attribute on a wrapped `.model`.
      3) Column in a `.glance` DataFrame.

    Args:
        m: Model object or wrapper.
        metric: Metric name (e.g., "aic", "bic", "r_squared").

    Returns:
        float: The extracted metric value.

    Raises:
        AttributeError: If the metric cannot be found anywhere.
    """
    metric = metric.lower()
    # direct attribute
    if hasattr(m, metric):
        return float(getattr(m, metric))
    # wrapped model
    if hasattr(m, "model") and hasattr(m.model, metric):
        return float(getattr(m.model, metric))
    # glance fallback
    if hasattr(m, "glance"):
        g = m.glance
        if isinstance(g, pd.DataFrame) and metric in g.columns:
            return float(g.iloc[0][metric])
    raise AttributeError(f"Could not find metric '{metric}' on model.")


def _lower_is_better(metric: str) -> bool:
    """Determine if the given metric is minimized.

    Args:
        metric: Metric name.

    Returns:
        bool: True if lower values indicate better fit (e.g., AIC, BIC, MSE).
    """
    metric = metric.lower()
    return metric in ("aic", "bic", "mse", "misclassification_probability")


def _compare_models(m1: Any, m2: Any, metrics: List[str], tol: float = 1e-12) -> Tuple[Any, Dict[str, str]]:
    """Compare two models across multiple metrics by majority vote.

    For each metric, the better model receives one vote. Ties on a metric
    give no votes. If the total vote is tied, the first metric in `metrics`
    is used as the final tie-breaker.

    Args:
        m1: First model to compare.
        m2: Second model to compare.
        metrics: List of metrics to evaluate (e.g., ["aic", "bic"]).
        tol: Absolute tolerance for considering two values a tie.

    Returns:
        Tuple[Any, Dict[str, str]]: (winner_model, per_metric_winner)
            - winner_model: The better of (m1, m2) by the voting rule.
            - per_metric_winner: Dict mapping metric → {"m1","m2","tie"}.
    """
    votes = {"m1": 0, "m2": 0, "ties": 0}
    per_metric = {}
    for met in metrics:
        try:
            v1 = _metric_value(m1, met)
            v2 = _metric_value(m2, met)
        except Exception:
            # if metric not available on either, skip
            continue
        better_is_lower = _lower_is_better(met)
        diff = v1 - v2
        if abs(diff) <= tol:
            votes["ties"] += 1
            per_metric[met] = "tie"
        else:
            if (diff < 0) == better_is_lower:  # m1 better
                votes["m1"] += 1
                per_metric[met] = "m1"
            else:
                votes["m2"] += 1
                per_metric[met] = "m2"

    if votes["m1"] > votes["m2"]:
        winner = m1
    elif votes["m2"] > votes["m1"]:
        winner = m2
    else:
        # tie across votes: fall back to first metric in the list (if available)
        first = metrics[0]
        try:
            v1 = _metric_value(m1, first)
            v2 = _metric_value(m2, first)
            better_is_lower = _lower_is_better(first)
            if (v1 - v2 < 0) == better_is_lower:
                winner = m1
            else:
                winner = m2
        except Exception:
            # if even that fails, default to m1
            winner = m1

    return winner, per_metric


def stepwise(
    model: Any,
    direction: str = "both",
    criterion: str = "aic",
    keep_vars: List[str] | None = None,
    min_features: int = 1,
    max_features: int | None = None,
    prefer_intercept: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Unified stepwise selection wrapper.

    Routes to forward, backward, or both directions and returns the
    best model by a vote across metrics when `direction="both"`.

    Args:
        model: Model object. Either:
            - Raw supervised_model with `.x` and `.y`.
            - SupervisedModel wrapper with `.x_vars`, `.y_var`, `.data`, `.model_type`.
        direction: Stepwise direction:
            - "forwards": Forward selection.
            - "backwards": Backward elimination.
            - "both": Run both and select the better.
        criterion: Model selection criterion ("aic", "bic", "mse", etc.).
        keep_vars: Variable names that must always be included.
        min_features: Minimum number of features (for backward).
        max_features: Maximum number of features (for forward).
        prefer_intercept: If True, prefer/include an intercept where applicable.
        verbose: If True, print selection progress.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "model": Best fitted model.
            - "selected_features": List of chosen features.
            - "history": pd.DataFrame of the chosen direction's history.
            - "direction_chosen": One of {"forwards","backwards"}.
            - "comparison": Dict of per-metric winners (only if direction="both").
    """
    direction = direction.lower()
    keep_vars = keep_vars or []

    if direction == "forwards":
        fwd = forward_stepwise(
            model,
            criterion=criterion,
            keep_vars=keep_vars,
            max_features=max_features,
            prefer_intercept=prefer_intercept,
            verbose=verbose,
        )
        return {
            "model": fwd["model"],
            "selected_features": fwd["selected_features"],
            "history": fwd["history"],
            "direction_chosen": "forwards",
        }

    if direction == "backwards":
        bwd = backward_stepwise(
            model,
            criterion=criterion,
            keep_vars=keep_vars,
            min_features=min_features,
            verbose=verbose,
        )
        return {
            "model": bwd["model"],
            "selected_features": bwd["selected_features"],
            "history": bwd["history"],
            "direction_chosen": "backwards",
        }

    if direction != "both":
        raise ValueError("direction must be one of {'forwards','backwards','both'}")

    # run both
    fwd = forward_stepwise(
        model,
        criterion=criterion,
        keep_vars=keep_vars,
        max_features=max_features,
        prefer_intercept=prefer_intercept,
        verbose=verbose,
    )
    bwd = backward_stepwise(
        model,
        criterion=criterion,
        keep_vars=keep_vars,
        min_features=min_features,
        verbose=verbose,
    )

    m_fwd = fwd["model"]
    m_bwd = bwd["model"]

    # build metric set: {criterion, aic, bic} (deduped, criterion first)
    metrics: List[str] = []
    for met in [criterion.lower(), "aic", "bic"]:
        if met not in metrics:
            metrics.append(met)

    winner_model, per_metric = _compare_models(m_fwd, m_bwd, metrics)

    if winner_model is m_fwd:
        chosen = {
            "model": fwd["model"],
            "selected_features": fwd["selected_features"],
            "history": fwd["history"],
            "direction_chosen": "forwards",
            "comparison": per_metric,
        }
    else:
        chosen = {
            "model": bwd["model"],
            "selected_features": bwd["selected_features"],
            "history": bwd["history"],
            "direction_chosen": "backwards",
            "comparison": per_metric,
        }
    return chosen


class LeastSquaresRegressor(PredictionModel):
    """Ordinary least squares (OLS) regression.

    A high-level wrapper around :class:`least_squares_regressor` that provides a
    formula interface and pandas-aware prediction methods. Fits a linear model
    by minimizing the sum of squared residuals.

    This class inherits from :class:`PredictionModel`, which handles parsing the
    formula, extracting variables from a DataFrame, and exposing tidy/glance
    summaries consistent with the rest of the package.

    Examples:
        Fit an OLS regression from a formula:

        >>> model = LeastSquaresRegressor("y ~ x1 + x2", data=df)
        >>> model.tidy         # coefficient table
        >>> model.glance       # model summary
        >>> preds = model.predict(df)

    Attributes:
        MODEL_TYPE (Type[supervised_model]): The underlying implementation
            (:class:`least_squares_regressor`).
        formula (str): Formula string used to specify the model.
        x_vars (list[str]): Predictor variable names.
        y_var (str): Response variable name.
        data (pd.DataFrame): Parsed DataFrame containing predictors and response.
        model (least_squares_regressor): Fitted underlying OLS model.
    """

    MODEL_TYPE = least_squares_regressor


class LogisticRegressor(PredictionModel):
    """Logistic regression for binary outcomes.

    A high-level wrapper around :class:`logistic_regressor` that provides a
    formula interface and pandas-aware prediction methods. Fits a generalized
    linear model with a logit link, estimating probabilities for binary response
    variables.

    This class inherits from :class:`PredictionModel`, which handles parsing the
    formula, extracting variables from a DataFrame, and exposing tidy/glance
    summaries consistent with the rest of the package.

    Examples:
        Fit a logistic regression model from a formula:

        >>> model = LogisticRegressor("y ~ x1 + x2", data=df)
        >>> model.tidy          # coefficient table with log-odds
        >>> model.glance        # model fit summary (AIC, log-likelihood, etc.)
        >>> probs = model.predict(df)   # predicted probabilities

    Attributes:
        MODEL_TYPE (Type[supervised_model]): The underlying implementation
            (:class:`logistic_regressor`).
        formula (str): Formula string used to specify the model.
        x_vars (list[str]): Predictor variable names.
        y_var (str): Response variable name.
        data (pd.DataFrame): Parsed DataFrame containing predictors and response.
        model (logistic_regressor): Fitted underlying logistic regression model.
    """

    MODEL_TYPE = logistic_regressor


class PoissonRegressor(PredictionModel):
    """Poisson regression for count outcomes.

    A high-level wrapper around :class:`poisson_regressor` that provides a
    formula interface and pandas-aware prediction methods. Fits a generalized
    linear model with a log link, appropriate for count data where the variance
    is proportional to the mean.

    This class inherits from :class:`PredictionModel`, which handles parsing the
    formula, extracting variables from a DataFrame, and exposing tidy/glance
    summaries consistent with the rest of the package.

    Examples:
        Fit a Poisson regression model from a formula:

        >>> model = PoissonRegressor("y ~ x1 + x2", data=df)
        >>> model.tidy          # coefficient table with log-incidence ratios
        >>> model.glance        # model summary (deviance, AIC, etc.)
        >>> rates = model.predict(df)   # expected counts

    Attributes:
        MODEL_TYPE (Type[supervised_model]): The underlying implementation
            (:class:`poisson_regressor`).
        formula (str): Formula string used to specify the model.
        x_vars (list[str]): Predictor variable names.
        y_var (str): Response variable name.
        data (pd.DataFrame): Parsed DataFrame containing predictors and response.
        model (poisson_regressor): Fitted underlying Poisson regression model.
    """

    MODEL_TYPE = poisson_regressor
