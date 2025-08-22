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
    def __init__(self, x, y, *args, **kwargs):
        super(linear_model, self).__init__(x, y)
        self.params = self._fit(x, y, *args, **kwargs)

    def _fit(self, x, y, *args, **kwargs): return np.linalg.solve(x.T @ x, x.T @ y)

    def predict(self, newdata): return newdata @ self.params

    def evaluate_lnL(self, pred):
        return -self.n_obs / 2 * (np.log(2 * np.pi * (self.y - pred).var()) + 1)

class logistic_regressor(linear_model,variance_model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.glance = pd.DataFrame(
            {'mcfaddens.r.squared': self.mcfaddens_r_squared,
             'ben.akiva.lerman.r.squared': self.ben_akiva_lerman_r_squared,
             'self.df': self.n_feat,
             'resid.df': self.degrees_of_freedom,
             'aic': self.aic,
             'bic': self.bic,
             'log.likelihood': self.log_likelihood,
             'deviance': self.deviance},
            index = ['']
        )
    def _fit(self,x,y):
        params,self.iters = newton(self.gradient,self.hessian,np.zeros(self.n_feat))
        return params
    @property
    def vcov_params(self) -> np.ndarray:
        H = self.hessian(self.params)
        try:
            return -np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return -np.linalg.pinv(H)

    def evaluate_lnL(self, pred: np.ndarray) -> float:
        eps = 1e-15  # or 1e-12 depending on precision needs
        pred = np.clip(pred, eps, 1 - eps)
        return self.y.T @ np.log(pred) + (1 - self.y).T @ np.log(1 - pred)
    def gradient(self,coefs):return self.x.T@(self.y-expit(self.x@coefs))
    def hessian(self,coefs):
        x = self.x
        if isinstance(x, (pd.DataFrame, pd.Series)): x = x.values
        Fx = expit(x@coefs)
        inside = np.diagflat(Fx*(1-Fx))
        return -x.T@inside@x
    def predict(self,target):return expit(target@self.params)
    @property
    def mcfaddens_r_squared(self): return 1-self.log_likelihood/self.null_likelihood
    @property
    def ben_akiva_lerman_r_squared(self):
        return (self.y.T@self.fitted+(1-self.y).T@(1-self.fitted))/self.n_obs

    # In glm.py, inside class logistic_regressor

    def marginal_effects(
            self,
            newx: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            average: bool = True,
    ) -> np.ndarray:
        X = self.x if newx is None else newx

        # Accept pandas
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        # Optional convenience: if model has an intercept column of 1s and
        # caller provided X with one fewer column, prepend ones.
        if X.ndim == 2 and X.shape[1] != self.n_feat:
            if self.x.ndim == 2 and np.allclose(self.x[:, 0], 1) and X.shape[1] == self.n_feat - 1:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            else:
                raise ValueError(f"newx has shape {X.shape}, but model expects {self.n_feat} features.")

        xb = X @ self.params
        Fx = expit(xb)
        slope = Fx * (1.0 - Fx)  # derivative of expit
        effects = slope.reshape(-1, 1) * self.params.reshape(1, -1)
        return effects.mean(0) if average else effects

class least_squares_regressor(linear_model, variance_model):
    def __init__(self, x, y, white: bool = False, hc: int = 3, *args, **kwargs):
        super(least_squares_regressor, self).__init__(x, y, *args, **kwargs)
        self.white = white
        self.hc = hc
        self.glance = pd.DataFrame({'r.squared': self.r_squared,
                                    'adjusted.r.squared': self.adjusted_r_squared,
                                    'self.df': self.n_feat,
                                    'resid.df': self.degrees_of_freedom,
                                    'aic': self.aic,
                                    'bic': self.bic,
                                    'log.likelihood': self.log_likelihood,
                                    'deviance': self.deviance,
                                    'resid.var': self.residual_variance}, index=[''])

    @property
    def vcov_params(self):
        if self.white:
            return self.__white(self.hc)
        return np.linalg.inv(self.x.T @ self.x) * self.residual_variance

    def __white(self, hc):
        e = self.residuals.values if type(self.residuals) == pd.Series else self.residuals
        esq = self.__hc_correction(e ** 2, hc)
        meat = np.diagflat(esq)
        bread = np.linalg.inv(self.x.T @ self.x) @ self.x.T
        return bread @ meat @ bread.T

    def __hc_correction(self, esq, hc):
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
            case _:
                pass
        return esq

    @property
    def _glance_dict(self):
        return



class poisson_regressor(linear_model):
    def _fit(self, x, y):
        params, self.iters = newton(self.gradient, self.hessian, np.zeros(self.n_feat))
        return params

    @property
    def vcov_params(self): return -np.linalg.inv(self.hessian(self.params))

    def evaluate_lnL(self, pred):
        return self.y.T @ np.log(pred) - np.ones((1, self.n_obs)) @ pred + np.ones((1, self.n_obs)) @ np.log(sp.special.factorial(self.y))

    def gradient(self, coefs): return self.x.T @ (self.y - np.exp(self.x @ coefs))

    def hessian(self, coefs):
        Fx = np.exp(self.x @ coefs)
        if type(Fx) == pd.DataFrame or type(Fx) == pd.Series: Fx = Fx.values
        return -self.x.T @ np.diagflat(Fx) @ self.x

    def predict(self, target): return np.exp(target @ self.params)

    @property
    def _glance_dict(self):
        return {'self.df': self.n_feat,
                'resid.df': self.degrees_of_freedom,
                'aic': self.aic,
                'bic': self.bic,
                'log.likelihood': self.log_likelihood,
                'deviance': self.deviance}


LeastSquaresRegressor = partial(PredictionModel,model_type=least_squares_regressor)
LogisticRegressor = partial(PredictionModel,model_type=logistic_regressor)
PoissonRegressor = partial(PredictionModel,model_type=poisson_regressor)

def backward_stepwise(model: Any,
                      criterion: str = "aic",
                      keep_vars: list[str] = None,
                      min_features: int = 1,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Generic backward stepwise for either supervised_model (x,y)
    or SupervisedModel wrapper (x_vars, y_var, data).
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

        j_best, sc_best, best_model = min(trial, key=lambda t: t[1]) if lower else max(trial, key=lambda t: t[1])
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

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

def _is_intercept_col(col: np.ndarray, name: Optional[str]) -> bool:
    if name is not None and name.strip().lower() == '(intercept)':
        return True
    return np.allclose(col, col[0])

def _score_generic(model: Any, criterion: str) -> tuple[float, bool]:
    c = criterion.lower()
    val = getattr(model, c)
    lower_is_better = c in ("aic", "bic", "mse", "misclassification_probability")
    return float(val), lower_is_better

def forward_stepwise(
    model: Any,
    criterion: str = "aic",
    keep_vars: Optional[List[str]] = None,
    max_features: Optional[int] = None,
    prefer_intercept: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Forward stepwise selection starting from intercept-only if present.
    Accepts either:
      - raw supervised_model with model.x, model.y
      - SupervisedModel wrapper with .x_vars, .y_var, .data, .model_type
    Returns: {'model', 'selected_features', 'history'}
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
    """
    Robustly extract metric from either a raw model or a SupervisedModel wrapper.
    Tries attribute on the object, then on .model, then in .glance dataframe.
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
    metric = metric.lower()
    return metric in ("aic", "bic", "mse", "misclassification_probability")

def _compare_models(m1: Any, m2: Any, metrics: List[str], tol: float = 1e-12) -> Tuple[Any, Dict[str, str]]:
    """
    Return the better of (m1, m2) by a majority vote across metrics.
    Each metric contributes one 'vote'. Ties use the first metric as tie-breaker.
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
    """
    Unified stepwise selector.
    - direction='forwards' → forward_stepwise
    - direction='backwards' → backward_stepwise
    - direction='both' → run both and return the better result by a vote across {criterion, aic, bic}
      (deduplicated if criterion is 'aic' or 'bic').

    Returns:
      {
        'model': best_model,
        'selected_features': list[str],
        'history': pd.DataFrame,           # from the chosen direction
        'direction_chosen': 'forwards'|'backwards',
        'comparison' : {'metric': 'winner', ...}  # only when direction='both'
      }
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
