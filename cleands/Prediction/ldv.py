"""
ldv.py — Limited Dependent Variable (LDV) Models
================================================

This module implements models for **limited dependent variables** where the
outcome is only partially observed due to censoring, truncation, or selection
processes. These models extend standard regression by explicitly accounting
for restricted observability of the dependent variable.

Models currently implemented
----------------------------
Tobit regression (two-limit censored normal)
    Latent variable:
        y* = Xβ + ε,   ε ~ N(0, σ²)

    Observed variable:
        y = L if y* ≤ L      (left-censored)
        y = y* if L < y* < R (uncensored)
        y = R if y* ≥ R      (right-censored)

    Features:
        * Supports left- and/or right-censoring (finite or infinite).
        * Fits parameters by maximum likelihood (L-BFGS-B).
        * Returns estimates of β and σ with variance-covariance matrix.
        * Provides log-likelihood, AIC, BIC, deviance, convergence info.
        * Includes:
            - predict() for latent mean μ = Xβ
            - expected_observed() for E[y | X] under censoring
            - censoring_probs() for P_left, P_uncensored, P_right.

Planned models
--------------
Truncated regression
    * Similar to Tobit but assumes data outside [L, R] are unobserved
      (not censored).
    * Log-likelihood excludes truncated cases entirely.
    * Useful for survey data where only responses in a restricted range
      are collected.

Heckman selection model (two-step / full MLE)
    * Jointly models outcome and selection equations to correct for
      sample selection bias.
    * Outcome observed only if selection variable exceeds threshold.
    * Widely applied in labor economics, health economics, and marketing.

Classes
-------
tobit_regressor
    Core implementation of two-limit Tobit regression with MLE fitting,
    prediction, and inference utilities.

Factory Aliases
---------------
TobitRegressor
    Partial wrapper that exposes `tobit_regressor` through the
    `PredictionModel` interface for pandas DataFrame/formula use.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from cleands.Prediction.ldv import TobitRegressor
>>> df = pd.DataFrame({"x1": np.random.randn(100), "y": np.random.randn(100)})
>>> model = TobitRegressor(x_vars=["x1"], y_var="y", data=df, L=0.0)
>>> model.glance
"""


import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from typing import Optional, Tuple
from functools import partial

from ..base import prediction_model, prediction_likelihood_model, variance_model, PredictionModel
from ..utils import hstack  # optional if you want to help users add an intercept


class tobit_regressor(prediction_model, prediction_likelihood_model, variance_model):
    """
    Two-limit Tobit (censored normal) regression model.

    Latent model:
        y* = Xβ + ε,  ε ~ N(0, σ²)

    Observed:
        - y = L if y* ≤ L   (left-censored)
        - y = y* if L < y* < R (uncensored)
        - y = R if y* ≥ R   (right-censored)

    Parameters are stored as:
        params = [β, σ] with σ > 0

    `predict(X)` returns the latent mean μ = Xβ.
    Use `expected_observed(X)` to compute E[y | X] under censoring.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        L: Optional[float] = 0.0,
        R: Optional[float] = None,
        add_intercept: bool = False,
        start: Optional[Tuple[np.ndarray, float]] = None,
        tol: float = 1e-8
    ) -> None:
        """
        Initialize a Tobit regression model and fit parameters via MLE.

        Args:
            x (np.ndarray): Design matrix of shape (n_obs, n_features).
            y (np.ndarray): Observed outcomes.
            L (float, optional): Left-censoring limit. Defaults to 0.0.
            R (float, optional): Right-censoring limit. Defaults to None (= +∞).
            add_intercept (bool, optional): Whether to prepend an intercept column of ones. Defaults to False.
            start (tuple, optional): Optional initial (β, σ) guess. Defaults to OLS-based warm start.
            tol (float, optional): Numerical tolerance for censoring classification. Defaults to 1e-8.
        """
        if add_intercept:
            x = hstack(np.ones(x.shape[0]), x)

        super().__init__(x, y)
        self.L = -np.inf if L is None else float(L)
        self.R = np.inf if R is None else float(R)
        self._tol = tol

        # Censoring masks
        self._left = (self.y <= self.L + self._tol)
        self._right = (self.y >= self.R - self._tol)
        self._unc = ~(self._left | self._right)

        # Initialize β and σ
        if start is not None:
            beta0, sigma0 = start
        else:
            try:
                beta0, *_ = np.linalg.lstsq(self.x, self.y, rcond=None)
            except np.linalg.LinAlgError:
                beta0 = np.zeros(self.n_feat)
            resid = self.y - self.x @ beta0
            sigma0 = float(np.sqrt(np.maximum(resid.var(), 1e-3)))

        theta0 = np.concatenate([beta0, np.array([np.log(sigma0)])])

        # Optimize log-likelihood
        res = minimize(self._objective_from_theta, theta0, method="L-BFGS-B", tol=1e-9)
        self.converged = bool(res.success)
        self.message = res.message
        self.nit = res.nit

        # Store fitted parameters
        self.beta = res.x[:-1]
        self.log_sigma = res.x[-1]
        self.sigma = float(np.exp(self.log_sigma))
        self.params = np.concatenate([self.beta, np.array([self.sigma])])

        # Covariance of parameters via delta method
        self._vcov_theta = None
        try:
            Hinv = res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else np.array(res.hess_inv)
            self._vcov_theta = np.array(Hinv)
        except Exception:
            self._vcov_theta = np.full((self.n_feat + 1, self.n_feat + 1), np.nan)

        r = self.n_feat
        V = np.zeros((r + 1, r + 1))
        V[:r, :r] = self._vcov_theta[:r, :r]
        V_beta_s = self._vcov_theta[:r, r] * self.sigma
        V[:r, r] = V_beta_s
        V[r, :r] = V_beta_s
        V[r, r] = (self.sigma ** 2) * self._vcov_theta[r, r]
        self._vcov_params = V

        # Summary glance DataFrame
        k = self.n_feat + 1
        self.glance = pd.DataFrame({
            'self.df': [k],
            'resid.df': [max(self.n_obs - k, 1)],
            'aic': [self.aic],
            'bic': [self.bic],
            'log.likelihood': [self.log_likelihood],
            'deviance': [self.deviance],
            'converged': [self.converged],
            'nit': [self.nit],
            'message': [str(self.message)]
        })

    def _objective_from_theta(self, theta: np.ndarray) -> float:
        """
        Internal optimizer hook. Computes negative log-likelihood at θ.

        Args:
            theta (np.ndarray): Candidate parameter vector [β, logσ].

        Returns:
            float: Negative log-likelihood value.
        """
        beta = theta[:-1]
        sigma = float(np.exp(theta[-1]))
        mu = self.x @ beta
        old_sigma, had_sigma = getattr(self, "sigma", None), hasattr(self, "sigma")
        self.sigma = sigma
        try:
            ll = self.evaluate_lnL(mu)
        finally:
            if had_sigma:
                self.sigma = old_sigma
            else:
                del self.sigma
        return -float(ll)

    def _split_sets(self, mu: np.ndarray, sigma: float):
        """
        Split observations into left-, right-, and uncensored sets.

        Args:
            mu (np.ndarray): Predicted latent mean values.
            sigma (float): Current standard deviation.

        Returns:
            tuple: (Lmask, Rmask, Umask, yU, muU, muL, muR)
        """
        Lmask, Rmask, Umask = self._left, self._right, self._unc
        yU, muU = self.y[Umask], mu[Umask]
        muL, muR = mu[Lmask], mu[Rmask]
        return (Lmask, Rmask, Umask, yU, muU, muL, muR)

    def predict(self, target: np.ndarray) -> np.ndarray:
        """
        Predict latent mean μ = Xβ.

        Args:
            target (np.ndarray): New design matrix.

        Returns:
            np.ndarray: Latent means.
        """
        return target @ self.beta

    def expected_observed(self, target: np.ndarray) -> np.ndarray:
        """
        Predict expected observed y under censoring.

        Args:
            target (np.ndarray): New design matrix.

        Returns:
            np.ndarray: Expected observed values, E[y | X].
        """
        mu = self.predict(target)
        sigma = self.sigma
        zL, zR = (self.L - mu) / sigma, (self.R - mu) / sigma
        PhiL = np.where(np.isfinite(self.L), sp.stats.norm.cdf(zL), 0.0)
        PhiR = np.where(np.isfinite(self.R), sp.stats.norm.cdf(zR), 1.0)
        phiL = np.where(np.isfinite(self.L), sp.stats.norm.pdf(zL), 0.0)
        phiR = np.where(np.isfinite(self.R), sp.stats.norm.pdf(zR), 0.0)
        return (self.L * PhiL) + (self.R * (1 - PhiR)) + mu * (PhiR - PhiL) + sigma * (phiL - phiR)

    def censoring_probs(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute probabilities of being left-censored, uncensored, or right-censored.

        Args:
            target (np.ndarray): New design matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (P_left, P_uncensored, P_right) for each observation.
        """
        mu = self.predict(target)
        sigma = self.sigma
        zL, zR = (self.L - mu) / sigma, (self.R - mu) / sigma
        P_left = np.where(np.isfinite(self.L), sp.stats.norm.cdf(zL), 0.0)
        P_right = np.where(np.isfinite(self.R), sp.stats.norm.sf(zR), 0.0)
        P_unc = 1.0 - P_left - P_right
        return P_left, P_unc, P_right

    def evaluate_lnL(self, pred: np.ndarray) -> float:
        """
        Evaluate the log-likelihood at a given latent mean μ.

        Args:
            pred (np.ndarray): Latent mean predictions μ = Xβ.

        Returns:
            float: Log-likelihood value.
        """
        mu = np.asarray(pred).reshape(-1)
        sigma = self.sigma
        Lmask, Rmask, Umask, yU, muU, muL, muR = self._split_sets(mu, sigma)
        ll = 0.0
        if Umask.any():
            t = (yU - muU) / sigma
            ll += (-0.5 * t**2 - np.log(sigma) - 0.5*np.log(2*np.pi)).sum()
        if Lmask.any() and np.isfinite(self.L):
            ll += sp.stats.norm.logcdf((self.L - muL) / sigma).sum()
        if Rmask.any() and np.isfinite(self.R):
            ll += sp.stats.norm.logsf((self.R - muR) / sigma).sum()
        return float(ll)

    @property
    def vcov_params(self) -> np.ndarray:
        """
        Variance-covariance matrix of parameter estimates.

        Returns:
            np.ndarray: (r+1) x (r+1) covariance matrix for [β, σ].
        """
        return self._vcov_params


class TobitRegressor(PredictionModel):
    """Convenience wrapper for Tobit regression.

    The Tobit model is used for censored dependent variables, where
    observations below (or above) a threshold are censored rather than
    fully observed. This wrapper provides a formula/DataFrame interface
    for the :class:`tobit_regressor`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`tobit_regressor`.

    Example:
        >>> model = TobitRegressor.from_formula("y ~ x1 + x2", data=df)
        >>> model.predict(df[["x1", "x2"]])
    """

    MODEL_TYPE = tobit_regressor


