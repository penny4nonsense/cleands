"""
Shrinkage and regularization models for regression.

Implements L1-regularized (Lasso) regression using quadratic programming
via `cvxopt`, along with a cross-validated variant to pick the penalty and
a bootstrap variant for uncertainty quantification.

Classes
-------
l1_regularization_regressor
    Lasso with a fixed L1 threshold.
l1_cross_validation_regressor
    Selects the L1 threshold by k-fold cross-validation.
l1_bootstrap_regressor
    Cross-validated Lasso with bootstrap-based variance estimation.

Factory Aliases
---------------
L1BootstrapRegressor
    `PredictionModel` wrapper for `l1_bootstrap_regressor`.
"""

from itertools import product
import numpy as np
import cvxopt
from ..utils import *
from ..base import *
from .glm import linear_model, least_squares_regressor
from typing import Optional
from functools import partial

cvxopt.solvers.options['show_progress'] = False


class l1_regularization_regressor(linear_model):
    """L1-regularized (Lasso) linear regression.

    Solves a least-squares problem subject to an L1 constraint on the
    coefficients. If an intercept column of ones is detected as the first
    or last column of `x`, it is treated without penalization.

    Attributes:
        threshold (float): L1 budget (sum of absolute coefficients).
    """

    def __init__(self, x, y, thresh: float, *args, **kwargs):
        """Initialize and fit a Lasso model.

        Args:
            x (np.ndarray): Design matrix of shape (n_obs, n_features).
            y (np.ndarray): Response vector of shape (n_obs,) or (n_obs, 1).
            thresh (float): L1 constraint level (sum |beta_j| ≤ thresh).
            *args: Forwarded to base classes.
            **kwargs: Forwarded to base classes.
        """
        super().__init__(x, y, thresh=thresh, *args, **kwargs)
        self.threshold = thresh

    def _fit(self, x, y, thresh: float, *args, **kwargs):
        """Estimate coefficients under an L1 constraint.

        Detects an intercept column (all ones) as first or last column and
        leaves it unpenalized by centering the remaining features/response,
        then recenters the intercept.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training response.
            thresh (float): L1 constraint level.

        Returns:
            np.ndarray: Estimated parameter vector of length `x.shape[1]`.
        """
        if np.all(x[:, 0] == 1):
            dx = x[:, 1:] - x[:, 1:].mean(0)
            dy = y - y.mean(0)
            outp = l1_regularization_regressor.solve_lasso(dx, dy, thresh)
            intc = y.mean(0) - x[:, 1:].mean(0) @ outp.reshape(-1, 1)
            return np.concatenate([intc, outp])
        elif np.all(x[:, -1] == 1):
            dx = x[:, :-1] - x[:, :-1].mean(0)
            dy = y - y.mean(0)
            outp = l1_regularization_regressor.solve_lasso(dx, dy, thresh)
            intc = y.mean(0) - x[:, -1:].mean(0) @ outp.reshape(-1, 1)
            return np.concatenate([outp, intc])
        else:
            return l1_regularization_regressor.solve_lasso(x, y, thresh)

    @staticmethod
    def solve_lasso(x, y, thresh):
        """Solve the Lasso via a QP with nonnegativity / L1 budget.

        Uses the standard positive/negative split of coefficients `b = b+ - b-`
        with `b+, b- ≥ 0` and `sum(b+ + b-) ≤ thresh`.

        Args:
            x (np.ndarray): Feature matrix (n_obs, n_features).
            y (np.ndarray): Response vector (n_obs,) or (n_obs, 1).
            thresh (float): L1 constraint level.

        Returns:
            np.ndarray: Coefficient vector (n_features,) for the original variables.
        """
        r = x.shape[1]
        P = np.kron(np.array([[1, -1], [-1, 1]]), x.T @ x)
        q = -np.kron(np.array([[1], [-1]]), x.T @ y.reshape(-1, 1))
        G = np.vstack([-np.eye(2 * r), np.ones((1, 2 * r))])
        h = np.vstack([np.zeros((2 * r, 1)), np.array([[thresh]])])
        b = np.array(cvxopt.solvers.qp(*[cvxopt.matrix(i) for i in [P, q, G, h]])['x'])
        return b[:r, 0] - b[r:, 0]


class l1_cross_validation_regressor(l1_regularization_regressor):
    """Cross-validated L1-regularized regression.

    Searches over a grid of λ values (scaled to `max_thresh`) and selects
    the one minimizing mean squared error via k-fold cross-validation.

    Attributes:
        statistic (float): Best cross-validated mean squared error.
        lambda_value (float): Selected λ in [0, 1] (scales `max_thresh`).
        max_threshold (float): Max L1 budget used to scale λ.
    """

    def __init__(self, x, y, max_thresh: Optional[int] = None, folds: int = 5, seed=None, *args, **kwargs):
        """Initialize and select λ by cross-validation, then fit.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training response.
            max_thresh (float, optional): Maximum L1 budget to scale λ in [0, 1].
                If None, defaults to `sum(abs(OLS(x, y).params[1:]))`.
            folds (int): Number of CV folds (default 5).
            seed (int, optional): Random seed for CV fold shuffling.
            *args: Forwarded to parent constructor.
            **kwargs: Forwarded to parent constructor.
        """
        default_state = cvxopt.solvers.options.get('show_progress', True)
        cvxopt.solvers.options['show_progress'] = False
        if max_thresh == None: max_thresh = np.abs(linear_model(x, y).params[1:]).sum()
        cv = k_fold_cross_validation(x, y, folds=folds, seed=seed)
        lam_values = np.linspace(0, 1, 100)
        mses = np.zeros(lam_values.shape[0])
        for i,lam in enumerate(lam_values):
            mses[i] = sum([l1_regularization_regressor(x_train, y_train, thresh=lam * max_thresh).out_of_sample_mean_squared_error(x_test, y_test) for x_train,x_test,y_train,y_test in cv])/folds
        i = mses.argmin()
        cvxopt.solvers.options['show_progress'] = default_state
        super().__init__(x, y, thresh=lam_values[i] * max_thresh, *args, **kwargs)
        self.statistic = mses[i]
        self.lambda_value = lam_values[i]
        self.max_threshold = max_thresh


class l1_bootstrap_regressor(l1_cross_validation_regressor,variance_model):
    """Bootstrap-augmented cross-validated Lasso.

    Fits the cross-validated Lasso then estimates parameter variability via
    bootstrap resampling and constructs a `.glance` summary table.

    Attributes:
        n_boot (int): Number of bootstrap replications.
        bootstraps (list[l1_regularization_regressor]): Fitted bootstrap models.
        bootstrap_params (np.ndarray): Coefficients from bootstrap models (n_boot, p).
        glance (pd.DataFrame): Summary metrics assembled post-fit.
    """

    def __init__(self,x,y,*args,bootstraps:int=1000,**kwargs):
        """Fit CV Lasso, then run bootstrap for variance estimation.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training response.
            bootstraps (int): Number of bootstrap draws (default 1000).
            *args: Forwarded to parent constructors.
            **kwargs: Forwarded to parent constructors.
        """
        super().__init__(x,y,*args,**kwargs)
        self.n_boot = bootstraps
        model = lambda x,y: l1_regularization_regressor(x,y,thresh=self.threshold)
        default_state = cvxopt.solvers.options.get('show_progress',True)
        cvxopt.solvers.options['show_progress'] = False
        self.bootstraps = bootstrap(model,x,y,bootstraps=bootstraps)
        cvxopt.solvers.options['show_progress'] = default_state
        self.bootstrap_params = np.array([item.params for item in self.bootstraps])
        self.glance = pd.DataFrame({'r.squared':self.r_squared,
                                        'adjusted.r.squared':self.adjusted_r_squared,
                                        'self.df':self.n_feat,
                                        'resid.df':self.degrees_of_freedom,
                                        'aic':self.aic,
                                        'bic':self.bic,
                                        'log.likelihood':self.log_likelihood,
                                        'deviance':self.deviance,
                                        'resid.var':self.residual_variance},index=[''])

    @property
    def vcov_params(self):
        """Bootstrap covariance of the parameter estimates.

        Returns:
            np.ndarray: (p x p) covariance matrix computed from bootstrap coefficients.
        """
        x = self.bootstrap_params-self.bootstrap_params.mean(0)
        return x.T@x/self.n_boot


class L1BootstrapRegressor(PredictionModel):
    """Convenience wrapper for L1-regularized regression with bootstrap inference.

    This model applies L1 (lasso) regularization to linear regression
    and uses bootstrapping to estimate parameter variability and
    robust standard errors. Provides a formula/DataFrame interface for
    the :class:`l1_bootstrap_regressor`.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`l1_bootstrap_regressor`.

    Example:
        >>> model = L1BootstrapRegressor.from_formula("y ~ x1 + x2", data=df, bootstraps=500)
        >>> model.tidy  # parameter estimates
        >>> model.glance  # model fit statistics
    """

    MODEL_TYPE = l1_bootstrap_regressor
