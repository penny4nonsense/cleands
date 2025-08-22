
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
    def __init__(self, x, y, thresh: float, *args, **kwargs):
        super().__init__(x, y, thresh=thresh, *args, **kwargs)
        self.threshold = thresh

    def _fit(self, x, y, thresh: float, *args, **kwargs):
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
        r = x.shape[1]
        P = np.kron(np.array([[1, -1], [-1, 1]]), x.T @ x)
        q = -np.kron(np.array([[1], [-1]]), x.T @ y.reshape(-1, 1))
        G = np.vstack([-np.eye(2 * r), np.ones((1, 2 * r))])
        h = np.vstack([np.zeros((2 * r, 1)), np.array([[thresh]])])
        b = np.array(cvxopt.solvers.qp(*[cvxopt.matrix(i) for i in [P, q, G, h]])['x'])
        return b[:r, 0] - b[r:, 0]


class l1_cross_validation_regressor(l1_regularization_regressor):
    def __init__(self, x, y, max_thresh: Optional[int] = None, folds: int = 5, seed=None, *args, **kwargs):
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
    def __init__(self,x,y,*args,bootstraps:int=1000,**kwargs):
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
        x = self.bootstrap_params-self.bootstrap_params.mean(0)
        return x.T@x/self.n_boot

L1BootstrapRegressor = partial(PredictionModel,model_type=l1_bootstrap_regressor)
