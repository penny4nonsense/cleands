import numpy as np

from .glm import least_squares_regressor, logistic_regressor
from .tree import recursive_partitioning_regressor
from ..utils import *
from ..base import PredictionModel
from functools import partial

class bootstrap_model(ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray, model_type: Type[supervised_model], seed: Optional[int] = None, bootstraps: int = 1000, *args, **kwargs) -> None:
        self.model_type: Type[supervised_model] = model_type
        self.model: supervised_model = model_type(x,y)
        self.seed: Optional[int] = seed
        self.n_boot: int = bootstraps
        self.bootstraps: list[supervised_model] = bootstrap(model_type, x, y, seed=seed, bootstraps=bootstraps)

class bagging_least_squares_regressor(least_squares_regressor):
    def __init__(self, x, y, seed=None, bootstraps=1000):
        super().__init__(x, y)
        self.seed = seed
        self.n_boot = bootstraps
        self.bootstraps = bootstrap(least_squares_regressor, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)

    def predict(self, newx):
        return np.array([item.predict(newx) for item in self.bootstraps]).mean(0)\

    @property
    def vcov_params(self):
        return (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot


class bagging_logistic_regressor(logistic_regressor):
    def __init__(self, x, y, seed=None, bootstraps=1000):
        super().__init__(x, y)
        self.seed = seed
        self.n_boot = bootstraps
        self.bootstraps = bootstrap(logistic_regressor, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)

    def predict(self, target: np.ndarray) -> np.ndarray:
        return np.array([item.predict(newx) for item in self.bootstraps]).mean(0)

    @property
    def vcov_params(self):
        return (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot


class bagging_recursive_partitioning_regressor(recursive_partitioning_regressor):
    def __init__(self, x, y, seed=None, bootstraps=1000, sign_level=0.95, max_level=None, random_x=False):
        super().__init__(x, y, max_level=1)
        self.seed, self.n_boot = seed, bootstraps
        model = lambda x, y: recursive_partitioning_regressor(x, y, sign_level=sign_level, max_level=max_level,
                                                              random_x=random_x)
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)
        fit = self.fitted
        indx = np.array([np.mean((item.fitted - fit) ** 2) for item in self.bootstraps]).argmin()
        model = self.bootstraps[indx]
        for key, value in vars(model).items(): setattr(self, key, value)

    def predict(self, newx, fitted=False):
        return np.array([item.predict(newx, fitted) for item in self.bootstraps]).mean(0)


random_forest_regressor = partial(bagging_recursive_partitioning_regressor, random_x=True)

BaggingLogisticRegressor = partial(PredictionModel, model_type=bagging_logistic_regressor)
BaggingRecursivePartitioningRegressor = partial(PredictionModel, model_type=bagging_recursive_partitioning_regressor)
RandomForestRegressor = partial(PredictionModel, model_type=random_forest_regressor)
