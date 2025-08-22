from ..base import *
from ..utils import *
from .glm import logistic_classifier
from .tree import recursive_partitioning_classifier
from functools import partial

class bagging_logistic_classifier(logistic_classifier):
    def __init__(self, x, y, probability=0.5, seed=None, bootstraps=1000):
        super(bagging_logistic_classifier, self).__init__(x, y, probability=probability)
        self.seed = seed
        self.n_boot = bootstraps
        model = lambda x, y: logistic_classifier(x, y, probability=probability)
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)
        self.bootstrap_params = np.array([item.model.params for item in self.bootstraps])
        self.params = self.bootstrap_params.mean(0)
        vcov_params = (self.bootstrap_params - self.params).T @ (self.bootstrap_params - self.params) / self.n_boot
        self.model = abstract_logistic_regressor(x, y, self.params, vcov_params)

    def predict_proba(self, newx):
        outp = (np.array([item.classify(newx) for item in self.bootstraps]).mean(0) > 0.5).astype(int)
        return hstack(1 - outp, outp)

# TEST AVERAGING PROBABILITIES VS USING THE CLASSIFICATIONS
class bagging_recursive_partitioning_classifier(recursive_partitioning_classifier):
    def __init__(self, x, y, seed=None, bootstraps=1000, sign_level=0.95, max_level=2, random_x=False, weights = None):
        super(bagging_recursive_partitioning_classifier, self).__init__(x, y, max_level=1)
        self.seed, self.n_boot = seed, bootstraps
        model = lambda x, y: recursive_partitioning_classifier(x, y, sign_level=sign_level, max_level=max_level,
                                                               random_x=random_x, weights=weights)
        self.bootstraps = bootstrap(model, x, y, seed=seed, bootstraps=bootstraps)
        fit = self.fitted
        indx = np.array([np.mean(item.fitted == fit) for item in self.bootstraps]).argmax()
        model = self.bootstraps[indx]
        for key, value in vars(model).items(): setattr(self, key, value)

    def predict_proba(self, target, fitted=False):
        return itemprob(np.array([item.classify(target, fitted=fitted) for item in self.bootstraps]), axis=0)


random_forest_classifier = partial(bagging_recursive_partitioning_classifier, random_x=True)
