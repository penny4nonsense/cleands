from ..Prediction import k_nearest_neighbors_regressor
from ..base import *
from ..utils import *


class k_nearest_neighbors_classifier(classification_model, k_nearest_neighbors_regressor):
    def __init__(self, x: np.array, y: np.array, k: int = 1) -> None:
        super(k_nearest_neighbors_classifier, self).__init__(x, y)
        self.k = k
        self.norms_train = (x ** 2).sum(1).reshape(-1, 1)

    def predict_proba(self, target: np.array) -> np.array:
        nearest_neighbors = self.neighbors(target, self.k)
        pred_vals = self.y[nearest_neighbors]
        return itemprob(pred_vals, 1)


class k_nearest_neighbors_cross_validation_classifier(k_nearest_neighbors_classifier):
    def __init__(self, x: np.array, y: np.array, k_max: int = 25, folds: int = 5, seed: Optional[int] = None) -> None:
        n = x.shape[0]
        outp = []
        deck = np.arange(n)
        if seed is not None: np.random.seed(seed)
        np.random.shuffle(deck)
        for i in range(folds):
            test = deck[int(i * n / folds):int((i + 1) * n / folds)]
            train_lower = deck[:int(i * n / folds)]
            train_upper = deck[int((i + 1) * n / folds):]
            train = np.concatenate((train_lower, train_upper))
            nearest_neighbors = k_nearest_neighbors_classifier(x[train], y[train], k_max + 1).neighbors(x[test],
                                                                                                        k_max + 1)
            outp += [[(y[test] == y[train][nearest_neighbors[:, :k]].mean(1)).mean() for k in range(1, k_max + 1)]]
        acc = np.array(outp).mean(0)
        k = acc.argmax() + 1
        super(k_nearest_neighbors_cross_validation_classifier, self).__init__(x, y, k)
