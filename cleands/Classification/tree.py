from ..Distribution import multinomial, two_sample, likelihood_ratio_test
from ..base import *
from ..utils import *


class recursive_partitioning_classifier(classification_model):
    def __init__(self, x, y, sign_level=0.95, max_level=None, random_x=False, level='', classes: Optional[int] = None, weights: Optional[np.ndarray] = None):
        self._col_indx = np.random.permutation(x.shape[1])[
                         :int(np.round(np.sqrt(x.shape[1])))] if random_x else np.arange(x.shape[1])
        x = x[:, self._col_indx]
        super(recursive_partitioning_classifier, self).__init__(x, y)
        if classes != None: self.n_classes = classes
        self.max_level, self.sign_level, self._level, self.weights = max_level, sign_level, level, weights
        self._split_variable, self._split_value, self._p_value, self._left, self._right = np.nan, np.nan, np.nan, None, None
        self._terminal_prediction = multinomial(y, w_x=weights, classes=self.n_classes).params
        if max_level != None and len(level) + 1 == max_level: return
        outp = np.array([self.__get_acc(self.x[:, i]) for i in range(self.n_feat)])
        if np.isnan(outp[:, 0]).all(): return
        self._split_variable = np.nanargmax(outp[:, 0])
        self._split_value = outp[self._split_variable, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._p_value = self._get_p_value()
        if max_level == None and self._p_value > (1 - sign_level) / 2 ** (2 * len(level) + 1): return
        left_indx = x[:, self._split_variable] == 0 if np.isnan(self._split_value) else x[:,
                                                                                        self._split_variable] <= self._split_value
        self._left = recursive_partitioning_classifier(x[left_indx, :], y[left_indx], sign_level=sign_level,
                                                       max_level=max_level, random_x=False, level=level + 'L',
                                                       classes=self.n_classes, weights=weights[left_indx] if weights is not None else None)
        self._right = recursive_partitioning_classifier(x[~left_indx, :], y[~left_indx], sign_level=sign_level,
                                                        max_level=max_level, random_x=False, level=level + 'R',
                                                        classes=self.n_classes, weights=weights[~left_indx] if weights is not None else None)

    # MODIFY WEIGHTS HERE
    def __get_acc(self, var, binary=False):
        if var.var() == 0: return (np.nan, np.nan)
        if binary or np.logical_or(var == 0, var == 1).all():
            retp = [self.y[var == 0], self.y[var == 1]]
            if any([item.shape[0] == 0 for item in retp]): return np.nan, np.nan
            outp = [itemfreq(item).argmax() for item in retp]
            retp = [(retp[i] == outp[i]).sum() for i in range(2)]
            return (sum(retp) / self.n_obs, np.nan)
        splits = np.linspace(var.min(), var.max(), 10)
        acc = np.array([self.__get_acc(var > split, True)[0] for split in splits])
        if np.isnan(acc).all(): return np.full(2, np.nan)
        indx = np.nanargmax(acc)
        return acc[indx], splits[indx]

    def _get_p_value(self):
        xvar = (self.x[:, self._split_variable] == 1) if np.isnan(self._split_value) else (
                self.x[:, self._split_variable] > self._split_value)
        try:
            null = multinomial(self.y, w_x=self.weights)
            if self.weights == None:
                alt = two_sample(x=self.y[xvar], y=self.y[~xvar], model_type=multinomial)
            else:
                alt = two_sample(x=self.y[xvar], y=self.y[~xvar], w_x=self.weights[xvar], w_y=self.weights[~xvar], model_type=multinomial)
            p_value = likelihood_ratio_test(null,alt)
        except:
            return 1
        return p_value['p.value']

    def __str__(self):
        if self._left == None: return f'Level:{self._level} Accuracy:{self.accuracy}; n.obs:{self.n_obs}; Classification:{self._terminal_prediction}; Probability:{self.y.mean()}; p.value:{self._p_value}\n'
        return f'Level:{self._level} n.obs:{self.n_obs}; Variable:{self._col_indx[self._split_variable]}; Split:{self._split_value}; Accuracy:{self.accuracy}; p.value:{self._p_value}\n' + str(
            self._left) + str(self._right)

    def predict_proba(self, newx, fitted=False):
        outp = np.array([self._terminal_prediction] * newx.shape[0])
        if self._left == None: return outp
        if not fitted: newx = newx[:, self._col_indx]
        left_indx = newx[:, self._split_variable] == 0 if np.isnan(self._split_value) else newx[:,
                                                                                           self._split_variable] <= self._split_value
        outp[left_indx,:], outp[~left_indx,:] = self._left.predict_proba(newx[left_indx, :]).reshape(outp[left_indx,:].shape), self._right.predict_proba(newx[~left_indx, :]).reshape(outp[~left_indx,:].shape)
        return outp

    def classify(self, target, fitted=False):
        return self.predict_proba(target, fitted).argmax(1)

    @property
    def fitted(self):
        return self.classify(self.x, fitted=True)
