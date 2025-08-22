from ..Prediction import logistic_regressor
from ..base import *
from ..utils import *
from functools import partial

class logistic_classifier(classification_model):
    def __init__(self, x, y, probability=0.5):
        super().__init__(x, y)
        self.model = logistic_regressor(x, y)
        self.params = self.model.params
        self.probability = probability

    def predict_proba(self, target):
        Fx = self.model.predict(target)
        if isinstance(Fx, (pd.Series, pd.DataFrame)):
            Fx = Fx.values
        Fx = Fx.reshape(-1, 1)
        return np.hstack((1 - Fx, Fx))

    @property
    def tidy(self):
        return self.tidyci(ci=False)

    def tidyci(self, level=0.95, ci=True):
        n = self.n_feat
        if hasattr(self, 'x_vars'):
            df = [self.x_vars, self.model.params[:n], self.model.std_error[:n], self.model.t_statistic[:n],
                  self.model.p_value[:n]]
        else:
            df = [np.arange(n), self.model.params[:n], self.model.std_error[:n], self.model.t_statistic[:n],
                  self.model.p_value[:n]]
        cols = ['variable', 'estimate', 'std.error', 't.statistic', 'p.value']
        if ci:
            df += [self.model.conf_int(level)[:, :n]]
            cols += ['ci.lower', 'ci.upper']
        df = pd.DataFrame(np.vstack(df).T, columns=cols)
        return df

    @property
    def vcov_params(self):
        return self.model.vcov_params

    @property
    def glance(self):
        return pd.DataFrame(self._glance_dict, index=[''])

    @property
    def _glance_dict(self):
        return {'mcfaddens.r.squared': self.model.mcfaddens_r_squared,
                'ben.akiva.lerman.r.squared': self.model.ben_akiva_lerman_r_squared,
                'self.df': self.model.n_feat,
                'resid.df': self.model.degrees_of_freedom,
                'aic': self.model.aic,
                'bic': self.model.bic,
                'log.likelihood': self.model.log_likelihood,
                'deviance': self.model.deviance,
                'probability.threshold': self.probability,
                'accuracy': self.accuracy,
                'misclassification.probability': self.misclassification_probability}


class multinomial_classifier(classification_model):
    def __init__(self, x, y):
        super(multinomial_classifier, self).__init__(x, y)
        self.ohe_y = one_hot_encode(y)
        self.models = [logistic_regressor(x, self.ohe_y[:, i]) for i in range(self.n_classes)]

    def predict_proba(self, target):
        outp = [model.predict(target).reshape(-1, 1) for model in self.models]
        outp = np.hstack(outp)
        outp /= outp.sum(1).reshape(-1, 1)
        return outp

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        outp = [model.tidyci(level, ci) for model in self.models]
        outp = [pd.concat((pd.DataFrame({'model': np.full(model.shape[0], i)}), model), axis=1) for i, model in
                enumerate(outp)]
        outp = pd.concat(outp, axis=0, ignore_index=True)
        return outp

    @property
    def tidy(self) -> pd.DataFrame: return self.tidyci(ci=False)

    @property
    def glance(self) -> pd.DataFrame:
        outp = [model.glance for model in self.models]
        outp = pd.concat(outp, axis=0)
        outp.index = np.arange(outp.shape[0])
        return outp

LogisticClassifier = partial(PredictionModel,model_type=logistic_classifier)