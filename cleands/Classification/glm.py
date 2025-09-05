"""
Classification models based on generalized linear models (GLMs).

This module implements classifiers built on top of GLM regression models:
logistic regression for binary classification and multinomial logistic
regression for multi-class classification. Both models integrate with the
broader `cleands` framework and expose `tidy` and `glance` methods for
summarized model output.

Classes:
    logistic_classifier:
        Binary classification model using logistic regression.
    multinomial_classifier:
        Multi-class classification model using a one-vs-rest approach
        with multiple logistic regressions.

Factory Aliases:
    LogisticClassifier:
        Wrapper for constructing a logistic_classifier via ClassificationModel.
    MultinomialClassifier:
        Wrapper for constructing a multinomial_classifier via ClassificationModel.

Typical usage example:

    >>> from cleands.Classification.glm import LogisticClassifier
    >>> import numpy as np
    >>> x = np.random.randn(100, 3)
    >>> y = (x[:, 0] + x[:, 1] > 0).astype(int)
    >>> model = LogisticClassifier(x, y)
    >>> model.tidy
    >>> model.glance
"""

from ..Prediction import logistic_regressor
from ..base import *
from ..utils import *
from functools import partial
import pandas as pd
import numpy as np


class logistic_classifier(classification_model):
    """Logistic regression classifier.

    Wraps a logistic regression model for binary classification tasks.

    Attributes:
        model (logistic_regressor): Underlying logistic regression estimator.
        params (np.ndarray): Estimated model parameters.
        probability (float): Threshold probability for classification.
    """

    def __init__(self, x, y, probability: float = 0.5):
        """Initialize a logistic classifier.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Binary response vector of shape (n_samples,).
            probability (float, optional): Classification threshold. Defaults to 0.5.
        """
        super().__init__(x, y)
        self.model = logistic_regressor(x, y)
        self.params = self.model.params
        self.probability = probability

    def predict_proba(self, target):
        """Predict class probabilities for new observations.

        Args:
            target (np.ndarray or pd.DataFrame): Feature matrix for predictions.

        Returns:
            np.ndarray: Predicted probabilities with shape (n_samples, 2).
                Column 0 = probability of class 0, Column 1 = probability of class 1.
        """
        Fx = self.model.predict(target)
        if isinstance(Fx, (pd.Series, pd.DataFrame)):
            Fx = Fx.values
        Fx = Fx.reshape(-1, 1)
        return np.hstack((1 - Fx, Fx))

    @property
    def tidy(self) -> pd.DataFrame:
        """Return parameter estimates in tidy format without confidence intervals.

        Returns:
            pd.DataFrame: Table of variables, estimates, standard errors, test statistics, and p-values.
        """
        return self.tidyci(ci=False)

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        """Return parameter estimates with optional confidence intervals.

        Args:
            level (float, optional): Confidence level. Defaults to 0.95.
            ci (bool, optional): If True, include confidence intervals. Defaults to True.

        Returns:
            pd.DataFrame: Table of coefficient estimates and statistics.
        """
        n = self.n_feat
        if hasattr(self, 'x_vars'):
            df = [self.x_vars, self.model.params[:n], self.model.std_error[:n],
                  self.model.t_statistic[:n], self.model.p_value[:n]]
        else:
            df = [np.arange(n), self.model.params[:n], self.model.std_error[:n],
                  self.model.t_statistic[:n], self.model.p_value[:n]]
        cols = ['variable', 'estimate', 'std.error', 't.statistic', 'p.value']
        if ci:
            df += [self.model.conf_int(level)[:, :n]]
            cols += ['ci.lower', 'ci.upper']
        df = pd.DataFrame(np.vstack(df).T, columns=cols)
        return df

    @property
    def vcov_params(self) -> np.ndarray:
        """Variance–covariance matrix of parameters.

        Returns:
            np.ndarray: Estimated covariance matrix.
        """
        return self.model.vcov_params

    @property
    def glance(self) -> pd.DataFrame:
        """Return model-level summary statistics.

        Returns:
            pd.DataFrame: Summary with fit metrics, information criteria, and classification accuracy.
        """
        return pd.DataFrame({
            'mcfaddens.r.squared': self.model.mcfaddens_r_squared,
            'ben.akiva.lerman.r.squared': self.model.ben_akiva_lerman_r_squared,
            'self.df': self.model.n_feat,
            'resid.df': self.model.degrees_of_freedom,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'log.likelihood': self.model.log_likelihood,
            'deviance': self.model.deviance,
            'probability.threshold': self.probability,
            'accuracy': self.accuracy,
            'misclassification.probability': self.misclassification_probability
        }, index=[''])


class multinomial_classifier(classification_model):
    """Multinomial logistic regression classifier.

    Fits a one-vs-rest set of logistic regressions for multi-class classification.
    """

    def __init__(self, x, y):
        """Initialize a multinomial classifier.

        Args:
            x (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Response vector with integer class labels of shape (n_samples,).
        """
        super(multinomial_classifier, self).__init__(x, y)
        self.ohe_y = one_hot_encode(y)
        self.models = [logistic_regressor(x, self.ohe_y[:, i]) for i in range(self.n_classes)]

    def predict_proba(self, target):
        """Predict class probabilities for new observations.

        Args:
            target (np.ndarray or pd.DataFrame): Feature matrix for predictions.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        outp = [model.predict(target).reshape(-1, 1) for model in self.models]
        outp = np.hstack(outp)
        outp /= outp.sum(1).reshape(-1, 1)
        return outp

    def tidyci(self, level: float = 0.95, ci: bool = True) -> pd.DataFrame:
        """Return parameter estimates for each logistic regression.

        Args:
            level (float, optional): Confidence level. Defaults to 0.95.
            ci (bool, optional): If True, include confidence intervals. Defaults to True.

        Returns:
            pd.DataFrame: Stacked table of coefficient estimates for all models.
        """
        outp = [model.tidyci(level, ci) for model in self.models]
        outp = [pd.concat((pd.DataFrame({'model': np.full(model.shape[0], i)}),
                           model), axis=1) for i, model in enumerate(outp)]
        outp = pd.concat(outp, axis=0, ignore_index=True)
        return outp

    @property
    def tidy(self) -> pd.DataFrame:
        """Return parameter estimates without confidence intervals.

        Returns:
            pd.DataFrame: Table of estimates for all models.
        """
        return self.tidyci(ci=False)

    @property
    def glance(self) -> pd.DataFrame:
        """Return model-level summary statistics for each logistic regression.

        Returns:
            pd.DataFrame: Concatenated DataFrame of summaries.
        """
        outp = [model.glance for model in self.models]
        outp = pd.concat(outp, axis=0)
        outp.index = np.arange(outp.shape[0])
        return outp


class LogisticClassifier(ClassificationModel):
    """Convenience wrapper for binary logistic regression classifier.

    Provides a formula/DataFrame interface for the
    :class:`logistic_classifier`, which fits a logistic regression
    model for binary outcomes.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`logistic_classifier`.

    Example:
        >>> model = LogisticClassifier.from_formula("y ~ x1 + x2", data=df)
        >>> model.classify(df[["x1", "x2"]])
        >>> model.predict_proba(df[["x1", "x2"]])
    """

    MODEL_TYPE = logistic_classifier


class MultinomialClassifier(ClassificationModel):
    """Convenience wrapper for multinomial logistic regression classifier.

    Provides a formula/DataFrame interface for the
    :class:`multinomial_classifier`, which fits a multinomial logistic
    regression model for multi-class classification problems.

    Attributes:
        MODEL_TYPE: Underlying model type, fixed to
            :class:`multinomial_classifier`.

    Example:
        >>> model = MultinomialClassifier.from_formula("y ~ x1 + x2 + x3", data=df)
        >>> model.classify(df[["x1", "x2", "x3"]])
        >>> model.predict_proba(df[["x1", "x2", "x3"]])
    """

    MODEL_TYPE = multinomial_classifier
