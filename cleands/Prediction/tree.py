import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Type
from abc import ABC, abstractmethod

from ..base import prediction_model, SupervisedModel
from ..utils import *
from .glm import least_squares_regressor
from ..Distribution import multinomial, likelihood_ratio_test

class recursive_partitioning_regressor(prediction_model):
    def __init__(self, x, y, sign_level=0.95, max_level=None, random_x=False, level=''):
        self._col_indx = np.random.permutation(x.shape[1])[
                         :int(np.round(np.sqrt(x.shape[1])))] if random_x else np.arange(x.shape[1])
        x = x[:, self._col_indx]
        super(recursive_partitioning_regressor, self).__init__(x, y)
        self.max_level, self._level = max_level, level
        self._split_variable, self._split_value, self._p_value, self._left, self._right = np.nan, np.nan, np.nan, None, None
        self._terminal_prediction = y.mean()
        if max_level != None and len(level) + 1 == max_level: return
        outp = np.array([self.__get_RSS(self.x[:, i]) for i in range(self.n_feat)])
        if np.isnan(outp[:, 0]).all(): return
        self._split_variable = np.nanargmin(outp[:, 0])
        self._split_value = outp[self._split_variable, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._p_value = self._get_p_value()
        if max_level == None and self._p_value > (1 - sign_level) / 2 ** (2 * len(self._level) + 2): return
        left_indx = x[:, self._split_variable] == 0 if np.isnan(self._split_value) else x[:,self._split_variable] <= self._split_value
        self._left = recursive_partitioning_regressor(x[left_indx, :], y[left_indx], sign_level=sign_level, max_level=max_level, level=level + 'L')
        self._right = recursive_partitioning_regressor(x[~left_indx, :], y[~left_indx], sign_level=sign_level, max_level=max_level, level=level + 'R')

    def __get_RSS(self, var, binary=False):
        if var.var() == 0: return np.nan, np.nan
        if binary or np.logical_or(var == 0, var == 1).all():
            retp = [self.y[var == 0], self.y[var == 1]]
            if any([item.shape[0] == 0 for item in retp]): return np.nan, np.nan
            outp = [item.mean() for item in retp]
            retp = [((retp[i] - outp[i]) ** 2).sum() for i in range(2)]
            return sum(retp), np.nan
        splits = np.linspace(var.min(), var.max(), 10)
        RSS = np.array([self.__get_RSS(var > split, True)[0] for split in splits])
        if np.isnan(RSS).all(): return np.nan, np.nan
        indx = np.nanargmin(RSS)
        return RSS[indx], splits[indx]

    def _get_p_value(self):
        xvar = self.x[:, self._split_variable] == 1 if np.isnan(self._split_value) else self.x[:,
                                                                                        self._split_variable] > self._split_value
        xvar = hstack(np.ones(self.n_obs), xvar.astype(int))
        try:
            model = least_squares_regressor(xvar, self.y)
        except:
            return 1
        return model.p_value[1]

    def __str__(self):
        if self._left == None: return f'level:{self._level}; n.obs:{self.n_obs}; residual.ss:{self.residual_sum_of_squares}; variable:{np.nan}; split:{np.nan}; prediction:{self._terminal_prediction}; mean.squared.error:{self.mean_squared_error}; p.value:{self._p_value}\n'
        return f'level:{self._level}; n.obs:{self.n_obs}; residual.ss:{self.residual_sum_of_squares}; variable:{self._col_indx[self._split_variable]}; split:{self._split_value}; prediction:{np.nan}; mean.squared.error:{self.mean_squared_error}; p.value:{self._p_value}\n' + str(
            self._left) + str(self._right)

    def predict(self, newx, fitted=False):
        outp = np.full(shape=newx.shape[0], fill_value=self._terminal_prediction)
        if self._left == None: return outp
        if not fitted: newx = newx[:, self._col_indx]
        left_indx = newx[:, self._split_variable] == 0 if np.isnan(self._split_value) else newx[:,
                                                                                           self._split_variable] <= self._split_value
        outp[left_indx], outp[~left_indx] = self._left.predict(newx[left_indx, :]), self._right.predict(
            newx[~left_indx, :])
        return outp

    @property
    def fitted(self):
        return self.predict(self.x, fitted=True)

    @property
    def tidy(self):
        outp = str(self)
        for i, line in enumerate(outp.split('\n')[:-1]):
            line = line.split(';')
            line = [item.strip().split(':') for item in line]
            mydict = {key: value for key, value in line}
            level = mydict.pop('level')
            newdf = pd.DataFrame(mydict, index=[level])
            if i == 0:
                df = newdf
            else:
                df = pd.concat([df, newdf], axis=0)
        df.index.name = 'level'
        for column in df.columns:
            if column in ['n.obs']:
                df[column] = df[column].astype(int)
            elif column in ['variable']:
                df[column] = df[column].astype(str)
            else:
                df[column] = df[column].astype(float)
        return df

    @property
    def glance(self):
        return pd.DataFrame(self._glance_dict, index=[''])

    @property
    def _glance_dict(self):
        return {'resid.sum.squares': self.residual_sum_of_squares,
                'mean.squared.error': self.mean_squared_error,
                'root.mean.squared.error': self.root_mean_squared_error,
                'r.squared': self.r_squared,
                'adjusted.r.squared': self.adjusted_r_squared,
                'self.df': self.n_feat,
                'resid.df': self.degrees_of_freedom,
                'resid.var': self.residual_variance}