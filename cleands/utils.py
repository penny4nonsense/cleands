import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Iterable, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .base import supervised_model

def one_hot_encode(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    ohe = np.zeros((n, x.max() + 1))
    ohe[np.arange(n), x] = 1
    return ohe


def itemfreq(x: np.ndarray, axis: Optional[int] = None, classes: Optional[int] = None) -> np.ndarray:
    return np.array([(x == i).sum(axis=axis) for i in range(x.max() + 1 if classes==None else classes)]).T

def itemprob(x: np.ndarray, axis: Optional[int] = None, classes: Optional[int] = None) -> np.ndarray:
    return np.array([(x == i).mean(axis=axis) for i in range(x.max() + 1 if classes==None else classes)]).T


def expit(x: np.ndarray) -> np.ndarray:
    # Use a piecewise definition for numerical stability
    out = np.empty_like(x, dtype=np.float64)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    # For positive x, compute normally
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # For negative x, rewrite sigmoid to avoid overflow in exp(-x)
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)

    return out

def hstack(*args: Tuple[np.ndarray])->np.ndarray:
    return np.hstack([item if item.ndim==2 else item.reshape(-1, 1) for item in args])


def vstack(*args: np.ndarray)->np.ndarray:
    return np.vstack([item if item.ndim==2 else item.reshape(-1, 1) for item in args])


def bind(*args: np.ndarray)->np.ndarray:
    return np.concatenate(args)


def grid_search(func: Callable[[np.ndarray],np.ndarray], space: np.ndarray, axis: Optional[int] = None, maximize: bool = False):
    return func(space).argmax(axis=axis) if maximize else func(space).argmin(axis=axis)


def gradient_descent(gradient: Callable[[np.ndarray],np.ndarray], init_x: np.ndarray, learning_rate: float = 0.005, threshold: float = 1e-10, max_reps: int = 10000, maximize: bool = False):
    x = init_x.copy()
    for i in range(max_reps):
        gx = gradient(x)
        x += gx * learning_rate if maximize else -gx * learning_rate
        if np.abs(gx).sum() < threshold: return x, True
    return x, False


def newton(gradient: Callable[[np.ndarray],np.ndarray], hessian: Callable[[np.ndarray],np.ndarray], init_x: np.ndarray, max_reps: int = 100, tolerance: float = 1e-14):
    x = init_x.copy()
    for i in range(max_reps):
        hess = hessian(x)
        grad = gradient(x)
        try:
            update = -np.linalg.solve(hess, grad)
        except:
            return (x, i - 1)
        x += update
        if np.abs(update).sum() < tolerance: return (x, i)
    raise Exception('Newton did not converge')


def add_intercept(x_vars: List[np.ndarray], y_var: str, data: pd.DataFrame):
    newdf = data.copy()
    x_vars = ['(intercept)'] + x_vars
    newdf['(intercept)'] = np.ones((data.shape[0],))
    return (x_vars, y_var, newdf)



def test_train_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.1, seed: Optional[int] = None):
    n = x.shape[0]
    n_test = int(n * test_ratio)
    n_train = n - n_test
    if seed != None: np.random.seed(seed)
    shuffle = np.random.permutation(n)
    x_test = x[shuffle, :][:n_test, :]
    x_train = x[shuffle, :][n_test:, :]
    y_test = y[shuffle][:n_test]
    y_train = y[shuffle][n_test:]
    return (x_train, x_test, y_train, y_test)


def test_split_pandas(data: pd.DataFrame, seed: Optional[int] = None, test_ratio: float = 0.1):
    n = data.shape[0]
    n_test = int(n * test_ratio)
    if seed != None: np.random.seed(seed)
    shuffle = np.random.permutation(n)
    x_test = data.iloc[shuffle[:n_test], :]
    x_train = data.iloc[shuffle[n_test:], :]
    return x_train, x_test


def k_fold_cross_validation(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: Optional[int]=None):
    n = x.shape[0]
    deck = np.arange(n)
    if seed is not None: np.random.seed(seed)
    np.random.shuffle(deck)
    outp = []
    for i in range(folds):
        test = deck[int(i * n / folds):int((i + 1) * n / folds)]
        train_lower = deck[:int(i * n / folds)]
        train_upper = deck[int((i + 1) * n / folds):]
        train = np.concatenate((train_lower, train_upper))
        outp += [(x[train,:],x[test,:],y[train],y[test])]
    return outp

def bootstrap(model:Type[supervised_model],x:np.ndarray,y:np.ndarray,seed:Optional[int]=None,bootstraps:int=1000):
    outp = []
    if seed is not None: np.random.seed(seed)
    for i in range(bootstraps):
        sample = np.random.randint(x.shape[0],size=(x.shape[0],))
        outp += [model(x[sample],y[sample])]
    return outp

def set_product(*args: Tuple[Iterable]) -> list:
    def bind_item(item, this_iterable):
        return [bind_item(item, item) for item in this_iterable] if isinstance(this_iterable[0][0], Iterable) else [(item,) + item for item in this_iterable]
    return [(x,) for x in args[0]] if len(args) == 1 else [bind_item(x,set_product(*args[1:])) for x in args[0]]

def intercept(x: np.ndarray) -> np.ndarray:
    return hstack(np.ones(x.shape[0]),x)

def C(n: int, r: int) -> np.ndarray:
    if n==r: return np.ones((1,r))
    if r==1: return np.eye(n)
    if n<r: raise Exception('Invalid input n<r')
    top_right = C(n-1,r-1)
    bottom_right = C(n-1,r)
    top = cds.hstack(np.ones(top_right.shape[0]),top_right)
    bottom = cds.hstack(np.zeros(bottom_right.shape[0]),bottom_right)
    return cds.vstack(top,bottom)