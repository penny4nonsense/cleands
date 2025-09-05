"""
Utility functions for statistical modeling and machine learning.

This module provides helper routines for array manipulation, encoding,
optimization, resampling, and dataset splitting. Many of these utilities
are thin wrappers around NumPy/SciPy/Pandas functionality but standardized
for consistent use across the CleanDS framework.

Key features:
    • Encoding: one-hot, frequency, and probability encoding of categorical arrays.
    • Math helpers: numerically stable sigmoid (`expit`), horizontal/vertical stack.
    • Optimization: gradient descent, Newton’s method, grid search.
    • Resampling: train/test split, k-fold cross-validation, bootstrap sampling.
    • Combinatorics: set product and binomial coefficient matrix constructor (`C`).
    • Intercept helpers: add or append intercept columns to data matrices.
"""

import numpy as np
import scipy as sp
import pandas as pd
import warnings
from typing import Optional, Protocol, Callable, List, Dict, Iterable, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .base import supervised_model


def one_hot_encode(x: np.ndarray) -> np.ndarray:
    """Convert integer vector to one-hot encoded matrix.

    Args:
        x (np.ndarray): Integer labels of shape (n,).

    Returns:
        np.ndarray: One-hot encoded array of shape (n, k),
            where k = x.max() + 1.
    """
    n = x.shape[0]
    ohe = np.zeros((n, x.max() + 1))
    ohe[np.arange(n), x] = 1
    return ohe


def itemfreq(x: np.ndarray, axis: Optional[int] = None, classes: Optional[int] = None) -> np.ndarray:
    """Frequency count of integer labels.

    Args:
        x (np.ndarray): Integer labels.
        axis (Optional[int]): Axis to count over. Defaults to None (flattened).
        classes (Optional[int]): Number of classes to assume. Defaults to x.max()+1.

    Returns:
        np.ndarray: Frequency counts.
    """
    return np.array([(x == i).sum(axis=axis) for i in range(x.max() + 1 if classes==None else classes)]).T


def itemprob(x: np.ndarray, axis: Optional[int] = None, classes: Optional[int] = None) -> np.ndarray:
    """Relative frequencies (probabilities) of integer labels.

    Args:
        x (np.ndarray): Integer labels.
        axis (Optional[int]): Axis to compute proportions along.
        classes (Optional[int]): Number of classes to assume. Defaults to x.max()+1.

    Returns:
        np.ndarray: Probabilities across classes.
    """
    return np.array([(x == i).mean(axis=axis) for i in range(x.max() + 1 if classes==None else classes)]).T


def expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid-transformed values in (0,1).
    """
    out = np.empty_like(x, dtype=np.float64)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    # For positive x, compute normally
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # For negative x, rewrite sigmoid to avoid overflow in exp(-x)
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)

    return out


def hstack(*args: Tuple[np.ndarray]) -> np.ndarray:
    """Column-wise stack with auto-reshaping of 1D arrays.

    Args:
        *args (np.ndarray): Arrays to stack.

    Returns:
        np.ndarray: Horizontally stacked 2D array.
    """
    return np.hstack([item if item.ndim==2 else item.reshape(-1, 1) for item in args])


def vstack(*args: np.ndarray) -> np.ndarray:
    """Row-wise stack with auto-reshaping of 1D arrays.

    Args:
        *args (np.ndarray): Arrays to stack.

    Returns:
        np.ndarray: Vertically stacked 2D array.
    """
    return np.vstack([item if item.ndim==2 else item.reshape(-1, 1) for item in args])


def bind(*args: np.ndarray) -> np.ndarray:
    """Concatenate arrays along first axis.

    Args:
        *args (np.ndarray): Arrays to concatenate.

    Returns:
        np.ndarray: Concatenated array.
    """
    return np.concatenate(args)


def grid_search(func: Callable[[np.ndarray], np.ndarray], space: np.ndarray, axis: Optional[int] = None, maximize: bool = False):
    """Perform grid search by evaluating a function on a search space.

    Args:
        func (Callable): Function mapping space -> score array.
        space (np.ndarray): Search points.
        axis (Optional[int]): Axis to reduce along.
        maximize (bool): Whether to maximize (default False).

    Returns:
        np.ndarray: Index of optimum along axis.
    """
    return func(space).argmax(axis=axis) if maximize else func(space).argmin(axis=axis)


def gradient_descent(
    gradient: Callable[[np.ndarray], np.ndarray],
    init_x: np.ndarray,
    learning_rate: float = 0.005,
    threshold: float = 1e-10,
    max_reps: int = 10000,
    maximize: bool = False,
    obj: Optional[Callable[[np.ndarray], float]] = None,
    step_shrink: float = 0.5,
    min_step: float = 1e-12,
    tol_step: float = 1e-12,
    armijo_c1: float = 1e-4,
) -> Tuple[np.ndarray, bool]:
    """
    Basic gradient method with robust stopping and optional Armijo backtracking.

    Args:
        gradient: Gradient function g(x).
        init_x: Initial point.
        learning_rate: Initial step size.
        threshold: Convergence threshold on ||g(x)||_2.
        max_reps: Maximum iterations.
        maximize: If True, performs gradient ascent.
        obj: Optional objective function f(x). If provided, perform Armijo backtracking.
        step_shrink: Backtracking multiplier in (0,1) when Armijo fails.
        min_step: Minimum allowable step size during backtracking before giving up.
        tol_step: Convergence threshold on parameter step size ||Δx||_2.
        armijo_c1: Armijo constant (typically 1e-4).

    Returns:
        (x, converged): Final iterate and convergence flag.
    """
    x = np.array(init_x, dtype=float, copy=True)

    for _ in range(max_reps):
        g = gradient(x)
        if not np.all(np.isfinite(g)):
            # Bail out if gradient is invalid
            return x, False

        gnorm = np.linalg.norm(g, ord=2)
        if gnorm <= threshold:
            return x, True

        # Descent/ascent direction
        d = g if maximize else -g

        # Propose step
        step = learning_rate

        if obj is None:
            # Fixed step
            x_new = x + step * d
        else:
            # Armijo backtracking line search
            f0 = obj(x)
            # Directional derivative <g, d>
            dir_deriv = float(np.dot(g.ravel(), d.ravel()))
            # For descent: require f(x + t d) <= f(x) + c1 t <g,d>
            # For ascent:  require f(x + t d) >= f(x) + c1 t <g,d>
            while True:
                x_new = x + step * d
                f1 = obj(x_new)
                sufficient = (
                    f1 >= f0 + armijo_c1 * step * dir_deriv
                    if maximize
                    else f1 <= f0 + armijo_c1 * step * dir_deriv
                )
                if sufficient or step < min_step:
                    break
                step *= step_shrink

        # Check step size convergence
        delta = x_new - x
        if not np.all(np.isfinite(delta)):
            return x, False

        if np.linalg.norm(delta, ord=2) <= tol_step:
            x = x_new
            return x, True

        x = x_new

    return x, False


def newton(gradient: Callable[[np.ndarray], np.ndarray], hessian: Callable[[np.ndarray], np.ndarray],
           init_x: np.ndarray, max_reps: int = 100, tolerance: float = 1e-14):
    """Newton-Raphson optimizer for root-finding/maximum likelihood.

    Args:
        gradient (Callable): Gradient function.
        hessian (Callable): Hessian function.
        init_x (np.ndarray): Initial guess.
        max_reps (int): Max iterations.
        tolerance (float): Stopping threshold for update size.

    Returns:
        Tuple[np.ndarray, int]: (solution, iterations) if converged.

    Raises:
        Exception: If convergence not achieved.
    """
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
    """Add an intercept column to a DataFrame and feature list.

    Args:
        x_vars (list[str]): Feature variable names.
        y_var (str): Target variable name.
        data (pd.DataFrame): Input dataset.

    Returns:
        Tuple[list[str], str, pd.DataFrame]: Updated (x_vars, y_var, new data).
    """
    newdf = data.copy()
    x_vars = ['(intercept)'] + x_vars
    newdf['(intercept)'] = np.ones((data.shape[0],))
    return (x_vars, y_var, newdf)


def test_train_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.1, seed: Optional[int] = None):
    """Split arrays into train/test sets.

    Args:
        x (np.ndarray): Features.
        y (np.ndarray): Targets.
        test_ratio (float): Proportion for test set.
        seed (Optional[int]): RNG seed.

    Returns:
        Tuple[np.ndarray]: (x_train, x_test, y_train, y_test).
    """
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
    """Split a Pandas DataFrame into train/test subsets.

    Args:
        data (pd.DataFrame): Dataset.
        seed (Optional[int]): RNG seed.
        test_ratio (float): Proportion for test set.

    Returns:
        Tuple[pd.DataFrame]: (x_train, x_test).
    """
    n = data.shape[0]
    n_test = int(n * test_ratio)
    if seed != None: np.random.seed(seed)
    shuffle = np.random.permutation(n)
    x_test = data.iloc[shuffle[:n_test], :]
    x_train = data.iloc[shuffle[n_test:], :]
    return x_train, x_test


def k_fold_cross_validation(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: Optional[int]=None):
    """Generate train/test splits for k-fold cross-validation.

    Args:
        x (np.ndarray): Features.
        y (np.ndarray): Targets.
        folds (int): Number of folds.
        seed (Optional[int]): RNG seed.

    Returns:
        list[Tuple[np.ndarray]]: List of (x_train, x_test, y_train, y_test).
    """
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


def bootstrap(model: Type[supervised_model], x: np.ndarray, y: np.ndarray,
              seed: Optional[int]=None, bootstraps: int=1000):
    """Generate bootstrap resamples and fit a model on each.

    Args:
        model (Type[supervised_model]): Model class to fit.
        x (np.ndarray): Features.
        y (np.ndarray): Targets.
        seed (Optional[int]): RNG seed.
        bootstraps (int): Number of bootstrap samples.

    Returns:
        list[supervised_model]: Fitted models from bootstrap samples.
    """
    outp = []
    if seed is not None: np.random.seed(seed)
    for i in range(bootstraps):
        sample = np.random.randint(x.shape[0],size=(x.shape[0],))
        outp += [model(x[sample],y[sample])]
    return outp


def set_product(*args: Tuple[Iterable]) -> list:
    """Cartesian product of multiple iterables.

    Args:
        *args (Tuple[Iterable]): Iterables to combine.

    Returns:
        list[Tuple]: Cartesian product tuples.
    """
    def bind_item(item, this_iterable):
        return [bind_item(item, item) for item in this_iterable] if isinstance(this_iterable[0][0], Iterable) else [(item,) + item for item in this_iterable]
    return [(x,) for x in args[0]] if len(args) == 1 else [bind_item(x,set_product(*args[1:])) for x in args[0]]


def intercept(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to feature matrix.

    Args:
        x (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Feature matrix with intercept column added.
    """
    return hstack(np.ones(x.shape[0]),x)


def C(n: int, r: int) -> np.ndarray:
    """Construct combinatorial matrix for n choose r.

    Args:
        n (int): Number of elements.
        r (int): Number chosen.

    Returns:
        np.ndarray: Binary matrix representing subsets.

    Raises:
        Exception: If n < r.
    """
    if n==r: return np.ones((1,r))
    if r==1: return np.eye(n)
    if n<r: raise Exception('Invalid input n<r')
    top_right = C(n-1,r-1)
    bottom_right = C(n-1,r)
    top = cds.hstack(np.ones(top_right.shape[0]),top_right)
    bottom = cds.hstack(np.zeros(bottom_right.shape[0]),bottom_right)
    return cds.vstack(top,bottom)
