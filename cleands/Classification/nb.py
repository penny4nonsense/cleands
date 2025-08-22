import numpy as np
from ..base import classification_model
from ..utils import itemprob, hstack
from typing import Optional


class naive_bayes(classification_model):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        super(naive_bayes, self).__init__(X, y)

        # --- fit priors (optionally weighted) ---
        if priors is None:
            if sample_weight is None:
                self.priors = itemprob(self.y)  # (K,)
            else:
                w = np.asarray(sample_weight, dtype=float).reshape(-1)
                if w.shape[0] != self.n_obs:
                    raise ValueError("sample_weight must have shape (n_obs,).")
                pri = np.array([w[self.y == k].sum() for k in range(self.n_classes)], dtype=float)
                s = pri.sum()
                if s <= 0:
                    raise ValueError("sum of sample weights must be > 0 for prior estimation.")
                self.priors = pri / s
        else:
            self.priors = np.asarray(priors, dtype=float).reshape(-1)
            if self.priors.shape[0] != self.n_classes:
                raise ValueError("priors must have length equal to number of classes.")
            s = self.priors.sum()
            if s <= 0:
                raise ValueError("priors must sum to a positive value.")
            self.priors = self.priors / s

        self._log_priors = np.log(self.priors + 1e-300)  # safety epsilon

    # --- subclass contract ---
    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement log_likelihood(target).")

    # --- common helpers ---
    def _check_target(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        if target.shape[1] != self.n_feat:
            raise ValueError(f"target must have shape (?, {self.n_feat})")
        return target

    def predict_proba(self, target: np.ndarray) -> np.ndarray:
        target = self._check_target(target)
        log_lik = self.log_likelihood(target)  # (n, K)
        log_post = log_lik + self._log_priors  # add log-priors
        m = np.max(log_post, axis=1, keepdims=True)
        stabilized = np.exp(log_post - m)
        probs = stabilized / stabilized.sum(axis=1, keepdims=True)
        return probs

class gaussian_naive_bayes(naive_bayes):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        reg: float = 1e-9,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.reg = float(reg)
        super(gaussian_naive_bayes, self).__init__(X, y, priors=priors, sample_weight=sample_weight)

        # --- fit per-class means/vars ---
        self.means = np.zeros((self.n_classes, self.n_feat), dtype=float)
        self.vars  = np.zeros((self.n_classes, self.n_feat), dtype=float)

        if sample_weight is None:
            for k in range(self.n_classes):
                Xk = self.x[self.y == k]
                if Xk.shape[0] == 0:
                    continue
                self.means[k, :] = Xk.mean(axis=0)
                self.vars[k,  :] = Xk.var(axis=0) + self.reg
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            for k in range(self.n_classes):
                mask = (self.y == k)
                Xk = self.x[mask]
                wk = w[mask]
                if Xk.shape[0] == 0 or wk.sum() == 0:
                    continue
                mk = (wk.reshape(-1, 1) * Xk).sum(axis=0) / wk.sum()
                diff = Xk - mk
                vk = (wk.reshape(-1, 1) * (diff ** 2)).sum(axis=0) / wk.sum()
                self.means[k, :] = mk
                self.vars[k,  :] = vk + self.reg

        # Precompute constants for log-density
        self._log_const = -0.5 * (np.log(2.0 * np.pi) + np.log(self.vars))  # (K, p)

    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        target = self._check_target(target)
        n = target.shape[0]
        K = self.n_classes
        out = np.empty((n, K), dtype=float)
        for c in range(K):
            diff = target - self.means[c, :]
            term = self._log_const[c, :] - 0.5 * (diff ** 2) / self.vars[c, :]
            out[:, c] = term.sum(axis=1)
        return out

class multinomial_naive_bayes(naive_bayes):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        if np.any(np.asarray(X) < 0):
            raise ValueError("Multinomial NB requires nonnegative features.")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        self.alpha = float(alpha)

        super(multinomial_naive_bayes, self).__init__(X, y, priors=priors, sample_weight=sample_weight)

        # --- accumulate (possibly weighted) feature counts per class ---
        self.feature_counts = np.zeros((self.n_classes, self.n_feat), dtype=float)

        if sample_weight is None:
            for k in range(self.n_classes):
                Xk = self.x[self.y == k]
                if Xk.size == 0:
                    continue
                self.feature_counts[k, :] = Xk.sum(axis=0)
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != self.n_obs:
                raise ValueError("sample_weight must have shape (n_obs,).")
            for k in range(self.n_classes):
                mask = (self.y == k)
                if not np.any(mask):
                    continue
                Xk = self.x[mask]
                wk = w[mask].reshape(-1, 1)
                self.feature_counts[k, :] = (wk * Xk).sum(axis=0)

        # --- Laplace smoothing and normalization per class ---
        smoothed = self.feature_counts + self.alpha
        class_totals = smoothed.sum(axis=1, keepdims=True)  # (K, 1)
        class_totals[class_totals == 0.0] = 1.0  # guard (empty class, alpha==0)
        self.feature_probs = smoothed / class_totals  # (K, p)
        self._log_feature_probs = np.log(self.feature_probs)  # cache

    def log_likelihood(self, target: np.ndarray) -> np.ndarray:
        target = self._check_target(target)
        if np.any(target < 0):
            raise ValueError("Multinomial NB requires nonnegative features.")
        # (n, p) @ (p,) per class -> (n, K)
        n = target.shape[0]
        K = self.n_classes
        out = np.empty((n, K), dtype=float)
        for c in range(K):
            out[:, c] = target @ self._log_feature_probs[c, :].T
        return out