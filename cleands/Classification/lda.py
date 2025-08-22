from ..base import *
from ..utils import *

def _quad_form_rows(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    # einsum is fast and stable for this pattern
    return np.einsum("ij,jk,ik->i", X, A, X)


class linear_discriminant_analysis(classification_model):
    def __init__(self, x: np.array, y: np.array) -> None:
        super(linear_discriminant_analysis, self).__init__(x, y)
        self.mean_vectors = [x[y == i].mean(0).reshape(-1, 1) for i in range(self.n_classes)]
        self.priors = itemprob(y)
        self.Sigma_within = np.zeros((self.n_feat, self.n_feat))
        for i, mean in enumerate(self.mean_vectors):
            for row in x[y == i]:
                row = row.reshape(-1, 1)
                self.Sigma_within += (row - mean) @ (row - mean).T
        self.overall_mean = x.mean(0).reshape(-1, 1)
        self.Sigma_between = np.zeros((self.n_feat, self.n_feat))
        for i, mean in enumerate(self.mean_vectors):
            n_class = x[y == i, :].shape[0]
            self.Sigma_between += n_class * (mean - self.overall_mean) @ (mean - self.overall_mean).T
        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.linalg.solve(self.Sigma_within, self.Sigma_between))
        self.eigenvalues = np.flip(self.eigenvalues)
        self.eigenvectors = np.fliplr(self.eigenvectors)[:, :self.n_classes - 1]
        self.eigenvectors = self.eigenvectors

    def discriminant(self, target: np.array) -> np.array:
        return target @ self.eigenvectors

    def predict_proba(self, target: np.array) -> np.array:
        outp = np.zeros((target.shape[0], self.n_classes))
        Sw_inv = np.linalg.inv(self.Sigma_within)
        for i, mean_vec in enumerate(self.mean_vectors):
            for j, x in enumerate(target):
                tmp = x.reshape(-1, 1) - mean_vec
                outp[j, i] = -0.5 * tmp.T @ Sw_inv @ tmp + np.log(self.priors[i])
        outp -= outp.max(1).reshape(-1, 1)
        outp = np.exp(outp)
        outp /= outp.sum(1).reshape(-1, 1)
        return outp

class quadratic_discriminant_analysis(classification_model):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        priors: Optional[np.ndarray] = None,
        reg: float = 1e-6,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(x, y)

        # Validate sample weights
        if sample_weight is None:
            w = np.ones(self.n_obs, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.ndim != 1 or w.shape[0] != self.n_obs:
                raise ValueError("sample_weight must be shape (n_obs,).")
            if np.sum(w) <= 0:
                raise ValueError("sample_weight must sum to a positive value.")

        # Classes and encoded indices
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes = classes                    # store labels
        self.n_classes = classes.size             # <-- keep this an INT
        k_classes = self.n_classes

        # Priors
        if priors is None:
            cw = np.bincount(y_idx, weights=w, minlength=k_classes).astype(float)
            priors = cw / cw.sum()
        else:
            priors = np.asarray(priors, dtype=float)
            if priors.shape != (k_classes,):
                raise ValueError(f"priors must be shape ({k_classes},)")
            if np.any(priors < 0) or priors.sum() <= 0:
                raise ValueError("priors must be nonnegative and sum to a positive value.")
            priors = priors / priors.sum()
        self.priors = priors

        # Means, covariances, inverses, log determinants
        self.means = np.zeros((k_classes, self.n_feat))
        self.covs = np.zeros((k_classes, self.n_feat, self.n_feat))
        self.inv_covs = np.zeros_like(self.covs)
        self.log_dets = np.zeros(k_classes)

        for k in range(k_classes):
            mask = (y_idx == k)
            wk = w[mask]
            Xk = x[mask]
            W = wk.sum()
            if W <= 0:
                raise ValueError(f"Class {classes[k]} has zero total weight.")

            mu = np.average(Xk, axis=0, weights=wk)
            self.means[k] = mu

            Xc = Xk - mu
            cov = (Xc.T * wk) @ Xc / W  # MLE covariance
            if reg > 0:
                cov = cov + reg * np.eye(self.n_feat)

            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + max(reg, 1e-8) * np.eye(self.n_feat)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError(
                        f"Covariance for class {classes[k]} is not PD even after regularization."
                    )

            self.covs[k] = cov
            self.inv_covs[k] = np.linalg.inv(cov)
            self.log_dets[k] = logdet

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if x.shape[1] != self.n_feat:
            raise ValueError(f"X has {x.shape[1]} features; expected {self.n_feat}")

        k_classes = self.n_classes  # int
        scores = np.zeros((x.shape[0], k_classes))
        for k in range(k_classes):
            mu = self.means[k]
            inv_cov = self.inv_covs[k]
            quad = _quad_form_rows(x - mu, inv_cov)  # needs to exist in utils
            scores[:, k] = np.log(self.priors[k]) - 0.5 * self.log_dets[k] - 0.5 * quad
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        m = scores.max(axis=1, keepdims=True)
        P = np.exp(scores - m)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def _check_is_fitted(self) -> None:
        attrs = ("means", "covs", "inv_covs", "log_dets", "priors")
        missing = [a for a in attrs if not hasattr(self, a)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} is not fitted. Missing: {', '.join(missing)}"
            )
