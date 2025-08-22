from ..utils import *
from ..base import dimension_reduction_model, supervised_dimension_reduction_model


class principal_components_analysis(dimension_reduction_model):
    def __init__(self, x: np.ndarray, k: int, *, center: bool = True, scale: bool = False) -> None:
        super().__init__(x)

        if not (1 <= k <= min(self.n_obs, self.n_feat)):
            raise ValueError("k must be between 1 and min(n_obs, n_feat)")

        # training stats
        self.mean = x.mean(0) if center else np.zeros(self.n_feat, dtype=float)
        std = x.std(0, ddof=1)
        if scale:
            # avoid divide-by-zero for constant columns
            self.scale = np.where(std == 0.0, 1.0, std)
        else:
            self.scale = np.ones(self.n_feat, dtype=float)

        x_cs = (x - self.mean) / self.scale

        # SVD: X = U S V^T  -> PCs = columns of V, scores = U S
        U, S, Vt = np.linalg.svd(x_cs, full_matrices=False)
        q = S.shape[0]
        self.singular_values = S
        self.rotation = Vt.T[:, :k]          # (p, k)
        self.components = (U * S)[:, :k]     # (n, k)
        self.k = k

        # variance bookkeeping
        denom = max(self.n_obs - 1, 1)
        ev = (S**2) / denom
        self.explained_variance = ev                       # length q
        total_var = x_cs.var(axis=0, ddof=1).sum()
        self.explained_variance_ratio = ev / total_var if total_var > 0 else np.zeros_like(ev)

    def reduce(self, target: np.ndarray) -> np.ndarray:
        if target.ndim != 2 or target.shape[1] != self.rotation.shape[0]:
            raise ValueError(f"target must be shape (m, {self.rotation.shape[0]})")
        x_cs = (target - self.mean) / self.scale
        return x_cs @ self.rotation


def select_k(eigs: np.ndarray, k_max: int | None = None, include_zero: bool = True, allow_zero: bool = False) -> int:
    lam = np.asarray(eigs, dtype=float)
    lam = lam[~np.isnan(lam)]
    if lam.size < 2:
        return 0 if allow_zero else 1

    lam = np.sort(lam)[::-1]  # λ₁ ≥ λ₂ ≥ ...
    if k_max is None:
        k_max = lam.size
    k_max = max(2, min(k_max, lam.size))  # need at least two to form a ratio

    lam_use = lam[:k_max]                      # [λ₁..λ_{k_max}]
    if include_zero:
        lam0 = lam.sum()/k_max                  # λ₀ := sum(λ₁..λ_{n})/k_max
        lam_aug = np.concatenate(([lam0], lam_use))  # [λ₀, λ₁, …, λ_{k_max}]
        ratios = lam_aug[:-1] / np.maximum(lam_aug[1:], 1e-15)
        # indices 0..k_max-1 correspond to k=0..k_max-1
        k_hat = int(np.argmax(ratios))
        if not allow_zero:
            k_hat = max(1, k_hat)              # enforce ≥1 factor if desired
        return k_hat
    else:
        ratios = lam_use[:-1] / np.maximum(lam_use[1:], 1e-15)  # k=1..k_max-1
        return int(np.argmax(ratios) + 1)


class canonical_correlation_analysis(supervised_dimension_reduction_model):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        k: int,
        *,
        center: bool = True,
        scale: bool = False,
        reg: float = 1e-6,
    ) -> None:
        # initialize supervised base
        super().__init__(X, Y)

        n, p = X.shape
        n2, q = Y.shape
        if n != n2:
            raise ValueError("X and Y must have the same number of rows.")
        k_max = min(p, q, max(n - 1, 1))
        if not (1 <= k <= k_max):
            raise ValueError(f"k must be in [1, {k_max}] given shapes X{X.shape}, Y{Y.shape}.")

        # training stats
        self.mean_x = X.mean(0) if center else np.zeros(p, dtype=float)
        self.mean_y = Y.mean(0) if center else np.zeros(q, dtype=float)

        std_x = X.std(0, ddof=1)
        std_y = Y.std(0, ddof=1)
        self.scale_x = np.where(std_x == 0.0, 1.0, std_x) if scale else np.ones(p, dtype=float)
        self.scale_y = np.where(std_y == 0.0, 1.0, std_y) if scale else np.ones(q, dtype=float)

        Xc = (X - self.mean_x) / self.scale_x
        Yc = (Y - self.mean_y) / self.scale_y

        # sample covariance blocks (ddof=1)
        d = max(n - 1, 1)
        Sxx = (Xc.T @ Xc) / d
        Syy = (Yc.T @ Yc) / d
        Sxy = (Xc.T @ Yc) / d

        # --- Whitening matrices Wx = (Sxx+reg I)^(-1/2), Wy = (Syy+reg I)^(-1/2) ---
        # symmetrize then eigendecompose
        Sxx = (Sxx + Sxx.T) * 0.5
        Syy = (Syy + Syy.T) * 0.5

        evalx, evecx = np.linalg.eigh(Sxx)
        evaly, evecy = np.linalg.eigh(Syy)

        # regularize and clamp to avoid zeros/negatives
        tiny = np.finfo(float).tiny
        evalx = np.clip(evalx + reg, tiny, None)
        evaly = np.clip(evaly + reg, tiny, None)

        inv_sqrt_x = evecx @ (np.diag(1.0 / np.sqrt(evalx)) @ evecx.T)
        inv_sqrt_y = evecy @ (np.diag(1.0 / np.sqrt(evaly)) @ evecy.T)
        # re‑symmetrize to remove numerical drift
        inv_sqrt_x = (inv_sqrt_x + inv_sqrt_x.T) * 0.5
        inv_sqrt_y = (inv_sqrt_y + inv_sqrt_y.T) * 0.5

        # whitened cross-covariance
        M = inv_sqrt_x @ Sxy @ inv_sqrt_y

        # --- SVD on whitened matrix ---
        U, Sigma, Vt = np.linalg.svd(M, full_matrices=False)

        # canonical correlations
        self.canonical_correlations = Sigma[:k]

        # canonical directions in original variable space
        A = inv_sqrt_x @ U[:, :k]   # (p, k)
        B = inv_sqrt_y @ Vt.T[:, :k]  # (q, k)

        # Normalize so that a_j^T Sxx a_j = 1 and b_j^T Syy b_j = 1 (unit variance CVs)
        AtS = A.T @ Sxx
        Aj_var = np.sqrt(np.maximum(np.diag(AtS @ A), tiny))
        A = A / Aj_var.reshape(1, -1)
        BtS = B.T @ Syy
        Bj_var = np.sqrt(np.maximum(np.diag(BtS @ B), tiny))
        B = B / Bj_var.reshape(1, -1)

        self.rotation_x = A
        self.rotation_y = B
        self.k = k

        # canonical variates (scores on training data)
        self.components_x = Xc @ self.rotation_x
        self.components_y = Yc @ self.rotation_y

    # --- API ---
    def reduce_X(self, x_new: np.ndarray) -> np.ndarray:
        if x_new.ndim != 2 or x_new.shape[1] != self.rotation_x.shape[0]:
            raise ValueError(f"x_new must be shape (m, {self.rotation_x.shape[0]}).")
        xc = (x_new - self.mean_x) / self.scale_x
        return xc @ self.rotation_x

    def reduce_Y(self, y_new: np.ndarray) -> np.ndarray:
        if y_new.ndim != 2 or y_new.shape[1] != self.rotation_y.shape[0]:
            raise ValueError(f"y_new must be shape (m, {self.rotation_y.shape[0]}).")
        yc = (y_new - self.mean_y) / self.scale_y
        return yc @ self.rotation_y