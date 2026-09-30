"""Second-order factorization machine: y = w0 + w.x + sum_{i<j} <v_i, v_j> x_i x_j on binary x.

Fitted by full-batch L-BFGS-B (scipy, float64, analytic gradient) on mean squared error with L2 on w
and V. The fitted model is a QUBO: linear term w, upper-triangular couplings Q_ij = <v_i, v_j>.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FM:
    w0: float
    w: np.ndarray
    v: np.ndarray

    def predict(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xv = x @ self.v
        inter = 0.5 * ((xv * xv).sum(1) - (x @ (self.v * self.v)).sum(1))
        return self.w0 + x @ self.w + inter

    def qubo(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(c, Q, const) with f(x) = const + c.x + x'Qx, Q strictly upper triangular."""
        return self.w.copy(), np.triu(self.v @ self.v.T, 1), float(self.w0)


def fit_fm(x, y, rank: int = 6, l2_w: float = 1e-4, l2_v: float = 1e-3, seed: int = 0, iters: int = 2000,
           init: FM | None = None) -> FM:
    """L-BFGS-B (scipy) on mean squared error + L2, with the analytic gradient; V starts at N(0, 0.01^2)."""
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = x.shape
    x2 = x * x
    if init is not None and init.v.shape == (n, rank):
        theta0 = np.concatenate([[init.w0], init.w, init.v.ravel()])
    else:
        v0 = 0.01 * np.random.default_rng(seed).standard_normal((n, rank))
        theta0 = np.concatenate([[y.mean() if m else 0.0], np.zeros(n), v0.ravel()])

    def fun(theta):
        w0, w, v = theta[0], theta[1:n + 1], theta[n + 1:].reshape(n, rank)
        xv = x @ v
        r = w0 + x @ w + 0.5 * ((xv * xv).sum(1) - x2 @ (v * v).sum(1)) - y
        loss = (r @ r) / m + l2_w * (w @ w) + l2_v * (v * v).sum()
        g_v = 2.0 / m * (x.T @ (r[:, None] * xv) - (x2.T @ r)[:, None] * v) + 2 * l2_v * v
        grad = np.concatenate([[2.0 * r.mean()], 2.0 / m * (x.T @ r) + 2 * l2_w * w, g_v.ravel()])
        return loss, grad

    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", options={"maxiter": iters, "gtol": 1e-12, "ftol": 1e-15})
    t = res.x
    return FM(float(t[0]), t[1:n + 1].copy(), t[n + 1:].reshape(n, rank).copy())


def cv_rank(x, y, ranks=(4, 6, 8), folds: int = 5, seed: int = 0, **kw) -> tuple[int, dict[int, float]]:
    """Pick the rank with the lowest k-fold mean absolute error."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = np.random.default_rng(seed).permutation(len(y))
    parts = np.array_split(idx, folds)
    err = {}
    for r in ranks:
        e = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            m = fit_fm(x[tr], y[tr], rank=r, seed=seed, **kw)
            e.append(np.abs(m.predict(x[te]) - y[te]))
        err[r] = float(np.mean(np.concatenate(e)))
    return min(err, key=err.get), err
