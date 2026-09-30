"""Pair-interaction models for layer compression, fitted on measured single and pair ΔPPL (percent).

The Mac planner chose among the same four forms (docs/PROPOSAL-v3.md, "Planner with measured pairwise
interactions"); its module is not in this copy, so they are re-implemented here from that description:
  additive        d_ij = a_i + a_j
  multiplicative  d_ij = a_i + a_j + a_i a_j / 100   (identical to log-additivity for a pair)
  scaled          d_ij = a_i + a_j + g a_i a_j / 100
  distance        d_ij = a_i + a_j + (g0 + g1 exp(-|i-j| / l)) a_i a_j / 100
Leave-one-pair-out absolute error in percentage points picks the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELLS = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class PairModel:
    kind: str
    g0: float = 0.0
    g1: float = 0.0
    ell: float = 1.0

    def interaction(self, i: int, j: int, a_i: float, a_j: float) -> float:
        prod = a_i * a_j / 100.0
        if self.kind == "additive":
            return 0.0
        if self.kind == "multiplicative":
            return prod
        if self.kind == "scaled":
            return self.g0 * prod
        return (self.g0 + self.g1 * np.exp(-abs(i - j) / self.ell)) * prod

    def predict(self, i: int, j: int, a_i: float, a_j: float) -> float:
        return a_i + a_j + self.interaction(i, j, a_i, a_j)


def fit(kind: str, pairs: list[tuple[int, int, float, float, float]]) -> PairModel:
    """pairs: (i, j, a_i, a_j, d_ij) in percent."""
    if kind in ("additive", "multiplicative"):
        return PairModel(kind)
    inter = np.array([d - ai - aj for _, _, ai, aj, d in pairs])
    prod = np.array([ai * aj / 100.0 for _, _, ai, aj, _ in pairs])
    if kind == "scaled":
        g = float(inter @ prod / (prod @ prod)) if prod @ prod > 0 else 0.0
        return PairModel("scaled", g0=g)
    best, best_err = PairModel("distance"), np.inf
    for ell in ELLS:
        dist = np.array([np.exp(-abs(i - j) / ell) for i, j, *_ in pairs])
        a = np.stack([prod, dist * prod], 1)
        coef, *_ = np.linalg.lstsq(a, inter, rcond=None)
        err = float(np.abs(a @ coef - inter).mean())
        if err < best_err:
            best, best_err = PairModel("distance", g0=float(coef[0]), g1=float(coef[1]), ell=ell), err
    return best


def loo(kind: str, pairs) -> np.ndarray:
    errs = []
    for k in range(len(pairs)):
        m = fit(kind, pairs[:k] + pairs[k + 1:])
        i, j, ai, aj, d = pairs[k]
        errs.append(abs(m.predict(i, j, ai, aj) - d))
    return np.array(errs)


def choose(pairs, kinds=("additive", "multiplicative", "scaled", "distance")) -> tuple[PairModel, list[dict]]:
    table = []
    for kind in kinds:
        e = loo(kind, pairs)
        table.append({"kind": kind, "loo_median_abs_pp": float(np.median(e)), "loo_mean_abs_pp": float(e.mean()),
                      "loo_max_abs_pp": float(e.max())})
    best = min(table, key=lambda t: t["loo_mean_abs_pp"])["kind"]
    return fit(best, pairs), table


def subset_qubo(a: dict[int, float], model: PairModel, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Linear and upper-triangular pair terms of the pairwise ΔPPL prediction over `layers`."""
    c = np.array([a[L] for L in layers])
    q = np.zeros((len(layers), len(layers)))
    for x, i in enumerate(layers):
        for y in range(x + 1, len(layers)):
            j = layers[y]
            q[x, y] = model.interaction(i, j, a[i], a[j])
    return c, q


def log_additive_pct(values: list[float]) -> float:
    """ΔPPL (%) if log(PPL / PPL_dense) adds over layers: 100 (prod (1 + a_i / 100) - 1)."""
    return float(100.0 * (np.prod([1.0 + v / 100.0 for v in values]) - 1.0))
