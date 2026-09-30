"""Pair-interaction sensitivity models and leave-one-out model selection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ELLS = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class Pair:
    i: int
    j: int
    ai: float
    aj: float
    d_ij: float

    def __iter__(self):
        return iter((self.i, self.j, self.ai, self.aj, self.d_ij))


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


def fit(kind: str, pairs: list[Pair]) -> PairModel:
    if kind == "additive":
        return PairModel("additive")
    if kind == "multiplicative":
        return PairModel("multiplicative")
    
    inter = np.array([p.d_ij - p.ai - p.aj for p in pairs])
    prod = np.array([p.ai * p.aj / 100.0 for p in pairs])
    
    if kind == "scaled":
        denom = float(prod @ prod)
        g = float(inter @ prod / denom) if denom > 0 else 0.0
        return PairModel("scaled", g0=g)
    
    best_m, best_err = PairModel("distance"), np.inf
    for ell in ELLS:
        dist = np.array([np.exp(-abs(p.i - p.j) / ell) for p in pairs])
        A = np.stack([prod, dist * prod], axis=1)
        coef, *_ = np.linalg.lstsq(A, inter, rcond=None)
        err = float(np.abs(A @ coef - inter).mean())
        if err < best_err:
            best_m = PairModel("distance", g0=float(coef[0]), g1=float(coef[1]), ell=ell)
            best_err = err
    return best_m


def loo(kind: str, pairs: list[Pair]) -> dict[str, float]:
    errs = []
    n = len(pairs)
    for k in range(n):
        train = pairs[:k] + pairs[k + 1:]
        m = fit(kind, train)
        target = pairs[k]
        pred = m.predict(target.i, target.j, target.ai, target.aj)
        errs.append(abs(pred - target.d_ij))
    arr = np.array(errs)
    return {
        "kind": kind,
        "loo_median_abs_pp": float(np.median(arr)),
        "loo_mean_abs_pp": float(arr.mean()),
        "loo_max_abs_pp": float(arr.max()),
    }


def choose(pairs: list[Pair], kinds=("additive", "multiplicative", "scaled", "distance")) -> tuple[PairModel, list[dict]]:
    table = [loo(k, pairs) for k in kinds]
    best_kind = min(table, key=lambda t: t["loo_mean_abs_pp"])["kind"]
    return fit(best_kind, pairs), table
