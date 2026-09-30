"""Search space: per-layer compression options as one-hot binaries under an exact bytes budget.

Each layer is dense or takes one of `keeps`; binary x[li * len(keeps) + oi] = 1 when layer li takes
keeps[oi]. `units[oi]` is the bytes that option saves on one MLP, in common units; for Qwen3 MLPs
(1024 x 3072, ranks multiple of 32) keep 0.9 saves 2 units and keep 0.8 saves 5, so a budget of
5 K units matches K MLPs at keep 0.8 exactly. A plan is feasible when every layer takes at most one
option and the units saved equal the budget.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .oracle import keep_fraction_rank


def saved_units(keeps: tuple[float, ...], d_in: int = 1024, d_out: int = 3072) -> tuple[int, ...]:
    """Parameters saved per option relative to dense, divided by their greatest common divisor."""
    dense = d_in * d_out
    saved = [dense - keep_fraction_rank(d_in, d_out, k) * (d_in + d_out) for k in keeps]
    g = math.gcd(*saved)
    return tuple(s // g for s in saved)


@dataclass(frozen=True)
class Space:
    layers: tuple[int, ...]
    keeps: tuple[float, ...]
    units: tuple[int, ...]
    budget: int

    @property
    def n(self) -> int:
        return len(self.layers) * len(self.keeps)

    @property
    def unit_vector(self) -> np.ndarray:
        return np.tile(np.asarray(self.units, dtype=float), len(self.layers))

    def encode(self, plan: dict[int, float]) -> np.ndarray:
        x = np.zeros(self.n, dtype=np.int8)
        for L, k in plan.items():
            if k < 1.0:
                x[self.layers.index(L) * len(self.keeps) + self.keeps.index(k)] = 1
        return x

    def decode(self, x) -> dict[int, float]:
        x = np.asarray(x).reshape(len(self.layers), len(self.keeps))
        return {L: self.keeps[int(np.argmax(row))] for L, row in zip(self.layers, x) if row.any()}

    def saved(self, x) -> int:
        return round(float(self.unit_vector @ np.asarray(x, dtype=float)))

    def feasible(self, x) -> bool:
        x = np.asarray(x).reshape(len(self.layers), len(self.keeps))
        return bool((x.sum(1) <= 1).all()) and self.saved(x.reshape(-1)) == self.budget

    @cached_property
    def compositions(self) -> list[tuple[int, ...]]:
        """Counts per option (c_0, c_1, ...) that meet the budget with at most one option per layer."""
        out = []
        for counts in itertools.product(*(range(len(self.layers) + 1) for _ in self.keeps)):
            if sum(counts) <= len(self.layers) and sum(c * u for c, u in zip(counts, self.units)) == self.budget:
                out.append(counts)
        return out

    def count(self) -> int:
        total = 0
        for counts in self.compositions:
            left, ways = len(self.layers), 1
            for c in counts:
                ways *= math.comb(left, c)
                left -= c
            total += ways
        return total

    def random(self, rng: np.random.Generator) -> np.ndarray:
        """A feasible plan: composition drawn in proportion to its number of plans, then layers uniformly."""
        weights = []
        for counts in self.compositions:
            left, ways = len(self.layers), 1
            for c in counts:
                ways *= math.comb(left, c)
                left -= c
            weights.append(ways)
        p = np.asarray(weights, dtype=float) / sum(weights)
        counts = self.compositions[rng.choice(len(self.compositions), p=p)]
        order = rng.permutation(len(self.layers))
        x = np.zeros((len(self.layers), len(self.keeps)), dtype=np.int8)
        pos = 0
        for oi, c in enumerate(counts):
            x[order[pos:pos + c], oi] = 1
            pos += c
        return x.reshape(-1)

    def neighbours(self, x) -> list[np.ndarray]:
        """Budget-preserving moves: move an option to a dense layer; swap two layers' options."""
        m = np.asarray(x).reshape(len(self.layers), len(self.keeps)).copy()
        opt = [int(np.argmax(r)) if r.any() else -1 for r in m]
        out = []
        for a, b in itertools.combinations(range(len(self.layers)), 2):
            if opt[a] == opt[b]:
                continue
            y = m.copy()
            y[a], y[b] = m[b], m[a]
            out.append(y.reshape(-1))
        return out

    def enumerate(self, limit: int = 5_000_000):
        """Every feasible plan, when there are at most `limit`."""
        if self.count() > limit:
            raise ValueError(f"{self.count():,} feasible plans exceed the enumeration limit {limit:,}")
        n_l, n_o = len(self.layers), len(self.keeps)
        for counts in self.compositions:
            yield from _assign(n_l, n_o, counts)


def _assign(n_l: int, n_o: int, counts: tuple[int, ...]):
    def rec(oi: int, free: tuple[int, ...], m: np.ndarray):
        if oi == n_o:
            yield m.reshape(-1).copy()
            return
        for chosen in itertools.combinations(free, counts[oi]):
            m2 = m.copy()
            m2[list(chosen), oi] = 1
            rest = tuple(i for i in free if i not in chosen)
            yield from rec(oi + 1, rest, m2)

    yield from rec(0, tuple(range(n_l)), np.zeros((n_l, n_o), dtype=np.int8))


def hamming(a, b) -> int:
    return int(np.sum(np.asarray(a) != np.asarray(b)))
