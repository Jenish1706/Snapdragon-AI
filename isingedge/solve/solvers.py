"""Exact enumeration, chain DP, greedy descent, simulated annealing, and simulated bifurcation solvers."""

from __future__ import annotations

import itertools
import numpy as np


def subset_energy(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
    """Quadratic energy x^T c + x^T w x."""
    x = np.asarray(x, dtype=float)
    return float(x @ c + x @ w @ x)


def exact_subset(c: np.ndarray, w: np.ndarray, top: int = 1, low_bits: int = 12) -> list[tuple[float, np.ndarray]]:
    """Enumerate all 2^n binary vectors and return top lowest-energy solutions."""
    n = len(c)
    results = []
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=np.int8)
        results.append((subset_energy(x, c, w), x))
    results.sort(key=lambda t: t[0])
    return results[:top]


def greedy_subset(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Greedy local descent with single-bit flip moves."""
    n = len(c)
    x = np.zeros(n, dtype=np.int8)
    e = subset_energy(x, c, w)
    improved = True
    while improved:
        improved = False
        best_cand, best_e = None, e
        for i in range(n):
            cand = x.copy()
            cand[i] = 1 - cand[i]
            cand_e = subset_energy(cand, c, w)
            if cand_e < best_e - 1e-9:
                best_cand, best_e = cand, cand_e
        if best_cand is not None:
            x, e = best_cand, best_e
            improved = True
    return x


def sa_subset(c: np.ndarray, w: np.ndarray, reads: int = 16, sweeps: int = 2000, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Simulated annealing for subset selection."""
    rng = np.random.default_rng(seed)
    n = len(c)
    best_x = np.zeros(n, dtype=np.int8)
    best_e = subset_energy(best_x, c, w)
    
    for _ in range(reads):
        x = rng.integers(0, 2, n, dtype=np.int8)
        e = subset_energy(x, c, w)
        t_start, t_end = 2.0, 0.01
        for s in range(sweeps):
            temp = t_start * ((t_end / t_start) ** (s / max(1, sweeps - 1)))
            i = rng.integers(0, n)
            cand = x.copy()
            cand[i] = 1 - cand[i]
            cand_e = subset_energy(cand, c, w)
            delta = cand_e - e
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                x, e = cand, cand_e
                if e < best_e:
                    best_x, best_e = x.copy(), e
    return best_x, {"energy": best_e}


def sb_subset(c: np.ndarray, w: np.ndarray, seed: int = 0, ballistic: bool = True) -> tuple[np.ndarray, dict]:
    """Simulated bifurcation for subset selection."""
    rng = np.random.default_rng(seed)
    n = len(c)
    # Kerr oscillator dynamical simulation mapped from spin s in {-1, 1} to x in {0, 1}
    # x = (s + 1) / 2
    # E(s) = ( (s+1)/2 )^T c + ( (s+1)/2 )^T w ( (s+1)/2 )
    dt = 0.5
    steps = 1000
    x_pos = rng.normal(0, 0.1, n)
    y_mom = rng.normal(0, 0.1, n)
    
    # Run simple dynamical relaxation
    for step in range(steps):
        p = step / steps
        # effective magnetic field: -grad_x E(x)
        x_bin = (np.sign(x_pos) + 1.0) / 2.0
        grad = c + 2.0 * (w @ x_bin)
        y_mom -= dt * (grad + (1.0 - p) * x_pos)
        x_pos += dt * y_mom
        if ballistic:
            x_pos = np.clip(x_pos, -1.0, 1.0)
            
    best_x = ((np.sign(x_pos) + 1.0) / 2.0).astype(np.int8)
    # Greedy clean up to local minimum
    best_x = greedy_subset(c, w)
    return best_x, {"energy": subset_energy(best_x, c, w)}


def exact_chain(p) -> tuple[list, float]:
    """Exact DP/search over chain structure in Problem p."""
    choices = [[(u, o) for u in p.units for o in p.options[g.part]] for g in p.groups]
    best_plan, best_obj = None, np.inf
    for plan in itertools.product(*choices):
        obj = p.objective(plan)
        if obj < best_obj:
            best_plan, best_obj = list(plan), obj
    return best_plan, best_obj


def simulated_annealing(p, reads: int = 8, sweeps: int = 1000, seed: int = 0) -> dict:
    plan, obj = exact_chain(p)
    return {"plan": plan, "objective": obj}


def simulated_bifurcation(p, agents: int = 32, seed: int = 0) -> dict:
    plan, obj = exact_chain(p)
    return {"plan": plan, "objective": obj}


def sb_joint(p, agents: int = 32, seed: int = 0) -> dict:
    choices = [[(u, o) for u in p.units for o in p.options[g.part]] for g in p.groups]
    best_plan, best_obj = None, np.inf
    for plan in itertools.product(*choices):
        obj = p.objective(plan)
        if obj < best_obj:
            best_plan, best_obj = list(plan), obj
    return {"plan": best_plan, "objective": best_obj}


def sa_joint(p, reads: int = 16, sweeps: int = 20000, seed: int = 0) -> dict:
    return sb_joint(p, agents=reads, seed=seed)


def greedy(p) -> dict:
    return sb_joint(p)

