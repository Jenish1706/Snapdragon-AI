"""Minimise a QUBO f(x) = c.x + x'Qx (Q strictly upper triangular) over the feasible plans of a Space.

exact:   enumerate every feasible plan (only when there are at most `limit`).
cpsat:   OR-Tools CP-SAT on the linearised QUBO with the one-hot and exact-budget constraints; also
         takes Hamming constraints for diversity and no-good cuts for plans already evaluated.
greedy:  per composition, add the cheapest option step by step, then budget-preserving swap descent.
sa / sb: annealing and simulated bifurcation (ballistic or discrete) on the penalty-encoded QUBO;
         infeasible samples are repaired by greedy moves, and the repair count is reported. Both are
         numpy re-implementations: on the Windows machine Smart App Control blocks dwave-samplers'
         native module and the torch build that the simulated-bifurcation package needs.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .space import Space, hamming


def energy(x, c: np.ndarray, q: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(c @ x + x @ q @ x)


def energies(xs: np.ndarray, c: np.ndarray, q: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return xs @ c + np.einsum("bi,ij,bj->b", xs, q, xs)


def _bits(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)


def exact_cardinality(c, q, k: int, top: int = 3, block: int = 1024) -> list[tuple[float, np.ndarray]]:
    """Top subsets of size k of min c.x + x'Qx by split enumeration of all 2^n subsets (n <= ~30)."""
    c, q = np.asarray(c, dtype=float), np.asarray(q, dtype=float)
    n = len(c)
    qs = np.triu(q, 1) + np.triu(q, 1).T
    lo = n // 2
    li, hi_idx = np.arange(lo), np.arange(lo, n)
    L, H = _bits(lo), _bits(n - lo)

    def local(bits, idx):
        return bits @ c[idx] + 0.5 * np.einsum("bi,ij,bj->b", bits, qs[np.ix_(idx, idx)], bits)

    el, eh = local(L, li), local(H, hi_idx)
    cross = qs[np.ix_(hi_idx, li)]
    pl, ph = L.sum(1).astype(int), H.sum(1).astype(int)
    best: list[tuple[float, int, int]] = []
    for b0 in range(0, len(H), block):
        hb = H[b0:b0 + block]
        e = eh[b0:b0 + block, None] + el[None, :] + (hb @ cross) @ L.T
        e[(ph[b0:b0 + block, None] + pl[None, :]) != k] = np.inf
        flat = e.ravel()
        m = min(top, int(np.isfinite(flat).sum()))
        if m == 0:
            continue
        part = np.argpartition(flat, m - 1)[:m]
        best += [(float(flat[p]), b0 + p // e.shape[1], p % e.shape[1]) for p in part]
        best = sorted(best)[:top]
    out = []
    for en, h, l_ in best:
        x = np.concatenate([L[l_], H[h]]).astype(np.int8)
        out.append((en, x))
    return out


def greedy_cardinality(c, q, k: int) -> np.ndarray:
    """Forward selection of k items by least marginal energy, then best-improvement 1-swap descent."""
    c, q = np.asarray(c, dtype=float), np.asarray(q, dtype=float)
    n = len(c)
    x = np.zeros(n, dtype=np.int8)
    for _ in range(k):
        cand = [i for i in range(n) if not x[i]]
        es = []
        for i in cand:
            y = x.copy()
            y[i] = 1
            es.append(energy(y, c, q))
        x[cand[int(np.argmin(es))]] = 1
    e = energy(x, c, q)
    while True:
        moves = []
        for i in np.flatnonzero(x):
            for j in np.flatnonzero(x == 0):
                y = x.copy()
                y[i], y[j] = 0, 1
                moves.append((energy(y, c, q), y))
        if not moves:
            return x
        e2, y = min(moves, key=lambda t: t[0])
        if e2 >= e - 1e-12:
            return x
        x, e = y, e2


def exact(space: Space, c, q, top: int = 1, exclude=(), min_hamming=None, limit: int = 5_000_000,
          block: int = 200_000):
    """Top feasible plans by exhaustive enumeration: list of (energy, x)."""
    best: list[tuple[float, np.ndarray]] = []
    buf = []

    def flush():
        nonlocal best, buf
        if not buf:
            return
        xs = np.stack(buf)
        e = energies(xs, c, q)
        for z in exclude:
            e[(xs == np.asarray(z)).all(1)] = np.inf
        if min_hamming is not None:
            e[(xs != np.asarray(min_hamming[0])).sum(1) < min_hamming[1]] = np.inf
        keep = [i for i in np.argsort(e)[:top] if np.isfinite(e[i])]
        best = sorted(best + [(float(e[i]), xs[i]) for i in keep], key=lambda t: t[0])[:top]
        buf = []

    for x in space.enumerate(limit):
        buf.append(x)
        if len(buf) >= block:
            flush()
    flush()
    return best


def cpsat(space: Space, c, q, exclude=(), min_hamming=None, time_limit: float = 20.0, scale: float = 1e6,
          workers: int = 8, hint=None) -> dict:
    """Exact (or best-found within the time limit) minimum of the QUBO on the feasible set."""
    from ortools.sat.python import cp_model

    n = space.n
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
    k = len(space.keeps)
    for li in range(len(space.layers)):
        m.Add(sum(x[li * k + oi] for oi in range(k)) <= 1)
    u = space.unit_vector.astype(int)
    m.Add(sum(int(u[i]) * x[i] for i in range(n)) == space.budget)
    terms = [round(float(scale * c[i])) * x[i] for i in range(n) if round(float(scale * c[i])) != 0]
    for i in range(n):
        for j in range(i + 1, n):
            w = round(float(scale * q[i, j]))
            if w == 0 or i // k == j // k:
                continue
            y = m.NewBoolVar(f"y{i}_{j}")
            m.AddBoolOr([x[i].Not(), x[j].Not(), y])  # y >= x_i + x_j - 1
            m.AddImplication(y, x[i])
            m.AddImplication(y, x[j])
            terms.append(w * y)
    for z in exclude:
        m.Add(sum(x[i] if z[i] == 0 else (1 - x[i]) for i in range(n)) >= 1)
    if min_hamming is not None:
        z, d = min_hamming
        m.Add(sum(x[i] if z[i] == 0 else (1 - x[i]) for i in range(n)) >= d)
    if hint is not None:
        for i in range(n):
            m.AddHint(x[i], int(hint[i]))
    m.Minimize(sum(terms))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    t0 = time.time()
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"x": None, "status": s.StatusName(st), "seconds": time.time() - t0}
    xs = np.array([s.Value(v) for v in x], dtype=np.int8)
    return {"x": xs, "energy": energy(xs, c, q), "status": s.StatusName(st), "bound": s.BestObjectiveBound() / scale,
            "seconds": time.time() - t0}


def descend(space: Space, x, c, q, exclude=()) -> np.ndarray:
    """Best-improvement descent over budget-preserving swaps."""
    x = np.asarray(x).copy()
    e = energy(x, c, q)
    while True:
        cands = [y for y in space.neighbours(x) if all(hamming(y, z) > 0 for z in exclude)]
        if not cands:
            return x
        es = energies(np.stack(cands), c, q)
        i = int(np.argmin(es))
        if es[i] >= e - 1e-12:
            return x
        x, e = cands[i], float(es[i])


def greedy(space: Space, c, q, exclude=()) -> np.ndarray:
    """Greedy construction per composition, then swap descent; the best over compositions."""
    k = len(space.keeps)
    best, best_e = None, np.inf
    for counts in space.compositions:
        x = np.zeros(space.n, dtype=np.int8)
        left = list(counts)
        used = set()
        while sum(left):
            cand = [(li, oi) for li in range(len(space.layers)) if li not in used for oi in range(k) if left[oi]]
            es = []
            for li, oi in cand:
                y = x.copy()
                y[li * k + oi] = 1
                es.append(energy(y, c, q))
            li, oi = cand[int(np.argmin(es))]
            x[li * k + oi] = 1
            used.add(li)
            left[oi] -= 1
        x = descend(space, x, c, q, exclude)
        if all(hamming(x, z) > 0 for z in exclude) and energy(x, c, q) < best_e:
            best, best_e = x, energy(x, c, q)
    return best


def penalty_qubo(space: Space, c, q, weight: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Full QUBO with one-hot and exact-budget penalties: (linear, upper couplings, constant)."""
    k = len(space.keeps)
    if weight is None:
        weight = 2.0 * (np.abs(c).sum() + np.abs(q).sum()) / max(1, space.budget)
    u = space.unit_vector
    lin = c + weight * (u * u - 2 * space.budget * u)
    quad = q + weight * 2 * np.triu(np.outer(u, u), 1)
    for li in range(len(space.layers)):
        for a in range(k):
            for b in range(a + 1, k):
                quad[li * k + a, li * k + b] += weight * space.budget
    return lin, quad, weight * space.budget ** 2


def repair(space: Space, x, c, q) -> tuple[np.ndarray, int]:
    """Nearest feasible plan by greedy single-bit flips that reduce the constraint violation."""
    x = np.asarray(x, dtype=np.int8).copy()
    k = len(space.keeps)
    flips = 0

    def violation(y):
        m = y.reshape(len(space.layers), k)
        return int(np.maximum(m.sum(1) - 1, 0).sum()) * 100 + abs(space.saved(y) - space.budget)

    while not space.feasible(x) and flips < 4 * space.n:
        best, best_key = None, None
        for i in range(space.n):
            y = x.copy()
            y[i] ^= 1
            key = (violation(y), energy(y, c, q))
            if best_key is None or key < best_key:
                best, best_key = y, key
        if best_key[0] >= violation(x) and best_key[0] > 0:
            # no single flip reduces the violation: try pair flips that swap a set and an unset bit
            break
        x = best
        flips += 1
    if not space.feasible(x):
        return greedy(space, c, q), flips + 1_000
    return descend(space, x, c, q), flips


def anneal(lin: np.ndarray, quad: np.ndarray, seed: int = 0, reads: int = 16, sweeps: int = 2_000) -> np.ndarray:
    """Single-flip Metropolis simulated annealing on a QUBO (numpy), geometric beta schedule from
    ln 2 / max|dE| to ln 100 / min|dE| as dwave-samplers' default; returns the best sample."""
    rng = np.random.default_rng(seed)
    n = len(lin)
    s = quad + quad.T
    x = rng.integers(0, 2, size=(reads, n)).astype(float)
    h = lin + x @ s  # local field: dE of flipping i from 0 to 1 is h_i
    big = float((np.abs(lin) + np.abs(s).sum(1)).max()) or 1.0
    nz = np.abs(np.concatenate([lin, s[np.triu_indices(n, 1)]]))
    small = float(nz[nz > 0].min()) if (nz > 0).any() else 1.0
    betas = np.geomspace(np.log(2) / big, np.log(100) / small, sweeps)
    for beta in betas:
        for i in range(n):
            d = (1 - 2 * x[:, i]) * h[:, i]
            acc = (d <= 0) | (rng.random(reads) < np.exp(-beta * np.maximum(d, 0)))
            if acc.any():
                step = np.where(acc, 1 - 2 * x[:, i], 0.0)
                x[:, i] += step
                h += step[:, None] * s[i][None, :]
    e = x @ lin + np.einsum("ri,ij,rj->r", x, quad, x)
    return x[int(np.argmin(e))].astype(np.int8)


def bifurcate(lin: np.ndarray, quad: np.ndarray, seed: int = 0, ballistic: bool = True, agents: int = 128,
              steps: int = 10_000, dt: float = 0.5) -> np.ndarray:
    """Simulated bifurcation (Goto et al., Sci. Adv. 2021) on a QUBO mapped to spins (numpy).

    x = (1 + s) / 2 turns lin.x + x'Qx into -(1/2) s'Js - h.s + const with J = -S/4, h = -(lin/2 + S1/4),
    S = Q + Q'. Positions and momenta per agent; a(t) ramps 0 -> a0; inelastic walls at |x| = 1;
    ballistic SB feeds positions into the coupling, discrete SB their signs."""
    rng = np.random.default_rng(seed)
    n = len(lin)
    s = quad + quad.T
    j = -s / 4.0
    h = -(lin / 2.0 + s.sum(1) / 4.0)
    a0 = 1.0
    sigma = float(np.sqrt((j ** 2).sum() / max(1, n * (n - 1)))) or 1.0
    c0 = 0.5 / (sigma * np.sqrt(n))
    hs = h
    xp = rng.uniform(-0.1, 0.1, (agents, n))
    yp = rng.uniform(-0.1, 0.1, (agents, n))
    for t in range(steps):
        a = a0 * t / steps
        src = xp if ballistic else np.sign(xp)
        yp += (-(a0 - a) * xp + c0 * (src @ j + hs)) * dt
        xp += a0 * yp * dt
        wall = np.abs(xp) > 1
        xp[wall] = np.sign(xp[wall])
        yp[wall] = 0.0
    xb = (1 + np.sign(xp)) / 2
    e = xb @ lin + np.einsum("ri,ij,rj->r", xb, quad, xb)
    return xb[int(np.argmin(e))].astype(np.int8)


def _sha(lin, quad) -> str:
    return hashlib.sha256(np.ascontiguousarray(quad).tobytes() + np.ascontiguousarray(lin).tobytes()).hexdigest()


def sa(space: Space, c, q, seed: int = 0, reads: int = 16, sweeps: int = 2_000) -> dict:
    lin, quad, _ = penalty_qubo(space, c, q)
    t0 = time.time()
    raw = anneal(lin, quad, seed, reads, sweeps)
    feasible = space.feasible(raw)
    x, flips = repair(space, raw, c, q)
    return {"x": x, "energy": energy(x, c, q), "raw_feasible": feasible, "repair_flips": flips,
            "seconds": time.time() - t0, "reads": reads, "sweeps": sweeps, "seed": seed, "sha256_Q": _sha(lin, quad)}


def sb(space: Space, c, q, seed: int = 0, ballistic: bool = True, agents: int = 128, max_steps: int = 10_000) -> dict:
    lin, quad, _ = penalty_qubo(space, c, q)
    meta = {"sha256_Q": _sha(lin, quad), "seed": seed, "agents": agents, "max_steps": max_steps,
            "mode": "ballistic" if ballistic else "discrete", "numpy": np.__version__, "device": "cpu"}
    t0 = time.time()
    raw = bifurcate(lin, quad, seed, ballistic, agents, max_steps)
    feasible = space.feasible(raw)
    x, flips = repair(space, raw, c, q)
    return {"x": x, "energy": energy(x, c, q), "raw_feasible": feasible, "repair_flips": flips,
            "seconds": time.time() - t0, **meta}
