"""The measure -> fit -> solve loop and its same-budget baselines.

Each round refits the factorization machine on every row so far (warm start plus the loop's own
evaluations), minimises it on the feasible set (exact enumeration when there are at most 5e6 plans,
otherwise CP-SAT), and evaluates the top-1 plus one diversity pick at Hamming distance >= `diversity`
from it. Plans already evaluated are excluded by no-good cuts. Greedy, SA and SB solve the same
surrogate every round, and their gap to the exact / CP-SAT energy is recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import acquire
from .fm import fit_fm
from .space import Space, hamming


@dataclass
class Eval:
    method: str
    seed: int
    index: int
    x: np.ndarray
    y: float
    pred: float | None = None
    info: dict = field(default_factory=dict)


def minimise(space: Space, c, q, exclude=(), min_hamming=None, time_limit: float = 20.0) -> dict:
    if space.count() <= 5_000_000:
        top = acquire.exact(space, c, q, top=1, exclude=exclude, min_hamming=min_hamming)
        if not top:
            return {"x": None, "status": "INFEASIBLE"}
        return {"x": top[0][1], "energy": top[0][0], "status": "OPTIMAL (enumeration)"}
    return acquire.cpsat(space, c, q, exclude=exclude, min_hamming=min_hamming, time_limit=time_limit)


def solver_gaps(space: Space, c, q, ref: float, seed: int, exclude=(), heuristics=("greedy", "sa", "sb-ballistic",
                                                                                   "sb-discrete")) -> dict:
    """Energy excess of each heuristic over the reference minimum, in percent of |ref| (surrogate units)."""
    out = {}
    for h in heuristics:
        t0 = time.time()
        if h == "greedy":
            x = acquire.greedy(space, c, q, exclude=exclude)
            r = {"x": x, "energy": acquire.energy(x, c, q)}
        elif h == "sa":
            r = acquire.sa(space, c, q, seed=seed)
        else:
            r = acquire.sb(space, c, q, seed=seed, ballistic=h.endswith("ballistic"))
        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
        out[h] = {"gap_pct": 100.0 * (r["energy"] - ref) / denom, "seconds": time.time() - t0,
                  "raw_feasible": r.get("raw_feasible", True), "repair_flips": r.get("repair_flips", 0),
                  **{k: r[k] for k in ("sha256_Q", "agents", "max_steps", "mode", "torch", "device") if k in r}}
    return out


def fit_scaled(xs, ys, rank: int, seed: int, subset: float | None = None):
    """FM on the standardised target (note A: FMQA stalls without output standardisation), returned as a
    QUBO in the original units: (fm, c, q, const). With `subset`, fit on that random fraction of the rows
    (FMQA's random-subset training, the loop's source of exploration)."""
    x, y = np.stack(xs).astype(float), np.asarray(ys, dtype=float)
    if subset is not None:
        idx = np.random.default_rng(seed).choice(len(y), size=max(2, int(subset * len(y))), replace=False)
        x, y = x[idx], y[idx]
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    fm = fit_fm(x, (y - mu) / sd, rank=rank, seed=seed)
    c, q, const = fm.qubo()
    return fm, c * sd, q * sd, mu + sd * const


def run_loop(space: Space, oracle, warm_x, warm_y, evals: int = 24, seed: int = 0, rank: int = 6,
             diversity: int = 4, compare_solvers: bool = True, time_limit: float = 20.0, subset: float = 0.8,
             log=print) -> list[Eval]:
    """Top-1 of the FM fitted on every row, plus the optimum of an FM fitted on a random `subset` of the rows
    at Hamming distance >= `diversity` from the top-1; no plan is evaluated twice."""
    xs, ys = [np.asarray(x) for x in warm_x], [float(y) for y in warm_y]
    seen = [x for x in xs if space.feasible(x)]
    out: list[Eval] = []
    rnd = 0
    while len(out) < evals:
        _, c, q, const = fit_scaled(xs, ys, rank, seed * 1000 + rnd)
        top = minimise(space, c, q, exclude=seen, time_limit=time_limit)
        if top["x"] is None:
            break
        picks = [("top1", top)]
        if len(out) + 1 < evals:
            _, c2, q2, _ = fit_scaled(xs, ys, rank, seed * 1000 + rnd + 500, subset=subset)
            div = minimise(space, c2, q2, exclude=seen + [top["x"]], min_hamming=(top["x"], diversity),
                           time_limit=time_limit)
            if div["x"] is not None:
                div["energy"] = acquire.energy(div["x"], c, q)  # predicted by the full-data FM
                picks.append(("diverse", div))
        gaps = solver_gaps(space, c, q, top["energy"], seed * 1000 + rnd, exclude=seen) if compare_solvers else {}
        for tag, r in picks:
            y = float(oracle(r["x"]))
            xs.append(r["x"])
            ys.append(y)
            seen.append(r["x"])
            out.append(Eval("loop", seed, len(out), r["x"], y, pred=const + r["energy"],
                            info={"round": rnd, "pick": tag, "status": r["status"], "bound": r.get("bound"),
                                  "gaps": gaps if tag == "top1" else {}}))
            log(f"loop seed {seed} eval {len(out):2d} round {rnd} {tag}: y {y:.4f} (pred {const + r['energy']:.4f}, "
                f"{r['status']}) best {min(e.y for e in out):.4f}")
        rnd += 1
    return out


def top_plans(space: Space, c, q, n: int, time_limit: float = 10.0) -> list[np.ndarray]:
    """The n best distinct feasible plans of a QUBO (successive no-good cuts)."""
    if space.count() <= 5_000_000:
        return [x for _, x in acquire.exact(space, c, q, top=n)]
    out: list[np.ndarray] = []
    for _ in range(n):
        r = acquire.cpsat(space, c, q, exclude=out, time_limit=time_limit)
        if r["x"] is None:
            break
        out.append(r["x"])
    return out


def run_ranked(space: Space, oracle, plans, method: str, seed: int = 0, log=print) -> list[Eval]:
    out = []
    for i, x in enumerate(plans):
        out.append(Eval(method, seed, i, x, float(oracle(x))))
        log(f"{method} eval {i + 1:2d}: y {out[-1].y:.4f} best {min(e.y for e in out):.4f}")
    return out


def run_random(space: Space, oracle, evals: int, seed: int, log=print) -> list[Eval]:
    rng = np.random.default_rng(seed)
    plans: list[np.ndarray] = []
    while len(plans) < evals:
        x = space.random(rng)
        if all(hamming(x, z) > 0 for z in plans):
            plans.append(x)
    return run_ranked(space, oracle, plans, "random", seed, log)


def run_local(space: Space, oracle, start: np.ndarray, evals: int, seed: int, log=print) -> list[Eval]:
    """First-improvement local search on the measured objective, random budget-preserving swaps."""
    rng = np.random.default_rng(seed)
    x = np.asarray(start)
    y = float(oracle(x))
    out = [Eval("local", seed, 0, x, y)]
    tried = [x]
    while len(out) < evals:
        nb = [z for z in space.neighbours(x) if all(hamming(z, t) > 0 for t in tried)]
        if not nb:
            break
        z = nb[rng.integers(len(nb))]
        tried.append(z)
        yz = float(oracle(z))
        out.append(Eval("local", seed, len(out), z, yz))
        if yz < y:
            x, y = z, yz
        log(f"local seed {seed} eval {len(out):2d}: y {yz:.4f} best {y:.4f}")
    return out
