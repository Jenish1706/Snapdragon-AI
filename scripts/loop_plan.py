#!/usr/bin/env python3
"""Closed measure -> fit -> solve planner, against same-budget baselines.

Search space: MLPs of layers 2-25, each dense, keep 0.9 or keep 0.8 (activation-aware SVD), one-hot;
exact bytes budget = K MLPs at keep 0.8 (keep 0.9 saves 2 units, keep 0.8 saves 5).
Per K, with an equal budget of `--evals` measured plans each:
  loop          FM surrogate (rank 6) on log(PPL/PPL_dense), refit every round; top-1 + one diversity
                pick per round; seeds 0-2
  oneshot-pw    the top plans under the pairwise model fitted on singles and pairs
  oneshot-log   the top plans under log-additive single-layer costs
  random        uniform feasible plans; seeds 0-2
  local         first-improvement random swaps on the measured objective from the one-shot-pw best
Synthetic path (default): a mock oracle with planted super-additive interactions; writes to
cache/loop-synthetic/. --real: 40-chunk Qwen3-0.6B perplexity on reference GPU (scripts/torch_ppl.py),
warm-started from every single, pair and plan row measured before S4; writes
results/loop/ and docs/figures/loop_regret.svg. The full split is then measured for
every method's best plan, and paired bootstrap intervals compare each loop seed with oneshot-pw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from isingedge.loop.acquire import energy
from isingedge.loop.oracle import MockOracle
from isingedge.loop.pairmodels import choose
from isingedge.loop.run import (
    fit_scaled,
    run_local,
    run_loop,
    run_random,
    run_ranked,
    top_plans,
)
from isingedge.loop.space import Space, saved_units
from isingedge.loop.stats import paired_bootstrap, spearman

LAYERS = tuple(range(2, 26))
KEEPS = (0.9, 0.8)


def space_for(k: int) -> Space:
    units = saved_units(KEEPS)
    return Space(LAYERS, KEEPS, units, k * units[KEEPS.index(0.8)])


def keys(plan: dict[int, float]) -> str:
    return " ".join(f"{L}:{k:g}" for L, k in sorted(plan.items()) if k < 1)


class RealOracle:
    """40-chunk log(PPL / PPL_dense); a plan measured before (same layers and keeps) is not re-measured."""

    def __init__(self, space: Space, k: int):
        import torch_ppl as tp

        self.tp, self.space, self.k = tp, space, k
        self.cache = {r["keeps"]: float(r["log_ratio"]) for r in tp.read(tp.PLANS) if r["chunks"] == "40" and r["keeps"]}
        self.method, self.seed, self.new, self.hits = "", "", 0, 0

    def __call__(self, x) -> float:
        plan = self.space.decode(x)
        key = keys(plan)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        name = "lp_" + hashlib.sha1(key.encode()).hexdigest()[:10]
        spec = {"plan": name, "layers": {str(L): k for L, k in plan.items()}, "method": self.method, "K": self.k,
                "seed": self.seed}
        self.tp.measure_plans([spec])
        row = next(r for r in self.tp.read(self.tp.PLANS) if r["plan"] == name and r["chunks"] == "40")
        self.cache[key] = float(row["log_ratio"])
        self.new += 1
        return self.cache[key]


def warm_rows(space: Space) -> tuple[list[np.ndarray], list[float], dict]:
    """Rows measured on this machine before S4 whose compressed layers all lie in the space."""
    import torch_ppl as tp

    xs, ys = [np.zeros(space.n, dtype=np.int8)], [0.0]
    dense = float(next(r for r in tp.read(tp.SINGLES) if r["kind"] == "dense")["ppl"])
    single = {}
    for r in tp.read(tp.SINGLES):
        if r["kind"] == "svd" and r["scaling"] == "imatrix" and int(r["layer"]) in space.layers \
                and float(r["keep"]) in space.keeps:
            L, k = int(r["layer"]), float(r["keep"])
            single[(L, k)] = float(r["delta_pct"])
            xs.append(space.encode({L: k}))
            ys.append(float(np.log(float(r["ppl"]) / dense)))
    for r in tp.read(tp.PAIRS):
        i, j = int(r["layer_i"]), int(r["layer_j"])
        if i in space.layers and j in space.layers:
            xs.append(space.encode({i: 0.8, j: 0.8}))
            ys.append(float(np.log(float(r["ppl"]) / dense)))
    for r in tp.read(tp.PLANS):
        if r["chunks"] != "40" or not r["keeps"] or r["method"].startswith("S4"):
            continue
        plan = {int(t.split(":")[0]): float(t.split(":")[1]) for t in r["keeps"].split()}
        if all(L in space.layers and k in space.keeps for L, k in plan.items()):
            xs.append(space.encode(plan))
            ys.append(float(r["log_ratio"]))
    return xs, ys, single


def oneshot_qubos(space: Space, single: dict, pairs) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model, _ = choose(pairs)
    nk = len(space.keeps)
    a = np.array([single[(L, k)] for L in space.layers for k in space.keeps])
    q = np.zeros((space.n, space.n))
    for i in range(space.n):
        for j in range(i + 1, space.n):
            if i // nk != j // nk:
                q[i, j] = model.interaction(space.layers[i // nk], space.layers[j // nk], a[i], a[j])
    return {"oneshot-pw": (a, q), "oneshot-log": (np.log1p(a / 100.0), np.zeros((space.n, space.n)))}


def run_k(k: int, args, real: bool, out_dir: Path, log) -> dict:
    space = space_for(k)
    if real:
        import torch_ppl as tp

        oracle = RealOracle(space, k)
        xs, ys, single = warm_rows(space)
        pairs = [(int(r["layer_i"]), int(r["layer_j"]), float(r["dpct_i"]), float(r["dpct_j"]), float(r["dpct_ij"]))
                 for r in tp.read(tp.PAIRS)]
    else:
        mock = MockOracle(space.n, seed=args.mock_seed)
        oracle = mock
        rng = np.random.default_rng(args.mock_seed)
        xs = [np.zeros(space.n, dtype=np.int8)] + [np.eye(space.n, dtype=np.int8)[i] for i in range(space.n)]
        single = {(L, kk): 100 * np.expm1(mock.true(space.encode({L: kk}))) for L in space.layers for kk in space.keeps}
        pairs = []
        for _ in range(30):
            i, j = sorted(rng.choice(space.layers, 2, replace=False))
            xs.append(space.encode({int(i): 0.8, int(j): 0.8}))
            pairs.append((int(i), int(j), single[(int(i), 0.8)], single[(int(j), 0.8)],
                          100 * np.expm1(mock.true(xs[-1]))))
        ys = [mock.true(x) for x in xs]
    log(f"K {k}: {space.count():,} feasible plans; warm start {len(xs)} rows")
    evals = []

    def tagged(method, seed):
        if real:
            oracle.method, oracle.seed = f"S4 {method} K{k}", seed
        return oracle

    for name, (c, q) in oneshot_qubos(space, single, pairs).items():
        plans = top_plans(space, c, q, args.evals, time_limit=args.time_limit)
        evals += run_ranked(space, tagged(name, ""), plans, name, log=log)
    start = min((e for e in evals if e.method == "oneshot-pw"), key=lambda e: e.y).x
    for seed in range(args.seeds):
        evals += run_loop(space, tagged("loop", seed), xs, ys, evals=args.evals, seed=seed, rank=args.rank,
                          compare_solvers=not args.no_solvers, time_limit=args.time_limit, log=log)
        evals += run_random(space, tagged("random", seed), args.evals, seed, log=log)
        evals += run_local(space, tagged("local", seed), start, args.evals, seed, log=log)
    # surrogate calibration: final-round FM of each loop seed on the random plans it never saw
    calib = []
    rand = [e for e in evals if e.method == "random"]
    for seed in range(args.seeds):
        mine = [e for e in evals if e.method == "loop" and e.seed == seed]
        _, c, q, const = fit_scaled(xs + [e.x for e in mine], ys + [e.y for e in mine], args.rank, seed)
        held = [e for e in rand if not any(np.array_equal(e.x, m.x) for m in mine)]
        pred = np.array([const + energy(e.x, c, q) for e in held])
        truth = np.array([e.y for e in held])
        calib.append({"K": k, "seed": seed, "held_out": len(held), "mae_log": float(np.abs(pred - truth).mean()),
                      "spearman": spearman(pred, truth), "bias_log": float((pred - truth).mean())})
    write(out_dir, k, space, evals, calib, oracle if real else None)
    return {"space": space, "evals": evals, "calib": calib}


def write(out_dir: Path, k: int, space: Space, evals, calib, oracle) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in evals:
        rows.append({"K": k, "method": e.method, "seed": e.seed, "index": e.index, "keeps": keys(space.decode(e.x)),
                     "y_log_ratio": e.y, "delta_pct": 100 * np.expm1(e.y), "pred_log_ratio": e.pred,
                     "round": e.info.get("round", ""), "pick": e.info.get("pick", ""), "status": e.info.get("status", "")})
    new = not (out_dir / "evals.csv").exists()
    with (out_dir / "evals.csv").open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    gaps = [{"K": k, "seed": e.seed, "round": e.info["round"], "solver": s, **g} for e in evals
            if e.method == "loop" and e.info.get("gaps") for s, g in e.info["gaps"].items()]
    if gaps:
        cols = sorted({c for g in gaps for c in g}, key=lambda c: (c not in ("K", "seed", "round", "solver", "gap_pct"), c))
        new = not (out_dir / "solver_gaps.csv").exists()
        with (out_dir / "solver_gaps.csv").open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(gaps)
    new = not (out_dir / "calibration.csv").exists()
    with (out_dir / "calibration.csv").open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(calib[0]))
        if new:
            w.writeheader()
        w.writerows(calib)
    if oracle is not None:
        (out_dir / f"oracle_K{k}.json").write_text(json.dumps({"new_measurements": oracle.new, "cache_hits": oracle.hits}))


# Categorical slots 1-5 of the dataviz reference palette (light surface), fixed order; dashes are the second encoding.
SERIES = [("loop", "#2a78d6", ""), ("oneshot-pw", "#eb6834", "6 3"), ("oneshot-log", "#1baf7a", "2 3"),
          ("random", "#eda100", "8 3 2 3"), ("local", "#e87ba4", "1 2")]


def regret_figure(results: dict, path: Path) -> None:
    """Best measured ΔPPL so far against measured plans, one panel per K (hand-written SVG, no plotting library).

    Lines are the median over seeds, bands the min-max over seeds; every line is labelled at its end."""
    pw, ph, ml, mr, mt, mb = 430, 300, 58, 112, 64, 46
    width, height = len(results) * (pw + ml + mr), ph + mt + mb
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            'font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">'),
           f'<rect width="{width}" height="{height}" fill="#fcfcfb"/>']
    lx = ml
    for name, col, dash in SERIES:  # legend row
        out.append(f'<line x1="{lx}" y1="18" x2="{lx + 22}" y2="18" stroke="{col}" stroke-width="2" '
                   f'stroke-dasharray="{dash}"/><text x="{lx + 28}" y="22" fill="#52514e">{name}</text>')
        lx += 40 + 8 * len(name)
    for p, (k, res) in enumerate(sorted(results.items())):
        x0 = p * (pw + ml + mr) + ml
        curves = {}
        for name, _, _ in SERIES:
            runs = []
            for s in sorted({e.seed for e in res["evals"] if e.method == name}):
                ys = [100 * np.expm1(e.y) for e in res["evals"] if e.method == name and e.seed == s]
                runs.append(np.minimum.accumulate(ys))
            if runs:
                n = min(len(r) for r in runs)
                curves[name] = np.stack([r[:n] for r in runs])
        allv = np.concatenate([c.ravel() for c in curves.values()])
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.06 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        nmax = max(c.shape[1] for c in curves.values())

        def sx(i, x0=x0, nmax=nmax):
            return x0 + (i - 1) / max(1, nmax - 1) * pw

        def sy(v, lo=lo, hi=hi):
            return mt + (hi - v) / (hi - lo) * ph

        out.append(f'<text x="{x0}" y="{mt - 14}" fill="#0b0b0b" font-weight="600">K = {k}: {k} MLPs of bytes saved '
                   'at keep 0.8 (matched)</text>')
        for t in np.linspace(lo + pad, hi - pad, 5):
            out.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x0 + pw}" y2="{sy(t):.1f}" stroke="#e4e3df" stroke-width="1"/>'
                       f'<text x="{x0 - 8}" y="{sy(t) + 4:.1f}" fill="#52514e" text-anchor="end">{t:.0f}</text>')
        for i in (1, 6, 12, 18, 24):
            if i <= nmax:
                out.append(f'<text x="{sx(i):.1f}" y="{mt + ph + 18}" fill="#52514e" text-anchor="middle">{i}</text>')
        out.append(f'<line x1="{x0}" y1="{mt + ph}" x2="{x0 + pw}" y2="{mt + ph}" stroke="#52514e" stroke-width="1"/>'
                   f'<text x="{x0 + pw / 2}" y="{mt + ph + 38}" fill="#52514e" text-anchor="middle">measured plans</text>'
                   f'<text transform="translate({x0 - 44},{mt + ph / 2}) rotate(-90)" fill="#52514e" text-anchor="middle">'
                   'best ΔPPL so far (%, 40 chunks)</text>')
        ends = []
        for name, col, dash in SERIES:
            if name not in curves:
                continue
            c = curves[name]
            xs = np.arange(1, c.shape[1] + 1)
            if c.shape[0] > 1:
                band = [(sx(i), sy(v)) for i, v in zip(xs, c.max(0))] + [(sx(i), sy(v)) for i, v in zip(xs[::-1], c.min(0)[::-1])]
                out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in band) +
                           f'" fill="{col}" fill-opacity="0.12" stroke="none"/>')
            med_c = np.median(c, 0)
            out.append('<polyline points="' + " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in zip(xs, med_c)) +
                       f'" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}" stroke-linejoin="round"/>')
            ends.append([sy(med_c[-1]), name, col, med_c[-1], sx(xs[-1])])
        ends.sort()
        for j in range(1, len(ends)):  # spread end labels so they never overlap
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 14)
        for yl, name, col, v, xe in ends:
            out.append(f'<line x1="{xe + 4:.1f}" y1="{yl:.1f}" x2="{xe + 12:.1f}" y2="{yl:.1f}" stroke="{col}" stroke-width="2"/>'
                       f'<text x="{xe + 16:.1f}" y="{yl + 4:.1f}" fill="#0b0b0b">{name} {v:+.1f}%</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def full_split(results: dict, out_dir: Path, log) -> list[dict]:
    """Measure each method's best plan on the full split; CI of each loop seed against oneshot-pw."""
    import torch_ppl as tp

    specs, best = [], {}
    for k, res in results.items():
        space = res["space"]
        for m in ("loop", "random", "local", "oneshot-pw", "oneshot-log"):
            for s in sorted({e.seed for e in res["evals"] if e.method == m}):
                e = min((e for e in res["evals"] if e.method == m and e.seed == s), key=lambda e: e.y)
                plan = space.decode(e.x)
                name = "lp_" + hashlib.sha1(keys(plan).encode()).hexdigest()[:10]
                best[(k, m, s)] = name
                specs.append({"plan": name, "layers": {str(L): kk for L, kk in plan.items()}, "method": f"S4 best {m}",
                              "K": k, "seed": s})
    uniq = list({s["plan"]: s for s in specs}.values())
    tp.measure_plans(uniq, full=True)
    meas = {(r["plan"], r["chunks"]): float(r["measured_pct"]) for r in tp.read(tp.PLANS)}
    nll = {(r["plan"], r["chunks"]): (np.array([float(v) for v in r["nll_sums"].split()]), int(r["per_chunk"]))
           for r in tp.read(tp.CHUNK_NLL)}
    rows = []
    for (k, m, s), name in sorted(best.items(), key=lambda t: (t[0][0], t[0][1], str(t[0][2]))):
        row = {"K": k, "method": m, "seed": s, "plan": name, "best_40_pct": meas[(name, "40")],
               "full_pct": meas[(name, "all")]}
        ref = best.get((k, "oneshot-pw", 0))
        if m != "oneshot-pw" and ref and ref != name:
            (na, per), (nb, _) = nll[(name, "all")], nll[(ref, "all")]
            ci = paired_bootstrap(na, nb, per, dense_nll=nll[("dense", "all")][0])
            row.update(vs_oneshot_pw_pp=ci["diff"], ci95_lo_pp=ci["lo"], ci95_hi_pp=ci["hi"],
                       better_outside_ci=ci["hi"] < 0)
        elif m != "oneshot-pw":
            row.update(vs_oneshot_pw_pp=0.0, ci95_lo_pp=0.0, ci95_hi_pp=0.0, better_outside_ci=False)
        rows.append(row)
    cols = ["K", "method", "seed", "plan", "best_40_pct", "full_pct", "vs_oneshot_pw_pp", "ci95_lo_pp", "ci95_hi_pp",
            "better_outside_ci"]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for k in sorted(results):
        seeds = [r for r in rows if r["K"] == k and r["method"] == "loop"]
        wins = sum(bool(r.get("better_outside_ci")) for r in seeds)
        log(f"K {k}: loop beats oneshot-pw outside the CI in {wins} of {len(seeds)} seeds")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ks", type=int, nargs="*", default=[12, 18])
    ap.add_argument("--evals", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--no-solvers", action="store_true", help="skip the per-round greedy/SA/SB comparison")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--mock-seed", type=int, default=0)
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    out_dir = ROOT / ("results/loop" if args.real else "cache/loop-synthetic")
    results = {k: run_k(k, args, args.real, out_dir, log) for k in args.ks}
    fig = ROOT / ("docs/figures" if args.real else "cache/loop-synthetic") / "loop_regret.svg"
    regret_figure(results, fig)
    if args.real and not args.no_full:
        full_split(results, out_dir, log)
    for k, res in results.items():
        for m in ("loop", "oneshot-pw", "oneshot-log", "random", "local"):
            ys = [100 * np.expm1(e.y) for e in res["evals"] if e.method == m]
            if ys:
                log(f"K {k} {m:11s} best {min(ys):+.2f}%  median of evaluated {np.median(ys):+.2f}%  n={len(ys)}")


if __name__ == "__main__":
    main()
