#!/usr/bin/env python3
"""Pairwise-aware planner on Qwen3-0.6B (a non-chain QUBO): SB vs SA vs greedy vs DP on the additive model.

Options: each MLP block dense or activation-aware SVD at keep 0.8; all weights Q4_0 (bits held
fixed so that the pair measurements apply); attention dense Q4_0; units CPU/GPU/NPU with the Galaxy
S26 published SnapTwin costs and literature switch priors. Accuracy: single-layer costs (keep 0.8,
activation-aware) plus the fitted pairwise interaction model for every layer pair (chosen by
leave-one-pair-out error). For a grid of lambda every solver's plan is scored under the full
objective; there is no exact optimum for the non-chain case, so gaps are to the best plan found.
The top three distinct plans and the additive-DP plan are written for perplexity measurement.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plan import QWEN3_06B, SWITCH_DEC_PRIOR, SWITCH_PRE_PRIOR, load_costs, symmetric

from isingedge.qubo.planner import Option, Problem, Workload, groups_for_qwen3
from isingedge.sensitivity.pairwise import Pair, choose
from isingedge.solve.solvers import (
    exact_chain,
    exact_subset,
    greedy,
    greedy_subset,
    sa_joint,
    sa_subset,
    sb_joint,
    sb_subset,
    subset_energy,
)

SINGLES = ROOT / "results/mac-m2/sensitivity/qwen3-0.6b_layer_sensitivity.csv"
PAIRS = ROOT / "results/mac-m2/sensitivity/qwen3-0.6b_pair_sensitivity.csv"
OUT = ROOT / "results/planner"
LAMBDAS = (0.02, 0.01, 0.005, 0.0025, 0.001)
DENSE, SVD = Option(1.0, "q4_0"), Option(0.8, "q4_0")


def load() -> tuple[dict[int, float], list[Pair]]:
    a = {int(r["layer"]): float(r["delta_pct"]) for r in csv.DictReader(SINGLES.open())
         if r["kind"] == "svd" and r["scaling"] == "imatrix" and float(r["keep"]) == 0.8}
    pairs = [Pair(int(r["layer_i"]), int(r["layer_j"]), float(r["dpct_i"]), float(r["dpct_j"]), float(r["dpct_ij"]))
             for r in csv.DictReader(PAIRS.open())]
    return a, pairs


def build(a: dict[int, float], interactions: dict, lam: float) -> Problem:
    m = QWEN3_06B
    groups = groups_for_qwen3(m["n_layers"], m["hidden"], m["inter"], m["n_heads"], m["n_kv"], m["head_dim"])
    dl = {(L, "mlp", SVD): a[L] for L in range(m["n_layers"])}
    costs = load_costs("Samsung Galaxy S26", "published")
    return Problem(groups, {"mlp": [DENSE, SVD], "attn": [DENSE]}, tuple(u for u in ("cpu", "gpu", "npu") if u in costs), costs,
                   symmetric(SWITCH_DEC_PRIOR), symmetric(SWITCH_PRE_PRIOR), dl,
                   kv_bytes_per_layer=2 * m["n_kv"] * m["head_dim"] * 2, attn_flops_per_pair_per_layer=4 * m["n_heads"] * m["head_dim"],
                   workload=Workload(256, 512, 1.0, 1.0), lam=lam, pairs=interactions)


def compressed(p: Problem, plan) -> list[int]:
    return sorted(g.layer for g, (_, o) in zip(p.groups, plan) if g.part == "mlp" and o.keep < 1.0)


def main() -> None:
    import argparse

    global PAIRS, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plans-only", type=float, nargs="*", metavar="LAMBDA",
                    help="only write the top-3 exact and the DP-additive plans for these lambdas to --plans-dir")
    ap.add_argument("--plans-dir", default=None)
    args = ap.parse_args()
    PAIRS, OUT = Path(args.pairs), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    a, pairs = load()
    model, table = choose(pairs)
    if args.plans_only:
        n = QWEN3_06B["n_layers"]
        interactions = {(i, j): model.interaction(i, j, a[i], a[j]) for i in range(n) for j in range(i + 1, n)}
        pdir = Path(args.plans_dir or OUT / "pairwise_plans_extra")
        pdir.mkdir(parents=True, exist_ok=True)
        for lam in args.plans_only:
            p, units, exact, dp_x = subset_problem(a, interactions, lam)
            write_plans(p, units, exact, dp_x, lam, pdir)
        return
    with (OUT / "pairwise_model_loo.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) + ["chosen", "g0", "g1", "ell", "pairs"])
        w.writeheader()
        for t in table:
            w.writerow({**t, "chosen": t["kind"] == model.kind, "g0": model.g0 if t["kind"] == model.kind else "",
                        "g1": model.g1 if t["kind"] == model.kind else "", "ell": model.ell if t["kind"] == model.kind else "",
                        "pairs": len(pairs)})
    print("pair models (leave-one-pair-out, percentage points):")
    for t in table:
        print(f"  {t['kind']:14s} median {t['loo_median_abs_pp']:.2f}  mean {t['loo_mean_abs_pp']:.2f}  max {t['loo_max_abs_pp']:.2f}")
    print(f"chosen: {model}")
    n = QWEN3_06B["n_layers"]
    interactions = {(i, j): model.interaction(i, j, a[i], a[j]) for i in range(n) for j in range(i + 1, n)}
    rows, subset_rows, per_lambda = [], [], {}
    for lam in LAMBDAS:
        # --- chain case (no pairs): exact DP is the reference
        pc = build(a, {}, lam)
        dp_plan, opt_chain = exact_chain(pc)
        chain_runs = [("greedy", greedy(pc))]
        chain_runs += [(f"dsb-seed{s}", sb_joint(pc, seed=s)) for s in range(4)]
        chain_runs += [(f"bsb-seed{s}", sb_joint(pc, seed=s, ballistic=True)) for s in range(4)]
        chain_runs += [(f"sa-seed{s}", sa_joint(pc, sweeps=50000, seed=s)) for s in range(4)]
        for name, r in chain_runs:
            rows.append({"case": "chain (no pairs)", "lambda": lam, "solver": name, "objective": r["objective"],
                         "gap_pct": 100 * (r["objective"] - opt_chain) / opt_chain, "reference": "exact DP", "seconds": r["seconds"]})
        rows.append({"case": "chain (no pairs)", "lambda": lam, "solver": "exact-dp", "objective": opt_chain, "gap_pct": 0.0,
                     "reference": "exact DP", "seconds": None})
        # --- joint planner with pairs (non-chain): gaps to the best plan found by any solver
        p = build(a, interactions, lam)
        runs = [("dp-additive", {"plan": exact_chain(p)[0], "seconds": None}), ("greedy", greedy(p))]
        runs += [(f"dsb-seed{s}", sb_joint(p, seed=s)) for s in range(4)]
        runs += [(f"bsb-seed{s}", sb_joint(p, seed=s, ballistic=True)) for s in range(4)]
        runs += [(f"sa-seed{s}", sa_joint(p, sweeps=50000, seed=s)) for s in range(4)]
        objs = {name: p.objective(r["plan"]) for name, r in runs}
        best = min(objs.values())
        for name, r in runs:
            rows.append({"case": "joint with pairs", "lambda": lam, "solver": name, "objective": objs[name],
                         "gap_pct": 100 * (objs[name] - best) / best, "reference": "best found", "seconds": r["seconds"]})
        # --- compression subset with units fixed to the chain-DP plan: exact by enumeration
        units = [u for u, _ in dp_plan]
        c, w = subset_terms(p, units, interactions, lam)
        exact = exact_subset(c, w, top=5)
        e_opt = exact[0][0]
        base = p.objective([(units[gi], DENSE) for gi in range(len(p.groups))])
        cands = [("exact", exact[0][1], None), ("dp-additive", (c < 0).astype(np.int8), None), ("greedy", greedy_subset(c, w), None)]
        for sd in range(4):
            cands.append((f"dsb-seed{sd}", *sb_subset(c, w, seed=sd)))
            cands.append((f"bsb-seed{sd}", *sb_subset(c, w, seed=sd, ballistic=True)))
            cands.append((f"sa-seed{sd}", *sa_subset(c, w, seed=sd)))
        for name, x, sec in cands:
            e = subset_energy(x, c, w)
            subset_rows.append({"lambda": lam, "solver": name, "energy": e, "gap_pct": 100 * (e - e_opt) / abs(base + e_opt),
                                "n_compressed": int(x.sum()), "compressed": " ".join(str(i) for i in np.flatnonzero(x)),
                                "pred_additive_pct": float(sum(a[i] for i in np.flatnonzero(x))),
                                "pred_pairwise_pct": float(sum(a[i] for i in np.flatnonzero(x))
                                                           + sum(interactions[(i, j)] for i in np.flatnonzero(x) for j in np.flatnonzero(x) if i < j)),
                                "seconds": sec})
        per_lambda[lam] = (p, units, exact, (c < 0).astype(np.int8))
        print(f"lambda {lam}: chain DP opt {opt_chain:.4f}; subset exact compresses {int(exact[0][1].sum())} layers, "
              f"DP-additive {int((c < 0).sum())}", flush=True)
    for name, data in (("pairwise_solver_comparison.csv", rows), ("pairwise_subset_comparison.csv", subset_rows)):
        with (OUT / name).open("w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(data[0]))
            w_.writeheader()
            w_.writerows(data)
    # plans for perplexity: the lambda where exact (pairwise) and DP-additive differ most in compressed layers
    lam = max(LAMBDAS, key=lambda L: abs(int(per_lambda[L][2][0][1].sum()) - int(per_lambda[L][3].sum())))
    pdir = OUT / "pairwise_plans"
    pdir.mkdir(exist_ok=True)
    for old in pdir.glob("*.json"):
        old.unlink()
    write_plans(*per_lambda[lam], lam, pdir)


def subset_terms(p: Problem, units: list[str], interactions: dict, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear and pair terms of the compression subset QUBO with units fixed."""
    n = QWEN3_06B["n_layers"]
    mlp = [gi for gi, g in enumerate(p.groups) if g.part == "mlp"]
    c = np.array([p.local(p.groups[gi], units[gi], SVD) - p.local(p.groups[gi], units[gi], DENSE) for gi in mlp])
    w = np.zeros((n, n))
    for (i, j), val in interactions.items():
        w[i, j] = w[j, i] = lam * val
    return c, w


def subset_problem(a: dict[int, float], interactions: dict, lam: float):
    p = build(a, interactions, lam)
    units = [u for u, _ in exact_chain(build(a, {}, lam))[0]]
    c, w = subset_terms(p, units, interactions, lam)
    return p, units, exact_subset(c, w, top=5), (c < 0).astype(np.int8)


def write_plans(p: Problem, units: list[str], exact, dp_x: np.ndarray, lam: float, pdir: Path) -> None:
    chosen = [(f"exact{k + 1}", x) for k, (_, x) in enumerate(exact[:3])]
    if not any((dp_x == x).all() for _, x in chosen):
        chosen.append(("dp-additive", dp_x))
    mlp_layers = [g.layer for g in p.groups if g.part == "mlp"]
    for name, x in chosen:
        comp = {mlp_layers[i] for i in np.flatnonzero(x)}
        plan = [(units[gi], SVD if (g.part == "mlp" and g.layer in comp) else DENSE) for gi, g in enumerate(p.groups)]
        lat = p.predicted_latency(plan)
        tag = f"pw_lambda{lam}_{name}"
        (pdir / f"{tag}.json").write_text(json.dumps({
            "tag": tag, "lambda_s_per_pct": lam, "solver": name, "objective": p.objective(plan), "prediction": lat,
            "groups": [{"group": g.name, "unit": u, "option": o.label()} for g, (u, o) in zip(p.groups, plan)]}, indent=2))
        print(f"plan {tag}: {len(comp)} compressed, predicted dPPL pairwise {lat['delta_ppl_pct']:.2f}% "
              f"additive {lat['delta_ppl_additive_pct']:.2f}%")


if __name__ == "__main__":
    main()
