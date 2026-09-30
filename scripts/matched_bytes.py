#!/usr/bin/env python3
"""Matched-bytes plan comparison.

For K in {6, 12, 18, 24} MLPs compressed with activation-aware SVD at keep 0.8:
  pairwise     best K-subset under the pair model chosen by leave-one-pair-out (exact enumeration)
  additive     K smallest single-layer costs (also the best under log-additivity)
  greedy       forward selection plus 1-swap descent under the pair model
  random0-2    uniform random K-subsets, seeds 0, 1, 2
Every distinct plan is measured on the 40 paired chunks; for each K the pairwise and additive plans
and the two best measured plans are also measured on the full split. Paired bootstrap intervals over
chunks compare pairwise with additive and the best with the second best.
Output: results/matched_bytes/summary.csv; plan rows go to the sensitivity plan CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch_ppl as tp

from isingedge.loop.acquire import exact_cardinality, greedy_cardinality
from isingedge.loop.pairmodels import choose, log_additive_pct, subset_qubo
from isingedge.loop.stats import paired_bootstrap

OUT = ROOT / "results/matched_bytes"
KS = (6, 12, 18, 24)


def candidates(a: dict[int, float], model, k: int) -> list[dict]:
    layers = sorted(a)
    c, q = subset_qubo(a, model, layers)

    def spec(name, chosen, method, seed=""):
        chosen = sorted(int(L) for L in chosen)
        vals = [a[L] for L in chosen]
        x = np.array([1 if L in chosen else 0 for L in layers])
        return {"plan": f"mb_K{k}_{name}", "layers": {str(L): 0.8 for L in chosen}, "method": method, "K": k,
                "seed": seed, "pred_additive_pct": sum(vals), "pred_log_pct": log_additive_pct(vals),
                "pred_pairwise_pct": float(c @ x + x @ q @ x)}

    out = []
    best = exact_cardinality(c, q, k, top=1)[0][1]
    out.append(spec("pairwise", [layers[i] for i in np.flatnonzero(best)], f"pairwise-{model.kind} exact"))
    out.append(spec("additive", sorted(layers, key=lambda L: a[L])[:k], "additive (linear = log)"))
    g = greedy_cardinality(c, q, k)
    out.append(spec("greedy", [layers[i] for i in np.flatnonzero(g)], f"greedy on pairwise-{model.kind}"))
    for s in range(3):
        rng = np.random.default_rng(s)
        out.append(spec(f"random{s}", rng.choice(layers, size=k, replace=False), "random", s))
    uniq, seen = [], {}
    for s in out:  # identical subsets are measured once; later methods point at the first one
        key = tuple(sorted(s["layers"]))
        if key in seen:
            seen[key]["method"] += f" = {s['method']}"
            continue
        seen[key] = s
        uniq.append(s)
    return uniq


def nll_rows() -> dict[tuple[str, str], tuple[np.ndarray, int]]:
    return {(r["plan"], r["chunks"]): (np.array([float(v) for v in r["nll_sums"].split()]), int(r["per_chunk"]))
            for r in tp.read(tp.CHUNK_NLL)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="*", default=list(KS))
    ap.add_argument("--no-full", action="store_true")
    args = ap.parse_args()
    _, single = tp.singles_table()
    a = {L: v for (L, k), v in single.items() if k == 0.8}
    pairs = [(int(r["layer_i"]), int(r["layer_j"]), float(r["dpct_i"]), float(r["dpct_j"]), float(r["dpct_ij"]))
             for r in tp.read(tp.PAIRS)]
    model, _ = choose(pairs)
    tp.log(f"pair model chosen by leave-one-pair-out: {model}")
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {k: candidates(a, model, k) for k in args.ks}
    for k in args.ks:
        tp.measure_plans(specs[k], full=False)
    meas = {(r["plan"], r["chunks"]): float(r["measured_pct"]) for r in tp.read(tp.PLANS)}
    full = []
    for k in args.ks:
        ranked = sorted(specs[k], key=lambda s: meas[(s["plan"], "40")])
        want = {f"mb_K{k}_pairwise", f"mb_K{k}_additive", ranked[0]["plan"], ranked[1]["plan"]}
        full += [s for s in specs[k] if s["plan"] in want]
    if not args.no_full:
        tp.measure_plans(full, full=True)
    meas = {(r["plan"], r["chunks"]): float(r["measured_pct"]) for r in tp.read(tp.PLANS)}
    nll = nll_rows()
    rows = []
    for k in args.ks:
        ranked = sorted(specs[k], key=lambda s: meas[(s["plan"], "40")])
        for s in specs[k]:
            rows.append({"K": k, "row": "plan", "plan": s["plan"], "method": s["method"],
                         "layers": " ".join(sorted(s["layers"], key=int)), "pred_additive_pct": s["pred_additive_pct"],
                         "pred_log_pct": s["pred_log_pct"], "pred_pairwise_pct": s["pred_pairwise_pct"],
                         "measured_40_pct": meas[(s["plan"], "40")], "measured_full_pct": meas.get((s["plan"], "all"), ""),
                         "rank_40": 1 + ranked.index(s)})
        comps = [(f"mb_K{k}_pairwise", f"mb_K{k}_additive"), (ranked[0]["plan"], ranked[1]["plan"])]
        for pa, pb in comps:
            for tag in ("40", "all"):
                if (pa, tag) not in nll or (pb, tag) not in nll or pa == pb:
                    continue
                (na, per), (nb, _) = nll[(pa, tag)], nll[(pb, tag)]
                ci = paired_bootstrap(na, nb, per, dense_nll=nll[("dense", tag)][0])
                rows.append({"K": k, "row": f"CI {tag} chunks", "plan": f"{pa} - {pb}", "method": "paired bootstrap, 4000",
                             "measured_40_pct" if tag == "40" else "measured_full_pct": ci["diff"],
                             "ci95_lo_pp": ci["lo"], "ci95_hi_pp": ci["hi"], "excludes_zero": ci["excludes_zero"]})
    cols = ["K", "row", "plan", "method", "layers", "pred_additive_pct", "pred_log_pct", "pred_pairwise_pct",
            "measured_40_pct", "measured_full_pct", "rank_40", "ci95_lo_pp", "ci95_hi_pp", "excludes_zero"]
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.4f}" if isinstance(v, float) else v) for kk, v in r.items()})
    for r in rows:
        if r["row"] != "plan":
            print(f"K {r['K']:2d} {r['row']:14s} {r['plan']}: {r.get('measured_40_pct', r.get('measured_full_pct')):+.2f} pp "
                  f"[{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}]")


if __name__ == "__main__":
    main()
