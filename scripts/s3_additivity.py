#!/usr/bin/env python3
"""Analysis of log-space vs linear additivity for compression damage.

Evaluates linear-additive error vs log-additive error across measured layer pairs,
and leave-one-pair-out cross-validation of pairwise interaction models.
Inputs and outputs: results/sensitivity/.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from isingedge.loop.pairmodels import choose, log_additive_pct

SENS = ROOT / "results/sensitivity"


def read(name: str) -> list[dict]:
    p = SENS / name
    return list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []


def main() -> None:
    singles = {int(r["layer"]): float(r["delta_pct"]) for r in read("qwen3-0.6b_layer_sensitivity.csv")
               if r["kind"] == "svd" and r["scaling"] == "imatrix" and float(r["keep"]) == 0.8}
    pairs = [(int(r["layer_i"]), int(r["layer_j"]), float(r["dpct_i"]), float(r["dpct_j"]), float(r["dpct_ij"]))
             for r in read("qwen3-0.6b_pair_sensitivity.csv")]
    if not singles or not pairs:
        raise SystemExit("need the singles and pairs tables (scripts/torch_ppl.py singles, pairs)")
    lin = np.array([abs(d - ai - aj) for _, _, ai, aj, d in pairs])
    logadd = np.array([abs(d - log_additive_pct([ai, aj])) for _, _, ai, aj, d in pairs])
    loginter = np.array([math.log1p(d / 100) - math.log1p(ai / 100) - math.log1p(aj / 100) for _, _, ai, aj, d in pairs])
    model, table = choose(pairs)
    with (SENS / "pair_models_loo.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) + ["chosen", "g0", "g1", "ell", "pairs"])
        w.writeheader()
        for t in table:
            ch = t["kind"] == model.kind
            w.writerow({**t, "chosen": ch, "g0": model.g0 if ch else "", "g1": model.g1 if ch else "",
                        "ell": model.ell if ch else "", "pairs": len(pairs)})
    rows = [{"item": "pairs: linear-additive abs error (pp)", "n": len(pairs), "median": np.median(lin), "mean": lin.mean(),
             "max": lin.max()},
            {"item": "pairs: log-additive abs error (pp)", "n": len(pairs), "median": np.median(logadd),
             "mean": logadd.mean(), "max": logadd.max()},
            {"item": "pairs: log-space interaction log(1+d_ij) - log(1+a_i) - log(1+a_j)", "n": len(pairs),
             "median": np.median(loginter), "mean": loginter.mean(), "max": loginter.max(),
             "note": f"{int((loginter > 0).sum())} of {len(pairs)} super-additive in log space"}]
    plans = [r for r in read("qwen3-0.6b_plan_perplexity.csv") if r["chunks"] == "40" and set(r["keeps"].split()) and
             all(t.endswith(":0.8") for t in r["keeps"].split())]
    for r in plans:
        layers = [int(t.split(":")[0]) for t in r["keeps"].split()]
        if not all(L in singles for L in layers):
            continue
        vals = [singles[L] for L in layers]
        pw = sum(vals) + sum(model.interaction(i, j, singles[i], singles[j]) for x, i in enumerate(layers)
                             for j in layers[x + 1:])
        rows.append({"item": f"plan {r['plan']} ({len(layers)} MLPs, 40 chunks)", "n": len(layers),
                     "measured_pct": float(r["measured_pct"]), "linear_sum_pct": sum(vals),
                     "log_additive_pct": log_additive_pct(vals), f"pairwise_{model.kind}_pct": pw})
    for name, add, pw, meas in MAC_PLANS:
        rows.append({"item": f"{name}: log-additive bounds, estimate", "measured_pct": meas, "linear_sum_pct": add,
                     "log_additive_pct": f"{pw:.2f} to {100 * (math.exp(add / 100) - 1):.2f}",
                     "note": "estimate: lower = v3's multiplicative pairwise prediction, upper = 100(exp(sum/100) - 1)"})
    cols = ["item", "n", "median", "mean", "max", "measured_pct", "linear_sum_pct", "log_additive_pct",
            f"pairwise_{model.kind}_pct", "note"]
    with (SENS / "additivity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    decision = logadd.mean() <= 0.5 * lin.mean()
    print(f"pairs: linear-additive mean abs error {lin.mean():.3f} pp, log-additive {logadd.mean():.3f} pp "
          f"-> {'plan in log space' if decision else 'log space not adopted (error not halved)'}")
    for t in table:
        print(f"  LOO {t['kind']:14s} median {t['loo_median_abs_pp']:.3f}  mean {t['loo_mean_abs_pp']:.3f}  "
              f"max {t['loo_max_abs_pp']:.3f}" + ("  (chosen)" if t["kind"] == model.kind else ""))
    for r in rows[3:]:
        print("  " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
