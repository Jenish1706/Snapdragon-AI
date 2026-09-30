#!/usr/bin/env python3
"""Print docs/RESULTS.md rows (markdown) for the measurements under results/win-rtx3060/, straight from the CSVs.

Paste the output under the newest date in docs/RESULTS.md; every row names its CSV and script.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(rel: str) -> list[dict]:
    p = RESULTS / rel
    if not p.exists():
        p = RESULTS / "win-rtx3060" / rel
    return list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    sens = read("sensitivity/qwen3-0.6b_layer_sensitivity.csv")
    if sens:
        dense = next(r for r in sens if r["kind"] == "dense")
        rows.append(("Dense, 40 chunks", f"PPL {float(dense['ppl']):.4f} ± {float(dense['ppl_err']):.4f}",
                     "sensitivity/qwen3-0.6b_layer_sensitivity.csv · `torch_ppl.py singles`"))
        for keep in (0.9, 0.8):
            v = {int(r["layer"]): float(r["delta_pct"]) for r in sens if r["kind"] == "svd" and float(r["keep"]) == keep}
            if v:
                lo, hi = min(v, key=v.get), max(v, key=v.get)
                rows.append((f"Single MLP, activation-aware SVD keep {keep}, {len(v)} layers, 40 chunks",
                             f"median {statistics.median(v.values()):+.2f} %; best L{lo} {v[lo]:+.2f} %, worst L{hi} {v[hi]:+.2f} %",
                             "same"))
    comp = read("compression/ppl.csv")
    for r in comp:
        rows.append((f"All-layer {r['tag']} (layers {r['layers']}), full split ({r['chunks']} chunks)",
                     f"PPL {float(r['ppl']):.3f} ± {float(r['ppl_err']):.3f} ({float(r['delta_pct']):+.1f} %)",
                     "compression/ppl.csv · `torch_ppl.py allayer`"))
    pairs = read("sensitivity/qwen3-0.6b_pair_sensitivity.csv")
    if pairs:
        inter = [float(r["interaction_pct"]) for r in pairs]
        top = max(pairs, key=lambda r: float(r["interaction_pct"]))
        rows.append((f"{len(pairs)} layer pairs (v3's seed-0 pairs), both keep 0.8, 40 chunks",
                     f"median interaction {statistics.median(inter):+.2f} pp; {sum(v > 0 for v in inter)} of {len(inter)} "
                     f"super-additive; largest {float(top['interaction_pct']):+.2f} pp (L{top['layer_i']}+L{top['layer_j']})",
                     "sensitivity/qwen3-0.6b_pair_sensitivity.csv · `torch_ppl.py pairs`"))
    for r in read("sensitivity/pair_models_loo.csv"):
        rows.append((f"Pair model {r['kind']}, leave-one-pair-out abs. error", f"median {float(r['loo_median_abs_pp']):.3f} / "
                     f"mean {float(r['loo_mean_abs_pp']):.3f} / max {float(r['loo_max_abs_pp']):.3f} pp"
                     + (" (chosen)" if r["chosen"] == "True" else ""), "sensitivity/pair_models_loo.csv · `s3_additivity.py`"))
    for r in read("sensitivity/qwen3-0.6b_kl_proxy_summary.csv"):
        rows.append((f"KL proxy {r['proxy']} vs 40-chunk ΔPPL, {r['layers']} layers", f"Spearman ρ {float(r['rho']):.3f} "
                     f"[{float(r['ci95_lo']):.3f}, {float(r['ci95_hi']):.3f}]",
                     "sensitivity/qwen3-0.6b_kl_proxy_summary.csv · `torch_ppl.py kl`"))
    for r in read("matched_bytes/summary.csv"):
        if r["row"] == "plan":
            full = f", full {float(r['measured_full_pct']):+.2f} %" if r["measured_full_pct"] else ""
            rows.append((f"Matched bytes K {r['K']}: {r['plan'][5:]} ({r['method']})",
                         f"40 chunks {float(r['measured_40_pct']):+.2f} %{full}", "matched_bytes/summary.csv · `matched_bytes.py`"))
        else:
            v = r["measured_40_pct"] or r["measured_full_pct"]
            rows.append((f"Matched bytes K {r['K']}: {r['plan']} ({r['row']})",
                         f"{float(v):+.2f} pp [{float(r['ci95_lo_pp']):+.2f}, {float(r['ci95_hi_pp']):+.2f}]", "same"))
    for r in read("loop/summary.csv"):
        ci = (f"; vs one-shot pairwise {float(r['vs_oneshot_pw_pp']):+.2f} pp [{float(r['ci95_lo_pp']):+.2f}, "
              f"{float(r['ci95_hi_pp']):+.2f}]" if r.get("vs_oneshot_pw_pp") else "")
        rows.append((f"S4 K {r['K']} {r['method']} seed {r['seed']}: best of 24 measured plans",
                     f"40 chunks {float(r['best_40_pct']):+.2f} %, full {float(r['full_pct']):+.2f} %{ci}",
                     "loop/summary.csv · `loop_plan.py --real`"))
    print("| Measurement | Value | Source (results/ CSV · script) |\n| --- | --- | --- |")
    for m, v, s in rows:
        print(f"| {m} | {v} | {s} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
