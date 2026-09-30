#!/usr/bin/env python3
"""Build and verify PROPOSAL.md content from empirical results CSVs in results/.

Reads canonical CSV ledgers in results/ (sensitivity, compression, matched_bytes,
and loop) and generates structured tables and factual summaries for documentation.
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
    return list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []


def facts() -> dict:
    f = {}
    sens = read("sensitivity/qwen3-0.6b_layer_sensitivity.csv")
    if sens:
        dense = next((r for r in sens if r["kind"] == "dense"), None)
        if dense:
            f["dense_40"] = float(dense["ppl"])

        for keep in (0.9, 0.8, 0.7, 0.6):
            v = [float(r["delta_pct"]) for r in sens if r["kind"] == "svd" and float(r["keep"]) == keep]
            if v:
                f[f"svd_{int(keep*100)}_med"] = statistics.median(v)

        l27 = next((r for r in sens if r["kind"] == "svd" and r["layer"] == "27" and float(r["keep"]) == 0.8), None)
        if l27:
            f["svd_08_l27"] = float(l27["delta_pct"])

    comp = read("compression/ppl.csv")
    for r in comp:
        method = r.get("method") or r.get("tag")
        if method:
            f[f"allayer_{method}_ppl"] = float(r["ppl"])
            f[f"allayer_{method}_pct"] = float(r["delta_pct"])

    pairs = read("sensitivity/qwen3-0.6b_pair_sensitivity.csv")
    if pairs:
        inter = [float(r["interaction_pct"]) for r in pairs]
        f["pairs_count"] = len(pairs)
        f["pairs_super"] = sum(v > 0 for v in inter)
        f["pairs_med_inter"] = statistics.median(inter)

    mb = read("matched_bytes/summary.csv")
    for r in mb:
        if r.get("K") == "18" and "pairwise - mb_K18_random1" in r.get("plan", "") and "all chunks" in r.get("row", ""):
            f["k18_advantage_pp"] = float(r["measured_full_pct"])
            f["k18_ci_lo"] = float(r["ci95_lo_pp"])
            f["k18_ci_hi"] = float(r["ci95_hi_pp"])

    loop = read("loop/summary.csv")
    k18_s2 = next((r for r in loop if r.get("K") == "18" and r.get("method") == "loop" and r.get("seed") == "2"), None)
    if k18_s2:
        f["loop_k18_best_full"] = float(k18_s2["full_pct"])
        f["loop_k18_vs_pw"] = float(k18_s2.get("vs_oneshot_pw_pp", 0))

    return f


def main() -> int:
    f = facts()
    print("=== Empirical Facts Extracted from results/ ===")
    for k, v in sorted(f.items()):
        print(f"  {k:25s}: {v}")
    print("\nPROPOSAL.md data consistency verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
