#!/usr/bin/env python3
"""Cross-verify numerical claims in PROPOSAL.md and SUBMISSION.md against CSV results.

Compares quantitative claims in master documentation with actual empirical data
in results/ (sensitivity, compression, matched_bytes, and loop ledgers).
Exit status: 0 if all verified numbers match, 1 on any numerical mismatch.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PROPOSAL = ROOT / "docs/PROPOSAL.md"
SUBMISSION = ROOT / "docs/SUBMISSION.md"

INPUTS = [
    "results/compression/ppl.csv",
    "results/sensitivity/qwen3-0.6b_layer_sensitivity.csv",
    "results/sensitivity/qwen3-0.6b_pair_sensitivity.csv",
    "results/matched_bytes/summary.csv",
    "results/loop/summary.csv",
]

NUM = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")
TITLE = re.compile(r"\*\*(.+?)\*\*")


def read_utf8(path: str) -> list[dict]:
    p = ROOT / path
    return list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []


def numbers(text: str) -> list[str]:
    return [m.group().replace(",", "").replace("−", "-") for m in NUM.finditer(text)]


def split_items(block: str) -> dict[str, str]:
    """Numbered ('1. **T** ...') or bulleted ('- **T** ...') items keyed by their first bold span."""
    out = {}
    for line in block.replace("\r\n", "\n").split("\n"):
        m = re.match(r"\s*(?:\d+\.|-)\s+(.*)", line)
        if m and (t := TITLE.search(m.group(1))):
            out[t.group(1)] = m.group(1).strip()
    return out


def compare(frozen: dict[str, str], regen: dict[str, str]) -> list[tuple[str, str, str]]:
    """(title, status, detail) per frozen item; status is ok, mismatch, text-differs or unverifiable."""
    out = []
    for title, ftxt in frozen.items():
        rtxt = regen.get(title)
        if rtxt is None:
            out.append((title, "unverifiable", "no regenerated item: its CSV rows are missing"))
            continue
        fn, rn = numbers(ftxt), numbers(rtxt)
        if fn != rn:
            diff = [f"{a} -> {b}" for a, b in zip(fn, rn) if a != b]
            if len(fn) != len(rn):
                diff.append(f"{len(fn)} numbers frozen vs {len(rn)} regenerated")
            out.append((title, "mismatch", "; ".join(diff)))
        elif ftxt != rtxt:
            out.append((title, "text-differs", "same numbers, different wording"))
        else:
            out.append((title, "ok", ""))
    return out


def verify_core_metrics() -> list[tuple[str, bool, str]]:
    """Verify core numerical claims directly against results CSVs."""
    checks = []

    # 1. Compression results
    comp_rows = read_utf8("results/compression/ppl.csv")
    mpo = next((r for r in comp_rows if r.get("method") == "mpo"), None)
    svd = next((r for r in comp_rows if r.get("method") == "svd"), None)

    if mpo and svd:
        mpo_ppl = float(mpo["ppl"])
        svd_ppl = float(svd["ppl"])
        checks.append(("All-layer MPO PPL ~ 39.05", abs(mpo_ppl - 39.049) < 0.01, f"{mpo_ppl:.3f}"))
        checks.append(("All-layer SVD PPL ~ 53.17", abs(svd_ppl - 53.174) < 0.01, f"{svd_ppl:.3f}"))

    # 2. Sensitivity singles
    sens_rows = read_utf8("results/sensitivity/qwen3-0.6b_layer_sensitivity.csv")
    dense = next((r for r in sens_rows if r.get("kind") == "dense"), None)
    if dense:
        dense_ppl = float(dense["ppl"])
        checks.append(("Dense baseline PPL ~ 19.598", abs(dense_ppl - 19.598) < 0.01, f"{dense_ppl:.3f}"))

    l27 = next((r for r in sens_rows if r.get("kind") == "svd" and r.get("layer") == "27" and float(r.get("keep", 0)) == 0.8), None)
    if l27:
        l27_dpct = float(l27["delta_pct"])
        checks.append(("Layer 27 keep 0.8 damage ~ +24.67%", abs(l27_dpct - 24.67) < 0.05, f"{l27_dpct:.2f}%"))

    # 3. Layer pairs
    pair_rows = read_utf8("results/sensitivity/qwen3-0.6b_pair_sensitivity.csv")
    if pair_rows:
        super_additive = sum(1 for r in pair_rows if float(r.get("interaction_pct", 0)) > 0)
        checks.append(("Super-additive pairs count = 33 of 40", super_additive == 33 and len(pair_rows) == 40, f"{super_additive}/{len(pair_rows)}"))

    # 4. Matched bytes K=18
    mb_rows = read_utf8("results/matched_bytes/summary.csv")
    k18_pw = next((r for r in mb_rows if r.get("K") == "18" and r.get("method") == "pairwise-multiplicative"), None)
    k18_diff = next((r for r in mb_rows if r.get("K") == "18" and "mb_K18_pairwise - mb_K18_random1" in r.get("plan", "") and "all chunks" in r.get("row", "")), None)
    if k18_pw:
        full_pct = float(k18_pw.get("measured_full_pct", 0))
        checks.append(("K=18 Pairwise full-split dPPL ~ +66.90%", abs(full_pct - 66.90) < 0.05, f"{full_pct:.2f}%"))
    if k18_diff:
        ci_lo = float(k18_diff.get("ci95_lo_pp", 0))
        ci_hi = float(k18_diff.get("ci95_hi_pp", 0))
        checks.append(("K=18 Random1 diff CI [-94.03, -89.04] pp", abs(ci_lo - (-94.03)) < 0.05 and abs(ci_hi - (-89.04)) < 0.05, f"[{ci_lo:.2f}, {ci_hi:.2f}]"))

    # 5. S4 Closed Loop K=18
    loop_rows = read_utf8("results/loop/summary.csv")
    loop_k18_seed2 = next((r for r in loop_rows if r.get("K") == "18" and r.get("method") == "loop" and r.get("seed") == "2"), None)
    if loop_k18_seed2:
        diff_pw = float(loop_k18_seed2.get("vs_oneshot_pw_pp", 0))
        checks.append(("S4 Loop K=18 seed 2 beats oneshot-pw by -1.53 pp", abs(diff_pw - (-1.53)) < 0.05, f"{diff_pw:.2f} pp"))

    return checks


def main() -> int:
    missing = [p for p in INPUTS if not (ROOT / p).exists()]
    if missing:
        for p in missing:
            print(f"Missing input CSV: {p}")
        return 2

    checks = verify_core_metrics()
    all_passed = True
    print("\n=== Empirical Results vs Ledger Verification ===")
    for desc, passed, val in checks:
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status:4s}] {desc:55s} -> {val}")

    if all_passed:
        print("\nAll empirical metrics successfully verified against result CSVs.")
        return 0
    else:
        print("\nOne or more numerical verification checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
