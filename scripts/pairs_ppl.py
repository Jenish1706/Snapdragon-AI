#!/usr/bin/env python3
"""Pairwise interactions of layer compression on Qwen3-0.6B, and perplexity of whole plans.

pairs: compress the MLPs of two layers together (activation-aware SVD, keep 0.8) for N random
       layer pairs (seed 0) and measure perplexity on the same first 40 WikiText-2 test chunks as
       the single-layer table, so interaction = dPPL(i, j) - dPPL(i) - dPPL(j) is paired.
plans: apply every MLP compression a plan chooses (activation-aware SVD at the plan's keep) and
       measure its perplexity the same way (optionally on the full test split too).
Weights are patched in place in one working copy of the F16 GGUF and restored after each run.
Apple M2 (Metal), llama.cpp b11256 llama-perplexity, context 512. Only CSVs are kept.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compress_ppl import BIN, F16, MODELS, WIKI, approximate, ensure_imatrix
from sensitivity_ppl import CHUNKS, log, tensor_info
from sensitivity_ppl import CSV as SINGLE_CSV

from isingedge.profile.mlp_graphs import relative_error

OUT = ROOT / "results/mac-m2/sensitivity"
PAIRS_CSV = OUT / "qwen3-0.6b_pair_sensitivity.csv"
PLANS_CSV = OUT / "qwen3-0.6b_plan_perplexity.csv"
WORK = MODELS / "Qwen3-0.6B-F16-pairs.gguf"


def perplexity(model: Path, chunks: int | None) -> tuple[float, float]:
    import re
    import subprocess

    cmd = [str(BIN / "llama-perplexity"), "-m", str(model), "-f", str(WIKI), "-c", "512", "-ngl", "99", "-b", "2048"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", res.stderr + res.stdout)
    if res.returncode != 0 or not m:
        (OUT / "last_failure.log").write_text(res.stderr + res.stdout)
        raise SystemExit("llama-perplexity failed; see results/mac-m2/sensitivity/last_failure.log")
    return float(m.group(1)), float(m.group(2))


def singles() -> tuple[float, dict[int, float]]:
    rows = list(csv.DictReader(SINGLE_CSV.open()))
    dense = float(next(r for r in rows if r["kind"] == "dense")["ppl"])
    single = {int(r["layer"]): float(r["delta_pct"]) for r in rows
              if r["kind"] == "svd" and r["scaling"] == "imatrix" and float(r["keep"]) == 0.8}
    return dense, single


def patch_and_measure(layers_keep: dict[int, float], info, scales, chunks: int | None) -> tuple[float, float, float]:
    """Compress the MLPs of the given layers at their keep fractions, measure, restore."""
    names = [(f"blk.{L}.ffn_{p}.weight", k) for L, k in layers_keep.items() for p in ("gate", "up", "down")]
    errs = []
    with open(F16, "rb") as pristine, open(WORK, "r+b") as work:
        try:
            for n, keep in names:
                off, shape, dtype = info[n]
                assert dtype == "F16", (n, dtype)
                pristine.seek(off)
                w = np.frombuffer(pristine.read(2 * shape[0] * shape[1]), dtype=np.float16).astype(np.float64).reshape(shape)
                w_hat, _, _ = approximate(w, "svd", keep, "A", scales[n])
                errs.append(relative_error(w, w_hat))
                work.seek(off)
                work.write(np.asarray(w_hat, dtype=np.float16).tobytes())
            work.flush()
            ppl, err = perplexity(WORK, chunks)
        finally:
            for n, _ in names:
                off, shape, _ = info[n]
                pristine.seek(off)
                work.seek(off)
                work.write(pristine.read(2 * shape[0] * shape[1]))
            work.flush()
    return ppl, err, float(np.mean(errs)) if errs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pairs", "plans"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plans", nargs="*", default=[])
    ap.add_argument("--full", action="store_true", help="plans: also measure on the full test split")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    info, scales = tensor_info(), ensure_imatrix()
    dense, single = singles()
    shutil.copyfile(F16, WORK)
    try:
        if args.mode == "pairs":
            done = set()
            if PAIRS_CSV.exists():
                done = {(int(r["layer_i"]), int(r["layer_j"])) for r in csv.DictReader(PAIRS_CSV.open())}
            rng = np.random.default_rng(args.seed)
            all_pairs = list(itertools.combinations(range(28), 2))
            pairs = [all_pairs[k] for k in sorted(rng.choice(len(all_pairs), size=args.n, replace=False))]
            new = not PAIRS_CSV.exists()
            with PAIRS_CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["layer_i", "layer_j", "keep", "scaling", "dpct_i", "dpct_j", "additive_pct",
                                                  "ppl", "ppl_err", "dense_ppl", "dpct_ij", "interaction_pct", "rel_error",
                                                  "chunks", "build"])
                if new:
                    w.writeheader()
                for i, j in pairs:
                    if (i, j) in done:
                        continue
                    ppl, err, rel = patch_and_measure({i: 0.8, j: 0.8}, info, scales, CHUNKS)
                    dij = 100 * (ppl - dense) / dense
                    w.writerow({"layer_i": i, "layer_j": j, "keep": 0.8, "scaling": "imatrix", "dpct_i": single[i],
                                "dpct_j": single[j], "additive_pct": single[i] + single[j], "ppl": ppl, "ppl_err": err,
                                "dense_ppl": dense, "dpct_ij": dij, "interaction_pct": dij - single[i] - single[j],
                                "rel_error": rel, "chunks": CHUNKS, "build": "llama.cpp b11256"})
                    f.flush()
                    log(f"pair L{i:02d}+L{j:02d}: dPPL {dij:+.2f}% vs additive {single[i] + single[j]:+.2f}% "
                        f"(interaction {dij - single[i] - single[j]:+.2f}%)")
        else:
            new = not PLANS_CSV.exists()
            with PLANS_CSV.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["plan", "compressed_layers", "n_compressed", "pred_additive_pct",
                                                  "pred_pairwise_pct", "chunks", "ppl", "ppl_err", "dense_ppl", "measured_pct"])
                if new:
                    w.writeheader()
                dense_full = None
                for path in args.plans:
                    plan = json.loads(Path(path).read_text())
                    layers = {int(g["group"].split(".")[1]): float(g["option"].split("-")[0][1:])
                              for g in plan["groups"] if g["group"].endswith(".mlp") and not g["option"].startswith("k1-")}
                    for chunks, dref in ((CHUNKS, dense), *(((None, None),) if args.full else ())):
                        if chunks is None:
                            if dense_full is None:
                                dense_full, _ = perplexity(F16, None)
                            dref = dense_full
                        ppl, err, _ = patch_and_measure(layers, info, scales, chunks)
                        w.writerow({"plan": Path(path).stem, "compressed_layers": " ".join(map(str, sorted(layers))),
                                    "n_compressed": len(layers), "pred_additive_pct": plan["prediction"].get("delta_ppl_additive_pct"),
                                    "pred_pairwise_pct": plan["prediction"].get("delta_ppl_pct"), "chunks": chunks or "all",
                                    "ppl": ppl, "ppl_err": err, "dense_ppl": dref, "measured_pct": 100 * (ppl - dref) / dref})
                        f.flush()
                        log(f"{Path(path).stem} ({len(layers)} layers, chunks {chunks or 'all'}): measured "
                            f"{100 * (ppl - dref) / dref:+.2f}% vs predicted pairwise {plan['prediction'].get('delta_ppl_pct')} "
                            f"/ additive {plan['prediction'].get('delta_ppl_additive_pct')}")
    finally:
        WORK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
