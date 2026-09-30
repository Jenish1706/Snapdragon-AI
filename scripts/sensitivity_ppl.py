#!/usr/bin/env python3
"""Per-layer accuracy sensitivity of Qwen3-0.6B: the planner's accuracy input (PLAN.md §0 step 5).

One layer at a time, one option at a time, paired against the dense model on the same chunks:
  * rank: SVD of the layer's MLP (gate, up, down) at keep fractions 0.9/0.8/0.7/0.6, data-free and
    activation-aware (imatrix from the wikitext-2 train split);
  * bits: ggml Q4_0 round-trip of the layer's MLP, and separately of its attention (q, k, v, o);
  * MPO: activation-aware 3-core MPO at equal parameters to the SVD, keep 0.9 and 0.7.
Tensors are patched in place in one working copy of the F16 GGUF and restored from the pristine
file after each run, so no per-variant model is kept. llama-perplexity (b11256, Metal) on the
first CHUNKS chunks of wikitext-2-raw test at context 512, Apple M2 Mac. Only the CSV is kept.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compress_ppl import (
    BIN,
    F16,
    MODELS,
    WIKI,
    approximate,
    ensure_imatrix,
    ensure_inputs,
)

from isingedge.profile.mlp_graphs import relative_error
from isingedge.sensitivity.quant import q4_0_roundtrip, q8_0_roundtrip

OUT = ROOT / "results/mac-m2/sensitivity"
CSV = OUT / "qwen3-0.6b_layer_sensitivity.csv"
WORK = MODELS / "Qwen3-0.6B-F16-work.gguf"
CHUNKS = 40
N_LAYERS = 28
COLS = ["kind", "part", "layer", "keep", "scaling", "rank", "bond", "params", "dense_params", "rel_error",
        "ppl", "ppl_err", "dense_ppl", "delta_ppl", "delta_pct", "chunks", "seconds", "build"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tensor_info() -> dict[str, tuple[int, tuple[int, int], str]]:
    from gguf import GGUFReader

    r = GGUFReader(str(F16))
    return {t.name: (int(t.data_offset), tuple(int(x) for x in t.data.shape), t.tensor_type.name)
            for t in r.tensors}


def perplexity(model: Path) -> tuple[float, float, float]:
    t0 = time.time()
    res = subprocess.run([str(BIN / "llama-perplexity"), "-m", str(model), "-f", str(WIKI), "-c", "512", "-ngl", "99",
                          "-b", "2048", "--chunks", str(CHUNKS)], capture_output=True, text=True, check=False)
    text = res.stderr + res.stdout
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    if res.returncode != 0 or not m:
        (OUT / "last_failure.log").write_text(text)
        raise SystemExit("llama-perplexity failed; see results/mac-m2/sensitivity/last_failure.log")
    return float(m.group(1)), float(m.group(2)), time.time() - t0


def plan() -> list[dict]:
    jobs = []
    for keep in (0.8, 0.6, 0.9, 0.7):  # interleaved so partial results cover the grid early
        for scaling in ("none", "imatrix"):
            jobs += [{"kind": "svd", "part": "mlp", "layer": i, "keep": keep, "scaling": scaling} for i in range(N_LAYERS)]
    for part in ("mlp", "attn"):
        jobs += [{"kind": "q4_0", "part": part, "layer": i, "keep": 1.0, "scaling": "-"} for i in range(N_LAYERS)]
    jobs += [{"kind": "q8_0", "part": p, "layer": i, "keep": 1.0, "scaling": "-"} for p in ("mlp", "attn") for i in (1, 14)]
    for keep in (0.9, 0.7):
        jobs += [{"kind": "mpo", "part": "mlp", "layer": i, "keep": keep, "scaling": "imatrix"} for i in range(N_LAYERS)]
    return jobs


def key(j: dict) -> tuple:
    return (j["kind"], j["part"], int(j["layer"]), float(j["keep"]), j["scaling"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_inputs()
    scales = ensure_imatrix()
    info = tensor_info()
    done: dict[tuple, dict] = {}
    if CSV.exists():
        with CSV.open() as f:
            for row in csv.DictReader(f):
                done[(row["kind"], row["part"], int(row["layer"]), float(row["keep"]), row["scaling"])] = row
    shutil.copyfile(F16, WORK)  # always start from a clean working copy
    if ("dense", "-", -1, 1.0, "-") in done:
        dense_ppl = float(done[("dense", "-", -1, 1.0, "-")]["ppl"])
    else:
        ppl, err, sec = perplexity(WORK)
        dense_ppl = ppl
        write_row({"kind": "dense", "part": "-", "layer": -1, "keep": 1.0, "scaling": "-", "ppl": ppl, "ppl_err": err,
                   "dense_ppl": ppl, "delta_ppl": 0.0, "delta_pct": 0.0, "chunks": CHUNKS, "seconds": round(sec, 1)})
        log(f"dense: PPL {ppl:.4f} +/- {err:.4f} ({CHUNKS} chunks)")
    parts = {"mlp": ("ffn_gate", "ffn_up", "ffn_down"), "attn": ("attn_q", "attn_k", "attn_v", "attn_output")}
    with open(F16, "rb") as pristine, open(WORK, "r+b") as work:
        for j in plan():
            if key(j) in done:
                continue
            names = [f"blk.{j['layer']}.{p}.weight" for p in parts[j["part"]]]
            errs, params, dense_params, extra = [], 0, 0, {}
            for n in names:
                off, shape, dtype = info[n]
                assert dtype == "F16", (n, dtype)
                pristine.seek(off)
                w = np.frombuffer(pristine.read(2 * shape[0] * shape[1]), dtype=np.float16).astype(np.float64).reshape(shape)
                if j["kind"] == "q4_0":
                    w_hat, p = q4_0_roundtrip(w), w.size
                elif j["kind"] == "q8_0":
                    w_hat, p = q8_0_roundtrip(w), w.size
                else:
                    scale = scales[n] if j["scaling"] == "imatrix" else None
                    w_hat, p, extra = approximate(w, j["kind"], j["keep"], "A", scale)
                errs.append(relative_error(w, w_hat))
                params, dense_params = params + p, dense_params + w.size
                work.seek(off)
                work.write(np.asarray(w_hat, dtype=np.float16).tobytes())
            work.flush()
            try:
                ppl, err, sec = perplexity(WORK)
            finally:
                for n in names:  # restore the pristine bytes
                    off, shape, _ = info[n]
                    pristine.seek(off)
                    work.seek(off)
                    work.write(pristine.read(2 * shape[0] * shape[1]))
                work.flush()
            row = {**j, **extra, "params": params, "dense_params": dense_params, "rel_error": float(np.mean(errs)),
                   "ppl": ppl, "ppl_err": err, "dense_ppl": dense_ppl, "delta_ppl": ppl - dense_ppl,
                   "delta_pct": 100 * (ppl - dense_ppl) / dense_ppl, "chunks": CHUNKS, "seconds": round(sec, 1)}
            write_row(row)
            done[key(j)] = row
            log(f"{j['kind']} {j['part']} L{j['layer']:02d} keep {j['keep']} {j['scaling']}: "
                f"PPL {ppl:.3f} (d {ppl - dense_ppl:+.3f}, {100 * (ppl - dense_ppl) / dense_ppl:+.2f}%) err {np.mean(errs):.3f}")
    WORK.unlink(missing_ok=True)
    log("sensitivity sweep finished")


def write_row(row: dict) -> None:
    new = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**row, "build": "llama.cpp b11256"})


if __name__ == "__main__":
    main()
