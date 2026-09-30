#!/usr/bin/env python3
"""Step 5: SVD vs MPO at equal parameters on Qwen3-0.6B, WikiText-2 perplexity (llama.cpp).

Compression of the MLP matrices (ffn_gate, ffn_up, ffn_down) in layers 2-25:
  * SVD: truncated SVD at keep fractions of the 2-core break-even rank (ranks multiple of 32);
  * MPO: 3-core TT-SVD with an equal bond chosen to match the SVD parameter count per matrix.
Two scalings: "none" (data-free, W itself) and "imatrix" (activation-aware: W diag(s) is factorised
and unscaled, s_i = RMS of input channel i from llama-imatrix on the wikitext-2 *train* split).
The dense reconstruction W_hat is written in place into a copy of the F16 GGUF, so perplexity
measures the approximation error only; this experiment makes no speed claim. Perplexity comes
from llama-perplexity (llama.cpp b11256, Metal) on the wikitext-2-raw test split at context 512,
measured on an Apple M2 Mac. Results: results/mac-m2/compression/.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.profile.mlp_graphs import (
    MPOSpec,
    keep_fraction_rank,
    match_mpo_bond,
    mpo_cores,
    mpo_dense,
    mpo_param_count,
    relative_error,
)

BIN = ROOT / "cache/bin/llama-b11256-bin-macos-arm64/llama-b11256"
MODELS = ROOT / "cache/models"
BF16 = MODELS / "Qwen3-0.6B-BF16.gguf"
F16 = MODELS / "Qwen3-0.6B-F16.gguf"
DATA = ROOT / "cache/data"
WIKI_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKI = DATA / "wikitext-2-raw" / "wiki.test.raw"
OUT = ROOT / "results/mac-m2/compression"
IMATRIX = ROOT / "cache/models/Qwen3-0.6B-F16.imatrix.gguf"
TRAIN = DATA / "wikitext-2-raw" / "wiki.train.raw"
LAYERS = range(2, 26)
HID, INTER = 1024, 3072
KEEPS = (0.9, 0.8, 0.7, 0.6, 0.5)
SHAPES = {
    "A": ((16, 4, 16), (16, 12, 16)),  # (modes of 1024, modes of 3072): big outer modes
    "B": ((8, 16, 8), (16, 12, 16)),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_inputs() -> dict:
    if not F16.exists():
        log("converting BF16 -> F16 with llama-quantize")
        subprocess.run([str(BIN / "llama-quantize"), str(BF16), str(F16), "F16"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not WIKI.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        z = DATA / "wikitext-2-raw-v1.zip"
        urllib.request.urlretrieve(WIKI_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA)
    return {"wiki_sha256": hashlib.sha256(WIKI.read_bytes()).hexdigest(), "wiki_bytes": WIKI.stat().st_size,
            "f16_bytes": F16.stat().st_size}


def tensor_table() -> dict[str, tuple[int, tuple[int, int]]]:
    from gguf import GGUFReader

    reader = GGUFReader(str(F16))
    table = {}
    for t in reader.tensors:
        if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)\.weight", t.name):
            if t.tensor_type.name != "F16":
                raise SystemExit(f"{t.name} is {t.tensor_type.name}, expected F16")
            off = int(t.data_offset)
            shape = tuple(int(x) for x in t.data.shape)
            assert bytes(reader.data[off:off + 16]) == t.data.reshape(-1)[:8].tobytes(), "offset check failed"
            table[t.name] = (off, shape)
    return table


def modes_for(shape: tuple[int, int], key: str):
    small, big = SHAPES[key]
    d_out, d_in = shape
    return (small, big) if (d_in, d_out) == (HID, INTER) else (big, small)


def ensure_imatrix() -> dict[str, np.ndarray]:
    """RMS input activation per column for every ffn matrix, from llama-imatrix on the train split."""
    from gguf import GGUFReader

    if not IMATRIX.exists():
        log("computing imatrix on wikitext-2 train (128 chunks of 512)")
        subprocess.run([str(BIN / "llama-imatrix"), "-m", str(F16), "-f", str(TRAIN), "-o", str(IMATRIX),
                        "-c", "512", "--chunks", "128", "-ngl", "99", "--no-ppl"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = GGUFReader(str(IMATRIX))
    t = {x.name: np.asarray(x.data, dtype=np.float64).reshape(-1) for x in r.tensors}
    out = {}
    for name, v in t.items():
        if name.endswith(".in_sum2") and ".ffn_" in name:
            base = name[: -len(".in_sum2")]
            count = float(t[base + ".counts"][0])
            out[base] = np.sqrt(v / max(count, 1.0)) + 1e-6
    return out


def approximate(w: np.ndarray, method: str, keep: float, shape_key: str, scale: np.ndarray | None):
    """Return (w_hat, params, extra) for one matrix w [d_out, d_in]."""
    d_out, d_in = w.shape
    ws = w * scale[None, :] if scale is not None else w
    if method == "svd-full":
        u, s, vt = np.linalg.svd(ws, full_matrices=False)
        w_hat, params, extra = (u * s) @ vt, w.size, {"rank": int(s.size)}
    elif method == "svd":
        rank = keep_fraction_rank(d_in, d_out, keep)
        u, s, vt = np.linalg.svd(ws, full_matrices=False)
        w_hat, params, extra = (u[:, :rank] * s[:rank]) @ vt[:rank], rank * (d_in + d_out), {"rank": rank}
    else:
        rank = keep_fraction_rank(d_in, d_out, keep)
        in_modes, out_modes = modes_for(w.shape, shape_key)
        chi = match_mpo_bond(rank * (d_in + d_out), in_modes, out_modes)
        w_hat = mpo_dense(*mpo_cores(ws, MPOSpec(in_modes, out_modes, (chi, chi))))
        params, extra = mpo_param_count(in_modes, out_modes, (chi, chi)), {"bond": chi}
    if scale is not None:
        w_hat = w_hat / scale[None, :]
    return w_hat, params, extra


def perplexity(model: Path, tag: str, chunks: int | None) -> dict:
    t0 = time.time()
    cmd = [str(BIN / "llama-perplexity"), "-m", str(model), "-f", str(WIKI), "-c", "512", "-ngl", "99", "-b", "2048"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = res.stderr + res.stdout
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    if res.returncode != 0 or not m:
        (OUT / f"{tag}.failure.log").write_text(text)
        raise SystemExit(f"perplexity failed for {tag}; see {OUT / (tag + '.failure.log')}")
    n = re.search(r"calculating perplexity over (\d+) chunks", text)
    return {"ppl": float(m.group(1)), "ppl_err": float(m.group(2)), "chunks": int(n.group(1)) if n else None,
            "ppl_seconds": round(time.time() - t0, 1)}


PLAN = (
    [("dense-f16", "dense", 1.0, "-", "none", LAYERS)]
    + [("sanity-svd-fullrank", "svd-full", 1.0, "-", "none", LAYERS)]
    + [("sanity-svd-k0.9-l14only", "svd", 0.9, "-", "none", range(14, 15))]
    + [(f"svd-k{k}", "svd", k, "-", "none", LAYERS) for k in (0.9, 0.8, 0.5)]
    + [(f"mpoA-k{k}", "mpo", k, "A", "none", LAYERS) for k in (0.9, 0.8, 0.5)]
    + [(f"imx-svd-k{k}", "svd", k, "-", "imatrix", LAYERS) for k in KEEPS]
    + [(f"imx-mpoA-k{k}", "mpo", k, "A", "imatrix", LAYERS) for k in KEEPS]
    + [("imx-mpoB-k0.5", "mpo", 0.5, "B", "imatrix", LAYERS)]
)


CSV_COLS = ["tag", "method", "keep", "mpo_shape", "scaling", "layers", "rank", "bond", "compressed_params",
            "dense_params", "param_ratio", "rel_error_mean", "rel_error_max", "ppl", "ppl_err", "chunks", "ppl_seconds",
            "factorise_seconds", "wiki_sha256"]


def append_csv(path: Path, rec: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)


def main() -> None:
    import argparse

    from gguf import GGUFReader

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated tags")
    ap.add_argument("--chunks", type=int, help="limit perplexity chunks (sanity checks only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ensure_inputs()
    table = tensor_table()
    scales = ensure_imatrix()
    csv_path = OUT / "ppl.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            done = {row["tag"] for row in csv.DictReader(f)}
    reader = GGUFReader(str(F16))
    for tag, method, keep, shape_key, scaling, layers in PLAN:
        if args.only and tag not in args.only.split(","):
            continue
        if args.chunks:
            tag = f"{tag}-c{args.chunks}"
        if tag in done:
            log(f"skip {tag}")
            continue
        rec = {"tag": tag, "method": method, "keep": keep, "mpo_shape": shape_key, "scaling": scaling,
               "layers": f"{layers.start}-{layers.stop - 1}", **meta}
        model = F16
        if method != "dense":
            model = MODELS / f"Qwen3-0.6B-F16-{tag}.gguf"
            shutil.copyfile(F16, model)
            errs, params, dense_params, extra, t0 = [], 0, 0, {}, time.time()
            with open(model, "r+b") as f:
                for i in layers:
                    for part in ("gate", "up", "down"):
                        n = f"blk.{i}.ffn_{part}.weight"
                        off, shape = table[n]
                        w = np.array(reader.data[off:off + 2 * shape[0] * shape[1]].view(np.float16),
                                     dtype=np.float64).reshape(shape)
                        w_hat, p, extra = approximate(w, method, keep, shape_key, scales[n] if scaling == "imatrix" else None)
                        errs.append(relative_error(w, w_hat))
                        params, dense_params = params + p, dense_params + w.size
                        f.seek(off)
                        f.write(w_hat.astype(np.float16).tobytes())
                        del w, w_hat
            rec.update(extra)
            rec.update({"compressed_params": params, "dense_params": dense_params, "param_ratio": params / dense_params,
                        "rel_error_mean": float(np.mean(errs)), "rel_error_max": float(np.max(errs)),
                        "factorise_seconds": round(time.time() - t0, 1)})
        log(f"{tag}: perplexity")
        try:
            rec.update(perplexity(model, tag, args.chunks))
        finally:
            if model != F16:
                model.unlink(missing_ok=True)
        append_csv(csv_path, rec)
        extra_txt = (f" | params {rec['param_ratio']:.3f} of dense, rel. err {rec['rel_error_mean']:.3f}"
                     if method != "dense" else "")
        log(f"{tag}: PPL {rec['ppl']:.3f} +/- {rec['ppl_err']:.3f}{extra_txt}")
    log("done")


if __name__ == "__main__":
    main()
