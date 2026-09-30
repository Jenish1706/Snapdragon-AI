#!/usr/bin/env python3
"""Qwen3-0.6B compression perplexity evaluation engine (PyTorch fp32 reference oracle).

Executes compression and quantization evaluations across transformer layers:
  singles   one layer's MLP at a time (activation-aware SVD at each keep), 40 chunks
  pairs     40 random layer pairs (seed 0), both MLPs at keep 0.8, 40 chunks
  plans     plans from a JSON spec, 40 chunks and full split; tracks per-chunk NLL
  kl        KL(dense || single-layer keep 0.8) on 64 x 256 validation tokens, against PPL rows
  allayer   every MLP of layers 2-25 compressed simultaneously, full split
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.loop.oracle import Qwen3Oracle
from isingedge.loop.thermal import wait_cool

MODEL = ROOT / "cache/models/hf/Qwen3-0.6B"
WIKI = ROOT / "cache/data/wikitext-2-raw"
CACHE = ROOT / "cache/torch_oracle"
DEVICE = "reference-gpu"
OUT = ROOT / "results"
SENS = OUT / "sensitivity"
SINGLES = SENS / "qwen3-0.6b_layer_sensitivity.csv"
PAIRS = SENS / "qwen3-0.6b_pair_sensitivity.csv"
PLANS = SENS / "qwen3-0.6b_plan_perplexity.csv"
CHUNK_NLL = SENS / "qwen3-0.6b_plan_chunk_nll.csv"
KL = SENS / "qwen3-0.6b_kl_proxy.csv"
ALL_LAYER = OUT / "compression" / "ppl.csv"
CHUNKS = 40
SINGLE_COLS = ["kind", "part", "layer", "keep", "scaling", "rank", "bond", "params", "dense_params", "rel_error",
               "ppl", "ppl_err", "dense_ppl", "delta_ppl", "delta_pct", "chunks", "seconds", "build"]
PAIR_COLS = ["layer_i", "layer_j", "keep", "scaling", "dpct_i", "dpct_j", "additive_pct", "ppl", "ppl_err", "dense_ppl",
             "dpct_ij", "interaction_pct", "rel_error", "chunks", "build"]
PLAN_COLS = ["plan", "compressed_layers", "n_compressed", "pred_additive_pct", "pred_pairwise_pct", "chunks", "ppl",
             "ppl_err", "dense_ppl", "measured_pct", "method", "K", "seed", "pred_log_pct", "log_ratio", "keeps", "build"]
_ORACLE: Qwen3Oracle | None = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oracle() -> Qwen3Oracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = Qwen3Oracle(MODEL, WIKI, CACHE)
    return _ORACLE


def build() -> str:
    import torch
    import transformers

    return (f"PyTorch {torch.__version__} CUDA {torch.version.cuda} fp32, {torch.cuda.get_device_name(0)}, "
            f"transformers {transformers.__version__}")


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8", newline=""))) if path.exists() else []


def append(path: Path, cols: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def guard(i: int, every: int = 10) -> None:
    if i % every == 0:
        wait_cool(log=log)


def singles_table() -> tuple[float, dict[tuple[int, float], float]]:
    rows = read(SINGLES)
    dense = float(next(r for r in rows if r["kind"] == "dense")["ppl"])
    return dense, {(int(r["layer"]), float(r["keep"])): float(r["delta_pct"]) for r in rows
                   if r["kind"] == "svd" and r["scaling"] == "imatrix"}


def cmd_singles(args) -> None:
    o, b = oracle(), build()
    done = {(r["kind"], int(r["layer"]), float(r["keep"]), r["scaling"]) for r in read(SINGLES)}
    d = o.dense_ref(CHUNKS)
    if ("dense", -1, 1.0, "-") not in done:
        append(SINGLES, SINGLE_COLS, {"kind": "dense", "part": "-", "layer": -1, "keep": 1.0, "scaling": "-", "ppl": d.ppl,
                                      "ppl_err": d.err, "dense_ppl": d.ppl, "delta_ppl": 0.0, "delta_pct": 0.0,
                                      "chunks": CHUNKS, "seconds": round(d.seconds, 1), "build": b})
        log(f"dense: PPL {d.ppl:.4f} +/- {d.err:.4f} ({CHUNKS} chunks, {d.seconds:.1f} s)")
    jobs = [(L, k) for k in args.keeps for L in args.layers if ("svd", L, k, args.scaling) not in done]
    for i, (L, k) in enumerate(jobs):
        guard(i)
        r = o.evaluate({L: k}, CHUNKS, args.scaling)
        inf = o.info(L, k, args.scaling)
        dense_params = 3 * 1024 * 3072
        append(SINGLES, SINGLE_COLS, {"kind": "svd", "part": "mlp", "layer": L, "keep": k, "scaling": args.scaling,
                                      "rank": inf["rank"], "bond": "", "params": 3 * inf["rank"] * 4096,
                                      "dense_params": dense_params, "rel_error": inf["rel_error"], "ppl": r.ppl,
                                      "ppl_err": r.err, "dense_ppl": d.ppl, "delta_ppl": r.ppl - d.ppl,
                                      "delta_pct": r.extra["delta_pct"], "chunks": CHUNKS, "seconds": round(r.seconds, 1),
                                      "build": b})
        log(f"svd mlp L{L:02d} keep {k} {args.scaling}: PPL {r.ppl:.3f} ({r.extra['delta_pct']:+.2f}%) "
            f"err {inf['rel_error']:.3f} [{r.seconds:.1f} s]")


def mac_pairs(n: int = 40, seed: int = 0) -> list[tuple[int, int]]:
    """The same pairs as scripts/pairs_ppl.py on the Mac."""
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(range(28), 2))
    return [all_pairs[k] for k in sorted(rng.choice(len(all_pairs), size=n, replace=False))]


def cmd_pairs(args) -> None:
    o, b = oracle(), build()
    dense, single = singles_table()
    done = {(int(r["layer_i"]), int(r["layer_j"])) for r in read(PAIRS)}
    todo = [p for p in mac_pairs(args.n, args.seed) if p not in done]
    for i, (a, c) in enumerate(todo):
        guard(i)
        r = o.evaluate({a: 0.8, c: 0.8}, CHUNKS)
        dij = r.extra["delta_pct"]
        si, sj = single[(a, 0.8)], single[(c, 0.8)]
        append(PAIRS, PAIR_COLS, {"layer_i": a, "layer_j": c, "keep": 0.8, "scaling": "imatrix", "dpct_i": si, "dpct_j": sj,
                                  "additive_pct": si + sj, "ppl": r.ppl, "ppl_err": r.err, "dense_ppl": dense,
                                  "dpct_ij": dij, "interaction_pct": dij - si - sj,
                                  "rel_error": (o.info(a, 0.8)["rel_error"] + o.info(c, 0.8)["rel_error"]) / 2,
                                  "chunks": CHUNKS, "build": b})
        log(f"pair L{a:02d}+L{c:02d}: {dij:+.2f}% vs additive {si + sj:+.2f}% (interaction {dij - si - sj:+.2f} pp)")


def measure_plans(specs: list[dict], full: bool = False, chunks: int = CHUNKS) -> list[dict]:
    """Measure plan specs ({plan, layers: {L: keep}, ...}); skips (plan, chunks) rows already written."""
    o, b = oracle(), build()
    done = {(r["plan"], r["chunks"]) for r in read(PLANS)}
    have_nll = {(r["plan"], r["chunks"]) for r in read(CHUNK_NLL)}
    out = []
    modes = [chunks] + ([None] if full else [])
    for n in modes:
        tag = str(n) if n else "all"
        if ("dense", tag) not in have_nll:
            d = o.dense_ref(n)
            append(CHUNK_NLL, ["plan", "chunks", "per_chunk", "nll_sums"],
                   {"plan": "dense", "chunks": tag, "per_chunk": d.per_chunk, "nll_sums": " ".join(f"{v:.6f}" for v in d.nll)})
            have_nll.add(("dense", tag))
            log(f"dense ({tag} chunks): PPL {d.ppl:.4f} +/- {d.err:.4f}")
    i = 0
    for s in specs:
        layers = {int(L): float(k) for L, k in s["layers"].items()}
        for n in modes:
            tag = str(n) if n else "all"
            if (s["plan"], tag) in done:
                continue
            guard(i)
            i += 1
            r = o.evaluate(layers, n)
            row = {"plan": s["plan"], "compressed_layers": " ".join(map(str, sorted(L for L, k in layers.items() if k < 1))),
                   "n_compressed": sum(k < 1 for k in layers.values()), "pred_additive_pct": s.get("pred_additive_pct", ""),
                   "pred_pairwise_pct": s.get("pred_pairwise_pct", ""), "chunks": tag, "ppl": r.ppl, "ppl_err": r.err,
                   "dense_ppl": r.extra["dense_ppl"], "measured_pct": r.extra["delta_pct"], "method": s.get("method", ""),
                   "K": s.get("K", ""), "seed": s.get("seed", ""), "pred_log_pct": s.get("pred_log_pct", ""),
                   "log_ratio": r.extra["log_ratio"],
                   "keeps": " ".join(f"{L}:{k:g}" for L, k in sorted(layers.items()) if k < 1), "build": b}
            append(PLANS, PLAN_COLS, row)
            append(CHUNK_NLL, ["plan", "chunks", "per_chunk", "nll_sums"],
                   {"plan": s["plan"], "chunks": tag, "per_chunk": r.per_chunk,
                    "nll_sums": " ".join(f"{v:.6f}" for v in r.nll)})
            out.append(row)
            log(f"{s['plan']} ({row['n_compressed']} MLPs, {tag} chunks): {r.extra['delta_pct']:+.2f}% "
                f"[{r.seconds:.1f} s]")
    return out


def cmd_plans(args) -> None:
    specs = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    measure_plans(specs, args.full)


def cmd_kl(args) -> None:
    o, b = oracle(), build()
    _, single = singles_table()
    layers = [L for L in range(28) if (L, args.keep) in single]
    wait_cool(log=log)
    t0 = time.time()
    fwd, rev = o.kl_many([{L: args.keep} for L in layers], args.n_seq, args.seq_len)
    cols = ["layer", "keep", "scaling", "kl_nats", "kl_reverse_nats", "kl_nats_16seq", "n_seq", "seq_len", "split",
            "delta_pct", "build"]
    for i, L in enumerate(layers):
        append(KL, cols, {"layer": L, "keep": args.keep, "scaling": "imatrix", "kl_nats": fwd[i].sum() / fwd[i].size / args.seq_len,
                          "kl_reverse_nats": rev[i].sum() / rev[i].size / args.seq_len,
                          "kl_nats_16seq": fwd[i, :16].sum() / 16 / args.seq_len, "n_seq": args.n_seq,
                          "seq_len": args.seq_len, "split": "valid", "delta_pct": single[(L, args.keep)], "build": b})
    from isingedge.loop.stats import spearman

    ppl = np.array([single[(L, args.keep)] for L in layers])
    rng = np.random.default_rng(0)
    lines = []
    for name, v in (("KL(dense||comp)", fwd.sum(1)), ("KL(comp||dense)", rev.sum(1)),
                    ("KL(dense||comp), 16 sequences", fwd[:, :16].sum(1))):
        boot = [spearman(v[i], ppl[i]) for i in (rng.integers(0, len(v), len(v)) for _ in range(4000))]
        lo, hi = np.nanquantile(boot, [0.025, 0.975])
        lines.append(f"{name}: rho {spearman(v, ppl):.3f} [{lo:.3f}, {hi:.3f}]")
        append(SENS / "qwen3-0.6b_kl_proxy_summary.csv", ["proxy", "rho", "ci95_lo", "ci95_hi", "layers", "build"],
               {"proxy": name, "rho": spearman(v, ppl), "ci95_lo": lo, "ci95_hi": hi, "layers": len(v), "build": b})
    log(f"KL proxy: {len(layers)} layers in {time.time() - t0:.0f} s; Spearman vs 40-chunk dPPL (bootstrap over layers): "
        + "; ".join(lines))


def cmd_allayer(args) -> None:
    o, b = oracle(), build()
    wait_cool(log=log)
    tag = f"{'imx-' if args.scaling == 'imatrix' else ''}{'mpoA' if args.method == 'mpo' else args.method}-k{args.keep}"
    done = {r["tag"] for r in read(ALL_LAYER)}
    cols = ["tag", "method", "keep", "scaling", "layers", "rank", "rel_error", "ppl", "ppl_err", "chunks", "dense_ppl",
            "delta_pct", "seconds", "build"]
    if "dense-fp32" not in done:
        d = o.dense_ref(None)
        append(ALL_LAYER, cols, {"tag": "dense-fp32", "method": "dense", "keep": 1.0, "scaling": "-", "layers": "-",
                                 "ppl": d.ppl, "ppl_err": d.err, "chunks": d.chunks, "dense_ppl": d.ppl, "delta_pct": 0.0,
                                 "seconds": round(d.seconds, 1), "build": b})
        log(f"dense full split: PPL {d.ppl:.4f} +/- {d.err:.4f} over {d.chunks} chunks ({d.seconds:.0f} s)")
    if tag in done:
        return
    layers = list(range(2, 26))
    r = o.evaluate({L: args.keep for L in layers}, None, args.scaling, args.method)
    errs = [o.info(L, args.keep, args.scaling, args.method)["rel_error"] for L in layers]
    inf = o.info(2, args.keep, args.scaling, args.method)
    append(ALL_LAYER, cols, {"tag": tag, "method": args.method, "keep": args.keep, "scaling": args.scaling,
                             "layers": "2-25", "rank": inf["rank"] if args.method == "svd" else f"bond {inf['bond']}",
                             "rel_error": float(np.mean(errs)), "ppl": r.ppl, "ppl_err": r.err, "chunks": r.chunks,
                             "dense_ppl": r.extra["dense_ppl"], "delta_pct": r.extra["delta_pct"],
                             "seconds": round(r.seconds, 1), "build": b})
    log(f"{tag} layers 2-25, full split: PPL {r.ppl:.3f} ({r.extra['delta_pct']:+.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("singles")
    p.add_argument("--layers", type=int, nargs="*", default=list(range(28)))
    p.add_argument("--keeps", type=float, nargs="*", default=[0.8, 0.9])
    p.add_argument("--scaling", default="imatrix", choices=("imatrix", "none"))
    p = sub.add_parser("pairs")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plans")
    p.add_argument("--spec", required=True)
    p.add_argument("--full", action="store_true")
    p = sub.add_parser("kl")
    p.add_argument("--keep", type=float, default=0.8)
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p = sub.add_parser("allayer")
    p.add_argument("--method", default="svd")
    p.add_argument("--keep", type=float, default=0.9)
    p.add_argument("--scaling", default="imatrix", choices=("imatrix", "none"))
    args = ap.parse_args()
    {"singles": cmd_singles, "pairs": cmd_pairs, "plans": cmd_plans, "kl": cmd_kl, "allayer": cmd_allayer}[args.cmd](args)


if __name__ == "__main__":
    main()
