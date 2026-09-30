#!/usr/bin/env python3
"""Step 4: context packing on HotpotQA (distractor) — QUBO vs top-k vs MMR vs full context.

For each question the candidate units are the 10 paragraphs' sentences. Relevance and redundancy
are TF-IDF cosines; token lengths come from the served model's tokenizer. Each method selects
sentences under a token budget; the packed prompt goes to a llama-server (/completion,
temperature 0, no prompt cache, Qwen3 non-thinking template) which reports prefill time and the
answer. Metrics: EM/F1 (official HotpotQA normalisation), supporting-sentence recall, tokens
kept, prefill ms, TTFT ms, selection (solve) ms. --label names the device that served the model.
Results: results/<label>/packer/hotpotqa_packer.csv (CSV only).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.packer.core import PrefillCost, f1_em, mmr, pack, tfidf_scores, top_k

DATA = ROOT / "cache/data/hotpotqa/validation.parquet"
SYSTEM = "Answer the question using only the context. Reply with a short answer: a few words, or yes or no."
COLS = ["qid", "type", "level", "method", "budget", "n_units", "units_kept", "tokens_kept", "sf_recall", "prompt_tokens",
        "prefill_ms", "ttft_ms", "decode_tps", "solve_ms", "em", "f1", "answer", "prediction"]


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def load_subset(n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(DATA).to_pylist()
    rng = np.random.default_rng(seed)
    return [rows[i] for i in sorted(rng.choice(len(rows), size=n, replace=False))]


def units_of(q: dict) -> list[dict]:
    sf = set(zip(q["supporting_facts"]["title"], q["supporting_facts"]["sent_id"]))
    out = []
    for title, sents in zip(q["context"]["title"], q["context"]["sentences"]):
        for k, s in enumerate(sents):
            if s.strip():
                out.append({"title": title, "k": k, "text": s.strip(), "support": (title, k) in sf})
    return out


def prompt_for(question: str, units: list[dict], x: np.ndarray) -> str:
    by_title: dict[str, list[str]] = {}
    for u, keep in zip(units, x):
        if keep:
            by_title.setdefault(u["title"], []).append(u["text"])
    ctx = "\n".join(f"{t}: {' '.join(s)}" for t, s in by_title.items())
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nContext:\n{ctx}\n\nQuestion: {question}"
            f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--label", required=True, help="device that serves the model, e.g. pad4 or mac-m2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the SnapTwin prefill-time term (relevance per s)")
    ap.add_argument("--alpha", type=float, default=0.0, help="SnapTwin prefill s/token for the serving unit")
    ap.add_argument("--beta", type=float, default=0.0, help="SnapTwin prefill s/token^2 (attention)")
    ap.add_argument("--methods", default="full,topk,mmr,qubo,qubo-cost")
    ap.add_argument("--pad", action="store_true", help="serve the model on the Pad 4 and time the QUBO solve on the device")
    ap.add_argument("--pad-unit", default="npu", choices=("cpu", "gpu", "npu"))
    ap.add_argument("--pad-model", default="Qwen3-1.7B-Q4_0.gguf")
    args = ap.parse_args()
    serial = None
    if args.pad:
        from isingedge.bench import adb
        from isingedge.packer import device

        pads = [d for d in adb.devices() if d.state == "device" and d.label == "pad4"]
        if not pads:
            sys.exit("OnePlus Pad 4 not visible to adb")
        serial = pads[0].serial
        device.ensure_solver(serial)
        device.start_server(serial, args.pad_model, args.pad_unit)
        args.server, args.label = "http://127.0.0.1:8092", "pad4"
    out = ROOT / "results" / args.label / "packer" / f"hotpotqa_packer_b{args.budget}{'_' + args.pad_unit if args.pad else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["qid"], r["method"]) for r in csv.DictReader(f)}
    new = not out.exists()
    subset = load_subset(args.n, args.seed)
    n0 = len(post(args.server + "/tokenize", {"content": prompt_for("", [], np.zeros(0))})["tokens"])
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for qi, q in enumerate(subset):
            units = units_of(q)
            lens = np.array([len(post(args.server + "/tokenize", {"content": " " + u["text"]})["tokens"]) for u in units])
            rel, sim = tfidf_scores(q["question"], [f"{u['title']} {u['text']}" for u in units])
            cost = PrefillCost(args.alpha, args.beta, n0 + len(q["question"]) // 4)
            for method in args.methods.split(","):
                if (q["id"], method) in done:
                    continue
                t0 = time.perf_counter()
                if method == "full":
                    x = np.ones(len(units), dtype=np.int8)
                elif method == "topk":
                    x = top_k(rel, lens, args.budget)
                elif method == "mmr":
                    x = mmr(rel, sim, lens, args.budget)
                elif serial is not None:  # QUBO methods solved on the device itself
                    from isingedge.packer import device

                    lam = args.lam if method == "qubo-cost" else 0.0
                    x, dev_ms = device.pack_on_device(serial, rel, sim, lens, args.budget, lam=lam, alpha=args.alpha,
                                                      beta=args.beta, n0=cost.n0, seed=args.seed)
                elif method == "qubo":
                    x, _, _ = pack(rel, sim, lens, args.budget, seed=args.seed)
                else:  # qubo-cost: same QUBO plus the SnapTwin prefill-time term
                    x, _, _ = pack(rel, sim, lens, args.budget, lam=args.lam, cost=cost, seed=args.seed)
                solve_ms = 1000 * (time.perf_counter() - t0)
                if serial is not None and method.startswith("qubo"):
                    solve_ms = dev_ms
                res = post(args.server + "/completion", {"prompt": prompt_for(q["question"], units, x), "n_predict": 32,
                                                          "temperature": 0, "cache_prompt": False, "stop": ["<|im_end|>", "\n"]})
                t = res.get("timings", {})
                pred = res.get("content", "").strip()
                f1, em = f1_em(pred, q["answer"])
                sup = [u["support"] for u in units]
                per_tok = t.get("predicted_ms", 0) / max(1, t.get("predicted_n", 1))
                w.writerow({"qid": q["id"], "type": q["type"], "level": q["level"], "method": method, "budget": args.budget,
                            "n_units": len(units), "units_kept": int(x.sum()), "tokens_kept": int(lens @ x),
                            "sf_recall": round(sum(s and k for s, k in zip(sup, x)) / max(1, sum(sup)), 4),
                            "prompt_tokens": t.get("prompt_n"), "prefill_ms": round(t.get("prompt_ms", 0), 2),
                            "ttft_ms": round(t.get("prompt_ms", 0) + per_tok, 2),
                            "decode_tps": round(t.get("predicted_per_second", 0), 2), "solve_ms": round(solve_ms, 3),
                            "em": em, "f1": round(f1, 4), "answer": q["answer"], "prediction": pred})
                f.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {qi + 1}/{len(subset)} {q['id']}", flush=True)
    if serial is not None:
        from isingedge.packer import device

        device.stop_server(serial)


if __name__ == "__main__":
    main()
