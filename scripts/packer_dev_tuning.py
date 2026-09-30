#!/usr/bin/env python3
"""Reproduce the packer's design choice on the dev sample (seed 1, disjoint from the eval subset):
supporting-sentence recall and tokens kept at a 256-token budget. Needs a llama-server only for
tokenisation. Output: results/mac-m2/packer/dev_tuning_b256.csv (Apple M2, no model answers)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from packer_eval import load_subset, post, units_of

from isingedge.packer.core import (
    mmr,
    pack,
    qubo_pack,
    qubo_pack_lagrange,
    tfidf_scores,
    top_k,
)


def main(server: str = "http://127.0.0.1:8091", budget: int = 256) -> None:
    cache = []
    for q in load_subset(50, seed=1):
        units = units_of(q)
        lens = np.array([len(post(server + "/tokenize", {"content": " " + u["text"]})["tokens"]) for u in units])
        rel, sim = tfidf_scores(q["question"], [f"{u['title']} {u['text']}" for u in units])
        cache.append((rel, sim, lens, np.array([u["support"] for u in units])))

    def evaluate(fn):
        out = [(float((fn(r, s, l).astype(bool) & sup).sum() / max(1, sup.sum())), float(fn(r, s, l) @ l)) for r, s, l, sup in cache]
        a = np.array(out)
        return round(float(a[:, 0].mean()), 3), round(float(a[:, 1].mean()))

    def thresholded(s, tau):
        s = np.clip(s - tau, 0, None)
        np.fill_diagonal(s, 0)
        return s

    methods = {
        "top-k (budget 128)": lambda r, s, l: top_k(r, l, 128),
        "top-k (budget 192)": lambda r, s, l: top_k(r, l, 192),
        "top-k": lambda r, s, l: top_k(r, l, budget),
        "MMR (lambda 0.7)": lambda r, s, l: mmr(r, s, l, budget),
        "QUBO, slack-bit budget, mu 0.5, tau 0.5": lambda r, s, l: qubo_pack(r / r.max(), thresholded(s, 0.5), l, budget, mu=0.5)[0],
        "QUBO, Lagrangian budget, mu 0.25, tau 0.3": lambda r, s, l: qubo_pack_lagrange(r / r.max(), thresholded(s, 0.3), l, budget, mu=0.25)[0],
        "QUBO, relevance mass, mu 0.1, tau 0.3 (chosen)": lambda r, s, l: pack(r, s, l, budget)[0],
    }
    out = ROOT / "results/mac-m2/packer/dev_tuning_b256.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sf_recall", "tokens_kept", "questions", "seed"])
        w.writeheader()
        for name, fn in methods.items():
            sf, tok = evaluate(fn)
            w.writerow({"method": name, "sf_recall": sf, "tokens_kept": tok, "questions": len(cache), "seed": 1})
            print(f"{name:48s} recall {sf:.3f} tokens {tok}", flush=True)


if __name__ == "__main__":
    main()
