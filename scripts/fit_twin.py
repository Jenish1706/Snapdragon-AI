#!/usr/bin/env python3
"""Fit SnapTwin on published rows and, when present, on device sweeps; predict the PC rows.

Stage A (published only): per device and unit, fit decode and prefill on Qualcomm's published
GenieX llama.cpp rows for Qwen3-0.6B/1.7B/4B Q4_0 (data/published/...); report leave-one-model-out
error; then fit on a phone (Galaxy S26 / S25) and predict the held-out X Elite and X2 Elite CRD
rows by rescaling the bandwidth-bound terms with nominal bandwidth (compute ratio 1 for prefill).
Stage B (devices): the same transfer from fits on results/<device>/llama-bench sweeps, plus the
Galaxy S26 cross-check for Pad 4 measurements. Outputs CSVs in results/twin/.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.twin.model_facts import load_facts
from isingedge.twin.snaptwin import (
    Sample,
    ape,
    fit,
    leave_one_model_out,
    predict,
    samples_from_llama_bench,
)

PUB = ROOT / "data/published/geniex_llamacpp_qwen3_q4_0.csv"
BW = {r["device"]: float(r["nominal_gbs"]) for r in csv.DictReader((ROOT / "data/specs/bandwidth.csv").open())}
OUT = ROOT / "results/twin"
TARGETS = ("Snapdragon X Elite CRD", "Snapdragon X2 Elite CRD")
SOURCES = ("Samsung Galaxy S26", "Samsung Galaxy S25")
DEVICE_LABELS = {"pad4": "OnePlus Pad 4", "nordce4": "OnePlus Nord CE4"}


def published_samples() -> dict[tuple[str, str], list[Sample]]:
    out: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for r in csv.DictReader(PUB.open()):
        n, key = int(r["prompt_tokens"]), (r["device"], r["unit"])
        out[key].append(Sample(r["model"], "pp", n, 0, n / float(r["prefill_tps"])))
        out[key].append(Sample(r["model"], "tg", int(r["gen_tokens"]), n, 1 / float(r["decode_tps"])))
    return out


def device_samples() -> dict[tuple[str, str], list[Sample]]:
    out: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for label, name in DEVICE_LABELS.items():
        for f in sorted((ROOT / "results" / label / "llama-bench").glob("Qwen3-*__*__*.jsonl")):
            model, unit, test = f.stem.split("__")
            if test not in ("pp", "tg"):
                continue
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            out[(name, unit)] += samples_from_llama_bench(rows, model)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(recs: list[dict], by: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        groups[tuple(r[k] for k in by)].append(r["ape"])
    return [{**dict(zip(by, k)), "cells": len(v), "median_ape_pct": round(100 * float(np.median(v)), 1),
             "max_ape_pct": round(100 * float(np.max(v)), 1)} for k, v in sorted(groups.items())]


def transfer(fits: dict, samples: dict, sources, targets, stage: str) -> list[dict]:
    facts = load_facts()
    recs = []
    for src in sources:
        for tgt in targets:
            for bw_tgt_name, bw_tgt in ((tgt, BW[tgt]),) + ((("X2 Elite Extreme 228", 228.0),) if "X2" in tgt else ()):
                for unit in ("cpu", "gpu", "npu"):
                    if (src, unit) not in fits or (tgt, unit) not in samples:
                        continue
                    pred_fit = fits[(src, unit)].scaled(bw_ratio=bw_tgt / BW[src], compute_ratio=1.0)
                    for s in samples[(tgt, unit)]:
                        p = predict(pred_fit, s, facts)
                        recs.append({"stage": stage, "source": src, "target": tgt, "target_bw_gbs": bw_tgt,
                                     "bw_label": bw_tgt_name, "unit": unit, "model": s.model, "kind": s.kind,
                                     "n": s.n, "depth": s.depth, "published_s": s.seconds, "predicted_s": p,
                                     "published_tps": (s.n if s.kind == "pp" else 1) / s.seconds,
                                     "predicted_tps": (s.n if s.kind == "pp" else 1) / p, "ape": ape(p, s.seconds)})
    return recs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    facts = load_facts()
    pub = published_samples()
    dev = device_samples()
    fits, lomo, params = {}, [], []
    for stage, table in (("published", pub), ("device", dev)):
        for key, ss in sorted(table.items()):
            if len({s.model for s in ss}) < 2:
                continue
            f = fit(ss, facts)
            fits[key] = f
            params.append({"stage": stage, "device": key[0], "unit": key[1], **{k: f"{v:.4e}" for k, v in f.as_dict().items()},
                           "eff_weight_bw_gbs": round(f.effective_bandwidth_gbs(), 1),
                           "bw_utilisation_pct": round(100 * f.effective_bandwidth_gbs() / BW[key[0]], 1) if key[0] in BW else "",
                           "in_sample_median_ape_pct": round(100 * float(np.median([ape(predict(f, s, facts), s.seconds) for s in ss])), 1)})
            for r in leave_one_model_out(ss, facts):
                lomo.append({"stage": stage, "device": key[0], "unit": key[1], **r})
    write_csv(OUT / "fits.csv", params)
    write_csv(OUT / "lomo.csv", lomo)
    write_csv(OUT / "lomo_summary.csv", summarize(lomo, ("stage", "device", "unit", "kind")))
    tr = transfer(fits, pub, SOURCES, TARGETS, "published->PC")
    dev_sources = tuple(DEVICE_LABELS[k] for k in DEVICE_LABELS if (DEVICE_LABELS[k], "cpu") in fits)
    tr += transfer(fits, pub, dev_sources, TARGETS, "device->PC")
    tr += transfer(fits, pub, dev_sources, ("Samsung Galaxy S26",), "device->S26 cross-check") if dev_sources else []
    write_csv(OUT / "transfer.csv", tr)
    write_csv(OUT / "transfer_summary.csv", summarize(tr, ("stage", "source", "target", "bw_label", "unit", "kind")))
    print(f"fits {len(params)}, lomo cells {len(lomo)}, transfer cells {len(tr)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
