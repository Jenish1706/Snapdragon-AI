#!/usr/bin/env python3
# ruff: noqa: ISC004
"""Packer Pareto curves on the HotpotQA evaluation subset (100 questions, seed 0), no LLM needed.

Supporting-sentence recall against tokens kept and against predicted TTFT, for top-k, MMR and the
QUBO packer over a budget sweep, and the cost-weighted QUBO over its weight at a 512-token budget.
Predicted TTFT is SnapTwin's prefill time for Qwen3-1.7B Q4_0 from the Galaxy S26 NPU published fit
(the Pad 4's SoC family): a prediction, not a measurement. Tokens come from the served model's
tokenizer (llama-server). QUBO and MMR carry a paired bootstrap 95% CI of their recall minus top-k's at
the same budget (10,000 resamples of the questions), and per-question better/worse counts with an
exact two-sided sign test. Outputs results/mac-m2/packer/pareto.csv and docs/figures/packer_pareto.svg.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from packer_eval import load_subset, post, prompt_for, units_of

from isingedge.packer.core import PrefillCost, mmr, pack, tfidf_scores, top_k
from isingedge.twin.model_facts import load_facts

BUDGETS = (64, 96, 128, 160, 192, 256, 320, 384, 512)
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
OUT_CSV = ROOT / "results/mac-m2/packer/pareto.csv"
OUT_SVG = ROOT / "docs/figures/packer_pareto.svg"


def s26_npu_prefill() -> tuple[float, float, float]:
    m = load_facts()["Qwen3-1.7B-Q4_0"]
    r = next(r for r in csv.DictReader((ROOT / "results/twin/fits.csv").open())
             if r["stage"] == "published" and r["device"] == "Samsung Galaxy S26" and r["unit"] == "npu")
    return float(r["b_f"]) * m.flops_per_token, float(r["b_attn"]) * m.attn_flops_per_pair, float(r["b0"])


def compute(server: str) -> list[dict]:
    alpha, beta, b0 = s26_npu_prefill()
    data = []
    for q in load_subset(100, seed=0):
        units = units_of(q)
        lens = np.array([len(post(server + "/tokenize", {"content": " " + u["text"]})["tokens"]) for u in units])
        n0 = len(post(server + "/tokenize", {"content": prompt_for(q["question"], [], np.zeros(0))})["tokens"])
        rel, sim = tfidf_scores(q["question"], [f"{u['title']} {u['text']}" for u in units])
        data.append((rel, sim, lens, np.array([u["support"] for u in units]), n0))

    def ttft_ms(n: float) -> float:
        return 1e3 * (alpha * n + beta * n * (n + 1) / 2 + b0)

    def point(method: str, setting: str, value: float, pick) -> dict:
        rec, tok, pred = [], [], []
        for rel, sim, lens, sup, n0 in data:
            x = pick(rel, sim, lens, n0)
            rec.append((x.astype(bool) & sup).sum() / max(1, sup.sum()))
            tok.append(float(lens @ x))
            pred.append(ttft_ms(n0 + float(lens @ x)))
        return {"method": method, "setting": setting, "value": value, "sf_recall": round(float(np.mean(rec)), 4),
                "tokens_kept": round(float(np.mean(tok)), 1), "pred_ttft_ms": round(float(np.mean(pred)), 1), "questions": len(data),
                "_rec": np.array(rec)}

    rows = [point("full context", "all", 0, lambda r, s, l, n0: np.ones(len(l), dtype=np.int8))]
    for b in BUDGETS:
        rows.append(point("top-k", "budget", b, lambda r, s, l, n0, b=b: top_k(r, l, b)))
        rows.append(point("MMR", "budget", b, lambda r, s, l, n0, b=b: mmr(r, s, l, b)))
        rows.append(point("QUBO", "budget", b, lambda r, s, l, n0, b=b: pack(r, s, l, b)[0]))
    for lam in LAMS:
        rows.append(point("QUBO + TTFT cost", "lambda", lam,
                          lambda r, s, l, n0, lam=lam: pack(r, s, l, 512, lam=lam, cost=PrefillCost(alpha, beta, n0))[0]))
    boot = np.random.default_rng(0).integers(0, len(data), size=(10000, len(data)))
    ref = {r["value"]: r["_rec"] for r in rows if r["method"] == "top-k"}
    for r in rows:
        r["d_recall_vs_topk"] = r["ci95_lo"] = r["ci95_hi"] = r["n_better"] = r["n_worse"] = r["sign_p"] = ""
        if r["method"] in ("QUBO", "MMR"):
            d = r["_rec"] - ref[r["value"]]
            means = d[boot].mean(axis=1)
            r["d_recall_vs_topk"] = round(float(d.mean()), 4)
            r["ci95_lo"], r["ci95_hi"] = (round(float(v), 4) for v in np.percentile(means, [2.5, 97.5]))
            r["n_better"], r["n_worse"] = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            r["sign_p"] = round(sign_test(r["n_better"], r["n_worse"]), 4)
        del r["_rec"]
    return rows


def sign_test(n_better: int, n_worse: int) -> float:
    """Two-sided exact sign test p-value over the questions where the two methods differ."""
    from math import comb

    n, k = n_better + n_worse, min(n_better, n_worse)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pareto(points: list[tuple[float, float]]) -> list[bool]:
    """True where no other point has <= x and >= recall with one strict."""
    out = []
    for i, (x, y) in enumerate(points):
        dom = any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for j, (x2, y2) in enumerate(points) if j != i)
        out.append(not dom)
    return out


def render(rows: list[dict]) -> str:
    """Three small multiples: recall vs tokens, recall vs predicted TTFT, recall minus top-k (paired CI)."""
    W, H, top, bottom, left, gap = 1000, 430, 92, 60, 60, 44
    pw = (W - left - 64 - 2 * gap) / 3
    ph = H - top - bottom
    series = [("top-k", 1, ""), ("MMR", 2, ""), ("QUBO", 3, ""), ("QUBO + TTFT cost", 3, "6 4")]
    css = ("<style>"
           ".bg{fill:#fcfcfb}.grid{stroke:#e1e0d9;stroke-width:1}.axis{stroke:#c3c2b7;stroke-width:1}"
           ".zero{stroke:#898781;stroke-width:1}.t1{fill:#0b0b0b}.t2{fill:#52514e}.tm{fill:#898781}"
           ".s1{stroke:#2a78d6}.s2{stroke:#eb6834}.s3{stroke:#1baf7a}.sm{stroke:#898781}"
           ".f1{fill:#2a78d6}.f2{fill:#eb6834}.f3{fill:#1baf7a}.fm{fill:#898781}.ring{stroke:#fcfcfb;stroke-width:2}"
           "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:12px}.h{font-size:13px;font-weight:600}"
           ".big{font-size:15px;font-weight:600}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.grid{stroke:#2c2c2a}.axis{stroke:#383835}"
           ".t1{fill:#ffffff}.t2{fill:#c3c2b7}.s1{stroke:#3987e5}.s2{stroke:#d95926}.s3{stroke:#199e70}"
           ".f1{fill:#3987e5}.f2{fill:#d95926}.f3{fill:#199e70}.ring{stroke:#1a1a19}}"
           "</style>")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           'aria-labelledby="t d">',
           '<title id="t">Packer Pareto curves on HotpotQA (100 questions)</title>',
           '<desc id="d">Supporting-sentence recall against tokens kept and against predicted TTFT for top-k, MMR and the '
           'QUBO packer, and QUBO and MMR recall minus top-k with paired bootstrap 95% intervals. Top-k and QUBO '
           'overlap; every QUBO interval touches or contains zero. Values are in results/mac-m2/packer/pareto.csv.</desc>',
           css, f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>',
           f'<text class="t1 big" x="{left}" y="24">Top-k and the QUBO packer tie on HotpotQA supporting-sentence recall</text>']
    # legend: one row under the title
    lx = left
    for name, c, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="46" y2="46" class="s{c}" stroke-width="2"{dash_attr}/>')
        out.append(f'<text class="t1" x="{lx + 28}" y="50">{name}</text>')
        lx += 28 + 8 * len(name) + 24
    out.append(f'<circle cx="{lx + 6}" cy="46" r="4" class="fm ring"/><text class="t1" x="{lx + 16}" y="50">full context</text>')

    def log_axis(ox, lo, hi):
        a, b = np.log10(lo), np.log10(hi)
        return lambda v: ox + (np.log10(v) - a) / (b - a) * pw

    full = next(r for r in rows if r["method"] == "full context")
    panels = [("tokens_kept", "Tokens kept (mean, log scale)", 40, 1600, (50, 100, 200, 500, 1000)),
              ("pred_ttft_ms", "Predicted TTFT, ms (Galaxy S26 NPU fit, log)", 50, 800, (50, 100, 200, 400, 800))]
    for k, (key, label, lo, hi, ticks) in enumerate(panels):
        ox = left + k * (pw + gap)
        sx = log_axis(ox, lo, hi)

        def sy(v):
            return top + (1 - v) * ph

        out.append(f'<text class="t1 h" x="{ox}" y="{top - 14}">Recall vs {"tokens kept" if k == 0 else "predicted TTFT"}</text>')
        for yv in (0.25, 0.5, 0.75, 1.0):
            out.append(f'<line class="grid" x1="{ox}" x2="{ox + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}"/>')
            out.append(f'<text class="tm" x="{ox - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:.2f}</text>')
        out.append(f'<line class="axis" x1="{ox}" x2="{ox + pw}" y1="{sy(0.25):.1f}" y2="{sy(0.25):.1f}"/>')
        for xv in ticks:
            out.append(f'<text class="tm" x="{sx(xv):.1f}" y="{sy(0.25) + 16:.1f}" text-anchor="middle">{xv:,}</text>')
        out.append(f'<text class="t2" x="{ox + pw / 2:.1f}" y="{sy(0.25) + 36:.1f}" text-anchor="middle">{label}</text>')
        if k == 0:
            out.append(f'<text class="t2" transform="translate({ox - 44},{top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
                       "Supporting-sentence recall</text>")
        for name, c, dash in series:
            pts = sorted(((r[key], r["sf_recall"], r) for r in rows if r["method"] == name), key=lambda t: t[0])
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            path = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}' for i, (x, y, _) in enumerate(pts))
            out.append(f'<path d="{path}" fill="none" class="s{c}" stroke-width="2" stroke-linejoin="round"{dash_attr}/>')
            if not dash:
                for x, y, r in pts:
                    tip = (f'{name}, budget {r["value"]:g}: recall {y:.3f}, {r["tokens_kept"]:.0f} tokens, '
                           f'{r["pred_ttft_ms"]:.0f} ms predicted')
                    out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" class="f{c} ring"><title>{tip}</title></circle>')
        fx, fy = sx(full[key]), sy(full["sf_recall"])
        out.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4" class="fm ring"><title>full context: recall {full["sf_recall"]:.3f}, '
                   f'{full["tokens_kept"]:.0f} tokens, {full["pred_ttft_ms"]:.0f} ms predicted</title></circle>')
        out.append(f'<text class="t2" x="{fx - 8:.1f}" y="{fy + 4:.1f}" text-anchor="end">full context</text>')
        anchor = next(r for r in rows if r["method"] == "QUBO" and r["value"] == 192)
        out.append(f'<text class="t2" x="{sx(anchor[key]):.1f}" y="{sy(0.45):.1f}">top-k lies under QUBO</text>')
    # third panel: paired difference to top-k at each budget
    ox = left + 2 * (pw + gap)
    sx = log_axis(ox, 56, 590)
    ylo, yhi = -0.08, 0.04

    def sy(v):
        return top + (yhi - v) / (yhi - ylo) * ph

    out.append(f'<text class="t1 h" x="{ox}" y="{top - 14}">Recall minus top-k, same budget</text>')
    for yv in (-0.08, -0.06, -0.04, -0.02, 0.02, 0.04):
        out.append(f'<line class="grid" x1="{ox}" x2="{ox + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}"/>')
    for yv in (-0.08, -0.04, 0.0, 0.04):
        out.append(f'<text class="tm" x="{ox - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:+.2f}</text>')
    out.append(f'<line class="zero" x1="{ox}" x2="{ox + pw}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    out.append(f'<line class="axis" x1="{ox}" x2="{ox + pw}" y1="{sy(ylo):.1f}" y2="{sy(ylo):.1f}"/>')
    for xv in (64, 128, 256, 512):
        out.append(f'<text class="tm" x="{sx(xv):.1f}" y="{sy(ylo) + 16:.1f}" text-anchor="middle">{xv}</text>')
    out.append(f'<text class="t2" x="{ox + pw / 2:.1f}" y="{sy(ylo) + 36:.1f}" text-anchor="middle">'
               "Token budget (log); bars are paired 95% CIs</text>")
    for name, c, nudge, dy in (("MMR", 2, 0.985, 14), ("QUBO", 3, 1.015, -8)):
        pts = [r for r in rows if r["method"] == name]
        for r in pts:
            x = sx(r["value"] * nudge)
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(r["ci95_lo"]):.1f}" y2="{sy(r["ci95_hi"]):.1f}" class="s{c}" '
                       'stroke-width="2" stroke-linecap="round"/>')
            tip = f'{name} minus top-k at budget {r["value"]:g}: {r["d_recall_vs_topk"]:+.3f} (95% CI {r["ci95_lo"]:+.3f} to {r["ci95_hi"]:+.3f})'
            out.append(f'<circle cx="{x:.1f}" cy="{sy(r["d_recall_vs_topk"]):.1f}" r="4" class="f{c} ring"><title>{tip}</title></circle>')
        last = max(pts, key=lambda r: r["value"])
        out.append(f'<text class="t2" x="{sx(last["value"] * nudge) + 8:.1f}" y="{sy(last["d_recall_vs_topk"]) + dy:.1f}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def load_rows() -> list[dict]:
    rows = []
    for r in csv.DictReader(OUT_CSV.open()):
        for k, v in r.items():
            if k not in ("method", "setting") and v not in ("", "True", "False"):
                r[k] = float(v)
        rows.append(r)
    return rows


def main(server: str = "http://127.0.0.1:8091") -> None:
    if "--render-only" in sys.argv:
        OUT_SVG.write_text(render(load_rows()))
        return
    rows = compute(server)
    flags = {}
    for key in ("tokens_kept", "pred_ttft_ms"):
        pts = [(r[key], r["sf_recall"]) for r in rows]
        for r, f in zip(rows, pareto(pts)):
            flags.setdefault(id(r), {})[key] = f
    for r in rows:
        r["pareto_tokens"] = flags[id(r)]["tokens_kept"]
        r["pareto_ttft"] = flags[id(r)]["pred_ttft_ms"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    OUT_SVG.write_text(render(rows))
    for r in rows:
        ci = f" d vs top-k {r['d_recall_vs_topk']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]" if r["ci95_lo"] != "" else ""
        print(f"{r['method']:18s} {r['setting']:6s} {r['value']:>6g}: recall {r['sf_recall']:.3f} tokens {r['tokens_kept']:7.1f} "
              f"pred TTFT {r['pred_ttft_ms']:7.1f} ms pareto {r['pareto_tokens']}{ci}")


if __name__ == "__main__":
    main()
