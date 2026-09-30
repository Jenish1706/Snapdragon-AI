#!/usr/bin/env python3
"""Build pitch_deck.pptx with the same slide order, titles, numbers and charts as pitch_deck.pdf.

Values come from generated/values.json and chart series from figures/data/*.csv (both written by
build_data.py from ../results), so the PPTX and the PDF cannot drift apart. Charts are native,
editable PowerPoint charts. Usage: python3 generate_pptx.py   (needs python-pptx)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "generated/values.json").read_text(encoding="utf-8"))
C = {k: RGBColor.from_string(v) for k, v in dict(ink="0B1F33", ink2="3E4C5E", muted="6B7686", rule="D8DEE6", panel="F3F6F9",
     blue="2A78D6", bluedk="1C5CAB", orange="EB6834", aqua="1BAF7A", gray="A8A7A0", good="006300", bad="D03B3B",
     white="FFFFFF", light="9FC3F0").items()}
FONT = "Calibri"
N = 11


def data(name):
    with (HERE / "figures/data" / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text(slide, x, y, w, h, runs, size=14, color="ink", bold=False, align=None):
    """runs: str or list of paragraphs; a paragraph is a str or a list of (text, dict) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for t, st in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run()
            r.text = t
            f = r.font
            f.name, f.size, f.bold = FONT, Pt(st.get("size", size)), st.get("bold", bold)
            f.color.rgb = C[st.get("color", color)]
            if st.get("sup"):
                r._r.get_or_add_rPr().set("baseline", "30000")
        para.space_after = Pt(4)
    return tb


def card(slide, x, y, w, h, big, small, fill="panel"):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = C[fill]
    s.line.fill.background()
    s.adjustments[0] = 0.05
    text(slide, x + 0.12, y + 0.06, w - 0.24, h - 0.1, [[(big, {"size": 24, "bold": True, "color": "bluedk"})],
                                                      [(small, {"size": 12, "color": "ink2"})]])


def frame(prs, kicker, title, n, source=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.55, 0.25, 12.2, 0.35, kicker.upper(), size=12, color="bluedk", bold=True)
    text(s, 0.55, 0.52, 12.2, 0.6, title, size=28, bold=True)
    ln = s.shapes.add_connector(1, Inches(0.6), Inches(1.18), Inches(12.75), Inches(1.18))
    ln.line.color.rgb = C["rule"]
    text(s, 0.55, 7.08, 8, 0.3, "Ising-Edge · Snapdragon AI Challenge 2026", size=9, color="muted")
    text(s, 11.9, 7.08, 1, 0.3, f"{n} / {N}", size=9, color="muted")
    if source:
        text(s, 0.55, 6.72, 12.2, 0.35, source, size=9, color="muted")
    return s


def bullets(slide, x, y, w, h, items, size=15):
    return text(slide, x, y, w, h, [[("■  ", {"color": "blue", "size": size - 4}), (t, {})] if isinstance(t, str) else t
                                    for t in items], size=size)


def chart(slide, x, y, w, h, kind, cats, series, colors, ytitle, xtitle, ymax=None, labels=None, legend=True):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.name, ch.font.size = FONT, Pt(11)
    ch.has_legend = legend
    if legend:
        ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = C["rule"]
    va.minimum_scale = 0
    if ymax:
        va.maximum_scale = ymax
    for ax, t in ((va, ytitle), (ch.category_axis, xtitle)):
        ax.has_title = True
        ax.axis_title.text_frame.text = t
        ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        ax.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    for s, col in zip(ch.series, colors):
        if kind == XL_CHART_TYPE.LINE:
            s.format.line.color.rgb, s.format.line.width, s.smooth = C[col], Pt(2.25), False
        else:
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = C[col]
    if kind != XL_CHART_TYPE.LINE:
        ch.plots[0].gap_width, ch.plots[0].overlap = 60, 0
    if labels:
        for si, pi, t, col in labels:
            dl = ch.series[si].points[pi].data_label
            dl.has_text_frame = True
            dl.text_frame.text = t
            dl.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
            dl.text_frame.paragraphs[0].runs[0].font.color.rgb = C[col]
    return ch


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = C["ink"]
    text(s, 0.6, 0.45, 12, 0.4, "SNAPDRAGON AI CHALLENGE 2026 · ON-DEVICE GENERATIVE AI ACCELERATION & OPTIMIZATION", 12, "light", True)
    text(s, 0.6, 1.25, 12, 1.2, "Ising-Edge", 66, "white", True)
    text(s, 0.6, 2.5, 12, 0.5, "Budget-exact compression planning for on-device LLMs", 26, "white")
    text(s, 0.6, 3.05, 12, 0.5, "Measure → fit → solve: measured perplexity, a quadratic surrogate, an exact solver", 18, "light")
    col = {"keep08": C["blue"], "dense": RGBColor(0x5B, 0x68, 0x78), "fixed": RGBColor(0x26, 0x37, 0x4B)}
    for r in data("plan_K18.csv"):
        L = int(r["layer"])
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6 + L * 0.42), Inches(4.0), Inches(0.34), Inches(0.34))
        b.fill.solid()
        b.fill.fore_color.rgb = col[r["state"]]
        b.line.fill.background()
    text(s, 0.6, 4.4, 12.2, 0.35, f"Example output: {V['motif.n']}-MLP byte-budget plan, layers 0–27 (blue = keep 0.8, grey = dense, "
         f"dark = not searched); full-split ΔPPL +{V['loop.K18.s2.full']}%", 11, "light")
    text(s, 0.6, 6.2, 8, 0.4, "Jenish Patel", 20, "white")
    text(s, 0.6, 6.65, 8, 0.3, "30 September 2026", 12, "light")

    # 2 problem
    s = frame(prs, "The problem", "Every layer needs a compression decision", 2,
              "Plan count from isingedge/loop/space.py (Space.count); llama.cpp placement per docs/research/hetero-placement-2026.md.")
    bullets(s, 0.6, 1.5, 7.2, 5, [
        "Edge devices impose tight memory and bandwidth limits on small language models.",
        "Low-rank compression of MLP weights saves bytes, but its cost depends on which layers are compressed and how strongly.",
        "On Snapdragon, each weight tensor can also be placed on CPU, Adreno GPU or Hexagon NPU (llama.cpp tensor overrides).",
        "The decision is a plan over all layers under a hard byte budget: a combinatorial problem."], 17)
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.6), Inches(4.4), Inches(2.1))
    b.fill.solid(); b.fill.fore_color.rgb = C["panel"]; b.line.fill.background(); b.adjustments[0] = 0.05
    text(s, 8.45, 1.68, 4.1, 2.0, [[(f"{V['loop.K12.count.mant']} × 10", {"size": 30, "bold": True, "color": "bluedk"}),
                                    (V["loop.K12.count.exp"], {"size": 30, "bold": True, "color": "bluedk", "sup": True})],
                                   [("feasible plans for one budget: 24 MLP blocks (layers 2–25) × {dense, keep 0.9, keep 0.8}, "
                                     "bytes of 12 MLPs at keep 0.8", {"size": 13, "color": "ink2"})]])

    # 3 sensitivity
    s = frame(prs, "Evidence 1 · single-layer sensitivity", "Where you compress matters", 3,
              f"Activation-aware SVD of one MLP (rank {V['rank.k08']}); WikiText-2, 40 chunks × 512 tokens. "
              "results/sensitivity/qwen3-0.6b_layer_sensitivity.csv")
    rows = data("sensitivity.csv")
    chart(s, 0.5, 1.35, 8.1, 5.3, XL_CHART_TYPE.COLUMN_CLUSTERED, [r["layer"] for r in rows],
          [("ΔPPL, one layer's MLP at keep 0.8", [float(r["keep08"]) for r in rows])], ["blue"], "ΔPPL vs dense (%)",
          "layer whose MLP is compressed alone (keep 0.8)", labels=[(0, 8, f"+{V['sens.k08.min']}%", "ink"), (0, 27, f"+{V['sens.k08.max']}%", "ink")],
          legend=False)
    card(s, 8.9, 1.6, 3.9, 1.6, f"+{V['sens.k08.min']}% to +{V['sens.k08.max']}%",
         f"damage range across the 28 layers (layer {V['sens.k08.min.layer']} vs layer {V['sens.k08.max.layer']})")
    card(s, 8.9, 3.4, 3.9, 1.6, f"+{V['sens.k08.median']}%", f"median layer at keep 0.8 (+{V['sens.k09.median']}% at keep 0.9)")

    # 4 compounding
    s = frame(prs, "Evidence 2 · interactions", "Damage compounds: per-layer costs do not add up", 4,
              "40 chunks. results/matched_bytes/summary.csv, results/sensitivity/qwen3-0.6b_pair_sensitivity.csv")
    rows = data("superadditivity.csv")
    chart(s, 0.5, 1.35, 7.4, 5.3, XL_CHART_TYPE.COLUMN_CLUSTERED, [r["K"] for r in rows],
          [("predicted (singles + pair model)", [round(float(r["pred_pairwise"]), 1) for r in rows]),
           ("measured", [round(float(r["measured_fast"]), 1) for r in rows])], ["gray", "blue"], "ΔPPL vs dense (%)",
          "MLPs compressed at keep 0.8 (model-chosen plan)", ymax=310,
          labels=[(si, pi, f"{float(rows[pi][k]):.1f}", "ink") for si, k in ((0, "pred_pairwise"), (1, "measured_fast")) for pi in range(4)])
    card(s, 8.2, 1.5, 4.6, 2.3, f"+{V['mb.K24.plan.fast']}% vs +{V['mb.K24.pred.lin']}%",
         f"24 MLPs: measured vs summed single-layer costs ({V['mb.K24.ratio']}×); log-additive +{V['mb.K24.pred.log']}%, "
         f"pair model +{V['mb.K24.pred.pw']}%")
    text(s, 8.2, 4.0, 4.6, 2.5, [f"{V['pairs.super']} of {V['pairs.n']} random layer pairs are super-additive (median "
                                 f"{V['pairs.inter.median']} pp; largest {V['pairs.inter.max']} pp, layers {V['pairs.inter.max.pair']}).",
                                 [("Plans must be measured, not only predicted.", {"bold": True})]], 14)

    # 5 approach
    s = frame(prs, "Our approach", "Ising-Edge: measure → fit → solve", 5)
    steps = [("1  Measure", "Compress the plan's MLPs (activation-aware SVD) and score WikiText-2 perplexity on 40 fixed chunks; keep per-chunk NLL."),
             ("2  Fit", f"Quadratic surrogate of log(PPL/PPL_dense) over one-hot plan bits: pair model (one-shot) or rank-{V['loop.fm.rank']} factorization machine (loop)."),
             ("3  Solve", "Minimise it exactly with CP-SAT: at most one option per layer, exact byte budget, no-good cuts on measured plans.")]
    for i, (h, t) in enumerate(steps):
        x = 0.6 + i * 4.2
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.5), Inches(3.8), Inches(2.4))
        b.fill.solid(); b.fill.fore_color.rgb = C["panel"]; b.line.fill.background(); b.adjustments[0] = 0.05
        text(s, x + 0.12, 1.58, 3.56, 2.3, [[(h, {"size": 18, "bold": True, "color": "bluedk"})], [(t, {"size": 14})]])
        if i < 2:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.83), Inches(2.55), Inches(0.33), Inches(0.3))
            a.fill.solid(); a.fill.fore_color.rgb = C["blue"]; a.line.fill.background()
    text(s, 0.6, 4.2, 12.2, 0.6, [[("min over x ∈ {0,1}", {"size": 18}), (V["loop.nbin"], {"size": 18, "sup": True}),
                                   ("   w0 + Σ wi·xi + Σ(i<j) ⟨vi, vj⟩·xi·xj    s.t.  Σo x(l,o) ≤ 1 for every layer l,  Σ ui·xi = B", {"size": 18})]])
    text(s, 0.6, 5.2, 12.2, 1.2, [[(f"Output: per-layer keep fractions, re-measured on the full {V['dense.full.chunks']}-chunk split with "
                                    "paired-bootstrap CIs. The same QUBO form admits latency (SnapTwin) and CPU/GPU/NPU placement terms: ", {}),
                                   ("implemented, not yet validated on a device.", {"color": "bad"})]], 14)

    # 6 implementation
    s = frame(prs, "Implementation", "Implementation: what is built and tested", 6,
              "Paths relative to isingedge/. Test run: python -m pytest tests/ (this snapshot).")
    rows6 = [("Component", "Module", "Status"), ("Plan space, exact byte budget", "loop/space.py", "evaluated"),
             ("Factorization machine (L-BFGS-B)", "loop/fm.py", "evaluated"), ("CP-SAT / exact / greedy / SA / SB", "loop/acquire.py", "evaluated"),
             ("Perplexity oracle, SVD + MPO", "loop/oracle.py", "evaluated"), ("Paired bootstrap, Spearman", "loop/stats.py", "evaluated"),
             ("SnapTwin prefill/decode model", "twin/snaptwin.py", "unit-tested"), ("Placement QUBO (CPU/GPU/NPU)", "qubo/planner.py", "unit-tested"),
             ("adb + llama-bench runners", "bench/, scripts/", "not yet run")]
    t = s.shapes.add_table(len(rows6), 3, Inches(0.6), Inches(1.45), Inches(7.2), Inches(4.6)).table
    for i, r in enumerate(rows6):
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = v
            c.fill.solid(); c.fill.fore_color.rgb = C["panel"] if i == 0 else C["white"]
            f = c.text_frame.paragraphs[0].runs[0].font
            f.name, f.size, f.bold = FONT, Pt(13), i == 0
            f.color.rgb = C["good"] if v == "evaluated" else C["bad"] if v == "not yet run" else C["ink"]
    bullets(s, 8.1, 1.45, 4.8, 5.2, [
        "Qwen3-0.6B, 28 layers; MLP 1024×3072 (gate, up, down)",
        f"keep 0.8 → rank {V['rank.k08']} ({V['frac.k08']}% of MLP params); keep 0.9 → rank {V['rank.k09']} ({V['frac.k09']}%)",
        f"WikiText-2 test, context {V['ctx']}; second half of each chunk scored",
        f"All results: {V['env.build']}",
        "Tests: 56 collected; 52 pass, 3 skip (optional torch/onnxruntime), 1 fails (smoke import of two subpackages missing from this snapshot)"], 13)

    # 7 matched bytes
    s = frame(prs, "Result 1 · matched bytes, full 584-chunk split", "At equal bytes, planned layers beat random ones (K ≤ 18)", 7,
              f"Paired bootstrap ({V['mb.boot.reps']} resamples), recomputed from results/sensitivity/qwen3-0.6b_plan_chunk_nll.csv; "
              "results/matched_bytes/summary.csv")
    rows = data("matched_bytes.csv")
    chart(s, 0.5, 1.35, 7.4, 5.3, XL_CHART_TYPE.COLUMN_CLUSTERED, [r["K"] for r in rows],
          [("model-chosen plan", [round(float(r["plan_full"]), 2) for r in rows]),
           ("best of 3 random plans", [round(float(r["random_full"]), 2) for r in rows])], ["blue", "gray"],
          "ΔPPL vs dense (%)", "K MLPs compressed at keep 0.8 (identical bytes)", ymax=330,
          labels=[(1, 0, f"{V['mb.K6.diff.short']} pp", "good"), (1, 1, f"{V['mb.K12.diff.short']} pp", "good"),
                  (1, 2, f"{V['mb.K18.diff.short']} pp", "good"), (0, 3, f"{V['mb.K24.diff.short']} pp (worse)", "bad")])
    card(s, 8.2, 1.5, 4.6, 1.7, f"{V['mb.K18.diff.short']} pp",
         f"K = 18: +{V['mb.K18.plan.full']}% vs +{V['mb.K18.rand.full']}%; 95% CI [{V['mb.K18.lo']}, {V['mb.K18.hi']}] pp")
    bullets(s, 8.2, 3.4, 4.6, 3.2, [
        f"K = 6 / 12: {V['mb.K6.diff']} pp [{V['mb.K6.lo']}, {V['mb.K6.hi']}] / {V['mb.K12.diff']} pp [{V['mb.K12.lo']}, {V['mb.K12.hi']}]",
        [("■  ", {"color": "blue", "size": 10}), ("K = 24: ", {}), ("model plan loses", {"color": "bad"}),
         (f" by {V['mb.K24.diff']} pp [{V['mb.K24.lo']}, {V['mb.K24.hi']}]: pair terms miss large-plan compounding", {})],
        "Pair, additive and log-additive models pick the same layers here"], 13)

    # 8 closed loop
    s = frame(prs, "Result 2 · closed loop, 24 measured plans per run", "The loop recovers the one-shot optimum; random trails by 20–85 pp", 8,
              "results/loop/evals.csv, results/loop/summary.csv. Plan budget in bytes of K MLPs at keep 0.8; options dense / keep 0.9 / keep 0.8 on layers 2–25.")
    rows = data("regret_K18.csv")
    ch = chart(s, 0.5, 1.35, 7.0, 5.3, XL_CHART_TYPE.LINE, [r["n"] for r in rows],
               [("one-shot (pair model)", [float(r["oneshot"]) for r in rows]), ("loop (median of 3)", [float(r["loop_med"]) for r in rows]),
                ("local search", [float(r["local_med"]) for r in rows]), ("random (median of 3)", [float(r["random_med"]) for r in rows])],
               ["blue", "orange", "aqua", "gray"], "best ΔPPL so far (%, 40 chunks)", "measured plans (K = 18 budget)", ymax=190)
    ch.value_axis.minimum_scale = 50
    t = s.shapes.add_table(3, 3, Inches(7.8), Inches(1.45), Inches(5.0), Inches(1.1)).table
    for i, r in enumerate([("K", "one-shot", "random (3 seeds)"),
                           ("12", f"+{V['loop.K12.one.full']}%", f"+{V['loop.K12.rand.full.min']}–{V['loop.K12.rand.full.max']}%"),
                           ("18", f"+{V['loop.K18.one.full']}%", f"+{V['loop.K18.rand.full.min']}–{V['loop.K18.rand.full.max']}%")]):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = v; c.fill.solid(); c.fill.fore_color.rgb = C["panel"] if i == 0 else C["white"]
            f = c.text_frame.paragraphs[0].runs[0].font; f.name, f.size, f.bold, f.color.rgb = FONT, Pt(13), i == 0, C["ink"]
    bullets(s, 7.8, 2.8, 5.0, 3.8, [
        f"Loop = one-shot plan in 3/3 seeds (K = 12) and {V['loop.K18.same']}/3 (K = 18); full-split ΔPPL above",
        f"K = 18 seed 2: one swapped layer, {V['loop.K18.s2.diff']} pp on the full split [{V['loop.K18.s2.lo']}, {V['loop.K18.s2.hi']}] "
        f"but {V['loop.K18.s2.fastgap']} pp on the 40-chunk objective",
        f"Every acquisition solved to optimality by CP-SAT ({V['loop.nbin']} binaries)"], 13)

    # 9 decisions
    s = frame(prs, "Evidence-based design choices", "What the measurements decided", 9,
              "results/loop/solver_gaps.csv, results/loop/calibration.csv, results/sensitivity/qwen3-0.6b_kl_proxy_summary.csv, results/compression/ppl.csv")
    card(s, 0.6, 1.45, 6.0, 2.45, f"{V['solve.greedy.opt']} / {V['solve.rounds']}",
         f"rounds where greedy + swap descent matched the exact surrogate optimum (median gap {V['solve.greedy.K12.median']}% at K = 12, "
         f"{V['solve.greedy.K18.median']}% at K = 18); ballistic-SB raw samples broke the constraints in {V['solve.sbb.infeasible']}/{V['solve.rounds']} ⇒ exact CP-SAT")
    card(s, 0.6, 4.1, 6.0, 2.45, f"ρ = {V['cal.K18.rho.min']}–{V['cal.K12.rho.max']}",
         f"surrogate rank correlation on 72 held-out random plans; it under-predicts damage (log-ratio bias {V['cal.bias.lo']} to "
         f"{V['cal.bias.hi']}) ⇒ re-measure every chosen plan")
    card(s, 6.8, 1.45, 6.0, 2.45, f"ρ = {V['kl.rho']}",
         f"Spearman of KL(dense‖compressed) vs single-layer ΔPPL (95% CI {V['kl.lo']}–{V['kl.hi']}), below the {V['kl.bar']} bar ⇒ perplexity stays the oracle")
    card(s, 6.8, 4.1, 6.0, 2.45, f"{V['mpo.ppl']} vs {V['svd.ppl']} PPL",
         f"3-core MPO vs SVD, layers {V['mpo.layers']} at equal params ({V['mpo.params.frac']}%), dense {V['dense.full.ppl']}; accuracy only, no speed claim")

    # 10 limitations
    s = frame(prs, "Honest status", "Limitations and next steps", 10)
    text(s, 0.6, 1.4, 6.5, 0.4, "Limitations", 18, bold=True)
    bullets(s, 0.6, 1.85, 6.6, 4.8, [
        "No Snapdragon measurements in this repository: latency, memory and energy not measured; SnapTwin and the placement QUBO are unit-tested only",
        "One model (Qwen3-0.6B), one metric (WikiText-2 perplexity), one runtime (PyTorch fp32, laptop GPU); no task accuracy",
        f"Post-training compression without healing: damage is large (+{V['mb.K18.plan.full']}% at K = 18)",
        "Pair model fails for large plans (K = 24); loop gains are small at 24 evaluations",
        "Plan-export and device scripts reference data not in this snapshot; one failing smoke test"], 14)
    text(s, 7.5, 1.4, 5.3, 0.4, "Next steps", 18, bold=True)
    text(s, 7.5, 1.85, 5.3, 4.8, [
        "1. Run the scripted llama-bench sweeps (llama.cpp b11256) on a Snapdragon device over adb: CPU, GPU, NPU",
        "2. Fit SnapTwin on device data; add latency terms to the same QUBO",
        "3. Export plans as llama.cpp tensor overrides; measure TTFT, decode rate, memory",
        "4. Add plan-size / higher-order terms; evaluate task accuracy and healing"], 14)

    # 11 takeaway
    s = frame(prs, "Takeaway", "Compression plans must be measured and solved jointly", 11)
    text(s, 0.6, 1.45, 12.2, 1.2, "Which layers to compress is a combinatorial choice with compounding costs. Measuring whole plans and solving "
         "exactly under a byte budget finds much better plans than random choice, and shows where simple models fail.", 18)
    card(s, 0.9, 3.0, 3.7, 1.7, f"{V['mb.K18.diff.short']} pp", "vs best random plan at equal bytes (K = 18)")
    card(s, 4.8, 3.0, 3.7, 1.7, f"{V['mb.K24.ratio']}×", "measured vs summed single-layer damage (24 MLPs)")
    card(s, 8.7, 3.0, 3.7, 1.7, f"{V['loop.K12.count.mant']} × 10^{V['loop.K12.count.exp']}", "plans per budget, solved to optimality by CP-SAT")
    text(s, 0.6, 5.2, 12.2, 0.5, [[("Measured: ", {"bold": True, "color": "good"}), ("accuracy-side planner     ", {}),
                                   ("Not yet validated: ", {"bold": True, "color": "bad"}), ("Snapdragon placement (SnapTwin + placement QUBO)     ", {}),
                                   ("Next: ", {"bold": True}), ("on-device sweeps", {})]], 14)
    text(s, 0.6, 5.8, 12.2, 0.4, "Jenish Patel · Ising-Edge (MIT) · Qwen3 (Apache-2.0) · llama.cpp (MIT)", 12, "muted")

    assert len(prs.slides) == N
    prs.save(HERE / "pitch_deck.pptx")
    print(f"wrote pitch_deck.pptx ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
