#!/usr/bin/env python3
"""
generate_pptx.py — Generate a PowerPoint (.pptx) version of the Ising-Edge pitch deck.

Requires: pip install python-pptx

This script creates a professional PPTX presentation matching the content
of pitch_deck.tex, for environments where LaTeX is unavailable.

Usage:
    python generate_pptx.py              # outputs pitch_deck.pptx
    python generate_pptx.py -o out.pptx  # custom output path
"""

import argparse
import sys

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    print("ERROR: python-pptx not installed. Run: pip install python-pptx")
    sys.exit(1)

# ── Colors ──────────────────────────────────────────────────────────────
TECH_BLUE   = RGBColor(0x00, 0x55, 0x87)
TEAL        = RGBColor(0x00, 0x7C, 0x92)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
BLACK       = RGBColor(0x20, 0x20, 0x20)
LIGHT_GRAY  = RGBColor(0xF5, 0xF5, 0xF5)
GOOD_GREEN  = RGBColor(0x2E, 0x7D, 0x32)
WARN_RED    = RGBColor(0xD3, 0x2F, 0x2F)
ACCENT_ORG  = RGBColor(0xF5, 0x7C, 0x00)

# ── Slide dimensions (16:9) ─────────────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_slide_bg(slide, color):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=BLACK, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri"):
    """Add a simple text box."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_frame(slide, left, top, width, height, bullets,
                     font_size=16, color=BLACK, bold_prefix=True):
    """Add a text box with bullet points."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(6)
        p.level = 0

        # Support bold prefix via "**prefix** rest" syntax
        if bold_prefix and "**" in bullet:
            parts = bullet.split("**")
            for j, part in enumerate(parts):
                if not part:
                    continue
                run = p.add_run()
                run.text = part
                run.font.size = Pt(font_size)
                run.font.color.rgb = color
                run.font.name = "Calibri"
                run.font.bold = (j % 2 == 1)  # odd indices are bold
        else:
            run = p.add_run()
            run.text = f"• {bullet}"
            run.font.size = Pt(font_size)
            run.font.color.rgb = color
            run.font.name = "Calibri"
    return txBox


def build_presentation(output_path: str):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank_layout = prs.slide_layouts[6]  # blank layout

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 1: Title
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, TECH_BLUE)

    add_text_box(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.5),
                 "Ising-Edge", font_size=48, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(3.2), Inches(11), Inches(1.2),
                 "Joint Compute Placement, Precision & Low-Rank Compression\n"
                 "via Quadratic Binary Optimisation for Snapdragon Edge AI",
                 font_size=22, color=WHITE, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(5.0), Inches(11), Inches(0.6),
                 "Jenish Patel", font_size=20, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(5.7), Inches(11), Inches(0.8),
                 "Snapdragon AI Challenge 2026\n"
                 "Track: On-Device Generative AI Acceleration & Optimization",
                 font_size=16, color=RGBColor(0xBB, 0xDE, 0xFB),
                 alignment=PP_ALIGN.CENTER)

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 2: The Problem
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "The Problem: No Single Engine Wins",
                 font_size=32, bold=True, color=TECH_BLUE)

    bullets = [
        "• Snapdragon SoCs have 3 compute engines: CPU, Adreno GPU, Hexagon NPU",
        "• Hexagon NPU → Up to 5× faster prefill (2,643 vs 353 tok/s)",
        "• CPU / GPU → Up to 2.5× faster decode (59.8 vs 26.9 tok/s)",
        "",
        "• Existing optimizers treat placement, quantization, and compression as disconnected",
        "• Compression damage is non-linear and coupled across layers",
    ]
    add_bullet_frame(slide, Inches(0.5), Inches(1.2), Inches(7), Inches(4),
                     bullets, font_size=18, bold_prefix=False)

    # Mini table
    add_text_box(slide, Inches(8), Inches(1.2), Inches(4.5), Inches(0.5),
                 "Galaxy S26 · Qwen3-1.7B Q4_0 · 512 ctx",
                 font_size=14, bold=True, color=TEAL, alignment=PP_ALIGN.CENTER)

    table_data = [
        ["Engine", "Prefill", "Decode"],
        ["NPU", "2,643", "45.9"],
        ["GPU", "665", "42.8"],
        ["CPU", "353", "47.9"],
    ]
    table_shape = slide.shapes.add_table(
        len(table_data), 3, Inches(8.2), Inches(1.8), Inches(4), Inches(2.2)
    )
    table = table_shape.table
    for r, row in enumerate(table_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(14)
                p.font.name = "Calibri"
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TECH_BLUE

    add_text_box(slide, Inches(8), Inches(4.2), Inches(4.5), Inches(0.4),
                 "Source: Qualcomm ai-hub-models/perf.yaml",
                 font_size=11, color=RGBColor(0x99, 0x99, 0x99),
                 alignment=PP_ALIGN.CENTER)

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 3: Why It Matters — Compression Trap
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "The Compression Damage Trap",
                 font_size=32, bold=True, color=TECH_BLUE)

    add_text_box(slide, Inches(0.5), Inches(1.2), Inches(8), Inches(0.6),
                 "Single-layer planning is mathematically invalid.",
                 font_size=22, bold=True, color=WARN_RED)

    bullets = [
        "• Compressing 24 MLP blocks (layers 2–25) independently:",
        "    Linear sum of single-layer damages predicts +57.8%",
        "    Log-additive model predicts +76.8%",
        "    Actual measured degradation: +275.6%",
        "",
        "• 4.8× underestimation when ignoring cross-layer interactions",
        "• Bilinear Jacobian cross-terms couple error across layers",
    ]
    add_bullet_frame(slide, Inches(0.5), Inches(2.0), Inches(7), Inches(4),
                     bullets, font_size=17, bold_prefix=False)

    # Callout box
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.5), Inches(2.0),
        Inches(4), Inches(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xFF, 0xEB, 0xEE)
    shape.line.color.rgb = WARN_RED
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4.8×"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = WARN_RED
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = "underestimation"
    p2.font.size = Pt(20)
    p2.font.color.rgb = WARN_RED
    p2.alignment = PP_ALIGN.CENTER
    p3 = tf.add_paragraph()
    p3.text = "\n+57.8% predicted\n+275.6% actual"
    p3.font.size = Pt(16)
    p3.font.color.rgb = BLACK
    p3.alignment = PP_ALIGN.CENTER

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 4: Our Solution
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Ising-Edge: Unified QUBO Optimization",
                 font_size=32, bold=True, color=TECH_BLUE)

    add_text_box(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.5),
                 "Jointly optimizes across every transformer block:",
                 font_size=20, bold=True, color=BLACK)

    bullets = [
        "1. Compute placement → CPU, GPU, NPU",
        "2. Weight precision → Q4_0, Q8_0",
        "3. Compression level → SVD & MPO rank",
        "",
        "All mapped into a single Ising Hamiltonian:",
        "   min H(x) = cᵀx + xᵀQx",
        "",
        "Output: Native llama.cpp flags",
        "   -dev HTP0,GPUOpenCL,none",
        "   -ot \"blk.[0-12].*=HTP0\"",
        "   -ctk q8_0 -ctv q4_0",
    ]
    add_bullet_frame(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(5),
                     bullets, font_size=17, bold_prefix=False)

    # Key box
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.5), Inches(1.5),
        Inches(5), Inches(3.5)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xE0, 0xF2, 0xF1)
    shape.line.color.rgb = TEAL
    tf = shape.text_frame
    tf.word_wrap = True
    for line in [
        ("cᵢ:", "Isolated latency + accuracy penalty", True),
        ("Qᵢⱼ:", "Switching costs + cross-layer coupling", True),
        ("Constraints:", "One-hot per layer, memory budget", True),
    ]:
        p = tf.add_paragraph()
        run = p.add_run()
        run.text = f"  {line[0]} "
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = TEAL
        run = p.add_run()
        run.text = line[1]
        run.font.size = Pt(16)
        run.font.color.rgb = BLACK

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 5: Key Result — Matched Bytes
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Result: Combinatorial Superiority at Matched Bytes",
                 font_size=30, bold=True, color=TECH_BLUE)

    # Results table
    table_data = [
        ["Budget", "Plan", "ΔPPL (Full Split)"],
        ["K=6", "Pairwise", "+6.76%"],
        ["K=6", "Random", "+17.37%"],
        ["K=12", "Pairwise", "+27.05%"],
        ["K=12", "Random", "+36.83%"],
        ["K=18", "Pairwise", "+66.90%"],
        ["K=18", "Random", "+158.51%"],
    ]
    table_shape = slide.shapes.add_table(
        len(table_data), 3, Inches(0.5), Inches(1.3), Inches(6.5), Inches(4)
    )
    table = table_shape.table
    for r, row in enumerate(table_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(15)
                p.font.name = "Calibri"
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                elif "Pairwise" in row[1]:
                    p.font.color.rgb = GOOD_GREEN
                    p.font.bold = True
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TECH_BLUE

    # Callout
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8), Inches(1.5),
        Inches(4.5), Inches(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xE0, 0xF2, 0xF1)
    shape.line.color.rgb = TEAL
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "91.61 pp"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = TEAL
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = "advantage at K=18"
    p2.font.size = Pt(20)
    p2.font.color.rgb = TEAL
    p2.alignment = PP_ALIGN.CENTER
    p3 = tf.add_paragraph()
    p3.text = "\n95% CI: [-94.03, -89.04] pp\np < 10⁻⁴"
    p3.font.size = Pt(14)
    p3.font.color.rgb = BLACK
    p3.alignment = PP_ALIGN.CENTER

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 6: MPO vs SVD
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Result: Tensor Network Scaling (MPO vs. SVD)",
                 font_size=30, bold=True, color=TECH_BLUE)

    add_text_box(slide, Inches(0.5), Inches(1.2), Inches(8), Inches(0.5),
                 "Across 24 layers (2–25) at keep 0.9, matched parameters:",
                 font_size=18, color=BLACK)

    table_data = [
        ["Method", "PPL", "ΔPPL"],
        ["Dense baseline", "21.884", "—"],
        ["3-Core MPO", "39.049", "+78.4%"],
        ["Activation SVD", "53.174", "+143.0%"],
    ]
    table_shape = slide.shapes.add_table(
        len(table_data), 3, Inches(0.5), Inches(2.0), Inches(6), Inches(2.5)
    )
    table = table_shape.table
    for r, row in enumerate(table_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(16)
                p.font.name = "Calibri"
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                elif r == 2:  # MPO row
                    p.font.bold = True
                    p.font.color.rgb = GOOD_GREEN
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TECH_BLUE

    # Why MPO wins
    bullets = [
        "• SVD errors align constructively → collapse representation",
        "• MPO distributes error isotropically across tensor cores",
        "• Errors cancel destructively over network depth",
        "",
        "• ~65 pp advantage for MPO over SVD",
    ]
    add_bullet_frame(slide, Inches(7.5), Inches(1.5), Inches(5), Inches(4),
                     bullets, font_size=16, bold_prefix=False)

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 7: Solver Performance
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Solver Performance: Discrete Dominates Continuous",
                 font_size=30, bold=True, color=TECH_BLUE)

    table_data = [
        ["Solver", "Optimality Gap", "Runtime", "Constraints"],
        ["CP-SAT (Exact)", "0.000%", "0.20–0.60 s", "Hard cuts"],
        ["Greedy 1-Swap", "0.000%", "< 5 ms", "Feasible manifold"],
        ["Ballistic SB", "5.36–13.08%", "0.31 s", "Penalty relaxation"],
        ["Discrete SB", "19.11–44.79%", "0.45 s", "Penalty relaxation"],
        ["Simulated Annealing", "20.46–36.21%", "2.10 s", "Penalty relaxation"],
    ]
    table_shape = slide.shapes.add_table(
        len(table_data), 4, Inches(0.5), Inches(1.3), Inches(12), Inches(3.5)
    )
    table = table_shape.table
    for r, row in enumerate(table_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(16)
                p.font.name = "Calibri"
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                elif r == 2:  # Greedy row
                    p.font.bold = True
                    p.font.color.rgb = GOOD_GREEN
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TECH_BLUE
            elif r == 2:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)

    add_text_box(slide, Inches(0.5), Inches(5.2), Inches(12), Inches(1.5),
                 "Key insight: Hard equality constraints (Σxᵢ = K) require penalty barriers "
                 "in continuous solvers, warping phase space. Greedy 1-swap operates "
                 "directly on the feasible manifold, evaluating swaps in O(1) amortized time.",
                 font_size=15, color=RGBColor(0x55, 0x55, 0x55))

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 8: Closed-Loop Planning
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Active Closed-Loop Planning",
                 font_size=30, bold=True, color=TECH_BLUE)

    add_text_box(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.5),
                 "Measure → Fit → Solve iterative refinement:",
                 font_size=20, bold=True, color=TEAL)

    bullets = [
        "• Rank-6 Factorization Machine surrogate",
        "• Fitted to ln(PPL / PPL_dense)",
        "• CP-SAT acquisition over 9.9×10⁹ feasible plans",
        "• Dual strategy: exploitation + exploration",
    ]
    add_bullet_frame(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(3),
                     bullets, font_size=17, bold_prefix=False)

    # Result boxes
    for i, (title, value, detail, color) in enumerate([
        ("Beats one-shot by", "−1.53 pp", "95% CI: [−2.16, −0.90]", TEAL),
        ("Defeats random by", "52–87 pp", "Statistically significant", ACCENT_ORG),
    ]):
        y = Inches(1.5) + Inches(i * 2.5)
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8), y,
            Inches(4.5), Inches(2)
        )
        shape.fill.solid()
        fill_color = RGBColor(0xE0, 0xF2, 0xF1) if i == 0 else RGBColor(0xFF, 0xF3, 0xE0)
        shape.fill.fore_color.rgb = fill_color
        shape.line.color.rgb = color
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.color.rgb = color
        p.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = value
        p2.font.size = Pt(36)
        p2.font.bold = True
        p2.font.color.rgb = color
        p2.alignment = PP_ALIGN.CENTER
        p3 = tf.add_paragraph()
        p3.text = detail
        p3.font.size = Pt(12)
        p3.font.color.rgb = BLACK
        p3.alignment = PP_ALIGN.CENTER

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 9: Deployment
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
                 "Deployment: Zero-Code, Open-Source",
                 font_size=30, bold=True, color=TECH_BLUE)

    bullets_left = [
        "• Qwen3-0.6B (Apache-2.0)",
        "• llama.cpp (MIT) with native Snapdragon backends",
        "• WikiText-2, HotpotQA (public datasets)",
        "",
        "• Plans emit standard CLI flags",
        "• No driver modifications required",
    ]
    add_bullet_frame(slide, Inches(0.5), Inches(1.3), Inches(6), Inches(4),
                     bullets_left, font_size=17, bold_prefix=False)

    bullets_right = [
        "• 56 unit tests, 100% pass rate",
        "• Automated numerical verification",
        "• Cross-runtime validation",
        "",
        "• Snapdragon X Elite / X2 Elite PCs",
        "• Snapdragon 8 Gen 3 / 8 Elite phones",
        "• Any llama.cpp-compatible device",
    ]
    add_bullet_frame(slide, Inches(7), Inches(1.3), Inches(5.5), Inches(4),
                     bullets_right, font_size=17, bold_prefix=False)

    # ═══════════════════════════════════════════════════════════════════
    # SLIDE 10: Summary
    # ═══════════════════════════════════════════════════════════════════
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, TECH_BLUE)

    add_text_box(slide, Inches(0.5), Inches(0.5), Inches(12), Inches(1),
                 "Ising-Edge: Unlocking Heterogeneous On-Device AI",
                 font_size=36, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    # Three columns
    col_data = [
        ("Unified\nOptimization",
         "Joint CPU/GPU/NPU\nplacement, quantization\n& compression\nin a single QUBO"),
        ("Verified\nAccuracy Gains",
         "91.6 pp matched-bytes\n65 pp MPO vs SVD\nAll with paired\nbootstrap CIs"),
        ("Real-Time\nEdge Planning",
         "0% gap in <5 ms\nNative llama.cpp\nZero-code deploy\n100% open-source"),
    ]
    for i, (title, desc) in enumerate(col_data):
        x = Inches(0.8 + i * 4.2)
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.2),
            Inches(3.5), Inches(3.5)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(0x00, 0x6F, 0x82)
        shape.line.color.rgb = TEAL
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = ""
        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(16)
        p3.font.color.rgb = RGBColor(0xBB, 0xDE, 0xFB)
        p3.alignment = PP_ALIGN.CENTER

    add_text_box(slide, Inches(0.5), Inches(6.2), Inches(12), Inches(0.7),
                 "Thank you.", font_size=28, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    # ── Save ────────────────────────────────────────────────────────────
    prs.save(output_path)
    print(f"Presentation saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ising-Edge pitch deck as PPTX"
    )
    parser.add_argument(
        "-o", "--output",
        default="pitch_deck.pptx",
        help="Output PPTX file path (default: pitch_deck.pptx)"
    )
    args = parser.parse_args()
    build_presentation(args.output)


if __name__ == "__main__":
    main()
