# Ising-Edge: Technical Presentation & Submission Package

**Competition:** Snapdragon AI Challenge 2026  
**Track:** On-Device Generative AI Acceleration & Optimization  
**Project:** Ising-Edge: Joint Compute Placement, Precision, and Low-Rank Compression via Quadratic Binary Optimisation for Snapdragon Edge AI  
**Author:** Jenish Patel  
**Submission Date:** September 30, 2026  

---

## 1. Overview

This directory contains the presentation slide deck, technical report, bibliographic records, and empirical fact-checking documentation for the **Ising-Edge** submission package.

### File Manifest

| File | Type | Description |
| :--- | :--- | :--- |
| [`pitch_deck.tex`](file:///e:/Snapdragon%20AI/presentation/pitch_deck.tex) | LaTeX / Beamer | 16:9 widescreen presentation pitch deck formatted with the `metropolis` theme and TikZ architectural diagrams for judges. |
| [`project_description.tex`](file:///e:/Snapdragon%20AI/presentation/project_description.tex) | LaTeX / Article | Extended technical document outlining theoretical foundations, QUBO formulations, hardware roofline models, and empirical findings. |
| [`references.bib`](file:///e:/Snapdragon%20AI/presentation/references.bib) | BibTeX | Peer-reviewed and industry references covering tensor networks, QUBO/Ising models, combinatorial solvers, benchmarks, and Snapdragon hardware. |
| [`FACT_CHECK.md`](file:///e:/Snapdragon%20AI/presentation/FACT_CHECK.md) | Markdown | Verification matrix matching every numerical claim and benchmark figure in the deck to raw CSV result files and test scripts. |
| [`README.md`](file:///e:/Snapdragon%20AI/presentation/README.md) | Markdown | Compilation instructions, package dependencies, Overleaf setup guide, and project manifest. |

---

## 2. Compilation Prerequisites

Both LaTeX documents can be built using standard TeX distributions:
* **TeX Distributions:** TeX Live 2022+ (Linux / macOS), MacTeX 2022+ (macOS), or MiKTeX 23+ (Windows).
* **Compilers:** `pdflatex` (recommended) or `xelatex`.
* **Bibliography Processor:** `bibtex`.
* **Automation:** `latexmk` (optional, for single-command builds).

### Required LaTeX Packages
The documents use standard, universally available CTAN packages:
* **Core Presentation & Layout:** `beamer` (with `metropolis` theme / `pgfcore`), `geometry`, `authblk`
* **Mathematics & Typography:** `amsmath`, `amssymb`, `inputenc` (utf8), `fontenc` (T1)
* **Tables & Graphics:** `booktabs`, `graphicx`, `xcolor`, `tikz`
* **TikZ Libraries:** `shapes.geometric`, `arrows.meta`, `positioning`, `calc`, `fit`, `backgrounds`
* **Hyperlinks & Citations:** `hyperref`, `cite`

---

## 3. Local Compilation Commands

### 3.1 Pitch Deck (`pitch_deck.tex`)
The pitch deck is a standalone Beamer document. Compile with `pdflatex` (two passes to resolve TikZ positioning and slide numbers):

```bash
cd presentation

# First pass
pdflatex pitch_deck.tex

# Second pass (resolves page counters and cross-references)
pdflatex pitch_deck.tex
```

*Using XeLaTeX:*
```bash
xelatex pitch_deck.tex
xelatex pitch_deck.tex
```

*Using latexmk:*
```bash
latexmk -pdf pitch_deck.tex
```

---

### 3.2 Extended Project Description (`project_description.tex`)
The project description includes bibliographic citations connected to `references.bib`. Compile with `pdflatex` + `bibtex`:

```bash
cd presentation

# 1. Initial pass
pdflatex project_description.tex

# 2. Compile bibliography
bibtex project_description

# 3. Resolve citations and cross-references
pdflatex project_description.tex
pdflatex project_description.tex
```

*Using XeLaTeX:*
```bash
xelatex project_description.tex
bibtex project_description
xelatex project_description.tex
xelatex project_description.tex
```

*Using latexmk:*
```bash
latexmk -pdf project_description.tex
```

---

## 4. Overleaf Compatibility

The files in this directory are fully compatible with [Overleaf](https://www.overleaf.com):

1. **Upload:** Create a new blank project in Overleaf, then upload all files from `presentation/`:
   * `pitch_deck.tex`
   * `project_description.tex`
   * `references.bib`
2. **Select Compiler:**
   * Open the Overleaf project settings (gear icon in the top left).
   * Ensure **Compiler** is set to **pdfLaTeX** (default) or **XeLaTeX**.
   * Ensure **TeX Live version** is set to **2023** or newer.
3. **Set Main Document:**
   * To build the pitch deck: set **Main document** to `pitch_deck.tex` and click **Recompile**.
   * To build the technical paper: set **Main document** to `project_description.tex` and click **Recompile**.
4. Both documents compile cleanly without missing package errors or font substitutions.

---

## 5. Verification & Numerical Accuracy

Every numerical claim presented in the slides and paper is cross-verified against real measurement ledgers in the `results/` directory.

To verify all numbers locally from the repository root:
```bash
python scripts/check_numbers.py
```

Refer to [`FACT_CHECK.md`](file:///e:/Snapdragon%20AI/presentation/FACT_CHECK.md) for the complete claim-by-claim verification table and confidence intervals.
