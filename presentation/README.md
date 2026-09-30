# Ising-Edge — submission package

| File | What |
|---|---|
| `pitch_deck.tex` → `pitch_deck.pdf` | 11-slide 16:9 Beamer deck (XeLaTeX, Inter font bundled in `assets/fonts`, OFL) |
| `pitch_deck.pptx` (`generate_pptx.py`) | Same slides, numbers and native editable charts (python-pptx) |
| `project_description.tex` → `.pdf` | 2-page A4 project description (`references.bib`) |
| `build_data.py` | Reads `../results`, recomputes and checks every value → `generated/`, `figures/data/`, `tables/` |
| `FACT_CHECK.md` | Claim → source file ledger, plus what was omitted and why |

## Rebuild everything (from `presentation/`)
```bash
pip install numpy python-pptx                 # plus the repo itself on the path (import isingedge)
python3 build_data.py                         # values, chart CSVs, table rows (+269 checks)
xelatex pitch_deck.tex && xelatex pitch_deck.tex
xelatex project_description.tex && bibtex project_description && xelatex project_description.tex && xelatex project_description.tex
python3 generate_pptx.py                      # pitch_deck.pptx
```
Requires TeX Live 2023+ with XeLaTeX, beamer, pgfplots, booktabs, natbib (no system fonts needed).
Figures are pgfplots charts drawn from `figures/data/*.csv`; the PPTX uses the same CSVs.
