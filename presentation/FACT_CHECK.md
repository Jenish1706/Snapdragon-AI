# FACT_CHECK — Ising-Edge submission documents

Every number in `pitch_deck.pdf`, `pitch_deck.pptx` and `project_description.pdf` is produced by
`build_data.py` from the raw files in `../results/` and stored in `generated/values.json`.
`generated/checks.txt` lists the 269 consistency checks that passed (the script stops on any failure).
Paired-bootstrap intervals were **recomputed** from per-chunk NLL with `isingedge.loop.stats.paired_bootstrap`
(seed 0, 4,000 resamples) and match the stored CSVs.

| Claim (as shown) | Value | Source file | Section / row |
|---|---|---|---|
| Environment of all measurements | PyTorch 2.14.1 + CUDA 12.6, fp32, RTX 3060 Laptop GPU | all `results/**.csv` | `build` column |
| Dense PPL, full split / 40 chunks | 21.88 (584 chunks) / 19.60 | `results/compression/ppl.csv`; `results/sensitivity/qwen3-0.6b_layer_sensitivity.csv` | `dense-fp32`; `kind=dense` |
| keep 0.8 = rank 608 (79.2%), keep 0.9 = rank 704 (91.7%) | ranks, params | `qwen3-0.6b_layer_sensitivity.csv` | `rank`, `params`, `dense_params` |
| Single-layer damage +0.45% (L8) to +24.67% (L27), median +2.68% (+1.68% at keep 0.9) | | `qwen3-0.6b_layer_sensitivity.csv` | `svd` rows |
| 33 of 40 pairs super-additive; median +0.21 pp; max +10.06 pp (L0+L1) | | `results/sensitivity/qwen3-0.6b_pair_sensitivity.csv` | `interaction_pct` |
| 24 MLPs: +275.60% measured vs +57.75% summed, +76.75% log, +73.62% pair model (4.8×) | 40 chunks | `results/matched_bytes/summary.csv` | `mb_K24_pairwise` |
| Matched bytes (full split), model plan vs best of 3 random: K=6 −10.61 [−11.24, −9.95]; K=12 −9.78 [−10.59, −8.94]; K=18 −91.61 [−94.03, −89.04] (+66.90% vs +158.51%); K=24 **+23.42 worse** [+17.24, +29.55] | pp of dense PPL | `results/matched_bytes/summary.csv` + `qwen3-0.6b_plan_chunk_nll.csv` | `CI all chunks` rows (recomputed) |
| Pair, additive, log-additive and greedy pick the same layers | | `results/matched_bytes/summary.csv` | `method` label |
| 9,928,631,398 feasible plans (K=12), 48 binaries | | `isingedge/loop/space.py` `Space.count()` | computed |
| Loop = one-shot plan in 3/3 (K=12) and 2/3 (K=18) seeds; one-shot full split +25.45% / +64.52%; random +45.68–47.50% / +115.37–149.85% (trails by 20.2–85.3 pp) | | `results/loop/summary.csv` | per method/seed |
| K=18 seed 2: −1.53 pp [−2.16, −0.90] full split, +0.83 pp on 40 chunks; one swapped layer (3 vs 4); plan was also in the one-shot top-24 | | `results/loop/summary.csv`, `qwen3-0.6b_plan_perplexity.csv` | `lp_c984076625`, `lp_ee56d68d3a` |
| All loop acquisitions CP-SAT OPTIMAL | | `results/loop/evals.csv` | `status` |
| Calibration ρ 0.78–0.81 (K=12), 0.52–0.63 (K=18); bias −0.25 to −0.18 | 72 held-out plans | `results/loop/calibration.csv` | all rows |
| Greedy matched exact optimum in 15/72 rounds (median gap 13.2% / 4.1%); ballistic SB raw infeasible 72/72 | | `results/loop/solver_gaps.csv` | `gap_pct`, `raw_feasible` |
| KL proxy ρ = 0.64 [0.35, 0.83] < 0.90 | 28 layers | `results/sensitivity/qwen3-0.6b_kl_proxy_summary.csv` | `KL(dense||comp)` |
| MPO 39.05 vs SVD 53.17 PPL, layers 2–25, equal params (91.7%), bond 240 vs rank 704 | full split | `results/compression/ppl.csv` | `imx-mpoA-k0.9`, `imx-svd-k0.9` |
| Tests: 56 collected, 52 pass, 3 skip, 1 fail | | `python -m pytest tests/` (this snapshot) | `test_smoke` imports missing `isingedge.export`, `isingedge.runtime` |
| Implemented-but-unvalidated: SnapTwin, placement QUBO, adb/llama-bench runners | | `isingedge/twin`, `isingedge/qubo`, `isingedge/bench`, `scripts/bench_sweep.py` | no device results in `results/` |

## Intentionally omitted (not verifiable from this repository)
- HotpotQA context-packing results (no result file; `qubo_pack` does not use its QUBO or TTFT term).
- Qualcomm `perf.yaml` throughput table and SnapTwin held-out errors (`docs/TWIN.md` sources `data/published/`, `results/twin/` are absent).
- Solver table in `docs/RESULTS.md` ("greedy 0.000% gap < 5 ms", SB 5.36–13.08%): contradicted by `results/loop/solver_gaps.csv`; `isingedge/solve/solvers.py` SA/SB are enumeration stubs.
- "Cross-runtime validation (Metal vs CUDA)", llama.cpp flag emission (`scripts/plan.py` cannot run against the current package), any on-device latency/energy/memory figure.
- MPO weight error "0.354" (`results/compression/ppl.csv` stores 39.06); keep 0.7/0.6, data-free SVD and Q4_0/Q8_0 per-layer rows (not in results); "p < 10⁻⁴" (not resolvable with 4,000 resamples).
- The previous `presentation/` package (commit 6630680) repeated these claims and was replaced.
