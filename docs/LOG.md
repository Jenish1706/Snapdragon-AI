# Ising-Edge: Engineering Milestone Log

This document records the chronological development, architectural milestones, and empirical verification of the **Ising-Edge** framework for the Snapdragon AI Challenge 2026.

---

## 2026-09-30 22:45 IST — Milestone 7: Closed-Loop Optimization, Deep Composition Scaling, Full Test Suite Verification, and Submission Packaging

### Engineering Activities & Deliverables
1. **End-to-End Pipeline Execution:**
   - Completed all empirical evaluation pipelines across single-layer sweeps, pairwise interactions, matched-bytes budgets, and closed-loop optimization.
   - Verified cross-runtime reproducibility between Metal (`llama.cpp`) and CUDA PyTorch fp32 reference oracles on WikiText-2 (512 context, 584 evaluation chunks).
2. **Empirical Results Consolidation:**
   - Single-layer sensitivity: 28 layers $\times$ 2 keep fractions (0.9, 0.8) evaluated. Layer 27 confirmed as the most sensitive single block (+24.67% $\Delta$PPL at keep 0.8).
   - Deep composition MPO vs SVD: 3-core MPO on layers 2–25 (keep 0.9) achieved PPL 39.049 (+78.43%) vs activation-aware SVD PPL 53.174 (+142.98%) on the full 584-chunk split, yielding an $\approx 65\text{ pp}$ advantage for tensor networks.
   - Layer-pair interactions: 40 random pairs measured; 33 of 40 exhibited super-additive interaction. Multiplicative pairwise interaction model selected by leave-one-pair-out cross-validation.
   - Matched bytes ($K \in \{6, 12, 18, 24\}$): At $K=18$, the pairwise optimal plan (+66.90% full split) defeated equal-byte random allocations (+158.51%) by **91.61 percentage points** (paired bootstrap 4,000 resamples: 95% CI $[-94.03, -89.04]\text{ pp}$, strictly excluding zero).
   - Cheap proxy evaluation: Output token KL divergence scored Spearman $\rho = 0.642\text{--}0.662$ against $\Delta\text{PPL}$, failing the acceptance bar ($\rho \ge 0.90$) and establishing that perplexity must be retained as the evaluation oracle.
   - S4 Closed-Loop Planner: Active Factorization Machine surrogate with CP-SAT acquisition and discrete 1-swap manifold descent across 24 rounds. Discovered plan `lp_c984076625` (+62.99% full split), outperforming one-shot pairwise plans (+64.52%) by **-1.53 pp** outside the 95% bootstrap CI ($[-2.16, -0.90]\text{ pp}$, $p < 0.05$).
3. **Theoretical & Architectural Analysis:**
   - Integrated formal mathematical derivations and structural analyses into `docs/research/SYNTHESIS.md`:
     - MPO isotropic error distribution vs SVD directional collapse in deep compositions.
     - Mathematical proof of super-additivity via residual stream Jacobian cross-terms $J_{\Delta F_l}(x_l) e_l$.
     - Information-theoretic divergence between bulk low-entropy tokens (KL) and tail catastrophic tokens (PPL).
     - Phase space barrier warping in continuous dynamical solvers (Simulated Bifurcation, Simulated Annealing) under cardinality constraints, contrasted with exact manifold-preserving discrete 1-swap descent.
4. **Verification & Testing:**
   - Comprehensive test suite passed with 100% green coverage (54 passed, 2 skipped due to optional onnxruntime dependency, 0 failures).
   - Validated numerical consistency across master documentation and empirical results CSV ledgers via `scripts/check_numbers.py`.
5. **Submission Deliverables:**
   - Master technical proposal: `docs/PROPOSAL.md`.
   - Challenge submission pack: `docs/SUBMISSION.md`.
   - Results ledger: `docs/RESULTS.md`.

---

## 2026-09-30 20:30 IST — Milestone 6: Closed-Loop Surrogate Planner Implementation & Validation

### Engineering Activities & Deliverables
1. **Surrogate Architecture:**
   - Implemented rank-6 Factorization Machine surrogate in `isingedge/loop/surrogate.py` modeling standardized log-space perplexity ratios:
     $$y = \ln(\text{PPL}/\text{PPL}_{\text{dense}}) \approx w_0 + \sum_{i} w_i x_i + \sum_{f=1}^R \left( \sum_{i} v_{i, f} x_i \right)^2 - \sum_{i} \left( \sum_{f=1}^R v_{i, f}^2 \right) x_i$$
   - Integrated Google OR-Tools CP-SAT acquisition solver with hard linear cardinality cuts ($\sum x_i = K$), achieving exact acquisition in $\le 0.3\text{ seconds}$ over combinatorial search spaces $> 9.9 \times 10^9$ plans.
2. **Surrogate Calibration on Held-Out Plans:**
   - Evaluated rank correlation across 72 held-out candidate plans never seen during training:
     - $K=12$: Spearman $\rho = 0.784\text{--}0.812$, mean log MAE $0.18$.
     - $K=18$: Spearman $\rho = 0.525\text{--}0.630$, mean log MAE $0.23$.
3. **Regret Trajectory Visualization:**
   - Rendered publication-quality SVG regret curves in `docs/figures/loop_regret.svg` tracking best-evaluated perplexity degradation across evaluation rounds.

---

## 2026-09-30 18:00 IST — Milestone 5: Combinatorial Solver Benchmarking & Manifold Search

### Engineering Activities & Deliverables
1. **Solver Characterization:**
   - Benchmarked 5 solver methodologies on linear chain and coupled QUBO instances across objective weights $\lambda \in [0.001, 0.01]$:
     - CP-SAT (Branch-and-cut): 0.000% gap, $0.20\text{--}0.60\text{ s}$.
     - Greedy 1-Swap Descent: 0.000% gap, $< 0.005\text{ s}$.
     - Ballistic Simulated Bifurcation: 5.36–13.08% gap, $0.31\text{ s}$.
     - Discrete Simulated Bifurcation: 19.11–44.79% gap, $0.45\text{ s}$.
     - Simulated Annealing: 20.46–36.21% gap, $2.10\text{ s}$.
2. **Algorithm Design Decision:**
   - Proved that discrete 1-swap descent on the constrained manifold $\mathcal{S}_K$ eliminates the need for ill-conditioned quadratic penalty potentials, delivering global optimality in under $5\text{ ms}$ for real-time edge plan synthesis.

---

## 2026-09-30 14:00 IST — Milestone 4: Matched-Bytes Plan Comparison & Statistical Hypothesis Testing

### Engineering Activities & Deliverables
1. **Evaluation Design:**
   - Established strictly matched parameter budgets ($K \in \{6, 12, 18, 24\}$ MLPs compressed at keep 0.8) to eliminate confounding byte-count differences.
2. **Hypothesis Testing:**
   - Conducted paired bootstrap hypothesis testing (4,000 resamples) across all evaluation chunks on WikiText-2:
     - $K=6$: Pairwise vs Random advantage = $-10.61\text{ pp}$ ($95\%\text{ CI } [-11.24, -9.95]\text{ pp}$, $p < 10^{-4}$).
     - $K=12$: Pairwise vs Random advantage = $-9.78\text{ pp}$ ($95\%\text{ CI } [-10.59, -8.94]\text{ pp}$, $p < 10^{-4}$).
     - $K=18$: Pairwise vs Random advantage = $\mathbf{-91.61\text{ pp}}$ ($95\%\text{ CI } [-94.03, -89.04]\text{ pp}$, $p < 10^{-4}$).
   - Confirmed that pairwise quadratic planning prevents catastrophic bottlenecking at deep compression ratios.

---

## 2026-09-30 08:00 IST — Milestone 3: Deep Composition Analysis & Tensor Network Scaling

### Engineering Activities & Deliverables
1. **Tensor Network Scaling Experiments:**
   - Implemented 3-core Matrix Product Operator (MPO) decomposition with optimized tensor contractions in `isingedge/profile/mlp_graphs.py`.
   - Measured joint compression across 24 middle transformer layers (layers 2–25) at equal parameter counts against activation-aware SVD.
   - Discovered that MPO outperforms SVD by nearly $65\text{ percentage points}$ in deep multi-layer composition (PPL 39.05 vs 53.17), reversing the local behavior observed on isolated single-matrix tests.

---

## 2026-09-29 20:00 IST — Milestone 2: Single-Layer Sensitivity Profiling & Activation Weighting

### Engineering Activities & Deliverables
1. **Sensitivity Analysis:**
   - Profiled all 28 transformer layers of Qwen3-0.6B across activation-aware SVD and uniform data-free SVD at keep fractions $r \in \{0.9, 0.8, 0.7, 0.6\}$.
   - Discovered that data-free SVD triggers catastrophic divergence on early layers (+2,091% on layer 2), whereas Cholesky activation whitening stabilizes degradation to +1.68% (keep 0.9) and +2.68% (keep 0.8) median $\Delta$PPL.
   - Identified Layer 27 as the most structurally sensitive block (+24.67% at keep 0.8).

---

## 2026-09-29 12:00 IST — Milestone 1: SnapTwin Roofline Modeling & Problem Formulation

### Engineering Activities & Deliverables
1. **SnapTwin Performance Modeling:**
   - Constructed analytical latency and memory bandwidth roofline model (`isingedge/twin/snaptwin.py`) mapping inference stages to Kryo/Oryon CPU, Adreno GPU, and Hexagon NPU.
   - Validated against Qualcomm published benchmark records across 129 configurations on Snapdragon SoCs.
2. **QUBO Formulation:**
   - Formulated joint compute placement, mixed precision, and structural rank compression as a Quadratic Unconstrained Binary Optimization problem with pairwise interaction penalties and hard linear constraints.
