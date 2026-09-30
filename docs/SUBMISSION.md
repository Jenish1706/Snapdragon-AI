# Ising-Edge: Submission Pack

**Snapdragon AI Challenge 2026**  
*Track: On-Device Generative AI Acceleration & Optimization*  
**Author:** Jenish Patel  
**Submission Date:** 30 September 2026

---

## Executive Summary (250 Words)

Deploying small language models on heterogeneous Snapdragon mobile and PC platforms requires navigating distinct trade-offs across three compute engines: CPU, Adreno GPU, and Hexagon NPU. Published benchmarks on Snapdragon SoCs demonstrate that no single processor dominates all phases: the Hexagon NPU achieves up to $5\times$ faster prefill, yet decodes up to $2.5\times$ slower than CPU or GPU cores due to memory-bandwidth and sequence serialization constraints. Existing optimizers treat compute unit selection, weight precision, and low-rank compression as disconnected decisions.

**Ising-Edge** is a unified, open-source combinatorial optimization framework that jointly determines compute unit placement ($\text{CPU}, \text{GPU}, \text{NPU}$), weight precision ($\text{Q4\_0}, \text{Q8\_0}$), and low-rank compression levels across every transformer block simultaneously. By mapping hardware latency profiles, synchronization costs, and compound cross-layer perplexity interactions into a single **Quadratic Unconstrained Binary Optimisation (QUBO) / Ising Hamiltonian**, Ising-Edge generates globally optimal execution plans emitted as native `llama.cpp` configuration flags.

Empirical evaluations across Qwen3-0.6B on standard benchmarks establish four core findings:
1. **Super-Additivity of Compression Damage:** Joint compression of 24 middle MLP blocks raises perplexity by **$+275.6\ \%$**, whereas their isolated costs sum to only **$+57.8\ \%$**, driven by bilinear error propagation through residual Jacobian products.
2. **Combinatorial Superiority at Matched Bytes:** Under strictly fixed parameter budgets ($K=18$ MLPs compressed at keep 0.8), the pairwise Ising plan delivers **$+66.9\ \%$** perplexity degradation on the full test split, defeating random allocations (**$+158.5\ \%$**) by **$91.6\text{ percentage points}$** ($95\%\text{ CI } [-94.0, -89.0]\text{ pp}$).
3. **Tensor Network Scaling Win (MPO vs SVD):** While isolated matrix truncation favors SVD, deep multi-layer composition demonstrates that 3-core MPO achieves **PPL $39.05$ ($+78.4\ \%$)** compared to SVD's **PPL $53.17$ ($+143.0\ \%$)**, suppressing cascading residual drift.
4. **Discrete Swap Descent Dominates Continuous Heuristics:** On cardinality-constrained hardware instances, discrete 1-swap greedy descent reaches exact global optima in milliseconds ($< 5\text{ ms}$), outperforming continuous penalty-relaxed Simulated Bifurcation ($+5.4\ \%\text{ to } +19.1\ \%$ gap).

---

## Core Empirical Results

All empirical measurements are established on Qwen3-0.6B on the standard WikiText-2 benchmark (512 context, 584 chunks) with cross-runtime reproducibility across Apple Silicon (Metal, `llama.cpp`) and NVIDIA RTX (CUDA, PyTorch fp32).

| # | Result | Quantitative Finding | Significance & Mechanism |
| :--- | :--- | :--- | :--- |
| **1** | **Super-Additivity of Compression Damage** | 24 MLPs (layers 2–25, keep 0.8) measured **+275.56 %** $\Delta$PPL together vs. **+57.78 %** summed linearly (+76.75 % log-additive). | Proves single-layer sensitivity tables are mathematically invalid for deep models; bilinear Jacobian cross-terms require 2nd-order pairwise interaction terms ($Q_{ij}$). |
| **2** | **Combinatorial Superiority at Strictly Matched Bytes** | At $K=18$ MLPs compressed (keep 0.8), Pairwise Plan scores **+66.90 %** $\Delta$PPL vs. Random Plan **+158.51 %** on full test split. | Statistically significant **91.61 pp advantage** (paired bootstrap 4,000 resamples: $95\%\text{ CI } [-94.03, -89.04]\text{ pp}$, strictly excluding zero). |
| **3** | **Tensor Network Composition Win (MPO vs SVD)** | Across 24 middle layers at keep 0.9: 3-Core MPO achieves **PPL 39.049 (+78.4 %)** vs. Activation-Aware SVD **PPL 53.174 (+143.0 %)**. | SVD error directions align constructively along the residual stream, collapsing representation rank; MPO distributes error isotropically across tensor cores, canceling destructively. |
| **4** | **Discrete Manifold Descent Outperforms Physical Solvers** | Greedy 1-Swap achieves **0.000 % optimality gap** in **< 5 ms**; Ballistic SB lags by **5.36–13.08 %**; Discrete SB lags by **19.11–44.79 %**. | Continuous dynamical solvers suffer from ill-conditioned quadratic penalty barriers $\lambda(\sum x_i - K)^2$; greedy descent preserves exact feasibility on the constrained manifold $\mathcal{S}_K$. |
| **5** | **Context Prompt Packing** | QUBO packing matches Top-K recall within $\pm 0.01$ (0.852 vs 0.841 at budget 320) while MMR drops recall significantly on multi-hop questions. | Incorporating SnapTwin TTFT cost balances context coverage with time-to-first-token latency. |
| **6** | **Closed-Loop Surrogate Optimization** | At $K=18$, adaptive closed-loop surrogate planning scores **+62.99 %** on full split vs. One-Shot Pairwise **+64.52 %** and Random **+149.85 %**. | Beating the one-shot frontier by **-1.53 pp** ($95\%\text{ CI } [-2.16, -0.90]\text{ pp}$, strictly excluding zero) and defeating random by **86.86 pp**. |

---

## Technical Architecture & Reproduction Guide

### Prerequisites & Test Suite Execution
The codebase provides synthetic pipelines that execute anywhere without external hardware dependencies, alongside full production runners:

```bash
# 1. Environment installation
pip install -e ".[dev,solve]"

# 2. Execute verified test suite (56 tests across 11 test modules covering models, solvers, and planning)
pytest tests/ -q

# 3. Verify empirical benchmark consistency against result ledgers
python scripts/check_numbers.py
```

### Reproducing Empirical Benchmarks
```bash
# Verify single-layer sensitivity and activation-aware SVD baseline
python scripts/torch_ppl.py singles

# Evaluate pairwise layer interaction coupling and super-additivity
python scripts/torch_ppl.py pairs

# Run matched-bytes plan comparison (K = 6, 12, 18, 24)
python scripts/matched_bytes.py

# Compare multi-layer MPO vs SVD deep composition
python scripts/torch_ppl.py allayer --method mpo --keep 0.9
python scripts/torch_ppl.py allayer --method svd --keep 0.9

# Run closed-loop surrogate optimization loop
python scripts/loop_plan.py --real
```

---

## Alignment with Judging Criteria

| Criterion | Evaluation Weight | Demonstrated Implementation in Ising-Edge |
| :--- | :--- | :--- |
| **Technical Implementation** | **Highest** | Complete, production-grade implementation: verified unit test coverage, exact CP-SAT and greedy combinatorial solvers, hardware roofline modeling (SnapTwin), activation-aware SVD and 3-core MPO tensor decompositions, and closed-loop Factorization Machine surrogate optimization. |
| **Application Use Case & Innovation** | **High** | First joint framework unifying heterogeneous unit placement, mixed precision, and structural rank compression in a single QUBO. Formulates and solves the fundamental problem of non-linear error compounding across transformer layers. |
| **Deployment & Accessibility** | **High** | Built entirely on permissively licensed open-source technologies (Qwen3 Apache-2.0, llama.cpp MIT). Generated execution plans emit native CLI flags (`-dev`, `-ot`, `-ctk`, `-ctv`) for immediate zero-code deployment on Snapdragon PCs and mobile devices. |
| **Presentation & Documentation** | **High** | Comprehensive, publication-grade documentation: master technical proposal ([docs/PROPOSAL.md](file:///e:/Snapdragon%20AI/docs/PROPOSAL.md)), full results ledger ([docs/RESULTS.md](file:///e:/Snapdragon%20AI/docs/RESULTS.md)), and research synthesis with theoretical proofs ([docs/research/SYNTHESIS.md](file:///e:/Snapdragon%20AI/docs/research/SYNTHESIS.md)). |

---

## Submission Checklist

Before finalizing submission on Unstop:
- [ ] **Author Identification:** Confirm author name and credentials match the submitting Unstop account profile.
- [ ] **Challenge Scope:** Completed within the official submission window (September 2026).
- [ ] **Licensing & Compliance:** 100% open-source models (Qwen3, Apache-2.0), runtimes (llama.cpp, MIT), and public datasets (WikiText-2, HotpotQA). No proprietary or restricted assets.
- [ ] **Originality & Independence:** All code, theoretical derivations, and experimental benchmarks represent original work developed for this challenge.
- [ ] **Eligibility & Format:** Submission prepared in English, fully formatted in GitHub-compliant Markdown.
