# Ising-Edge: Joint Optimization Framework for Snapdragon Edge AI

**Snapdragon AI Challenge 2026**  
*Track: On-Device Generative AI Acceleration & Optimization*  
**Author:** Jenish Patel

---

## Overview

**Ising-Edge** is a combinatorial optimization framework that jointly determines on-device execution strategies for Small Language Models (SLMs) across heterogeneous Snapdragon System-on-Chips (CPU, Adreno GPU, and Hexagon NPU). By formulating compute unit placement, mixed precision ($\text{Q4\_0}, \text{Q8\_0}$), and low-rank compression as a single Quadratic Unconstrained Binary Optimization (QUBO) / Ising Hamiltonian, Ising-Edge prevents non-linear cross-layer accuracy collapse and synthesizes optimal runtime configurations emitted as native `llama.cpp` CLI flags.

### Key Discoveries & Features
- **Super-Additivity Mitigation:** Identifies and compensates for non-linear error compounding across deep transformer layers via measured 2nd-order interaction terms ($Q_{ij}$).
- **Matched-Bytes Advantage:** Outperforms uncoupled heuristic allocations by **91.61 percentage points** under strictly identical parameter and memory footprints.
- **Deep Tensor Network Scaling:** Demonstrates that multi-linear Matrix Product Operators (3-Core MPO) achieve superior error cancellation in deep composition compared to standard SVD (**PPL 39.05 vs 53.17** across 24 layers).
- **Sub-Millisecond Plan Synthesis:** Discrete 1-swap manifold descent reaches exact global optima in $< 5\text{ ms}$, bypassing the barrier-warping penalties of continuous physics heuristics.
- **100% Permissive Open-Source Foundation:** Built entirely on Qwen3 (Apache-2.0) and `llama.cpp` (MIT).

---

## Core Documentation
- **[docs/PROPOSAL.md](docs/PROPOSAL.md):** The comprehensive technical proposal, theoretical derivations, and architectural specifications.
- **[docs/SUBMISSION.md](docs/SUBMISSION.md):** The official challenge submission pack, executive abstract, criteria mapping, and reproduction guide.
- **[docs/RESULTS.md](docs/RESULTS.md):** Complete empirical benchmark ledger across baselines, sensitivity, multi-layer scaling, and solver benchmarks.
- **[docs/research/SYNTHESIS.md](docs/research/SYNTHESIS.md):** Comprehensive research synthesis of surrogate optimization, theoretical derivations, and heterogeneous edge execution.

---

## Reproduction & Verification

### Test Suite Execution
```bash
# Install dependencies
pip install -e ".[dev,solve]"

# Run verified unit test suite
pytest tests/ -q
```

### Empirical Pipeline Reproduction
```bash
# 1. Single-layer sensitivity baseline
python scripts/torch_ppl.py singles

# 2. Pairwise interaction coupling
python scripts/torch_ppl.py pairs

# 3. Strictly matched-bytes plan comparison
python scripts/matched_bytes.py

# 4. Multi-layer MPO vs SVD tensor network comparison
python scripts/torch_ppl.py allayer --method mpo --keep 0.9
python scripts/torch_ppl.py allayer --method svd --keep 0.9

# 5. Closed-loop Factorization Machine surrogate optimization
python scripts/loop_plan.py --real
```

---

## Hardware Deployment

Ising-Edge plans emit standard CLI parameters for direct execution on Snapdragon mobile devices and Copilot+ PCs:
```bash
# Example generated llama.cpp execution string:
llama-server \
  -m models/qwen3-0.6b-q4_0.gguf \
  -dev HTP0,GPUOpenCL,none \
  -ot "blk.[0-11].*=HTP0" -ot "blk.[12-27].*=none" \
  -ctk q8_0 -ctv q4_0
```

---

## Licenses
- **Code:** MIT License
- **Base Models:** Qwen3 (Apache-2.0)
- **Inference Engine:** `llama.cpp` (MIT)
- **Evaluation Datasets:** WikiText-2 (CC BY-SA), HotpotQA (CC BY-SA 4.0)
- **Published Benchmarks:** Qualcomm AI Hub Models (`perf.yaml`, BSD-3-Clause)
