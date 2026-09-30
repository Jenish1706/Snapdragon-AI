# Research Synthesis: Algorithmic Architecture and Strategic Directions

This document synthesizes theoretical and empirical findings across the three core research tracks informing the **Ising-Edge** framework:
- `closed-loop-surrogate.md`: Factorization-machine surrogates, acquisition functions, evaluation budgets, and combinatorial optimization for neural architecture and parameter compression.
- `hetero-placement-2026.md`: Multi-engine hardware execution (CPU, GPU, NPU) across Snapdragon architectures and `llama.cpp` kernel dispatch capabilities.
- `thermal-and-proxies.md`: Sustained mobile device throughput, thermal throttling behavior, proxy metrics for sensitivity estimation, and joint quantization-compression planning.

---

## 1. Comparative Analysis of Optimization Directions

| Direction | Edge AI Impact | Technical Feasibility | Novelty & Differentiation | Empirical Validation & Kill Criteria | Overall Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1: Closed-Loop Surrogate Planner** (*Measure $\to$ Fit $\to$ Solve*) | **High:** Yields superior compression allocations under identical byte budgets through adaptive surrogate guidance. | **High:** Feasible via offline paired perplexity measurements without device-specific flashing. | **Significant:** First application of rank-6 Factorization Machines with CP-SAT and manifold greedy descent for transformer layer compression. | Validated against strictly budget-matched baselines; kills if 95% bootstrap CI fails to beat one-shot planning. | **Primary Focus (Implemented)** |
| **D2: Cheap Proxy Sensitivity Metrics** (*Log-Space / Output KL*) | **Moderate:** Aims to accelerate layer ranking without full-context perplexity evaluations. | **High:** Requires forward-pass logit caching over calibration token windows. | **Established:** KL divergence widely explored in post-training quantization literature. | Tested against ground-truth $\Delta\text{PPL}$. Spearman $\rho \ge 0.9$ threshold required; failed ($\rho \approx 0.65$), correctly retaining true perplexity oracle. | **Evaluated & Resolved** |
| **D3: Heterogeneous Stage Splitting** (*NPU Prefill + CPU/GPU Decode*) | **Critical:** Exploits asymmetric processor strengths (Hexagon NPU achieves up to $5\times$ TTFT speedup; CPU decodes faster on PC). | **Moderate:** Requires split-context runtime architectures and inter-engine KV-cache handoffs. | **Evolving:** Stage splitting recognized in academic frameworks; native support in upstream runtimes expanding. | Validated via Snapdragon hardware roofline modeling in SnapTwin. | **Architecture Roadmapped** |
| **D4: Thermally-Sustained Throughput Planning** | **High:** Mobile devices throttle significantly under continuous generative workloads. | **Moderate:** Requires physical battery and temperature telemetry. | **High:** Novel intersection of thermal decay laws with combinatorial QUBO scheduling. | Kills if sustained decode throughput stays within 10% of cold baseline across all compute units. | **Future Device Deployment** |

---

## 2. Core Architectural Decisions

### 2.1 Surrogate Search & Acquisition
1. **Target Normalization:** Surrogate regression operates on standardized logarithmic perplexity ratios $\ln(\text{PPL} / \text{PPL}_{\text{dense}})$, stabilizing Factorization Machine weight gradients across diverse compression depths.
2. **Dual Acquisition Strategy:** In each iteration of the closed loop, candidate plans are acquired via two complementary mechanisms:
   - *Exploitation:* CP-SAT global minimization on the full empirical dataset.
   - *Exploration:* CP-SAT minimization on a random 80% subsample of evaluated plans, avoiding premature convergence in local minima.
3. **Hard Constraint Enforcement:** Parameter budgets ($\sum x_i = K$) are enforced strictly as combinatorial linear cuts rather than soft quadratic penalty relaxations, ensuring 100% feasibility.

### 2.2 Deep Composition Dynamics: MPO vs. SVD
Isolated single-matrix truncation is fundamentally deceptive for deep neural architectures:
- SVD minimizes local Frobenius norm $\|W - AB\|_F$, but aligns error singular vectors sequentially, collapsing representation dimensionality across deep layers.
- Matrix Product Operators (MPO) distribute truncation across multi-linear tensor cores, acting as incoherent pseudo-random noise that cancels destructively across residual layer additions.
- Consequently, deep compression across 24 transformer blocks demonstrates a **$\approx 65\text{ pp}$ perplexity advantage** for 3-core MPO over activation-aware SVD at identical parameter savings.

### 2.3 Solvers: Discrete Manifold Descent vs. Continuous Physics Heuristics
Hard cardinality constraints $\sum x_i = K$ warp continuous phase space in Simulated Bifurcation (SB) and Simulated Annealing (SA) when penalized quadratically via $\lambda(\sum x_i - K)^2$. 

By contrast, **Greedy 1-Swap Descent** restricts its search space entirely to the feasible manifold $\mathcal{S}_K$. Because candidate swaps are evaluated in $O(1)$ amortized time via the differential energy equation:
$$\Delta E_{i \to j} = (c_j - c_i) + 2 \sum_{k \neq i, j} (Q_{jk} - Q_{ik}) x_k - 2 Q_{ij}$$
greedy descent reaches exact global optima in under $5\text{ ms}$, delivering instantaneous plan generation suitable for real-time edge deployment.
