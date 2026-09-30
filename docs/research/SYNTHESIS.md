# Research Synthesis: Algorithmic Architecture and Strategic Directions

This document synthesizes theoretical and empirical findings across the core research tracks informing the **Ising-Edge** framework:
- `closed-loop-surrogate.md`: Factorization-machine surrogates, acquisition functions, evaluation budgets, and combinatorial optimization for neural architecture and parameter compression.
- `hetero-placement-2026.md`: Multi-engine hardware execution (CPU, GPU, NPU) across Snapdragon architectures and `llama.cpp` kernel dispatch capabilities.
- `thermal-and-proxies.md`: Sustained mobile device throughput, thermal throttling behavior, proxy metrics for sensitivity estimation, and joint quantization-compression planning.

---

## 1. Comparative Analysis of Optimization Directions

| Direction | Edge AI Impact | Technical Feasibility | Novelty & Differentiation | Empirical Validation & Success Criteria | Overall Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1: Closed-Loop Surrogate Planner** (*Measure $\to$ Fit $\to$ Solve*) | **High:** Discovers superior compression allocations under identical byte budgets through adaptive surrogate guidance. | **High:** Feasible via offline paired perplexity measurements without device-specific flashing. | **Significant:** First application of rank-6 Factorization Machines with CP-SAT and manifold greedy descent for transformer layer compression. | Validated against strictly budget-matched baselines; beats one-shot planning outside 95% bootstrap CI. | **Primary Focus (Implemented)** |
| **D2: Proxy Sensitivity Metrics** (*Log-Space / Output KL*) | **Moderate:** Aims to accelerate layer ranking without full-context perplexity evaluations. | **High:** Requires forward-pass logit caching over calibration token windows. | **Established:** KL divergence widely explored in post-training quantization literature. | Tested against ground-truth $\Delta\text{PPL}$. Evaluated against Spearman $\rho \ge 0.9$ threshold ($\rho \approx 0.65$), confirming perplexity as the necessary ground-truth oracle. | **Evaluated & Integrated** |
| **D3: Heterogeneous Stage Splitting** (*NPU Prefill + CPU/GPU Decode*) | **Critical:** Exploits asymmetric processor strengths (Hexagon NPU achieves up to $5\times$ TTFT speedup; CPU decodes faster on PC). | **Moderate:** Requires split-context runtime architectures and inter-engine KV-cache handoffs. | **Evolving:** Stage splitting recognized in academic frameworks; native support in upstream runtimes expanding. | Validated via Snapdragon hardware roofline modeling in SnapTwin. | **Architecture Roadmapped** |
| **D4: Thermally-Sustained Throughput Planning** | **High:** Mobile devices throttle significantly under continuous generative workloads. | **Moderate:** Requires physical battery and temperature telemetry. | **High:** Novel intersection of thermal decay laws with combinatorial QUBO scheduling. | Quantifies plateau vs peak execution trade-offs across SoC compute engines. | **Future Device Deployment** |

---

## 2. Core Architectural & Theoretical Foundations

### 2.1 Surrogate Search & Acquisition
1. **Target Normalization:** Surrogate regression operates on standardized logarithmic perplexity ratios $\ln(\text{PPL} / \text{PPL}_{\text{dense}})$, stabilizing Factorization Machine weight gradients across diverse compression depths.
2. **Dual Acquisition Strategy:** In each iteration of the closed loop, candidate plans are acquired via two complementary mechanisms:
   - *Exploitation:* CP-SAT global minimization on the full empirical dataset.
   - *Exploration:* CP-SAT minimization on a random 80% subsample of evaluated plans, avoiding premature convergence in local minima.
3. **Hard Constraint Enforcement:** Parameter budgets ($\sum x_i = K$) are enforced strictly as combinatorial linear cuts rather than soft quadratic penalty relaxations, ensuring 100% feasibility.

### 2.2 Deep Composition Dynamics: MPO vs. SVD
Isolated single-matrix truncation behaves fundamentally differently from deep multi-layer composition:
- SVD minimizes local Frobenius norm $\|W - AB\|_F$, but aligns error singular vectors sequentially, collapsing representation dimensionality across deep layers.
- Matrix Product Operators (MPO) distribute truncation across multi-linear tensor cores, acting as incoherent pseudo-random noise that cancels destructively across residual layer additions.
- Consequently, deep compression across 24 transformer blocks demonstrates an **$\approx 65\text{ pp}$ perplexity advantage** for 3-core MPO over activation-aware SVD (**PPL 39.05 vs 53.17**, $+78.4\%$ vs $+143.0\%$) at identical parameter savings.

### 2.3 Solvers: Discrete Manifold Descent vs. Continuous Physics Heuristics
Hard cardinality constraints $\sum x_i = K$ warp continuous phase space in Simulated Bifurcation (SB) and Simulated Annealing (SA) when penalized quadratically via $\lambda(\sum x_i - K)^2$. 

By contrast, **Greedy 1-Swap Descent** restricts its search space entirely to the feasible manifold $\mathcal{S}_K$. Because candidate swaps are evaluated in $O(1)$ amortized time via the differential energy equation:
$$\Delta E_{i \to j} = (c_j - c_i) + 2 \sum_{k \neq i, j} (Q_{jk} - Q_{ik}) x_k - 2 Q_{ij}$$
greedy descent reaches exact global optima in under $5\text{ ms}$, delivering instantaneous plan generation suitable for real-time edge deployment.

### 2.4 Mathematical Derivation of Compound Residual Perturbations
In a transformer residual stream, layer $l$ computes $x_{l+1} = x_l + F_l(x_l; W_l)$. Let the compressed weight matrix be $\widetilde{W}_l = W_l + \Delta W_l$, let $\tilde{x}_l$ be the hidden state under compression, and define the state error as $e_l = \tilde{x}_l - x_l$.

Expanding the layer update via a first-order Taylor expansion around unperturbed state $x_l$:
$$F_l(\tilde{x}_l; W_l + \Delta W_l) \approx F_l(x_l; W_l) + J_{F_l}(x_l) e_l + \Delta F_l(x_l) + J_{\Delta F_l}(x_l) e_l$$
where $J_{F_l} = \frac{\partial F_l}{\partial x_l}$ is the Jacobian of the clean layer, $\Delta F_l(x_l)$ is the isolated layer perturbation, and $J_{\Delta F_l}(x_l) e_l$ is the **bilinear interaction cross-term**.

Subtracting clean recurrence $x_{l+1} = x_l + F_l(x_l)$ yields the exact error recurrence:
$$e_{l+1} = \left( I + J_{F_l}(x_l) + J_{\Delta F_l}(x_l) \right) e_l + \Delta F_l(x_l)$$

When multiple layers are compressed:
1. Cross-terms $J_{\Delta F_l} e_l$ represent a direct coupling between incoming representation error $e_l$ and weight distortion $\Delta W_l$. Degraded hidden states misroute tokens past non-linear SwiGLU gating thresholds into dead activation regions.
2. Because cross-entropy loss exponentiates under perplexity ($\text{PPL} = \exp(\mathcal{L})$), extreme logit perturbations in tail tokens dominate the expectation value, resulting in explosive super-additivity (+276% measured vs +58% summed linearly across 24 layers).
3. This proves that single-layer linear planning is mathematically invalid for LLMs, establishing the necessity of a **2nd-order QUBO / Ising formulation** with pairwise interaction coefficients $Q_{ij}$.

### 2.5 Evaluation Oracle Dynamics: Perplexity vs Token-Averaged KL Divergence
Output KL divergence over a sequence is defined as:
$$D_{\text{KL}}(P \| Q) = \sum_{t=1}^T \sum_{v \in \mathcal{V}} P(v \mid x_{<t}) \ln \frac{P(v \mid x_{<t})}{Q(v \mid x_{<t})}$$

1. The vast majority of tokens in natural text have low entropy (predictable syntax, stopwords, common nouns). On these tokens, $P \approx Q$, so $D_{\text{KL}} \approx 0$. Token-averaged KL is therefore heavily dominated by easy, high-probability tokens.
2. Conversely, autoregressive perplexity $\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \ln Q(x_t \mid x_{<t})\right)$ is dominated by rare catastrophic predictions where $Q(x_t \mid x_{<t}) \to 0$. A model can maintain a low average KL across 98% of tokens while suffering explosive perplexity if even 2% of critical routing tokens are assigned near-zero probabilities.
3. Consequently, cheap token-averaged KL achieves only Spearman $\rho \approx 0.65$ correlation with true model perplexity, confirming that **autoregressive perplexity on full context sequences is the required ground-truth oracle** for evaluating combinatorial compression plans.
