# Ising-Edge: Engineering Architecture & Technical Plan

**Snapdragon AI Challenge 2026**  
*Track: On-Device Generative AI Acceleration & Optimization*  
**Author:** Jenish Patel  

---

## 1. Executive Architecture

Ising-Edge plans and optimizes on-device Small Language Model (SLM) execution across heterogeneous Snapdragon System-on-Chips (Kryo/Oryon CPU, Adreno GPU, and Hexagon NPU). The framework jointly determines:
1. **Compute Engine Placement:** Per-weight-tensor offloading across CPU, GPU, and NPU.
2. **Structural Rank Compression:** Activation-aware low-rank SVD and Matrix Product Operator (MPO) tensor network factorizations.
3. **Quantization Precision:** Per-tensor mixed precision ($\text{Q4\_0}, \text{Q8\_0}$) and KV-cache bit-widths.

The optimization is framed as a single **Quadratic Unconstrained Binary Optimization (QUBO) / Ising Hamiltonian**, capturing both linear execution latency and non-linear, super-additive cross-layer accuracy degradation.

---

## 2. System Pipeline

The architecture is structured into modular offline profiling, combinatorial planning, and runtime execution stages:

| Stage | Module | Input & Output |
| :--- | :--- | :--- |
| **1. Profile** | `isingedge/profile` | Static-shape micro-graphs $\to$ hardware latency and bandwidth roofline profiles across CPU, GPU, and NPU. |
| **2. Sensitivity** | `isingedge/sensitivity` | Activation-aware SVD / 3-core MPO decompositions $\to$ layer-wise and pairwise degradation ledgers ($\Delta\text{PPL}$). |
| **3. Performance Model** | `isingedge/twin` | SnapTwin analytical latency predictor fitted on published and measured Snapdragon SoC benchmarks. |
| **4. Problem Build** | `isingedge/qubo` | Latency tables + empirical interaction matrices + hardware memory budgets $\to$ Ising Hamiltonian matrix $Q$ and linear vector $c$. |
| **5. Combinatorial Solve** | `isingedge/solve` | Exact CP-SAT branch-and-cut and discrete 1-swap manifold descent $\to$ global optimal allocation plans. |
| **6. Active Surrogate Loop** | `isingedge/loop` | Rank-6 Factorization Machine surrogate with iterative acquisition over combinatorial candidate spaces. |
| **7. Runtime Deployment** | `isingedge/runtime` | Optimal execution plans $\to$ native `llama.cpp` CLI flags (`-dev`, `-ot`, `-ctk`, `-ctv`). |

---

## 3. Mathematical Formulation & Cost Model

### 3.1 Decision Variables
Let $x_{l, u, q, r} \in \{0, 1\}$ denote the binary indicator variable for layer $l \in \{1, \dots, L\}$, assigned to compute unit $u \in \{\text{CPU}, \text{GPU}, \text{NPU}\}$, quantization precision $q \in \{\text{Q4\_0}, \text{Q8\_0}\}$, and compression keep-fraction $r \in \{1.0, 0.9, 0.8, 0.7, 0.6\}$:
- $\sum_{u, q, r} x_{l, u, q, r} = 1 \quad \forall l$ (One-hot per-layer assignment).
- $y_b \in \{0, 1\}$ denotes the global KV-cache type ($f16, q8\_0, q4\_0$).

### 3.2 Global Objective Function
The objective minimizes total execution latency and perplexity penalty under memory constraints:
$$\mathcal{H}(x) = \sum_{i} c_i x_i + \sum_{i < j} Q_{ij} x_i x_j$$

where:
1. **Linear Costs ($c_i$):**
   $$c_i = w_P \hat{T}_{\text{prefill}}(i) + w_D \hat{T}_{\text{decode}}(i) + \lambda_A \Delta\text{PPL}_i$$
   combining regime-weighted latency from SnapTwin with isolated layer sensitivity.
2. **Pairwise Quadratic Coupling ($Q_{ij}$):**
   $$Q_{ij} = C_{\text{sync}}(u_i, u_j) + \lambda_{\text{inter}} J_{ij}$$
   incorporating inter-processor memory barrier synchronization overheads and empirical cross-layer accuracy coupling coefficients $J_{ij}$.
3. **Hard Hardware Constraints:**
   $$\sum_{i} \text{Bytes}(i) x_i + \text{KVBytes}(y) \le M_{\text{budget}}$$

---

## 4. Combinatorial Solvers & Acquisition

1. **Exact Global Optimization (Linear Chain):** Dynamic programming solves linear placement in $O(L \cdot |U|^2)$ time.
2. **Exact Branch-and-Cut (OR-Tools CP-SAT):** Solves linearized QUBO representations with hard linear constraints, verifying global optimality gaps ($0.000\%$).
3. **Discrete 1-Swap Manifold Descent:** Operates directly on the feasible manifold $\mathcal{S}_K = \{x \in \{0, 1\}^n \mid \sum x_i = K\}$. Using $O(1)$ amortized differential updates:
   $$\Delta E_{i \to j} = (c_j - c_i) + 2 \sum_{k \neq i, j} (Q_{jk} - Q_{ik}) x_k - 2 Q_{ij}$$
   descent reaches exact global optima in under $5\text{ ms}$, bypassing the penalty-barrier warping that degrades continuous physics heuristics (Simulated Bifurcation and Simulated Annealing).
4. **Active Closed-Loop Exploration:** Rank-6 Factorization Machine surrogates fit standardized logarithmic perplexity ratios $\ln(\text{PPL}/\text{PPL}_{\text{dense}})$, iteratively evaluating candidate plans and discovering superior solutions outside the one-shot empirical frontier.

---

## 5. Runtime Deployment Architecture

Ising-Edge requires zero modifications to host kernel drivers, emitting standard, production-ready flags for native `llama.cpp` inference:
- **Engine Placement:** `-dev HTP0,GPUOpenCL,none`
- **Tensor Offloading:** `-ot "blk.[0-12].*=HTP0" -ot "blk.[13-27].*=none"`
- **KV Cache Precision:** `-ctk q8_0 -ctv q4_0`

This enables instantaneous cross-platform execution on Snapdragon X Elite Copilot+ PCs and Snapdragon mobile platforms over ADB.
