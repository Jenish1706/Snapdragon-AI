# Red team: PROPOSAL.md against SYNTHESIS.md

Written 2026-09-29. Three lists: claims that are weaker than we thought, steps the proposal omits, and costs it underestimates. Each item names the research note (and finding, F#) that carries the evidence. Use this list to correct the proposal before the 30 Sep 2026 submission and as the deviation record for `docs/PLAN.md`.

## A. Claims that are weaker than we thought

| # | Proposal claim (section) | What the evidence says | Source |
| --- | --- | --- | --- |
| A1 | "228 GB/s bandwidth for the X2 Elite family"; "caps decode near 150 tokens/s" (Problem) | 228 GB/s is the X2 Elite *Extreme* only; X2 Elite is 152 GB/s, X Elite 135 GB/s; HP OmniBooks ship X1E-78-100 or X2E-84/90-100. Ceilings at 4K: 65 / 73 / 110 tok/s. | npu-decode-bottleneck F11; baselines-and-metrics F21–F22 |
| A2 | "up to 85 TOPS" for the X2 Elite family (Problem) | 85 TOPS only on X2E-90/84-100; X2E-88-100 is 80; X Elite is 45. | npu-decode-bottleneck F11 |
| A3 | "about 1.5 GB per generated token" (Problem) | 1.61 GB at 4.0 bpw including the tied 394M lm_head; 1.81 GB at Q4_0's 4.5 bpw. | npu-decode-bottleneck arithmetic |
| A4 | Decode is "limited by memory bandwidth" on the NPU (Problem) | NPU decode on X Elite runs at 13–24% of the bandwidth ceiling; the stall is dispatch, sync and the preallocated context. Bandwidth-bound holds for CPU (~66% of 135 GB/s) and GPU. | npu-decode-bottleneck F5, F6, F10, F11 |
| A5 | "pays a fixed invocation cost per token" attributed to arXiv 2607.05475 (Problem) | That paper shows a context-size-dependent penalty (−20 to −24% from 16 to 4K tokens); the fixed dispatch cost is in arXiv 2605.27435 and HeteroInfer. | npu-decode-bottleneck Contradiction 3 |
| A6 | Bond dimension r ∈ {8, 16, 32, 64} (Proposed solution, Steps 1–2) | Retains 0.4–11% of each matrix; every accuracy-preserving result keeps ≥ 60%; Llama-3.2-3B at 40% retention gives PPL 56–762 without healing. Grid must be keep-fractions. | tn-compression-recipes F5, F11–F15, Table 3 |
| A7 | Tensor networks as the compression method of choice (Summary, Novelty 2) | "Tensor formats fail to outperform their matrix counterparts at matched compression ratios"; RTN 4/8-bit dominates; a 3-core MPO at χ = 64 needs 8–22× dense MACs. | tn-compression-recipes F5, F13, F17; qnn-op-support F14 |
| A8 | "CompactifAI/KARIPAP report GPU results" as two data points (Novelty) | KARIPAP's Table 1 and headline numbers reproduce CompactifAI's; not independent. | tn-compression-recipes F3 |
| A9 | "First evaluation of quantum-inspired TN compression on the Hexagon NPU" (Novelty 2) | INT4 TT cores were already run on an Ascend 310 NPU and an FPGA; say "first on Hexagon". | tn-compression-recipes F6, F7 |
| A10 | "the Hexagon NPU runs the same linear layer up to 1.62× faster in one equivalent shape" as justification for core-shape factorisations (Proposed solution) | The 1.62× is an activation-layout effect for an unchanged 2048² weight on 8 Gen 3/8 Gen 2 phones; it does not evidence MPO core shapes. | qnn-op-support F13, Contradiction 3 |
| A11 | "Express each core contraction as plain MatMul + Reshape + Transpose, the ops the Hexagon compiler already fuses" (Risks) | No Qualcomm document describes such fusion; Transpose is a real HTP op with rank ≤ 5 and restricted 5-D dtypes; FullyConnected rank ≤ 4. | qnn-op-support F2, F10, Contradiction 1 |
| A12 | "QNN/AI Hub compiler rejects MPO graphs (einsum, odd reshapes)" (Risks) | Einsum is accepted by decomposition; the real triggers are rank ≥ 5/6 intermediates, Gemm with transB = 0, dynamic Expand, bad quant-axis/encoding combinations. | qnn-op-support F1, F2, Contradiction 2 |
| A13 | Bits "per KV page 4, 8 or 16" (Proposed solution) | KV type is one per context in llama.cpp; Hexagon flash-attention accepts only F16/Q8_0 K/V; AI Hub bundles bake int8 KV; Qualcomm's native KV cache requires uint8. Per-page bits are not honourable anywhere today. | kv-and-memory F10–F11, F15; hexagon-runtime-status F7; qnn-op-support Contradiction 5 |
| A14 | Placement "per operator and regime", z over ~340 operators (Proposed solution, Formulation) | llama.cpp places weight tensors; non-weight ops follow their inputs; placement is fixed at load; the QNN path gives no placement control. | hexagon-runtime-status F9, F10, F15, F16 |
| A15 | "every cost is at most pairwise" (Formulation) | Ω squares a bilinear expression (bytes = rank × bits): degree 4. | qubo-solvers F17, Contradiction 1 |
| A16 | "the one stated approximation" (cubic rank × bits × unit dropped) (Formulation) | Unnecessary: a joint per-matrix one-hot v(w,r,b) makes every coupling exact and pairwise. | qubo-solvers F17 |
| A17 | T and T′ as two measured latencies (Formulation) | Each op's base latency is counted twice; the bytes term re-counts memory time inside a measured decode latency. | npu-decode-bottleneck F14, Contradiction 8 |
| A18 | "solves this size in milliseconds to seconds"; "re-plans in milliseconds" (Formulation, Proposed solution) | Dense SB at defaults: ~60 s laptop CPU, ~4 s discrete GPU; milliseconds only for the placement chain by exact DP; NPU re-placement needs a new compiled graph (seconds). | qubo-solvers F18; npu-decode-bottleneck F6–F8 |
| A19 | SB is "deterministic"; "the same profile always yields the same plan" (Summary, Formulation, Novelty 4) | Random initialisation; reproducible only with a pinned seed, device and torch version; seeded SA is equally deterministic. | qubo-solvers F3 |
| A20 | GSB and TESB "as drop-in upgrades" (Formulation) | TESB is an MIT standalone script with its own integrator; GSB has no licence file. | qubo-solvers F8, F9 |
| A21 | `dwave-neal` (Steps 3–4) | Deprecated in favour of `dwave-samplers`. | qubo-solvers F10 |
| A22 | "TT-SVD ... with tntorch" (Step 2) | tntorch has no TT-matrix/MPO type; tensorkrowch cannot take per-bond ranks; use tensorly `tensor_train_matrix` or a hand-written chain. | tn-compression-recipes F19, Contradiction 3 |
| A23 | "Optional 100-step LoRA healing per Minima's recipe" (Step 2) | Minima updates the cores for "a few thousand steps"; the 100-step LoRA is Zagitov et al.; the 40–100-step factor refinement is L3. | tn-compression-recipes F2, F5, F13, Contradiction 2 |
| A24 | "Compression must not cost accuracy" (Evaluation) | No post-training result on a 3B/1.7B model at ≥ 15% removal stays within 4 points 0-shot without fine-tuning. | tn-compression-recipes F14, Contradiction 6 |
| A25 | "GenieX bundle for NPU parts and GGUF for CPU/GPU parts; a thin scheduler drives GenieX's serve mode" (Step 5) | GenieX cannot pass activations between its two runtimes, exposes no per-tensor placement, and reloads on `compute` change. | hexagon-runtime-status F2–F5, F14 |
| A26 | Baseline 2 "GenieX's per-tensor scheduler across NPU/GPU/CPU" (Evaluation) | `hybrid` is HTP + CPU; GPU is a separate device list that no GenieX doc tests; the default alias is `npu`. | hexagon-runtime-status F1, F13 |
| A27 | Baseline 3 "HeteroInfer-style placement (matrix multiplies on NPU, normalisation on CPU)" (Evaluation) | HeteroInfer puts norms/SwiGLU on the GPU and makes the GPU primary in decode; its 6.02× is end-to-end, decode-only gains are 1.5–2.5×; GenieX cannot express the rules at all. | npu-decode-bottleneck F6, Contradiction 4; baselines-and-metrics F25 |
| A28 | Baseline 1 at 32K context; "Peak RAM at 32K" for NPU-only rows (Evaluation) | AI Hub bundle context is fixed at export (4096); F16 KV at 32K alone fills a ~3.5 GB HTP session. | kv-and-memory F12, F15; baselines-and-metrics F26 |
| A29 | "Llama-3.2-3B-Instruct (published on Qualcomm AI Hub for Snapdragon X targets)" (Technical plan) | Listed with numbers but not distributed as pre-compiled assets; must be exported with gated Meta weights and QAIRT. | baselines-and-metrics F7 |
| A30 | HeteroMosaic and PELM as Snapdragon evidence (References) | AMD Ryzen AI (XDNA) and NVIDIA Jetson respectively. | npu-decode-bottleneck F13 |
| A31 | "the 4 GiB HTP address-space investigation" (References) | ~3.5 GB window; 3000 MiB default on v73 (X Elite), 3200 MiB on v75+. | hexagon-runtime-status F8 |
| A32 | KVTC "up to 20× compression" listed with decode-time KV methods (References, Novelty 3) | KVTC decodes on decompressed caches; it reduces stored/transferred bytes, not bytes per decoded token. | kv-and-memory F4 |
| A33 | "Palu-style low-rank latent KV pages and page-wise ranks (PuzzleKV)" (Phase 2) | Palu's rank is per layer/head-group in the weights; PuzzleKV uses fixed uniform ranks with a runtime SVD; neither has a Hexagon kernel; JoLT's fused decode is 7.5–65.9× slower at batch 1. | kv-and-memory F1, F3, F6 |
| A34 | Lead's own earlier note that "Genie support will be deprecated soon" | Not found in ai-hub-apps at 2026-09-16 nor on the rendered page; unverified. | baselines-and-metrics F7 |

## B. Missing steps

| # | Missing step | Why it matters | Source |
| --- | --- | --- | --- |
| B1 | A plan for device access: no hosted device runs llama.cpp, so Path A's T tables and every mixed-placement end-to-end run need a physical Snapdragon PC. | Decides Path A vs B and whether Milestone B's 10% check can be run before week 4. | hexagon-runtime-status F16; PLAN.md §3 |
| B2 | The runtime fork itself: choose llama.cpp (`llama-server` + `-ot`/`--tensor-type`/`-ctk`) or ORT/QNN segment chaining. | Step 5 as written is not implementable. | SYNTHESIS §1 |
| B3 | A GenieX patch (or upstream `llama-server` harness) to consume per-tensor overrides. | GenieX has no field for them. | hexagon-runtime-status F2, F14 |
| B4 | Week-1 v73 smoke test: `test-backend-ops` for MUL_MAT / FLASH_ATTN_EXT / GATED_DELTA_NET with HMX on and off; `GGML_HEXAGON_VMEM=0` probe. | Issue #29473 may disable HMX on X Elite; the session window bounds NPU placement. | hexagon-runtime-status F8, F11 |
| B5 | Windows HTP prerequisites on the demo laptop: NPU driver, self-signed certificate, `bcdedit /set TESTSIGNING ON`, possibly Secure Boot off. | Blocks the llama.cpp NPU path and weakens the accessibility story unless disclosed. | baselines-and-metrics F11; hexagon-runtime-status F6 |
| B6 | Gated Llama export (`HF_TOKEN` with Llama access, `qai_hub_models ... export --chipset ...`, ≥ 12 GB RAM). | Baseline 1 and the NPU bundle depend on it; "one-command reproduction" needs the caveat. | baselines-and-metrics F7 |
| B7 | A calibration compute plan: 256 × 2048-token whitening and sequential-prefix activations require memory management with resident fp16 models. | Layer-wise streaming or GPU execution is required. | tn-compression-recipes F11, Open risks; SYNTHESIS |
| B8 | Joint measurement of ΔL(rank, bits) with fake-quantised factors, and per-matrix per-layer sensitivity (sources disagree on which layers are fragile). | INT4 on factors costs ~4× what it costs on dense weights. | tn-compression-recipes F12, F14, Open risks |
| B9 | A healing budget and recipe (Zagitov LoRA 100 steps; L3 40–100 steps) with attribution. | The proposal's "100-step LoRA per Minima" is mis-attributed and under-budgeted. | tn-compression-recipes F5, F13 |
| B10 | Terms in E for KV bytes, per-segment NPU invocation, static-shape chunking, preallocated-context penalty, energy, per-session cap; T′ as deltas. | Without them the plan cannot predict NPU decode within 10%. | npu-decode-bottleneck F14 |
| B11 | The joint one-hot v(w,r,b) and a linear budget; penalty-weight recipe; feasibility filter and repair for SB outputs; plan cache keyed by hash(Q). | Ω as written is quartic; SB emits infeasible agents. | qubo-solvers F17, Recommendation |
| B12 | The runtime re-planner as an exact chain DP over precompiled graph variants. | SB is too slow for per-step use and NPU graphs cannot be rebuilt per step. | qubo-solvers F18; npu-decode-bottleneck F6–F8 |
| B13 | The profiler's validation loop: compile → inference job with profiling → `execution_detail[*].compute_unit` → op-by-op and QHAS artifacts; `session.disable_cpu_ep_fallback=1` on ORT. | The per-layer time key is unverified; silent CPU fallback would poison the tables. | qnn-op-support F8, F11 |
| B14 | The ONNX pattern for MPO cores (FC chains, axis merging, rank ≤ 4) and the W4/W8 + A16 quantisation choice. | Naive contractions are rejected; FP16 activations are 2.3–2.6× slower. | qnn-op-support F2–F4, Recommendation |
| B15 | Prefill micro-graphs at chunk 128 (AI Hub's prompt processor), not only 256; decode at sequence 1. | Prefill cost is ceil(prompt/C)·T_C. | npu-decode-bottleneck F10, Contradiction 6 |
| B16 | An energy sampler (EMI rails via `npu-watt` or an IOCTL client) and a measurement protocol (warm-up, repetitions, medians, thermal settle, power mode, AC/DC, SKU, runtime version in every row). | "Device power counters" is otherwise unspecified. | baselines-and-metrics F13–F20, F27 |
| B17 | Fixed prompt-id files (1,024 / 8,192 tokens) and decode depths 4,096 / 32,768; RULER NIAH at 4K–32K to validate the 32K row. | Comparable, reproducible rows. | baselines-and-metrics F27, F35 |
| B18 | Concrete agent-suite sources (BFCL, AgentDojo, LongBench, NarrativeQA) and scoring (pass^3, J per completed task). | "Twenty local tasks" had no provenance or scoring rule. | baselines-and-metrics F28–F35 |
| B19 | AI Hub job caching and a job budget (first pass ≈ 224 jobs). | Quotas and queue time. | PLAN.md §5 |
| B20 | Complete verified empirical baseline and error quantification across hardware engines. | Schedule and execution rigor. | SYNTHESIS |

## C. Costs we underestimated

| # | Cost | Proposal's implicit estimate | Evidence-based estimate | Source |
| --- | --- | --- | --- | --- |
| C1 | Solver wall time | milliseconds | 4–60 s per offline solve on a laptop; ≤ 20 s with 32 agents × 3,000 steps; runtime DP in µs | qubo-solvers F18 |
| C2 | Sensitivity calibration | 256 sequences per (layer, rank, bits) | 196 matrices × 5 keeps × 2 bits with whitening on 256 × 2048 tokens: hours of streamed activations on standard hardware; use per-matrix reconstruction error as the proxy and validate a handful of plans end-to-end | tn-compression-recipes F11, F19; SYNTHESIS |
| C3 | Healing | "optional 100-step LoRA" | 100-step LoRA recovers only part of the loss (PPL 26 → 13 vs dense 8.9 in Zagitov); CompactifAI-class recovery is < 1 epoch on 8 GPUs; L3 refinement ~1 h per plan | tn-compression-recipes F5, F13, Open risks |
| C4 | Compute of tensor-network options | bytes down ⇒ time down | 3-core MPO at χ = 64: 8–22× dense MACs; two-GEMM low rank decodes at 0.68–0.94× dense on GPUs; on the NPU each extra op pays dispatch | tn-compression-recipes F13, F17; qnn-op-support F14; npu-decode-bottleneck F5 |
| C5 | Unit-switch overhead | one S term | 400 µs sync, 80–100 µs sleep floor, 730 RPC calls/token, 1–1.5× fallback cost, 10–13% of decode; plus a graph rebuild of seconds when NPU placement changes | npu-decode-bottleneck F5–F8, F12 |
| C6 | RAM for two regime plans | free | two loaded contexts = 2× weights on a 16 GB memory footprint, or a multi-second reload | hexagon-runtime-status F15, Open risks |
| C7 | NPU memory | "4 GiB" | ~3.5 GB window, 3000 MiB default on v73; 3B at Q8_0 or 32K F16 KV each exceed one session | hexagon-runtime-status F8; kv-and-memory F12 |
| C8 | Long-context decode | bytes = weights | at 32K, KV is 70% of the bytes for the 3B; 4-bit KV is worth more than any weight-side choice, and is CPU/GPU-only | npu-decode-bottleneck arithmetic; kv-and-memory F11 |
| C9 | AI Hub profiling | "compile and profile jobs" | ~224 jobs for the first pass, each a compile + inference job; per-layer timing needs artifact parsing; K-quants and FP16 need extra control runs | qnn-op-support F7, F8; PLAN.md §5 |
| C10 | Windows NPU setup | none | driver + certificate + test signing (+ Secure Boot) for the llama.cpp path; QAIRT runtime pinned per process | baselines-and-metrics F11; hexagon-runtime-status F5, F6 |
| C11 | Energy measurement | "device power counters" | EMI sampler at 250 ms with idle subtraction and ABBA ordering; battery runs for J/task; 12 h cap on non-Pro HWiNFO | baselines-and-metrics F14, F18, F20 |
| C12 | Accuracy loss | "must not cost accuracy" | −3 to −5 pp 0-shot at keep 0.85 without healing on our models | tn-compression-recipes F14 |
| C13 | Scheduling overhead | unconstrained | empirical sweeps require staged evaluation pipelines | SYNTHESIS |
| C14 | Model memory footprint | standard RAM | 3B models in full fp16 require sequential chunk streaming | SYNTHESIS |
