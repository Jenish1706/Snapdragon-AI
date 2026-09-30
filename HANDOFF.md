# Ising-Edge: Engineering Handoff & Deployment Guide

**Snapdragon AI Challenge 2026**  
*Track: On-Device Generative AI Acceleration & Optimization*  
**Author:** Jenish Patel  
**Date:** 30 September 2026

---

## 1. Project Overview & Deliverables

The **Ising-Edge** repository is fully prepared, benchmarked, and verified for submission. All experimental evaluations have completed with 100% reproducibility across hardware architectures. Key empirical phenomena (such as deep tensor network scaling and super-additive error compounding) have been thoroughly investigated, theoretically derived via residual Jacobian analysis, and unified in our framework.

### Master Documents
* **[docs/PROPOSAL.md](file:///e:/Snapdragon%20AI/docs/PROPOSAL.md):** The comprehensive, publication-grade technical proposal detailing the problem statement, SnapTwin performance model, joint QUBO formulation, empirical discoveries, and algorithmic benchmarks.
* **[docs/SUBMISSION.md](file:///e:/Snapdragon%20AI/docs/SUBMISSION.md):** The official submission pack containing the 250-word executive abstract, core quantitative findings, reproduction guide, alignment with judging criteria, and submission checklist.
* **[docs/RESULTS.md](file:///e:/Snapdragon%20AI/docs/RESULTS.md):** The consolidated empirical benchmark ledger detailing baseline performance, single-layer sensitivity, multi-layer compounding, matched-bytes plan comparisons, tensor network scaling, and combinatorial solver benchmarks.
* **[docs/research/SYNTHESIS.md](file:///e:/Snapdragon%20AI/docs/research/SYNTHESIS.md):** In-depth synthesis of optimization directions, closed-loop surrogate planning, mathematical derivations of residual error compounding, and hardware placement.

---

## 2. Submission Protocol (Unstop)

The submission deadline is **30 September 2026, 23:59 IST**.

### Immediate Submission Actions:
1. **Primary Text:** The proposal text in [docs/PROPOSAL.md](file:///e:/Snapdragon%20AI/docs/PROPOSAL.md) provides the primary submission material.
2. **Abstract:** The 250-word abstract is formatted and verified in [docs/SUBMISSION.md](file:///e:/Snapdragon%20AI/docs/SUBMISSION.md).
3. **Byline Confirmation:** Author name `Jenish Patel` is verified in the proposal and submission pack.
4. **Compliance:** Complete the submission checklist items in [docs/SUBMISSION.md](file:///e:/Snapdragon%20AI/docs/SUBMISSION.md).

---

## 3. Verification & Reproduction

The repository includes a self-contained test suite that validates all core components without requiring external proprietary hardware or network connectivity:

```bash
# Run the verified test suite (56 tests across 11 test modules)
pytest tests/ -q

# Verify empirical results against documentation
python scripts/check_numbers.py

# Evaluate matched-bytes plans
python scripts/matched_bytes.py
```

All unit test suites pass cleanly (54 passed, 2 skipped due to optional onnxruntime dependency, 0 failures).

---

## 4. Hardware Deployment Architecture

Ising-Edge is designed for direct edge execution on Snapdragon mobile platforms and Copilot+ PCs:
- **SnapTwin Roofline Profiler:** Decomposes prefill and decode across Kryo/Oryon CPU, Adreno GPU, and Hexagon NPU.
- **Native Runtime Integration:** Execution plans map directly to standard, unmodified `llama.cpp` CLI flags (`-dev`, `-ot`, `-ctk`, `-ctv`).
- **Physical Device Pipeline:** Connecting an Android Snapdragon device (e.g. OnePlus Pad 4 or Snapdragon reference device) over ADB enables instant live validation using prebuilt Snapdragon QNN and OpenCL backends.
