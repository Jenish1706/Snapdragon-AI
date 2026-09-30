# Stage-split placement (prefill on NPU, decode on GPU or CPU) on Snapdragon: prior work and llama.cpp b11256

Scope. This note tests candidate direction D3: run prefill on the Hexagon NPU and decode on the Adreno GPU or the CPU. It covers four things. (1) HeteroInfer (SOSP '25). (2) Two 2026 NPU measurement studies, arXiv 2605.27435 and arXiv 2607.05475. (3) The llama.cpp Hexagon backend and scheduler at our pinned release b11256 (commit `c85b92c`, 29 Sep 2026) and the Hexagon commits after it. (4) Other 2024–2026 systems that put prefill and decode on different units. All sources were read on 30 Sep 2026 through arXiv HTML, GitHub pages, the GitHub API and raw source at the named tag or commit. No device was run for this note. Every number below is the source's own; ratios marked "(arithmetic)" are ours. The repo's `hexagon-runtime-status.md` is missing from this copy, so the runtime facts are collected in "Deltas" below.

## Findings

1. **Seed check.** All five seed links exist. arXiv 2501.14794 is HeteroInfer. Its v2 (4 Oct 2025) is titled "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference", and the HTML carries the SOSP '25 header (DOI 10.1145/3731569.3764808). v1 (11 Jan 2025) was titled "HeteroLLM: …" and claimed 9.99× over MLC and 4.36× over MNN. arXiv 2605.27435 and 2607.05475 say what we cite, but both measured llama.cpp builds from before HMX support (finding 13). PR #26501 adds an asynchronous, multi-NPU Hexagon backend and has nothing to do with placement. Issue #29473 is an HMX correctness bug on a v73 7-series SoC, not a placement issue (findings 15–16). [verified]
2. **HeteroInfer: units and SoCs.** Only the GPU (the authors' own OpenCL kernels) and the NPU (QNN, FP16 compute on W4A16 weights) compute. The CPU is a control plane only, handling synchronisation and GPU kernel submission. Devices: Snapdragon 8 Gen 3 (all numbers unless noted) and Snapdragon 8 Elite. Batch size 1. Models: Llama-8B/7B/3B and InternLM-1.8B (§5.1). [verified]
3. **HeteroInfer: what is partitioned.** Stages and operators: in prefill, Matmul goes to the NPU and RMSNorm/SwiGLU to the GPU; in decode the GPU is the primary unit (§4.1, "layer-level"). Tensors, three ways (§4.2.1–4.2.3):
   - weight-centric: a row split of the weight for static shapes;
   - activation-centric: a split of the sequence dimension, where the NPU runs a pre-built standard length and the GPU runs the dynamic remainder;
   - hybrid: both at once.

   An offline profiler (under 20 min) and a solver minimise max(T_GPU, T_NPU) + T_sync + T_copy per operator, and the engine picks a strategy per sequence length at run time (§4.4). It does not pipeline layers across units, and it does not choose precision or compression. [verified]
4. **HeteroInfer: per-operator numbers that bear on D3** (Table 3, 8 Gen 3, µs). Decode mat-vec:
   - [4096,4096]: GPU 511, NPU 693, split 1:1;
   - [28672,4096]: GPU 1903, NPU 3886, split 3:1;
   - [4096,14336]: GPU 1467, NPU 6506, GPU only.

   Prefill [4096,4096]×128: GPU 7306, NPU 912, NPU only. [verified]
5. **HeteroInfer: speedups.**
   - End to end (§5.2, Table 4): from 1.34× (vs PowerInfer-2 with a sparse model, decode-heavy multi-turn dialogue, Llama-7B) to 6.02× (vs MNN-OpenCL, prefill-dominant LongBench).
   - Prefill at 256 tokens on Llama-8B (§5.3.1): Hetero-layer is 5.85×, 5.64× and 24.9× faster than MNN-OpenCL, MLC and llama.cpp, and 3.29× faster than PowerInfer-2-FP16. Hetero-tensor adds 30.2% on average and reaches 247.9 tok/s on Llama-8B and 1,092 tok/s on InternLM-1.8B.
   - Decode (§5.4): Hetero-tensor reaches 14.01 tok/s on Llama-8B: 1.50×, 2.53× and 1.52× over MNN-OpenCL, llama.cpp and MLC, 1.32× over PowerInfer-2, and 22.0% over Hetero-layer. The gain comes from bandwidth rising from 43.3 GB/s (GPU only) to 59.5 GB/s (GPU+NPU).
   - 8 Elite: about 10.5% more prefill than 8 Gen 3; decode is similar because that device had the same bandwidth (§5.3.1, §5.4). [verified]
6. **HeteroInfer: synchronisation.** `clFinish` costs a fixed ~400 µs (§3.1, GPU-2). The fix combines shared buffers with a predicted sleep followed by polling. It gives 15.8% (Hetero-layer) and 24.3% (Hetero-tensor) more prefill speed, and 2.2–4.01× more decode speed (§4.3, §5.5). [verified]
7. **HeteroInfer: code.** The paper has no repository link and no availability statement. A GitHub repository search for HeteroInfer/HeteroLLM (30 Sep 2026) found no SJTU/IPADS repository. Treat it as closed source. [inferred]
8. **arXiv 2605.27435 ("When NPUs Are Not Always Faster"): setup** (§III).
   - Device: one Snapdragon 8 Gen 3 phone (Hexagon v75, Android 15), 6 CPU threads.
   - Runtime: llama.cpp tag `b7588` with the Hexagon backend.
   - Units: CPU vs NPU only; no GPU.
   - Models: Q4_0 Llama-3.2-3B, Llama-3.1-8B, Qwen3-4B, Qwen3-8B.
   - Workload: prompt about 1,281 tokens, batch 128. The NPU share is set with `-ngl`. [verified]
9. **arXiv 2605.27435: results.**
   - Prefill: CPU-only beats every NPU offload level at every prompt length, while offloading more layers to the NPU speeds up decode (§IV-A, Fig. 2).
   - Per-call MUL_MAT: the NPU is 1.27–1.62× slower than the CPU in prefill and 1.55–1.67× faster in decode (Table III).
   - End-to-end decode gains only 1.05–1.20× (Table V). Two reasons: light ops pay 8–22× call-vs-compute overhead, and FLASH_ATTN_EXT falls back to the CPU (§IV-D).
   - NPU communication takes 0.2–3.2% of NPU time in prefill and 9.9–13.0% in decode (Table IV).
   - Energy: full offload raises battery drain by 22%, 32% and 51% at ~321-, 641- and 1,281-token prompts (§IV-E).
   - No model size or prompt length makes the NPU win prefill. Guideline G1 recommends CPU prefill and NPU decode, the reverse of D3 (§IV-F). [verified]
10. **arXiv 2607.05475 ("Is Your NPU Ready for LLMs?"): setup.**
    - Phones (Table 4): Xiaomi 17 and OnePlus 15 (SM8850, NPU V81), Xiaomi 15 (SM8750, V79), Xiaomi 14 (SM8650, V75), all with 16 GB.
    - Frameworks (Table 3): llama.cpp at commit `eadc418` (custom Hexagon kernels), MNN, MLC-LLM, MLLM, GENIE 2.39.0.
    - Workload (Table 5): 256-token prefill and decode on Qwen2.5-1.5B/7B and Llama3.2-1B/3B with 4-bit weights. [verified]
11. **arXiv 2607.05475: where the NPU loses.**
    - Prefill: the NPU wins only through QNN-based stacks, e.g. GENIE at 1,463.7 tok/s on OnePlus 15 with Qwen2.5-1.5B.
    - Decode: the CPU is usually best and the GPU nearly equivalent (§6.2). On OnePlus 15, Qwen2.5-1.5B decodes at 55.7 tok/s (llama.cpp CPU), 50.3 (llama.cpp GPU) and 23.0 (GENIE NPU) (Table 5).
    - Size exception: Qwen2.5-7B with GENIE W4A16 decodes at 16.8 tok/s vs 14.3 on the CPU (OnePlus 15, Table 5).
    - Context: QNN NPU decode falls 20.0–23.8% as context grows from 16 to 4,096 tokens, whatever the output length (§6.1, Fig. 7).
    - Kernels: at matrix width N = 256, QNN needs 26.9–30.7× fewer MatMul cycles than llama.cpp's Hexagon kernels; at N ≤ 4 they are comparable (§5.2, Fig. 3).
    - The paper does not build a stage split. "A runtime that dispatches prefill to the NPU and decode to the CPU" is listed as future work (§8). [verified]
12. **arXiv 2607.05475 on the Pad 4's SoC** (OnePlus 15 column, SM8850, Table 5). llama.cpp @`eadc418` is slowest on the NPU in both phases for Qwen2.5-1.5B, and the same ordering holds for Llama3.2-1B/3B and Qwen2.5-7B:
    - prefill: 115.1 (NPU), 299.9 (CPU), 754.8 (GPU) tok/s;
    - decode: 33.3 (NPU), 55.7 (CPU), 50.3 (GPU) tok/s.

    754.8 / 115.1 = 6.6× GPU over NPU prefill (arithmetic). §5.2 of the text gives 22.8 tok/s for the same llama.cpp NPU decode cell, which conflicts with Table 5's 33.3. [verified]
13. **Both 2026 studies predate HMX in llama.cpp.** Per the GitHub API, `b7588` is dated 2025-12-31 and `eadc418` 2026-01-05. HMX matmul landed in PR #20693 on 2026-03-19. More than 20 merged Hexagon PRs that mention HMX followed through Sep 2026 (GitHub search), among them HMX flash attention (#22347), 32×32 tiled repack (#24954) and K-quants (#28994). `ggml-hexagon.cpp` grew from 3,021 lines at `b7588` to 8,508 at `b11256`. [verified]
14. **So their llama.cpp NPU prefill verdicts do not carry over to b11256.** Two post-HMX reports point the other way:
    - issue #29473 (SM7750, v73, commit `86a24a1`): HMX pp512 at 1,749.4 tok/s vs CPU 216.3 for Llama-3.2-1B Q4_0, but with garbled output;
    - BigMoeOnEdge (v81 phone, llama.cpp 2026-09-26 base): 172 vs 22.2 tok/s at 1,418 tokens on a 35B MoE streamed from flash.

    Neither compares against the GPU on a dense model on SM8850. [inferred]
15. **PR #26501** ("hexagon: support for multi-NPU devices (IQ9, IQ10) and fully asynchronous backend"). Merged 2026-08-27 as `192067b`, so it is inside b11256 (`c85b92c`, 2026-09-29). It brings:
    - asynchronous graph compute, events and copies;
    - non-host buffers by default;
    - device names of the form `HTP<phys>:<virt>`;
    - tensor split through the meta backend;
    - new `scripts/snapdragon/build.py` and `run.py`.

    The description says nothing about prefill/decode splitting or `-ot`; a WebFetch summary that claimed otherwise was wrong. A maintainer reply in the thread (2026-09-18) says op dispatch uses a shared-memory FIFO (dspqueue) of fused op batches, and FastRPC only opens the session. [verified]
16. **Issue #29473** (open, filed 2026-09-26).
    - Symptom: on SM7750 (v73), HMX MUL_MAT returns inf for n ≥ 5 and FLASH_ATTN_EXT/GATED_DELTA_NET fail, so output is garbled. `GGML_HEXAGON_MM_SELECT=1` plus an op filter fixes the output but runs slower than the CPU.
    - Root cause (reporter, confirmed by the maintainer 2026-09-29): some lower-end v73 SKUs (4 MB VTCM) have no FP16 HMX. The maintainer's dual-v73 IQ9 runs clean.
    - Fix in progress: an integer-HMX PR (#29740) is open. With it the reporter measured pp512 at 172 tok/s vs 52 on the CPU for Qwen3.5-4B, at CPU-level perplexity.
    - Related: a similar HMX report on SM8850 (#25876, whisper.cpp) was closed as stale. By then the reporter had narrowed the remaining errors to ADD consuming HMX MUL_MAT output. [verified]
17. **`-ot` can pin a single weight tensor to HTP at b11256.**
    - Accepted names: the parser accepts each registered device's default buffer-type name (`common/arg.cpp` L252–284; `llama-bench.cpp` L941–957). For Hexagon that is the session name, `HTP0` when `GGML_HEXAGON_DEVICES` is unset. For Adreno it is `OpenCL`; the device itself is `GPUOpenCL`.
    - Loading: the loader regex-matches each tensor name and uses a non-CPU buffer type as given (`llama-model-loader.cpp` L1234–1257).
    - Scheduling: the scheduler runs the tensor's ops on the highest-priority backend that supports both that buffer type and the op (`ggml-backend.cpp` L888–908).
    - No CPU fallback: HTP's default buffer is non-host (L2359–2362) and HTP accepts only its own buffer types (L7813–7820), so the CPU cannot fall back to HTP-resident weights.
    - Weight types HTP multiplies: Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K, F16 and F32, within the VTCM budget (L5717–5797). [verified]
18. **No stage split inside one context.** Ops with weights run where the weights live. The only batch-size-dependent rule is op offload: ops whose weights sit in a host CPU buffer move to a higher-priority backend whose `offload_op()` returns true (`ggml-backend.cpp` L951–980, L644–651). Both Snapdragon accelerators set `offload_op = NULL` at b11256: Hexagon at L7834 and OpenCL at `ggml-opencl.cpp` L13010. Hexagon also had NULL at `b7588` (L2854) and still has it at its latest change `19e28a2770` (L7838). `common/arg.cpp` has no prefill or decode device option; `--spec-draft-device` is for a draft model. [verified]
19. **What llama.cpp does offer.**
    - (a) An automatic stage split inside the NPU: MUL_MAT uses HMX when the token rows M > `HTP_MM_HMX_MIN_NROWS` and HVX otherwise (L4672–4677, L4946–4955). The default `GGML_HEXAGON_MM_SELECT=2` tries HMX, then HVX, then CPU (L102).
    - (b) Op-type routing: `GGML_HEXAGON_OPFILTER` sends matching op types to the CPU or GPU (L7624–7627; Snapdragon README).
    - (c) A public API for two model loads with different devices and overrides (`llama.h` L319–324), and for moving a sequence's KV state between contexts with `llama_state_seq_get_data/set_data` (L882–901).
    - (d) Graph reuse: a graph with a repeated shape is reused without re-scheduling (`llama-context.cpp` L1410–1425); `LLAMA_GRAPH_REUSE_DISABLE` turns this off (L280–283). [verified]
20. **Two ways to do D3 in one process; neither is upstream.**
    - (i) Two contexts over two loads of the same GGUF: the prefill context keeps its weights on `HTP0`, the decode context on the CPU or `OpenCL`, and the KV state is copied between them. Cost: double weight memory and one state copy per prompt.
    - (ii) BigMoeOnEdge's per-graph rebinding of `tensor->buffer/data/extra`. It stays safe because prefill and decode graphs are kept to different widths, and the KV cache sits in the HTP host buffer (`GGML_HEXAGON_HOSTBUF=1`) so both phases share it. [inferred]
21. **llm.npu (ASPLOS '25, arXiv 2407.05858).** Prefill runs on the NPU (INT8, 256-token chunks, outliers on CPU/GPU); decode "utilizes the MLLM CPU backend" (§4). Phones: Snapdragon 8 Gen 3 and 8 Gen 2 (§4.1). Code is in UbiquitousLearning/mllm. This is D3 with CPU decode. Since Feb 2026 mllm also offers QNN AOT full-graph NPU execution (README). [verified]
22. **PowerInfer-2 (arXiv 2406.06282).** Prefill is NPU-centric (§4.1.1). Decode is hybrid: hot neurons on the NPU, sparse cold neurons on the CPU, and attention heads split between both (§4.1.2–4.1.3). Phones: OnePlus 12 (8 Gen 3) and Ace 2 (8+ Gen 1) (§7.1). The llm.npu paper (2024) says PowerInfer-2 is not open source. [verified]
23. **Phase Matters (arXiv 2606.27906, June 2026).** SM8750, LiteRT/QNN, FastVLM-0.5B. "Hybrid-PD" puts prefill on the NPU and the encoder and decode on the CPU. Latency (Table 4), warm / cold:
    - Hybrid-PD: 2,640 / 6,240 ms;
    - CPU-only: 3,412 / 4,180 ms;
    - NPU-only: 2,080 / 26,820 ms.

    Over the CPU, the NPU gives 1.64× on prefill and 1.18× on decode (§1). [verified]
24. **BigMoeOnEdge `--prefill-device`** (PR #200, merged 2026-09-28 as `10af539a636a`; doc at `3170385fad8e`).
    - Design: built on llama.cpp. Graphs at least `--prefill-min-tokens` wide (default 32) run on `HTP0`, and decode stays on the CPU. Placement needs no upstream patch; one hook commit handles expert streaming.
    - Measured on one 12 GB v81 phone, Qwen3.6-35B-A3B Q4_0 streamed from flash: prefill 172 vs 22.2 tok/s at 1,418 tokens (7.8×), 171 vs 18.0 tok/s at 1,921 tokens (9.5×), and parity at 121 tokens. Decode 3.25 vs 3.41 tok/s.
    - NPU decode was tried: it was 11% (adb) to 31% (app) slower, because each CPU–NPU crossing costs 0.3–0.5 ms.
    - This is D3 with CPU decode, in llama.cpp, on the Pad 4's NPU generation. [verified]
25. **GenieX "hybrid" is not a stage split.** qualcomm/GenieX `notes/run.md` (`ca874e4f18b3`) defines `hybrid` as llama.cpp's per-tensor HTP+CPU scheduler (empty `device_id`, ngl −1) and `npu` as pinned `HTP0`. On X1E80100 with Qwen3-1.7B-Q8_0 it quotes about 90/27 tok/s prefill/decode for hybrid vs about 60/22 for pinned. [verified]
26. **No Qualcomm source found for a cross-unit prefill/decode split on mobile.** Genie/QAIRT appear to run prefill (128-token chunks) and decode (1 token) as separate NPU graphs. That comes from a search summary of GenieX issue #1266, which was not opened. [unsure]
27. **Off Snapdragon.**
    - Agent.xpu (arXiv 2506.24045 v2) splits prefill and decode across the NPU and iGPU on Intel Core Ultra with OpenVINO.
    - SqueezeBits' Yetter engine (blog, 26 Aug 2025) runs prefill on the Apple Neural Engine (Core ML) and decode on the GPU (MLX), for Qwen3-0.6B on an iPhone 15 Pro.
    - HeRo (arXiv 2603.01661) maps whole agentic-RAG stages (several models) across xPUs on 8 Gen 3 and 8 Elite phones; it does not split one model's prefill and decode. [verified]
28. **The Pad 4 is v81.** GSMArena lists the OnePlus Pad 4 as SM8850-AC Snapdragon 8 Elite Gen 5 with 2×4.6 + 6×3.62 GHz Oryon V3, Adreno 840 and 8/12 GB. Those are the same CPU clocks as the SM8850 phones in 2607.05475 Table 4, which list the NPU as V81. That puts the Pad 4 inside llama.cpp's supported v73–v81 range (`ggml-hexagon.cpp` L8216–8228). [inferred]
29. **The Nord CE4 stays CPU-only.** b11256 raises any arch below v73 to v73 with a warning (L8222–8224) and forces a single session below v75 on Android (L8449–8454). BigMoeOnEdge reports that pre-v73 SoCs refuse the session. With the repo's v69 for the Snapdragon 7 Gen 3, the Nord CE4 has no usable HTP. [inferred]

## Deltas for hexagon-runtime-status.md

Verified runtime facts only. Line numbers refer to the file at the tag or commit in the link.

- **Pinned release.** `b11256` = commit `c85b92c` (committed 2026-09-29 12:32 UTC, released 13:06 UTC). It ships "Android arm64 (Snapdragon: CPU, Adreno GPU, Hexagon NPU)". https://github.com/ggml-org/llama.cpp/releases/tag/b11256
- **Async backend is in the release.** PR #26501 (merged 2026-08-27, `192067b`) is inside b11256. It brings the async backend, non-host HTP buffers by default, `HTP<phys>:<virt>` naming and `GGML_HEXAGON_DEVICES`; `GGML_HEXAGON_NDEV` is deprecated (L8265–8269). https://github.com/ggml-org/llama.cpp/pull/26501 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8265-L8269
- **Names.**
  - Hexagon default device and buffer type: `HTP0` when `GGML_HEXAGON_DEVICES` is unset (L8441–8447). The buffer type is named after the session (L2387–2400). `HTP0-HOST` exists only with `GGML_HEXAGON_HOSTBUF=1` (L7487–7493). https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8441-L8447
  - Adreno: buffer type `OpenCL` (L12778–12779), device `GPUOpenCL` (L12897). https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-opencl/ggml-opencl.cpp#L12778-L12779
- **Which names `-ot` accepts.** In llama-cli/server and in llama-bench, `-ot` accepts only each device's default buffer type. llama-bench separates overrides with `;` and test groups with `,`. `-dev` takes `/`-separated device lists and `none`. https://github.com/ggml-org/llama.cpp/blob/b11256/common/arg.cpp#L252-L284 ; https://github.com/ggml-org/llama.cpp/blob/b11256/tools/llama-bench/llama-bench.cpp#L941-L957 ; https://github.com/ggml-org/llama.cpp/blob/b11256/tools/llama-bench/llama-bench.cpp#L144-L173
- **How overrides apply.** `std::regex_search` runs on each tensor name and the first match wins. A non-CPU buffer type is used as given. A CPU override considers the extra (repack) buffer types and warns when mmap is on. https://github.com/ggml-org/llama.cpp/blob/b11256/src/llama-model-loader.cpp#L1234-L1257
- **Scheduler rule.** A weight's op runs on the highest-priority backend that supports both the weight's buffer type and the op. Op offload applies only to weights in host CPU buffers, and only toward backends whose `offload_op()` returns true; a NULL `offload_op` means false. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-backend.cpp#L888-L908 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-backend.cpp#L951-L984 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-backend.cpp#L644-L651
- **No batch-dependent (prefill vs decode) placement exists for Snapdragon backends.** `offload_op = NULL` for:
  - Hexagon at b11256: https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L7834
  - Hexagon at b7588: https://github.com/ggml-org/llama.cpp/blob/b7588/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2854
  - Hexagon at its latest change `19e28a2770`: https://github.com/ggml-org/llama.cpp/blob/19e28a2770/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L7838
  - OpenCL at b11256: https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-opencl/ggml-opencl.cpp#L13010
- **HTP buffers.** The default HTP buffer is non-host (`is_host` false), and HTP supports only its own buffer types. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2359-L2367 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L7813-L7820
- **Matmul engine choice.** MUL_MAT uses HMX when token rows M > `HTP_MM_HMX_MIN_NROWS` and HVX otherwise. `GGML_HEXAGON_MM_SELECT` defaults to 2 (HMX→HVX→CPU) and `GGML_HEXAGON_NHMX` to 1. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4672-L4677 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4946-L4955 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L93-L102
- **HTP MUL_MAT weight types.** Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K, F16, F32, rejected if the VTCM need exceeds the budget. K-quants arrived in #28994 (2026-09-16) and Q5_K in #29123 (2026-09-25). https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5717-L5797
- **Architecture range.**
  - Arch below v73 is raised to v73 with a warning; above v81 is capped at v81. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8216-L8228
  - Android forces one session below v75. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8449-L8454
  - DMA64 is on by default above v79, and WEIGHTS buffers are then mapped "extended". https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8241 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3936-L3946
- **Environment variables read at b11256.** VERBOSE, OPBATCH, OPQUEUE, OPPOLL, OPFUSION, OPFILTER, PROFILE, ETM, NHVX, NHMX, MM_SELECT, FA_SELECT, GDN_SELECT, AR_SELECT, NDEV, ARCH, VMEM, MBUF, OPTRACE, HOSTBUF, DMA64 and DEVICES, all prefixed `GGML_HEXAGON_`. `OPFILTER` rejects ops whose description matches the regex. https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L8193-L8213 ; https://github.com/ggml-org/llama.cpp/blob/b11256/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L7624-L7627
- **Snapdragon README at b11256.** Hexagon "behaves as a 'GPU' device" for `-ngl`. One NPU session has ~3.5 GB of virtual address space, and weights beyond that are mapped and unmapped automatically. Multi-device modes: layer split across virtual sessions, tensor split across physical NPUs, row split via `HTP0[0-1]`. https://github.com/ggml-org/llama.cpp/blob/b11256/docs/backend/snapdragon/README.md ("How to Run", "Multi-Device Execution Modes", "Environment variables")
- **Graph reuse.** A reused graph skips scheduler reset and allocation; `LLAMA_GRAPH_REUSE_DISABLE` disables reuse. https://github.com/ggml-org/llama.cpp/blob/b11256/src/llama-context.cpp#L1410-L1425 ; https://github.com/ggml-org/llama.cpp/blob/b11256/src/llama-context.cpp#L280-L283
- **Public API that a two-context handoff needs.** `llama_model_params.devices` and `tensor_buft_overrides`; `llama_context_params.op_offload`; `llama_state_seq_get_size/get_data/set_data`. https://github.com/ggml-org/llama.cpp/blob/b11256/include/llama.h#L319-L324 ; https://github.com/ggml-org/llama.cpp/blob/b11256/include/llama.h#L408 ; https://github.com/ggml-org/llama.cpp/blob/b11256/include/llama.h#L881-L901
- **No prefill or decode device flag.** `common/arg.cpp` at b11256 has none; `--prefill-assistant` is a chat-template flag and `--spec-draft-device` is for draft models. https://github.com/ggml-org/llama.cpp/blob/b11256/common/arg.cpp#L4210
- **Hexagon changes after b11256** (up to `19e28a2770`, 2026-09-29 21:59 UTC): only #29631 (GELU_ERF/GEGLU_ERF), #29673 (concat) and #29209 (f16 activation ops). None touches placement. https://github.com/ggml-org/llama.cpp/commits/master/ggml/src/ggml-hexagon
- **FP16 HMX is not guaranteed on v73.** HMX MUL_MAT correctness depends on FP16 HMX, which some v73 SKUs lack (SM7750, 4 MB VTCM). An integer-HMX PR, #29740, is open. https://github.com/ggml-org/llama.cpp/issues/29473 (comments of 2026-09-27, 2026-09-29, 2026-09-30)

## Citations

- HeteroInfer, arXiv 2501.14794 v2 HTML: https://arxiv.org/html/2501.14794v2 — Table 1 (engine classification), Table 2, §3.1 (GPU-2, ~400 µs `clFinish`), §3.3, §4.1, §4.2.1–4.2.3, §4.3, §4.4 with Table 3, §5.1, §5.2 with Table 4, §5.3.1, §5.4, §5.5, §6. Abstract and v1 title: https://arxiv.org/abs/2501.14794 and https://arxiv.org/abs/2501.14794v1. SOSP PDF (exists, not parsed): https://ipads.sjtu.edu.cn/zh/publications/sosp25-chen.pdf
- arXiv 2605.27435 HTML: https://arxiv.org/html/2605.27435 — Abstract, §II-A, §III, §IV-A Fig. 2, §IV-B Table III, §IV-C Table IV, §IV-D Table V, §IV-E, §IV-F.
- arXiv 2607.05475 v1 HTML: https://arxiv.org/html/2607.05475v1 — Table 3 (framework versions), Table 4 (devices), Table 5 (throughput and energy at 256 tokens), §4, §5.2 Fig. 3, §6.1 Fig. 7, §6.2, §8.
- llama.cpp PR #26501: https://github.com/ggml-org/llama.cpp/pull/26501 — description; maintainer comment of 2026-09-18 (dspqueue op batches).
- llama.cpp issue #29473: https://github.com/ggml-org/llama.cpp/issues/29473 — body (throughput table, workaround); comments of 2026-09-26 to 2026-09-30. Created 2026-09-26T09:01Z per https://api.github.com/repos/ggml-org/llama.cpp/issues/29473
- llama.cpp issue #25876: https://github.com/ggml-org/llama.cpp/issues/25876 — body and the reporter's 2026-07-22 update.
- llama.cpp PR #20693 (HMX added, merged 2026-03-19): https://github.com/ggml-org/llama.cpp/pull/20693 ; HMX PR list: https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+hexagon+HMX+is:pr+is:merged
- llama.cpp source at b11256, b7588 and `19e28a2770`: the line-anchored links in "Deltas" above. Commit dates for b7588 and eadc418: https://api.github.com/repos/ggml-org/llama.cpp/commits/b7588 and https://api.github.com/repos/ggml-org/llama.cpp/commits/eadc418
- llm.npu, arXiv 2407.05858 v2: https://arxiv.org/html/2407.05858v2 — §4 (implementation: "MLLM CPU backend for decoding stage"), §4.1 (devices); code https://github.com/UbiquitousLearning/mllm (README news of 2026-02-03).
- PowerInfer-2, arXiv 2406.06282: https://arxiv.org/html/2406.06282 — §4.1.1, §4.1.2, §4.1.3, §7.1.
- Phase Matters, arXiv 2606.27906 v1: https://arxiv.org/html/2606.27906v1 — §1 contributions, §2.1–2.3 with Table 1, §5.1 Table 4.
- BigMoeOnEdge: PR #200 https://github.com/Helldez/BigMoeOnEdge/pull/200 ; https://github.com/Helldez/BigMoeOnEdge/blob/3170385fad8e/docs/npu-prefill.md ("Measured", "How it works", "Why not decode") ; https://github.com/Helldez/BigMoeOnEdge/blob/main/docs/seam.md ("Per-graph placement").
- GenieX run notes at `ca874e4f18b3`: https://github.com/qualcomm/GenieX/blob/ca874e4f18b3/notes/run.md — "Compute-unit aliases", "NPU compute-unit selection (llama_cpp)".
- Agent.xpu, arXiv 2506.24045 v2: https://arxiv.org/html/2506.24045v2 — §1 ("Implementation and Evaluations").
- Yetter (SqueezeBits, 2025-08-26): https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176
- HeRo, arXiv 2603.01661: https://arxiv.org/html/2603.01661 — Table 1, Table 2, §3.
- OnePlus Pad 4 specifications: https://www.gsmarena.com/oneplus_pad_4-14630.php (Chipset, CPU, GPU, Memory rows).
- Our own numbers: `docs/TWIN.md` ("What the held-out numbers say", "Fitted effective weight bandwidth"); `docs/PLAN.md` L53, L118, L137, L154–158.

## Contradictions with our results

- **Per-regime placement cannot be exported to one llama.cpp process.** `docs/PLAN.md` L53 defines z(m, t, u) per regime m ∈ {P, D}, and the §7 "phase-split characterisation fallback" assumes the same. At b11256 a tensor has one buffer per process, and no Snapdragon backend implements `offload_op`. Per CLAUDE.md, the exporter must choose one of:
  - tie z(P,t,·) = z(D,t,·);
  - add a second-copy memory term plus a measured KV-handoff term (two contexts);
  - build the rebinding runner.
- **Decode on GPU does not fit our S26 fit.** Our published-row fit for the Galaxy S26 (sm8850-ad, the Pad 4 cross-check) gives effective decode bandwidth CPU 65.5, NPU 51.9 and GPU 51.6 GB/s (`docs/TWIN.md`), so the CPU is the better decode unit. That agrees with 2607.05475 (CPU best, GPU near-equivalent, §6.2 and Table 5). It contradicts D3's GPU-decode variant and HeteroInfer's GPU-primary decode, which compared the authors' own OpenCL kernels with QNN on 8 Gen 3.
- **Our NPU decode is not the paper's NPU decode.** On our S26 fit the NPU decodes at GPU level (51.9 vs 51.6 GB/s effective; t0 1.34 vs 1.59 ms). 2607.05475's pre-HMX llama.cpp NPU decoded 34% slower than its GPU on the same SoC (1 − 33.3/50.3 = 0.34, arithmetic). The two sources used different llama.cpp versions, and the commit inside Qualcomm's GenieX plugin is unknown to us. Do not pool them in one model. [inferred]
- **Consistent with SnapTwin, and it matters for D3.** SnapTwin says NPU decode does not transfer by bandwidth (58% median error) and prefill does not transfer with compute ratio 1 (47–61%). The literature agrees:
  - NPU decode is bound by per-call overhead: 2605.27435 Table IV, HeteroInfer §3.1, BigMoeOnEdge's 0.3–0.5 ms per crossing, and the context dependence in 2607.05475 Fig. 7.
  - NPU prefill is bound by kernel maturity, which changed by an order of magnitude within 2026 (findings 13–14).

  So D3's prefill gain has to be measured on the Pad 4 at b11256. It cannot be predicted from bandwidth or from papers that predate March 2026.
- **PROPOSAL.md misstates 2607.05475.** `docs/PROPOSAL.md` L189 summarises it as "prefill on NPU, decode on CPU". The paper measured a phase split in backend ranking and only proposed such a runtime (§8). 2605.27435 recommends the opposite split, but for pre-HMX llama.cpp.
- **Baseline 3 is not HeteroInfer.** `docs/PLAN.md` L118 lists "HeteroInfer-style rules via `llama-bench -ot`". `-ot` expresses one static placement per tensor, while HeteroInfer changes placement with the phase and splits rows or sequences. Call baseline 3 "HeteroInfer-inspired static placement".
- **The v73 HMX kill criterion rests on a SKU gap, not a v73-wide bug.** `docs/PLAN.md` L137 and L158 treat #29473 as a v73 HMX bug that might reproduce on X Elite. Its cause is missing FP16 HMX on some v73 SKUs, and the maintainer's dual-v73 IQ9 runs clean. Whether X Elite's HMX has FP16 was not checked here. For the Pad 4 (v81), the open risk is #25876-style errors in ADD after HMX MUL_MAT, so check perplexity before timing.
- **Confirmed.** REDTEAM A27 holds: HeteroInfer decode gains are 1.50–2.53×, decode is GPU-primary, and norms/SwiGLU run on the GPU (§4.1, §5.4). REDTEAM A5 holds: 2607.05475 reports a 20.0–23.8% decode penalty from 16 to 4,096 context (§6.1).

## Recommendation

1. **Expressible?** Not with stock b11256 flags: ops follow weight residency, Hexagon and OpenCL leave `offload_op` NULL, and there is no prefill-device option. D3 needs our own runner, either two model loads with a `llama_state_seq_get_data/set_data` KV handoff (double weight memory) or BigMoeOnEdge-style per-graph weight rebinding with `GGML_HEXAGON_HOSTBUF=1` and distinct graph widths.
2. **Measure first on the Pad 4** (b11256, Qwen3-0.6B/1.7B/4B Q4_0): `llama-bench -dev HTP0,GPUOpenCL,none -p 128,512,2048 -n 128 -d 0,1024,4096`, plus an HTP-vs-CPU perplexity and greedy-text check (HMX correctness reports #25876, #29473); time a KV state copy only if HTP wins prefill.
3. **Effect reported by others:** NPU prefill 7.8–9.5× the CPU at 1.4–1.9K tokens but parity at 121 tokens (BigMoeOnEdge, v81, llama.cpp 2026-09-26 base, 35B MoE). NPU prefill with CPU decode at 2,640 ms vs 3,412 ms CPU-only and 2,080 ms NPU-only, warm (Phase Matters Table 4). But on SM8850 with pre-HMX llama.cpp, the GPU prefilled 6.6× faster than the NPU (754.8 vs 115.1 tok/s, 2607.05475 Table 5).
4. **Kill criterion** (thresholds are ours): drop D3 on the Pad 4 if HTP pp512 is below 1.5× the faster of OpenCL and CPU for every Qwen3 size, or HTP perplexity exceeds CPU perplexity by more than 1%, or the KV handoff plus the second weight copy costs more time to first token than HTP prefill saves at the agent's median prompt length.
5. **Novelty:** splitting stages is not new (llm.npu, PowerInfer-2, HeteroInfer, Phase Matters, and BigMoeOnEdge's `--prefill-device` on llama.cpp v81; Agent.xpu and Yetter off Snapdragon). What may be new is choosing the stage split jointly with per-tensor bits and rank in one QUBO with a measured handoff term, on stock llama.cpp.
