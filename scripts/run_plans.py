#!/usr/bin/env python3
"""Step 3 on the device: run planner outputs on the Pad 4 and compare measured with predicted.

For each plan JSON (results/planner/plan_*_executable.json): quantise a GGUF with the plan's
per-tensor types from the local Qwen3-0.6B F16 (llama-quantize, macOS build) unless it is uniform
Q4_0, push it, and run llama-bench on the Pad with `-dev HTP0/GPUOpenCL -ngl 99 -ot <plan>` for
the plan's prompt length (prefill) and decode at the plan's depth. Calibration placements (all-CPU,
all-GPU, all-NPU, attention-NPU/MLP-CPU, attention-NPU/MLP-GPU) are run first to fit the switch
costs that the planner otherwise takes from literature priors. Results: results/pad4/plans/*.csv.

Run outside the sandbox (adb needs USB). Measured numbers are Pad 4 results, never PC results.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.bench import adb

MAC_BIN = ROOT / "cache/bin/llama-b11256-bin-macos-arm64/llama-b11256"
F16 = ROOT / "cache/models/Qwen3-0.6B-F16.gguf"
OUT = ROOT / "results/pad4/plans"
CALIBRATION = {
    "cal-all-cpu": ("^blk\\..*=CPU", "none"),
    "cal-all-gpu": ("", "GPUOpenCL"),
    "cal-all-npu": ("", "HTP0"),
    "cal-attn-npu-mlp-cpu": ("^blk\\.[0-9]+\\.ffn_.*=CPU", "HTP0"),
    "cal-attn-npu-mlp-gpu": ("^blk\\.[0-9]+\\.ffn_.*=OpenCL", "HTP0/GPUOpenCL"),
}


def quantise(plan: dict, name: str) -> Path:
    types = plan["tensor_types"]
    out = ROOT / "cache/models" / f"Qwen3-0.6B-plan-{name}.gguf"
    if all(t.endswith("=q4_0") for t in types):
        base = ROOT / "cache/models" / "Qwen3-0.6B-plan-uniform-q4_0.gguf"
        if not base.exists():
            subprocess.run([str(MAC_BIN / "llama-quantize"), str(F16), str(base), "Q4_0"], check=True, capture_output=True)
        return base
    tfile = ROOT / "cache/models" / f"tensor-types-{name}.txt"
    tfile.write_text("\n".join(types) + "\n")
    subprocess.run([str(MAC_BIN / "llama-quantize"), "--tensor-type-file", str(tfile), str(F16), str(out), "Q4_0"],
                   check=True, capture_output=True)
    return out


def bench(dev: adb.Device, model: str, ot: str, devices: str, prompt: int, depth: int, reps: int) -> dict:
    unit = "npu" if "HTP" in devices else ("gpu" if "GPU" in devices else "cpu")
    args = ["-m", f"{adb.MODEL_DIR}/{model}", "-dev", devices, "-ngl", "0" if devices == "none" else "99", "-t", "6",
            "-r", str(reps), "-o", "jsonl", "-p", str(prompt), "-n", "128", "-d", f"0,{depth}"]
    if "HTP" in devices:
        args += ["-ub", "1024"]
    if ot:
        args += ["-ot", ot]
    res = adb.adb("shell", adb.llama_cmd("llama-bench", args, unit), serial=dev.serial, check=False, timeout=3600)
    rows = [json.loads(line) for line in (res.stdout or "").splitlines() if line.startswith("{")]
    pp = next((r for r in rows if r["n_prompt"] == prompt and r["n_gen"] == 0), None)
    tg = next((r for r in rows if r["n_gen"] > 0 and r["n_depth"] == depth), None)
    return {"returncode": res.returncode, "prefill_tps": pp and pp["avg_ts"], "ttft_ms": pp and pp["avg_ns"] / 1e6,
            "decode_tps": tg and tg["avg_ts"], "stderr_tail": (res.stderr or "")[-400:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", nargs="*", default=sorted(str(p) for p in (ROOT / "results/planner").glob("plan_*_executable.json")))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    devs = [d for d in adb.devices() if d.state == "device" and d.label == "pad4"]
    if not devs:
        sys.exit("OnePlus Pad 4 not visible to adb")
    dev = devs[0]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    base = quantise({"tensor_types": ["x=q4_0"]}, "uniform")
    adb.adb("push", str(base), f"{adb.MODEL_DIR}/{base.name}", serial=dev.serial)
    for name, (ot, devices) in CALIBRATION.items():
        r = bench(dev, base.name, ot, devices, 1024, 1024, args.reps)
        rows.append({"kind": "calibration", "name": name, "devices": devices, "override": ot, **r})
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {r}", flush=True)
    for path in args.plans:
        plan = json.loads(Path(path).read_text())
        name = Path(path).stem
        gguf = quantise(plan, name)
        adb.adb("push", str(gguf), f"{adb.MODEL_DIR}/{gguf.name}", serial=dev.serial)
        w = plan["workload"]
        r = bench(dev, gguf.name, plan["override_tensor"], "HTP0/GPUOpenCL", w["prompt_tokens"], w["prompt_tokens"], args.reps)
        pred = plan["prediction"]
        rows.append({"kind": "plan", "name": name, "devices": "HTP0/GPUOpenCL", "override": plan["override_tensor"][:120],
                     "pred_ttft_ms": 1e3 * pred["ttft_s"], "pred_decode_tps": pred["decode_tps"], **r})
        print(f"[{time.strftime('%H:%M:%S')}] {name}: measured {r['ttft_ms']} ms / {r['decode_tps']} tok/s, "
              f"predicted {1e3 * pred['ttft_s']:.0f} ms / {pred['decode_tps']:.1f} tok/s", flush=True)
        if gguf.name != base.name:
            gguf.unlink(missing_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (OUT / "plans_measured_vs_predicted.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
