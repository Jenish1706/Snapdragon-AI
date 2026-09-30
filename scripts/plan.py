#!/usr/bin/env python3
"""Step 3: build the planner QUBO for Qwen3-0.6B, solve with exact DP, SB and SA, write plans.

Costs: SnapTwin per-unit coefficients from results/twin/fits.csv (--cost-device; default the
Galaxy S26 published fit, the Pad 4's SoC, until Pad sweeps exist). Accuracy: the per-layer
sensitivity table (percent perplexity increase vs F16; options missing from the table are
dropped). Unit-switch costs: priors from the literature (HeteroInfer clFinish ~400 us; see
docs/research/npu-decode-bottleneck.md) until calibrated on the device. Plans carry llama.cpp
`-ot` override strings and a tensor-type list for llama-quantize. Output: results/planner/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.qubo.planner import (
    Option,
    Problem,
    UnitCost,
    Workload,
    default_options,
    groups_for_qwen3,
)
from isingedge.solve.solvers import (
    exact_chain,
    sa_joint,
    sb_joint,
    simulated_annealing,
    simulated_bifurcation,
)

SENS = ROOT / "results/mac-m2/sensitivity/qwen3-0.6b_layer_sensitivity.csv"
FITS = ROOT / "results/twin/fits.csv"
OUT = ROOT / "results/planner"
BUFFER = {"cpu": "CPU", "gpu": "OpenCL", "npu": "HTP0"}  # llama.cpp buffer-type names on Android (verify on device)
SWITCH_DEC_PRIOR = {("cpu", "gpu"): 2e-4, ("cpu", "npu"): 3e-4, ("gpu", "npu"): 4e-4}
SWITCH_PRE_PRIOR = {k: 10 * v for k, v in SWITCH_DEC_PRIOR.items()}
QWEN3_06B = {"n_layers": 28, "hidden": 1024, "inter": 3072, "n_heads": 16, "n_kv": 8, "head_dim": 128}


def symmetric(d: dict) -> dict:
    return {**d, **{(b, a): v for (a, b), v in d.items()}}


def load_costs(device: str, stage: str) -> dict[str, UnitCost]:
    out = {}
    for r in csv.DictReader(FITS.open()):
        if r["device"] == device and r["stage"] == stage:
            out[r["unit"]] = UnitCost(*(float(r[k]) for k in ("a_w", "a_kv", "t0", "b_f", "b_attn", "b0")))
    if not out:
        sys.exit(f"no fits for {device} ({stage}) in {FITS}")
    return out


def load_sensitivity(scaling: str) -> tuple[dict, set]:
    """dL[(layer, part, Option)] in percent; returns (dl, available (part, keep, bits) options)."""
    svd, q4 = {}, {}
    for r in csv.DictReader(SENS.open()):
        L, part = int(r["layer"]), r["part"]
        if r["kind"] == "svd" and r["scaling"] == scaling:
            svd[(L, float(r["keep"]))] = float(r["delta_pct"])
        elif r["kind"] == "q4_0":
            q4[(L, part)] = float(r["delta_pct"])
    n = QWEN3_06B["n_layers"]
    # offer an option only if it was measured on every layer, so no unmeasured option looks free
    keeps = [1.0] + sorted((k for k in {k for (_, k) in svd} if all((L, k) in svd for L in range(n))), reverse=True)
    q4_ok = {part: all((L, part) in q4 for L in range(n)) for part in ("mlp", "attn")}
    dl, avail = {}, set()
    for L in range(QWEN3_06B["n_layers"]):
        for keep in keeps:
            for bits in ("q4_0", "q8_0"):
                if bits == "q4_0" and not q4_ok["mlp"]:
                    continue
                o = Option(keep, bits)
                dl[(L, "mlp", o)] = (svd.get((L, keep), 0.0) if keep < 1 else 0.0) + (q4.get((L, "mlp"), 0.0) if bits == "q4_0" else 0.0)
                avail.add(("mlp", keep, bits))
        for bits in ("q4_0", "q8_0"):
            if bits == "q4_0" and not q4_ok["attn"]:
                continue
            dl[(L, "attn", Option(1.0, bits))] = q4.get((L, "attn"), 0.0) if bits == "q4_0" else 0.0
            avail.add(("attn", 1.0, bits))
    return dl, avail


def override_string(p: Problem, plan: list) -> str:
    """llama-bench/llama-cli -ot value: one anchored alternation regex per unit, ';'-separated."""
    per_unit: dict[str, list[str]] = {}
    for g, (u, _) in zip(p.groups, plan):
        per_unit.setdefault(u, []).extend(t.replace(".", r"\.") for t in g.tensors)
    return ";".join(f"^({'|'.join(ts)})$={BUFFER[u]}" for u, ts in sorted(per_unit.items()))


def build(args, executable: bool) -> Problem:
    costs = load_costs(args.cost_device, args.cost_stage)
    dl, avail = load_sensitivity(args.scaling)
    opts = default_options(executable)
    opts = {part: [o for o in os_ if (part, o.keep, o.bits) in avail] for part, os_ in opts.items()}
    m = QWEN3_06B
    groups = groups_for_qwen3(m["n_layers"], m["hidden"], m["inter"], m["n_heads"], m["n_kv"], m["head_dim"])
    units = tuple(u for u in ("cpu", "gpu", "npu") if u in costs)
    return Problem(groups, opts, units, costs, symmetric(SWITCH_DEC_PRIOR), symmetric(SWITCH_PRE_PRIOR), dl,
                   kv_bytes_per_layer=2 * m["n_kv"] * m["head_dim"] * 2, attn_flops_per_pair_per_layer=4 * m["n_heads"] * m["head_dim"],
                   workload=Workload(args.prompt, args.gen, 1.0, 1.0), lam=args.lam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-device", default="Samsung Galaxy S26")
    ap.add_argument("--cost-stage", default="published")
    ap.add_argument("--scaling", default="imatrix", choices=("none", "imatrix"))
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.05, help="seconds of latency worth one percent of perplexity")
    ap.add_argument("--tag", default="agent")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for executable in (False, True):
        p = build(args, executable)
        _, idx, _ = p.qubo_joint()
        plan, opt = exact_chain(p)
        runs = [("exact-dp", {"plan": plan, "objective": opt, "seconds": None, "feasible_agents": None})]
        import time

        t0 = time.perf_counter()
        exact_chain(p)
        runs[0][1]["seconds"] = time.perf_counter() - t0
        for s in range(args.seeds):
            runs.append((f"sb-joint-seed{s}", sb_joint(p, seed=s)))
            runs.append((f"sa-joint-5k-seed{s}", sa_joint(p, sweeps=5000, seed=s)))
            runs.append((f"sa-joint-50k-seed{s}", sa_joint(p, sweeps=50000, seed=s)))
        # ablation: the split encoding (unit one-hot x option one-hot with pairwise costs)
        runs.append(("sb-split-seed0", simulated_bifurcation(p, seed=0)))
        runs.append(("sa-split-seed0", simulated_annealing(p, seed=0)))
        mode = "executable" if executable else "full"
        for name, r in runs:
            lat = p.predicted_latency(r["plan"])
            rows.append({"tag": args.tag, "mode": mode, "solver": name, "n_vars": len(idx), "objective": r["objective"],
                         "gap_pct": 100 * (r["objective"] - opt) / opt, "seconds": r["seconds"],
                         "feasible_agents": r["feasible_agents"], "pred_ttft_s": lat["ttft_s"], "pred_decode_tps": lat["decode_tps"],
                         "pred_delta_ppl_pct": lat["delta_ppl_pct"],
                         "units": "".join(u[0] for u, _ in r["plan"]), "options": " ".join(o.label() for _, o in r["plan"])})
        plan_json = {"tag": args.tag, "mode": mode, "cost_source": f"{args.cost_device} ({args.cost_stage} SnapTwin fit)",
                     "switch_costs": "literature priors, not yet calibrated", "workload": vars(p.workload), "lambda_s_per_pct": args.lam,
                     "scaling": args.scaling, "prediction": p.predicted_latency(plan),
                     "groups": [{"group": g.name, "unit": u, "option": o.label()} for g, (u, o) in zip(p.groups, plan)],
                     "override_tensor": override_string(p, plan),
                     "tensor_types": [f"{t}={o.bits}" for g, (_, o) in zip(p.groups, plan) for t in g.tensors]}
        (OUT / f"plan_{args.tag}_{mode}.json").write_text(json.dumps(plan_json, indent=2))
    path = OUT / "solver_comparison.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["solver"] in ("exact-dp", "sb-joint-seed0", "sa-joint-5k-seed0", "sa-joint-50k-seed0", "sb-split-seed0", "sa-split-seed0"):
            print(f"{r['mode']:10s} {r['solver']:9s} vars {r['n_vars']} gap {r['gap_pct']:.3f}% time {1e3 * r['seconds']:.1f} ms "
                  f"| pred TTFT {r['pred_ttft_s']:.3f}s decode {r['pred_decode_tps']:.1f} tok/s dPPL {r['pred_delta_ppl_pct']:.2f}% | {r['units']}")


if __name__ == "__main__":
    main()
