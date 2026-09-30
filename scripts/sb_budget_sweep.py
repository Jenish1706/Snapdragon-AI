#!/usr/bin/env python3
"""Does SB reach the exact optimum on the planner's chain case with a larger budget?

The pairwise planner instance without pair terms (Qwen3-0.6B, 56 blocks, Galaxy S26 published costs,
literature switch priors, 256 prompt + 512 generated tokens, lambda 0.005) has an exact DP optimum.
Sweeps SB agents, steps and dt for discrete and ballistic SB (seed 0 and 1) and records the gap to
the optimum and the wall time on this machine. Output: results/planner/sb_budget_sweep.csv.
"""

from __future__ import annotations

import csv
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plan_pairwise import build, load

from isingedge.solve.solvers import exact_chain, sb_joint

OUT = ROOT / "results/planner/sb_budget_sweep.csv"
LAM = 0.005
CONFIGS = [(64, 2000), (64, 10000), (64, 40000), (256, 10000)]


def main() -> None:
    a, _ = load()
    p = build(a, {}, LAM)
    plan, opt = exact_chain(p)
    counts = Counter(u for u, _ in plan)
    rows = [{"variant": "exact DP", "agents": "", "steps": "", "dt": "", "seed": "", "gap_pct": 0.0, "feasible_agents": "",
             "units": " ".join(f"{u}:{counts[u]}" for u in sorted(counts)),
             "n_compressed": sum(1 for g, (_, o) in zip(p.groups, plan) if g.part == "mlp" and o.keep < 1),
             "seconds": "", "lambda": LAM, "machine": platform.machine()}]
    for ballistic in (False, True):
        for agents, steps in CONFIGS:
            for dt in (0.5, 1.0):
                for seed in (0, 1):
                    r = sb_joint(p, agents=agents, steps=steps, dt=dt, seed=seed, ballistic=ballistic)
                    gap = 100 * (r["objective"] - opt) / opt
                    counts = Counter(u for u, _ in r["plan"])
                    rows.append({"variant": "ballistic" if ballistic else "discrete", "agents": agents, "steps": steps, "dt": dt,
                                 "seed": seed, "gap_pct": round(gap, 4), "feasible_agents": r["feasible_agents"],
                                 "units": " ".join(f"{u}:{counts[u]}" for u in sorted(counts)),
                                 "n_compressed": sum(1 for g, (_, o) in zip(p.groups, r["plan"]) if g.part == "mlp" and o.keep < 1),
                                 "seconds": round(r["seconds"], 2), "lambda": LAM, "machine": platform.machine()})
                    print(f"{rows[-1]['variant']:9s} agents {agents:3d} steps {steps:5d} dt {dt} seed {seed}: gap {gap:7.3f}% "
                          f"feasible {r['feasible_agents']}/{agents} {rows[-1]['units']} {r['seconds']:.1f}s", flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
