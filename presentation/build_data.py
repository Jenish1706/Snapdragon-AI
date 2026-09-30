#!/usr/bin/env python3
"""Extract every number, table row and chart series used by the submission documents from ../results.

Nothing in the pitch deck, the project description or the PPTX is typed in by hand: every value is
read from the repository's raw result files, re-derived where possible (paired bootstrap intervals are
recomputed from the per-chunk NLL sums with the repository's own isingedge.loop.stats), checked, and
written to:

  generated/values.tex    \\ievalue{key}{value}   -> used through \\V{key} in pitch_deck.tex and project_description.tex
  generated/values.json   {key: value}           -> used by generate_pptx.py and check_parity.py
  generated/checks.txt    every consistency check that passed (the script stops on the first failure)
  figures/data/*.csv      chart series (pgfplots in the PDFs, native charts in the PPTX)
  tables/*.tex            table rows shared by the deck and the description

Run from the repository root or from presentation/:  python3 presentation/build_data.py
Requires numpy only (plus the repository's isingedge package, imported from ../).
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from isingedge.loop.space import Space, saved_units  # noqa: E402
from isingedge.loop.stats import paired_bootstrap, ppl_from_chunks  # noqa: E402

GEN = HERE / "generated"
DATA = HERE / "figures" / "data"
TABLES = HERE / "tables"
MINUS = "\u2212"  # typographic minus, rendered by Inter (PDF) and Calibri/Carlito (PPTX)
NDASH = "\u2013"  # ranges

values: dict[str, str] = {}
checks: list[str] = []


# ----------------------------------------------------------------------------- helpers
def read(rel: str) -> list[dict]:
    with (RES / rel).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(x: float, nd: int = 2, sign: bool = False) -> str:
    """Fixed decimals, typographic minus, optional explicit plus sign."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def grouped(n: int) -> str:
    return f"{n:,}"


def put(key: str, value: str) -> None:
    if key in values and values[key] != value:
        raise SystemExit(f"value {key} defined twice with different content")
    values[key] = value


def check(cond: bool, what: str) -> None:
    if not cond:
        raise SystemExit(f"CHECK FAILED: {what}")
    checks.append(what)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def write_csv(name: str, cols: list[str], rows: list[list]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in r])


# ----------------------------------------------------------------------------- environment
comp = read("compression/ppl.csv")
builds = {r["build"] for r in comp}
check(len(builds) == 1, "all compression rows come from one build")
build = builds.pop()
check("RTX 3060 Laptop GPU" in build and "fp32" in build, "results were measured in PyTorch fp32 on an RTX 3060 Laptop GPU")
put("env.build", "PyTorch 2.14.1 + CUDA 12.6, fp32, NVIDIA GeForce RTX 3060 Laptop GPU")
check("PyTorch 2.14.1" in build and "CUDA 12.6" in build and "transformers 5.17.0" in build, "build string matches env.build")

# ----------------------------------------------------------------------------- dense baselines
dense_full = next(r for r in comp if r["method"] == "dense")
check(int(dense_full["chunks"]) == 584, "full test split = 584 chunks")
put("dense.full.ppl", num(float(dense_full["ppl"]), 2))
put("dense.full.err", num(float(dense_full["ppl_err"]), 2))
put("dense.full.chunks", dense_full["chunks"])

singles = read("sensitivity/qwen3-0.6b_layer_sensitivity.csv")
dense40 = next(r for r in singles if r["kind"] == "dense")
check(int(dense40["chunks"]) == 40, "paired fast split = 40 chunks")
put("dense.fast.ppl", num(float(dense40["ppl"]), 2))
put("dense.fast.err", num(float(dense40["ppl_err"]), 2))
put("dense.fast.chunks", dense40["chunks"])
put("ctx", "512")

# ----------------------------------------------------------------------------- ranks and parameters
svd_rows = [r for r in singles if r["kind"] == "svd"]
check({r["scaling"] for r in svd_rows} == {"imatrix"}, "single-layer rows are activation-aware (imatrix) SVD only")
check({r["keep"] for r in svd_rows} == {"0.8", "0.9"}, "single-layer rows cover keep 0.8 and 0.9 only")
rank = {k: {int(r["rank"]) for r in svd_rows if r["keep"] == k} for k in ("0.8", "0.9")}
check(rank == {"0.8": {608}, "0.9": {704}}, "keep 0.8 -> rank 608, keep 0.9 -> rank 704")
dense_params = {int(r["dense_params"]) for r in svd_rows}
check(dense_params == {9437184}, "dense MLP = 3 x 1024 x 3072 = 9,437,184 parameters")
params = {k: {int(r["params"]) for r in svd_rows if r["keep"] == k}.pop() for k in ("0.8", "0.9")}
check(params == {"0.8": 3 * 608 * 4096, "0.9": 3 * 704 * 4096}, "compressed MLP parameters = 3 x rank x (1024 + 3072)")
put("rank.k08", "608")
put("rank.k09", "704")
put("frac.k08", num(100 * params["0.8"] / 9437184, 1))
put("frac.k09", num(100 * params["0.9"] / 9437184, 1))
saved08 = 9437184 - params["0.8"]
saved09 = 9437184 - params["0.9"]
check(saved08 == 1966080 and saved09 == 786432, "saved parameters per MLP: 1,966,080 (keep 0.8), 786,432 (keep 0.9)")
check(tuple(saved_units((0.9, 0.8))) == (2, 5), "loop byte units: keep 0.9 saves 2, keep 0.8 saves 5")
put("mlp.total.params", grouped(28 * 9437184))

# ----------------------------------------------------------------------------- single-layer sensitivity
sens = {k: {int(r["layer"]): float(r["delta_pct"]) for r in svd_rows if r["keep"] == k} for k in ("0.8", "0.9")}
check(all(len(v) == 28 for v in sens.values()), "28 layers measured at each keep")
for k, tag in (("0.8", "k08"), ("0.9", "k09")):
    v = sens[k]
    lo, hi = min(v, key=v.get), max(v, key=v.get)
    put(f"sens.{tag}.median", num(st.median(v.values()), 2))
    put(f"sens.{tag}.min.layer", str(lo))
    put(f"sens.{tag}.min", num(v[lo], 2))
    put(f"sens.{tag}.max.layer", str(hi))
    put(f"sens.{tag}.max", num(v[hi], 2))
put("sens.k08.L26", num(sens["0.8"][26], 2))
put("sens.k08.L18", num(sens["0.8"][18], 2))
check(values["sens.k08.max.layer"] == "27" and values["sens.k08.min.layer"] == "8", "keep 0.8: most sensitive layer 27, least 8")
write_csv("sensitivity.csv", ["layer", "keep08", "keep09"], [[L, sens["0.8"][L], sens["0.9"][L]] for L in range(28)])

# ----------------------------------------------------------------------------- layer pairs
pairs = read("sensitivity/qwen3-0.6b_pair_sensitivity.csv")
check(len(pairs) == 40 and {r["keep"] for r in pairs} == {"0.8"}, "40 random layer pairs at keep 0.8")
inter = [float(r["interaction_pct"]) for r in pairs]
for r in pairs:
    check(close(float(r["interaction_pct"]), float(r["dpct_ij"]) - float(r["dpct_i"]) - float(r["dpct_j"]), 1e-9),
          f"pair {r['layer_i']}+{r['layer_j']}: interaction = measured - sum of singles")
imax = max(range(len(pairs)), key=lambda i: inter[i])
put("pairs.n", str(len(pairs)))
put("pairs.super", str(sum(x > 0 for x in inter)))
put("pairs.inter.median", num(st.median(inter), 2, sign=True))
put("pairs.inter.max", num(inter[imax], 2, sign=True))
put("pairs.inter.max.pair", f"{pairs[imax]['layer_i']}+{pairs[imax]['layer_j']}")
check(values["pairs.super"] == "33" and values["pairs.inter.max.pair"] == "0+1", "33 of 40 pairs super-additive; largest interaction at layers 0+1")
loo = {r["kind"]: r for r in read("sensitivity/pair_models_loo.csv")}
check(loo["multiplicative"]["chosen"] == "True", "leave-one-pair-out selects the multiplicative pair model")
put("pairs.loo.mult", num(float(loo["multiplicative"]["loo_mean_abs_pp"]), 2))
put("pairs.loo.add", num(float(loo["additive"]["loo_mean_abs_pp"]), 2))

# ----------------------------------------------------------------------------- matched bytes (K MLPs at keep 0.8)
mb = read("matched_bytes/summary.csv")
plans = read("sensitivity/qwen3-0.6b_plan_perplexity.csv")
plan_rows = {(r["plan"], r["chunks"]): r for r in plans}
nll = {}
for r in read("sensitivity/qwen3-0.6b_plan_chunk_nll.csv"):
    nll[(r["plan"], r["chunks"])] = (np.array([float(v) for v in r["nll_sums"].split()]), int(r["per_chunk"]))
dense_nll_all, per_chunk = nll[("dense", "all")]
check(len(dense_nll_all) == 584 and per_chunk == 255, "per-chunk NLL: 584 chunks x 255 scored tokens")
check(close(ppl_from_chunks(dense_nll_all, per_chunk), float(dense_full["ppl"]), 1e-6), "dense full-split PPL recomputed from chunk NLL")

mb_rows, sa_rows = [], []
for K in (6, 12, 18, 24):
    rows = [r for r in mb if r["K"] == str(K)]
    model = next(r for r in rows if r["row"] == "plan" and r["plan"].endswith("pairwise"))
    check(model["method"].startswith("pairwise-multiplicative exact = additive"),
          f"K={K}: pairwise, additive and greedy selections coincide (method label)")
    rnd = [r for r in rows if r["row"] == "plan" and "random" in r["plan"]]
    check(len(rnd) == 3, f"K={K}: three random plans")
    best_rnd = min(rnd, key=lambda r: float(r["measured_40_pct"]))
    ci_row = next(r for r in rows if r["row"] == "CI all chunks")
    a, b = ci_row["plan"].split(" - ")
    # recompute the interval from per-chunk NLL with the repository's own bootstrap (seed 0, 4000 resamples)
    ci = paired_bootstrap(nll[(a, "all")][0], nll[(b, "all")][0], per_chunk, dense_nll=dense_nll_all)
    check(close(ci["diff"], float(ci_row["measured_full_pct"]), 1e-3) and close(ci["lo"], float(ci_row["ci95_lo_pp"]), 1e-3)
          and close(ci["hi"], float(ci_row["ci95_hi_pp"]), 1e-3), f"K={K}: bootstrap CI recomputed from chunk NLL")
    # express every interval as (model plan - best random plan): negative = model plan better
    if a == model["plan"]:
        diff, lo, hi = ci["diff"], ci["lo"], ci["hi"]
        other = b
    else:
        diff, lo, hi = -ci["diff"], -ci["hi"], -ci["lo"]
        other = a
    check(other == best_rnd["plan"], f"K={K}: the full-split comparison uses the best of the three random plans")
    plan_full = float(model["measured_full_pct"])
    rnd_full = float(plan_rows[(other, "all")]["measured_pct"])
    check(close(plan_full - rnd_full, diff, 1e-3), f"K={K}: difference = model plan - best random plan (pp of dense PPL)")
    k = f"mb.K{K}"
    put(f"{k}.plan.full", num(plan_full, 2))
    put(f"{k}.rand.full", num(rnd_full, 2))
    put(f"{k}.plan.fast", num(float(model["measured_40_pct"]), 2))
    put(f"{k}.diff", num(diff, 2, sign=True))
    put(f"{k}.diff.short", num(diff, 1, sign=True))
    put(f"{k}.lo", num(lo, 2, sign=True))
    put(f"{k}.hi", num(hi, 2, sign=True))
    put(f"{k}.abs", num(abs(diff), 1))
    put(f"{k}.pred.pw", num(float(model["pred_pairwise_pct"]), 2))
    put(f"{k}.pred.lin", num(float(model["pred_additive_pct"]), 2))
    put(f"{k}.pred.log", num(float(model["pred_log_pct"]), 2))
    put(f"{k}.saved.params", num(K * saved08 / 1e6, 1))
    put(f"{k}.saved.frac", num(100 * K * saved08 / (28 * 9437184), 1))
    mb_rows.append([K, plan_full, rnd_full, diff, lo, hi, int(diff < 0)])
    sa_rows.append([K, float(model["pred_pairwise_pct"]), float(model["measured_40_pct"]), float(model["pred_additive_pct"]),
                    float(model["pred_log_pct"]), plan_full])
check(values["mb.K24.diff"].startswith("+"), "K=24: the model plan is worse than the best random plan")
write_csv("matched_bytes.csv", ["K", "plan_full", "random_full", "diff_pp", "lo_pp", "hi_pp", "plan_better"], mb_rows)
write_csv("superadditivity.csv", ["K", "pred_pairwise", "measured_fast", "pred_linear", "pred_log", "measured_full"], sa_rows)
k24 = next(r for r in mb if r["plan"] == "mb_K24_pairwise")
put("mb.K24.layers", "0, 2" + NDASH + "14, 16" + NDASH + "25")
check(k24["layers"] == " ".join(str(L) for L in [0] + list(range(2, 15)) + list(range(16, 26))),
      "K=24 model plan compresses layers 0, 2-14, 16-25")
check(close(float(k24["measured_40_pct"]) / float(k24["pred_additive_pct"]), 4.77, 0.01), "K=24: measured / linear sum = 4.77")
put("mb.K24.ratio", num(float(k24["measured_40_pct"]) / float(k24["pred_additive_pct"]), 1))
put("mb.boot.reps", "4,000")

# ----------------------------------------------------------------------------- closed loop (S4)
LAYERS, KEEPS = tuple(range(2, 26)), (0.9, 0.8)
units = saved_units(KEEPS)
for K in (12, 18):
    sp = Space(LAYERS, KEEPS, units, K * units[1])
    put(f"loop.K{K}.count", grouped(sp.count()))
    put(f"loop.K{K}.count.exp", str(int(math.log10(sp.count()))))
    put(f"loop.K{K}.count.mant", f"{sp.count() / 10 ** int(math.log10(sp.count())):.1f}")
    check(sp.n == 48, "loop search space: 24 MLP blocks x 2 compressed options = 48 binaries")
check(values["loop.K12.count"] == "9,928,631,398", "K=12 budget: 9,928,631,398 feasible plans")
put("loop.nbin", "48")

summary = read("loop/summary.csv")
evals = read("loop/evals.csv")
calib = read("loop/calibration.csv")
gaps = read("loop/solver_gaps.csv")


def srow(K: int, method: str, seed: str) -> dict:
    return next(r for r in summary if r["K"] == str(K) and r["method"] == method and r["seed"] == seed)


per_method = defaultdict(list)
for r in evals:
    per_method[(int(r["K"]), r["method"], r["seed"])].append(r)
check(all(len(v) == 24 for v in per_method.values()), "every method and seed measured exactly 24 plans")
put("loop.evals", "24")
put("loop.seeds", "3")
put("loop.fm.rank", "6")
check({r["status"] for r in evals if r["method"] == "loop"} == {"OPTIMAL"}, "every loop acquisition was solved to proven optimality (CP-SAT)")

for K in (12, 18):
    one = srow(K, "oneshot-pw", "0")
    put(f"loop.K{K}.one.full", num(float(one["full_pct"]), 2))
    put(f"loop.K{K}.one.fast", num(float(one["best_40_pct"]), 2))
    rnd = [srow(K, "random", s) for s in "012"]
    put(f"loop.K{K}.rand.full.min", num(min(float(r["full_pct"]) for r in rnd), 2))
    put(f"loop.K{K}.rand.full.max", num(max(float(r["full_pct"]) for r in rnd), 2))
    put(f"loop.K{K}.rand.gap.min", num(min(float(r["vs_oneshot_pw_pp"]) for r in rnd), 1))
    put(f"loop.K{K}.rand.gap.max", num(max(float(r["vs_oneshot_pw_pp"]) for r in rnd), 1))
    same = [s for s in "012" if srow(K, "loop", s)["plan"] == one["plan"]]
    put(f"loop.K{K}.same", str(len(same)))
    local_same = [s for s in "012" if srow(K, "local", s)["plan"] == one["plan"]]
    check(len(local_same) == 3, f"K={K}: local search from the one-shot best never improves on it")
check(values["loop.K12.same"] == "3" and values["loop.K18.same"] == "2", "loop = one-shot plan in 3/3 seeds (K=12) and 2/3 (K=18)")
check(values["loop.K12.rand.gap.min"] == "20.2" and values["loop.K18.rand.gap.max"] == "85.3", "random search trails one-shot by 20.2-85.3 pp")

s2 = srow(18, "loop", "2")
na, nb = nll[(s2["plan"], "all")][0], nll[(srow(18, "oneshot-pw", "0")["plan"], "all")][0]
ci = paired_bootstrap(na, nb, per_chunk, dense_nll=dense_nll_all)
check(close(ci["diff"], float(s2["vs_oneshot_pw_pp"]), 1e-3) and close(ci["lo"], float(s2["ci95_lo_pp"]), 1e-3)
      and close(ci["hi"], float(s2["ci95_hi_pp"]), 1e-3), "K=18 seed 2 vs one-shot: bootstrap CI recomputed from chunk NLL")
put("loop.K18.s2.full", num(float(s2["full_pct"]), 2))
put("loop.K18.s2.fast", num(float(s2["best_40_pct"]), 2))
put("loop.K18.s2.diff", num(float(s2["vs_oneshot_pw_pp"]), 2, sign=True))
put("loop.K18.s2.lo", num(float(s2["ci95_lo_pp"]), 2, sign=True))
put("loop.K18.s2.hi", num(float(s2["ci95_hi_pp"]), 2, sign=True))
put("loop.K18.s2.fastgap", num(float(s2["best_40_pct"]) - float(srow(18, "oneshot-pw", "0")["best_40_pct"]), 2, sign=True))
k1 = {t.split(":")[0] for t in plan_rows[(s2["plan"], "40")]["keeps"].split()}
k2 = {t.split(":")[0] for t in plan_rows[(srow(18, "oneshot-pw", "0")["plan"], "40")]["keeps"].split()}
check(k1 - k2 == {"3"} and k2 - k1 == {"4"}, "K=18: seed-2 plan and one-shot plan differ by one swapped layer (3 vs 4)")
check(plan_rows[(s2["plan"], "40")]["method"] == "S4 oneshot-pw K18", "the seed-2 plan was also among the one-shot top-24 plans")

# surrogate calibration on the 72 held-out random plans
for K in (12, 18):
    rows = [r for r in calib if r["K"] == str(K)]
    check(all(r["held_out"] == "72" for r in rows), f"K={K}: calibration on 72 held-out random plans")
    rho = [float(r["spearman"]) for r in rows]
    bias = [float(r["bias_log"]) for r in rows]
    mae = [float(r["mae_log"]) for r in rows]
    put(f"cal.K{K}.rho.min", num(min(rho), 2))
    put(f"cal.K{K}.rho.max", num(max(rho), 2))
    put(f"cal.K{K}.mae.min", num(min(mae), 2))
    put(f"cal.K{K}.mae.max", num(max(mae), 2))
    put(f"cal.K{K}.bias.min", num(min(bias), 2))
    put(f"cal.K{K}.bias.max", num(max(bias), 2))
    check(max(bias) < 0, f"K={K}: surrogate under-predicts damage on held-out plans (negative mean bias)")
put("cal.bias.lo", num(min(float(r["bias_log"]) for r in calib), 2))
put("cal.bias.hi", num(max(float(r["bias_log"]) for r in calib), 2))

# regret curves: best measured delta PPL so far (40 chunks) against measured plans
for K in (12, 18):
    cols = ["n", "oneshot", "loop_med", "loop_min", "loop_max", "local_med", "local_min", "local_max",
            "random_med", "random_min", "random_max"]
    curves = {}
    for m in ("oneshot-pw", "loop", "local", "random"):
        runs = []
        for s in sorted({r["seed"] for r in evals if r["K"] == str(K) and r["method"] == m}):
            ys = [float(r["delta_pct"]) for r in sorted(per_method[(K, m, s)], key=lambda r: int(r["index"]))]
            runs.append(np.minimum.accumulate(ys))
        curves[m] = np.stack(runs)
    rows = []
    for i in range(24):
        rows.append([i + 1, float(curves["oneshot-pw"][0, i])] +
                    [float(f(curves[m][:, i])) for m in ("loop", "local", "random") for f in (np.median, np.min, np.max)])
    write_csv(f"regret_K{K}.csv", cols, rows)
    first = int(np.argmax(np.isclose(np.median(curves["loop"], 0), float(srow(K, "oneshot-pw", "0")["best_40_pct"]))))
    put(f"loop.K{K}.loop.catchup", str(first + 1))
    put(f"loop.K{K}.one.first", str(int(np.argmax(np.isclose(curves["oneshot-pw"][0], float(srow(K, "oneshot-pw", "0")["best_40_pct"])))) + 1))
    put(f"loop.K{K}.rand.best.fast.min", num(float(curves["random"][:, -1].min()), 2))
    put(f"loop.K{K}.rand.best.fast.max", num(float(curves["random"][:, -1].max()), 2))

# acquisition solvers on the loop's surrogate QUBOs (reference: exact CP-SAT optimum)
g = defaultdict(list)
for r in gaps:
    g[(r["K"], r["solver"])].append(r)
check(all(len(v) == 36 for v in g.values()), "36 acquisition rounds per K and solver")
opt = sum(1 for (K, s), rs in g.items() if s == "greedy" for r in rs if abs(float(r["gap_pct"])) < 1e-9)
put("solve.greedy.opt", str(opt))
put("solve.rounds", "72")
for K in ("12", "18"):
    put(f"solve.greedy.K{K}.median", num(st.median(float(r["gap_pct"]) for r in g[(K, "greedy")]), 1))
    put(f"solve.greedy.K{K}.opt", str(sum(abs(float(r["gap_pct"])) < 1e-9 for r in g[(K, "greedy")])))
    put(f"solve.greedy.K{K}.sec", num(st.median(float(r["seconds"]) for r in g[(K, "greedy")]), 2))
for s, tag in (("sb-ballistic", "sbb"), ("sb-discrete", "sbd"), ("sa", "sa")):
    bad = sum(r["raw_feasible"] == "False" for K in ("12", "18") for r in g[(K, s)])
    put(f"solve.{tag}.infeasible", str(bad))
check(values["solve.sbb.infeasible"] == "72", "ballistic SB raw samples violated the constraints in 72 of 72 rounds")
check(all(r["solver"] != "greedy" or r["raw_feasible"] == "True" for r in gaps), "greedy moves stay feasible")

# ----------------------------------------------------------------------------- KL proxy
kl = {r["proxy"]: r for r in read("sensitivity/qwen3-0.6b_kl_proxy_summary.csv")}
f = kl["KL(dense||comp)"]
check(f["layers"] == "28", "KL proxy scored on the 28 single-layer keep-0.8 rows")
put("kl.rho", num(float(f["rho"]), 2))
put("kl.lo", num(float(f["ci95_lo"]), 2))
put("kl.hi", num(float(f["ci95_hi"]), 2))
put("kl.rev.rho", num(float(kl["KL(comp||dense)"]["rho"]), 2))
put("kl.bar", "0.90")
check(float(f["rho"]) < 0.9 and float(kl["KL(comp||dense)"]["rho"]) < 0.9, "both KL directions fall below the 0.90 acceptance bar")

# ----------------------------------------------------------------------------- MPO vs SVD (layers 2-25, keep 0.9, full split)
mpo = next(r for r in comp if r["method"] == "mpo")
svd = next(r for r in comp if r["method"] == "svd")
check(mpo["layers"] == svd["layers"] == "2-25" and mpo["keep"] == svd["keep"] == "0.9", "MPO and SVD both compress layers 2-25 at keep 0.9")
check(mpo["rank"] == "bond 240" and svd["rank"] == "704", "MPO bond 240 matched to SVD rank 704")
put("mpo.ppl", num(float(mpo["ppl"]), 2))
put("mpo.delta", num(float(mpo["delta_pct"]), 1, sign=True))
put("svd.ppl", num(float(svd["ppl"]), 2))
put("svd.delta", num(float(svd["delta_pct"]), 1, sign=True))
put("mpo.bond", "240")
# three-core MPO parameters with bond 240 vs rank-704 SVD, per matrix (modes (16,4,16) x (16,12,16))
mpo_p = 16 * 16 * 240 + 240 * 12 * 4 * 240 + 240 * 16 * 16
svd_p = 704 * (1024 + 3072)
check(abs(mpo_p - svd_p) / svd_p < 0.002, "MPO (bond 240) and SVD (rank 704) parameter counts agree within 0.2%")
put("mpo.params.frac", num(100 * svd_p / (1024 * 3072), 1))
put("mpo.layers", "2" + NDASH + "25")

# ----------------------------------------------------------------------------- example plan (title motif)
motif = {int(t.split(":")[0]): float(t.split(":")[1]) for t in plan_rows[(s2["plan"], "all")]["keeps"].split()}
write_csv("plan_K18.csv", ["layer", "state"], [[L, ("keep08" if motif.get(L) == 0.8 else "keep09" if motif.get(L) == 0.9
                                                     else "dense" if 2 <= L <= 25 else "fixed")] for L in range(28)])
put("motif.n", str(len(motif)))

# ----------------------------------------------------------------------------- tables shared by the deck and the description
TABLES.mkdir(parents=True, exist_ok=True)
with (TABLES / "matched_bytes_rows.tex").open("w", encoding="utf-8") as fh:
    fh.write("% generated by build_data.py from results/matched_bytes/summary.csv and plan_chunk_nll.csv\n")
    for K in (6, 12, 18, 24):
        k = f"mb.K{K}"
        fh.write(f"{K} & {values[k + '.plan.full']} & {values[k + '.rand.full']} & "
                 f"{values[k + '.diff']} & [{values[k + '.lo']}, {values[k + '.hi']}] \\\\\n")
with (TABLES / "closed_loop_rows.tex").open("w", encoding="utf-8") as fh:
    fh.write("% generated by build_data.py from results/loop/summary.csv\n")
    fh.write(f"12 & {values['loop.K12.one.full']} & {values['loop.K12.one.full']} (3/3 seeds) & "
             f"{values['loop.K12.rand.full.min']}{NDASH}{values['loop.K12.rand.full.max']} \\\\\n")
    fh.write(f"18 & {values['loop.K18.one.full']} & {values['loop.K18.one.full']} (2/3), {values['loop.K18.s2.full']} (1/3) & "
             f"{values['loop.K18.rand.full.min']}{NDASH}{values['loop.K18.rand.full.max']} \\\\\n")

# ----------------------------------------------------------------------------- write outputs
GEN.mkdir(parents=True, exist_ok=True)
(GEN / "values.json").write_text(json.dumps(values, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
with (GEN / "values.tex").open("w", encoding="utf-8") as fh:
    fh.write("% generated by build_data.py from ../results -- do not edit by hand\n")
    for k in sorted(values):
        v = values[k].replace("%", "\\%")
        check(not any(c in v for c in "#$&_^{}\\") or v.startswith("\\"), f"value {k} is LaTeX-safe")
        fh.write(f"\\ievalue{{{k}}}{{{v}}}\n")
(GEN / "checks.txt").write_text("\n".join(f"[ok] {c}" for c in checks) + "\n", encoding="utf-8")
print(f"{len(values)} values, {len(checks)} checks passed -> {GEN.relative_to(ROOT)}, {DATA.relative_to(ROOT)}, "
      f"{TABLES.relative_to(ROOT)}")
