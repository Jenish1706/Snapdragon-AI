#!/bin/zsh
# Runs after the per-layer sensitivity sweep finishes (GPU work is serialised on the Mac).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q "^exit=" results/mac-m2/sensitivity/run.txt 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] sweep finished: $(tail -1 results/mac-m2/sensitivity/run.txt)"
# pairwise interactions -> non-chain planner QUBO -> perplexity of the chosen plans
$PY scripts/pairs_ppl.py pairs --n 40 --seed 0 2>&1 | grep -v MallocStack
$PY scripts/plan_pairwise.py 2>&1 | tail -40
$PY scripts/pairs_ppl.py plans --full --plans results/planner/pairwise_plans/*.json 2>&1 | grep -v MallocStack
$PY scripts/packer_pareto.py 2>&1 | tail -40
$PY scripts/compress_ppl.py --only imx-svd-k0.9,imx-mpoA-k0.9 2>&1 | grep -v MallocStack
for spec in "agent 1024 128" "decode 256 512" "prefill 4096 32"; do
  set -- ${=spec}
  $PY scripts/plan.py --tag $1 --prompt $2 --gen $3 --lam 0.05 --seeds 4 2>&1 | tail -6
done
read -r ALPHA BETA <<< "$($PY - <<'PYEOF'
import csv, sys
sys.path.insert(0, ".")
from isingedge.twin.model_facts import load_facts
m = load_facts()["Qwen3-1.7B-Q4_0"]
r = next(r for r in csv.DictReader(open("results/twin/fits.csv")) if r["stage"] == "published" and r["device"] == "Samsung Galaxy S26" and r["unit"] == "npu")
print(float(r["b_f"]) * m.flops_per_token, float(r["b_attn"]) * m.attn_flops_per_pair)
PYEOF
)"
echo "[$(date +%H:%M:%S)] packer eval on the Mac server; qubo-cost priced with the Galaxy S26 NPU fit: alpha=$ALPHA beta=$BETA"
$PY scripts/packer_eval.py --label mac-m2 --n 100 --seed 0 --budget 256 --lam 2.0 --alpha "$ALPHA" --beta "$BETA" 2>&1 | tail -2
$PY scripts/fit_twin.py && $PY scripts/report_twin.py && $PY scripts/build_proposal.py
echo "[$(date +%H:%M:%S)] queue done"
