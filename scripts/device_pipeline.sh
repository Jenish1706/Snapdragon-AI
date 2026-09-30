#!/bin/zsh
# One command for everything that needs the devices, in the order PLAN.md §0 requires.
# Run outside any sandbox (adb needs USB). Each step is resumable; results land in results/<device>/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
adb start-server >/dev/null 2>&1
adb devices -l
echo "[$(date +%H:%M:%S)] 1/5 llama-bench sweeps (Pad 4 CPU/GPU/NPU, Nord CE4 CPU)"
$PY scripts/bench_sweep.py || exit 1
echo "[$(date +%H:%M:%S)] 2/5 SnapTwin refit on the devices; Galaxy S26 cross-check; PC transfer"
$PY scripts/fit_twin.py && $PY scripts/report_twin.py
echo "[$(date +%H:%M:%S)] 3/5 planner with Pad 4 costs, then plans on the Pad (switch costs calibrated first)"
for spec in "pad-agent 1024 128" "pad-decode 256 512" "pad-prefill 4096 32"; do
  set -- ${=spec}
  $PY scripts/plan.py --tag $1 --prompt $2 --gen $3 --lam 0.05 --seeds 2 --cost-device "OnePlus Pad 4" --cost-stage device | tail -3
done
$PY scripts/run_plans.py --plans results/planner/plan_pad-*_executable.json
echo "[$(date +%H:%M:%S)] 4/5 packer on the Pad (model served on the NPU, solve on the device)"
$PY scripts/packer_eval.py --pad --pad-unit npu --label pad4 --n 100 --seed 0 --budget 256
echo "[$(date +%H:%M:%S)] 5/5 rebuild docs"
$PY scripts/build_proposal.py
echo "[$(date +%H:%M:%S)] device pipeline done"
