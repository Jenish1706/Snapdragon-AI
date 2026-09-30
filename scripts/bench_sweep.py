#!/usr/bin/env python3
"""llama-bench sweeps of Qwen3 Q4_0 on Android devices over adb (step 1 of PLAN.md §0).

For each connected device: push the pinned llama.cpp Snapdragon build (b11256) and the models,
probe CPU thread counts, then run prompt-processing tests (pp128..pp4096 at depth 0) and decode
tests (tg128 at depths 512..8192) on every unit the device supports (Pad 4: CPU, GPU, NPU;
Nord CE4: CPU). Raw JSONL goes to results/<device>/llama-bench/, one file per model x unit x test,
with the exact command and device state in results/<device>/runs.jsonl. Re-running skips files
that already exist. Battery current and voltage are sampled during each run if readable.

Run outside the sandbox (adb needs USB):  .venv/bin/python scripts/bench_sweep.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.bench import adb

BUILD = ROOT / "cache/bin/llama-b11256-bin-android-arm64-snapdragon/llama-b11256"
BUILD_TAG = "b11256"
MODELS = ROOT / "cache/models"
QWEN = ("Qwen3-0.6B-Q4_0.gguf", "Qwen3-1.7B-Q4_0.gguf", "Qwen3-4B-Q4_0.gguf")
PP = "128,256,512,1024,2048,4096"
DEPTHS = "512,1024,2048,4096,8192"
UNIT_ARGS = {
    "cpu": ["-dev", "none", "-ngl", "0"],
    "gpu": ["-dev", "GPUOpenCL", "-ngl", "99"],
    "npu": ["-dev", "HTP0", "-ngl", "99", "-ub", "1024"],
}
LOCAL_TZ = dt.datetime.now(dt.UTC).astimezone().tzinfo


def now() -> str:
    return dt.datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def device_state(serial: str) -> dict:
    temps = adb.sh(serial, "for z in /sys/class/thermal/thermal_zone*; do "
                           "echo $(cat $z/type 2>/dev/null)=$(cat $z/temp 2>/dev/null); done | head -80", check=False)
    return {"time": now(), "battery": adb.battery_probe(serial),
            "thermal": dict(t.split("=", 1) for t in temps.split() if "=" in t and t.split("=", 1)[1])}


def ensure_package(serial: str) -> None:
    marker = f"{adb.LLAMA_DIR}/.build-{BUILD_TAG}"
    if adb.sh(serial, f"[ -f {marker} ] && echo yes || echo no").strip() == "yes":
        return
    log(f"{serial}: pushing llama.cpp {BUILD_TAG} Snapdragon build")
    adb.sh(serial, f"mkdir -p {adb.LLAMA_DIR} {adb.MODEL_DIR}")
    for sub in ("bin", "lib"):
        adb.adb("push", str(BUILD / sub), f"{adb.LLAMA_DIR}/", serial=serial)
    adb.sh(serial, f"chmod 755 {adb.LLAMA_DIR}/bin/* && touch {marker}")


def ensure_model(serial: str, name: str) -> None:
    local = MODELS / name
    remote = f"{adb.MODEL_DIR}/{name}"
    size = adb.sh(serial, f"stat -c %s {remote} 2>/dev/null || echo 0").strip()
    if size == str(local.stat().st_size):
        return
    log(f"{serial}: pushing {name} ({local.stat().st_size / 1e6:.0f} MB)")
    adb.adb("push", str(local), remote, serial=serial)


def start_energy(serial: str, tag: str) -> str | None:
    """Start a 5 Hz battery sampler on the device; returns the remote log path, or None if unreadable."""
    if not adb.battery_probe(serial)["readable"]:
        return None
    path = f"{adb.REMOTE}/energy_{tag}.txt"
    b = adb.BATTERY
    loop = (f"while true; do echo $(cut -d' ' -f1 /proc/uptime) $(cat {b}/current_now) "
            f"$(cat {b}/voltage_now) $(cat {b}/status); sleep 0.2; done")
    adb.sh(serial, f"rm -f {path} {path}.pid; nohup sh -c '{loop}' > {path} 2>/dev/null & echo $! > {path}.pid")
    return path


def stop_energy(serial: str, path: str | None, out: Path) -> None:
    if path is None:
        return
    adb.sh(serial, f"kill $(cat {path}.pid) 2>/dev/null; rm -f {path}.pid", check=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    adb.adb("pull", path, str(out), serial=serial, check=False)


def run_bench(dev: adb.Device, model: str, unit: str, test: str, threads: int, reps: int, force: bool,
              extra: list[str] | None = None, tag: str | None = None) -> Path:
    out_dir = ROOT / "results" / dev.label / "llama-bench"
    tag = tag or f"{Path(model).stem}__{unit}__{test}"
    out = out_dir / f"{tag}.jsonl"
    if out.exists() and out.stat().st_size > 0 and not force:
        log(f"{dev.label}: skip {tag} (exists)")
        return out
    test_args = ["-p", PP, "-n", "0"] if test == "pp" else ["-p", "0", "-n", "128", "-d", DEPTHS]
    args = ["-m", f"{adb.MODEL_DIR}/{model}", *UNIT_ARGS[unit], "-t", str(threads), "-r", str(reps),
            "--delay", "5", "-o", "jsonl", *test_args, *(extra or [])]
    cmd = adb.llama_cmd("llama-bench", args, unit)
    before = device_state(dev.serial)
    epath = start_energy(dev.serial, tag)
    t0 = time.time()
    log(f"{dev.label}: {tag}")
    res = adb.adb("shell", cmd, serial=dev.serial, check=False, timeout=6 * 3600)
    wall = time.time() - t0
    stop_energy(dev.serial, epath, ROOT / "results" / dev.label / "energy" / f"{tag}.txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.log").write_text(res.stderr or "")
    rows = [line for line in (res.stdout or "").splitlines() if line.startswith("{")]
    if rows:
        out.write_text("\n".join(rows) + "\n")
    append_jsonl(ROOT / "results" / dev.label / "runs.jsonl", {
        "tag": tag, "model": model, "unit": unit, "test": test, "threads": threads, "reps": reps,
        "command": cmd, "returncode": res.returncode, "wall_s": round(wall, 1), "rows": len(rows),
        "build": BUILD_TAG, "before": before, "after": device_state(dev.serial), "energy_samples": epath})
    log(f"{dev.label}: {tag} rc={res.returncode} rows={len(rows)} wall={wall:.0f}s")
    return out


def probe_threads(dev: adb.Device, reps: int) -> int:
    """Pick the CPU thread count with the best tg128 on the smallest model (probe kept in results)."""
    best, best_tps = 6, -1.0
    for t in (2, 4, 6, 8):
        out = run_bench(dev, QWEN[0], "cpu", "probe", t, reps, False,
                        extra=["-p", "512", "-n", "128", "-d", "0"], tag=f"probe_threads_t{t}")
        for line in out.read_text().splitlines() if out.exists() else []:
            row = json.loads(line)
            if row.get("n_gen", 0) > 0 and row.get("avg_ts", 0) > best_tps:
                best, best_tps = t, row["avg_ts"]
    log(f"{dev.label}: CPU threads -> {best} (tg128 {best_tps:.1f} tok/s)")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated device labels to run (default: all known)")
    ap.add_argument("--units", help="override units, e.g. cpu,npu")
    ap.add_argument("--models", default=",".join(QWEN))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    devs = [d for d in adb.devices() if d.state == "device"]
    if not devs:
        sys.exit("no authorised adb devices; connect the Pad 4 / Nord CE4 with USB debugging enabled")
    for dev in devs:
        if args.only and dev.label not in args.only.split(","):
            continue
        info_path = ROOT / "results" / dev.label / "device.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        keep = {k: v for k, v in dev.props.items() if k.startswith(("ro.soc", "ro.product", "ro.board", "ro.build.version",
                                                                     "ro.hardware", "ro.vendor.build.fingerprint"))}
        info = {"serial_hint": dev.serial[-4:], "label": dev.label, "props": keep,
                "meminfo": adb.sh(dev.serial, "head -3 /proc/meminfo", check=False),
                "cpu_max_khz": adb.sh(dev.serial, "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq", check=False).split(),
                "battery": adb.battery_probe(dev.serial), "recorded": now(), "build": BUILD_TAG}
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True))
        ensure_package(dev.serial)
        for m in args.models.split(","):
            ensure_model(dev.serial, m)
        threads = probe_threads(dev, args.reps)
        units = args.units.split(",") if args.units else adb.UNITS_BY_LABEL.get(dev.label, ("cpu",))
        for m in args.models.split(","):
            for unit in units:
                for test in ("pp", "tg"):
                    run_bench(dev, m, unit, test, threads, args.reps, args.force)
    log("sweep finished")


if __name__ == "__main__":
    main()
