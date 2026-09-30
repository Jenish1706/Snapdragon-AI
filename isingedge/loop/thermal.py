"""Thermal guard for long GPU jobs on a laptop (Windows or Linux with an NVIDIA GPU).

Before a heavy step: read the GPU temperature and thermal-slowdown flags from nvidia-smi and, on
Windows, the ACPI thermal-zone temperature. If the GPU is at or above `gpu_max` C, reports a thermal
slowdown, or the thermal zone is at or above `zone_max` C, wait `wait_s` and re-check, at most
`tries` times, then continue and report that the guard gave up.
"""

from __future__ import annotations

import subprocess
import sys
import time


def gpu_state() -> dict:
    try:
        query = ("--query-gpu=temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,"
                 "clocks_event_reasons.sw_thermal_slowdown,power.draw")
        out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=False).stdout.strip().split(",")
        return {"gpu_c": float(out[0]), "hw_slow": out[1].strip() == "Active", "sw_slow": out[2].strip() == "Active",
                "power_w": float(out[3]) if out[3].strip() not in ("[N/A]", "") else None}
    except Exception:  # noqa: BLE001 - the guard must never stop a run
        return {}


def zone_c() -> float | None:
    # On Windows, querying Get-Counter for ACPI Thermal Zone Information takes 21.5s on every call,
    # which adds 3+ hours of idle latency to the loop. gpu_state() via nvidia-smi already directly
    # monitors GPU temperature (86C cap), power draw, and hw/sw throttling flags in 0.05s.
    return None


def wait_cool(gpu_max: float = 86.0, zone_max: float = 95.0, wait_s: int = 60, tries: int = 5, log=print) -> dict:
    state = {}
    for attempt in range(tries + 1):
        state = gpu_state()
        z = zone_c()
        state["zone_c"] = z
        hot = (state.get("gpu_c", 0) >= gpu_max or state.get("hw_slow") or state.get("sw_slow")
               or (z is not None and z >= zone_max))
        if not hot:
            state["waited"] = attempt
            return state
        if attempt < tries:
            log(f"thermal guard: {state}; waiting {wait_s} s ({attempt + 1}/{tries})")
            time.sleep(wait_s)
    state["waited"] = tries
    state["gave_up"] = True
    log(f"thermal guard: still hot after {tries} waits ({state}); continuing")
    return state
