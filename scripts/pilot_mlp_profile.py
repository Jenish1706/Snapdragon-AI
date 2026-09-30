#!/usr/bin/env python3
"""Optional AI Hub pilot: one Qwen3-1.7B MLP layer, dense vs SVD vs MPO, on hosted X Elite / X2 Elite.

Dormant under the open-source-only direction (docs/PLAN.md §0); run it only if an AI Hub token appears.

  prepare  fetch the layer's weights with byte-range reads, factorise, write ONNX + prepare.json
  run      upload, compile (qnn_dlc, W4A16), profile on hosted devices, then report
  report   summarise cached profiles into docs/pilot-mlp-layer14.md and the marked doc blocks

The AI Hub token comes from the environment only (QAI_HUB_API_TOKEN) and is never printed or
written to disk. Weights come from the public Apache-2.0 Qwen3 repository; no Hugging Face token is used. Every AI Hub id is cached in cache/pilot/manifest.json, so a re-run
resumes instead of resubmitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isingedge.profile.hf_weights import bf16_to_float32, read_tensors
from isingedge.profile.mlp_graphs import (
    MPO,
    Dense,
    LowRank,
    MPOSpec,
    keep_fraction_rank,
    max_tensor_rank,
    mlp_apply,
    mlp_model,
    relative_error,
)

CACHE = ROOT / "cache" / "pilot"
LAYER = 14
HIDDEN, INTER = 2048, 6144
REPO = "Qwen/Qwen3-1.7B"
SHARD = "model-00001-of-00002.safetensors"
SVD_KEEP = 0.8
MPO_UP = MPOSpec(in_modes=(16, 8, 16), out_modes=(32, 6, 32), bonds=(64, 64))
MPO_DOWN = MPOSpec(in_modes=(32, 6, 32), out_modes=(16, 8, 16), bonds=(64, 64))
DEVICES = ("Snapdragon X Elite CRD", "Snapdragon X2 Elite CRD")
SEQ_LENS = (1, 128)
COMPILE_OPTIONS = "--target_runtime qnn_dlc --quantize_full_type w4a16 --quantize_io"
VARIANTS = ("dense", "svd", "mpo")
LABELS = {
    "dense": "Dense",
    "svd": f"SVD, keep {SVD_KEEP} (rank {keep_fraction_rank(HIDDEN, INTER, SVD_KEEP)})",
    "mpo": "MPO, 3 cores, bond 64",
}
PROPOSAL = ROOT / "docs" / "PROPOSAL-v2.md"
RESULTS = ROOT / "docs" / "RESULTS.md"
PILOT_DOC = ROOT / "docs" / "pilot-aihub-qwen3-mlp.md"
START, END = "<!-- pilot:start -->", "<!-- pilot:end -->"
LOCAL_TZ = dt.datetime.now(dt.UTC).astimezone().tzinfo


def today() -> str:
    return dt.datetime.now(LOCAL_TZ).date().isoformat()


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


# --------------------------------------------------------------------------- prepare


def load_layer_weights(repo: str | None) -> tuple[dict[str, np.ndarray], dict]:
    names = {p: f"model.layers.{LAYER}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")}
    npz, meta_path = CACHE / f"weights_l{LAYER}.npz", CACHE / f"weights_l{LAYER}.json"
    if npz.exists() and meta_path.exists():
        raw = np.load(npz)
        log(f"weights: cached {npz.relative_to(ROOT)}")
        return {p: bf16_to_float32(raw[p]) for p in names}, json.loads(meta_path.read_text())
    from huggingface_hub import HfApi

    repo = repo or REPO
    sha = HfApi().model_info(repo).sha
    log(f"weights: reading layer {LAYER} MLP from {repo}@{sha[:12]} ({SHARD}) with range requests")
    t0 = time.time()
    got = read_tensors(repo, SHARD, list(names.values()), revision=sha)
    w = {p: got[n] for p, n in names.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **{p: (a.view(np.uint32) >> 16).astype(np.uint16) for p, a in w.items()})
    meta = {"repo": repo, "revision": sha, "file": SHARD, "tensors": names, "layer": LAYER,
            "seconds": round(time.time() - t0, 1)}
    save_json(meta_path, meta)
    return w, meta


def build_variants(w: dict[str, np.ndarray]) -> dict[str, tuple]:
    r = keep_fraction_rank(HIDDEN, INTER, SVD_KEEP)
    return {
        "dense": (Dense.from_w(w["gate"]), Dense.from_w(w["up"]), Dense.from_w(w["down"])),
        "svd": tuple(LowRank.from_w(w[p], r) for p in ("gate", "up", "down")),
        "mpo": (MPO.from_w(w["gate"], MPO_UP), MPO.from_w(w["up"], MPO_UP), MPO.from_w(w["down"], MPO_DOWN)),
    }


def cmd_prepare(args) -> None:
    import onnx
    import onnxruntime as ort

    w, source = load_layer_weights(args.repo)
    log("factorising: SVD and MPO (float64)")
    variants = build_variants(w)
    x = np.random.default_rng(0).standard_normal((1, HIDDEN)).astype(np.float32)
    out = {"date": today(), "layer": LAYER, "source": source,
           "compile_options": COMPILE_OPTIONS, "devices": list(DEVICES), "seq_lens": list(SEQ_LENS),
           "variants": {}}
    (CACHE / "onnx").mkdir(parents=True, exist_ok=True)
    for kind, lins in variants.items():
        errs, num, den = {}, 0.0, 0.0
        for part, lin in zip(("gate", "up", "down"), lins):
            ref = w[part].astype(np.float64)
            errs[part] = relative_error(ref, lin.dense_w())
            num += np.linalg.norm(ref - lin.dense_w()) ** 2
            den += np.linalg.norm(ref) ** 2
        errs["all"] = float(np.sqrt(num / den))
        model = mlp_model(*lins, HIDDEN, name=f"qwen3_1p7b_l{LAYER}_mlp_{kind}")
        ranks = {str(s): max_tensor_rank(model, s) for s in SEQ_LENS}
        path = CACHE / "onnx" / f"mlp_l{LAYER}_{kind}.onnx"
        onnx.save(model, str(path))
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (y,) = sess.run(None, {"x": x})
        ort_err = float(np.max(np.abs(y - mlp_apply(x, *lins))) / (np.max(np.abs(y)) + 1e-12))
        params = sum(lin.params for lin in lins)
        out["variants"][kind] = {
            "label": LABELS[kind],
            "params": params,
            "bytes_w4": params // 2,
            "macs_per_token": sum(lin.macs_per_token() for lin in lins),
            "rel_weight_error": errs,
            "max_tensor_rank": ranks,
            "onnx": str(path.relative_to(ROOT)),
            "onnx_bytes": path.stat().st_size,
            "onnxruntime_vs_numpy_max_rel_diff": ort_err,
            "ops": sorted({n.op_type for n in model.graph.node}),
        }
        v = out["variants"][kind]
        log(f"{kind}: params {params:,} | rel. weight error {errs['all']:.4f} | MACs/token "
            f"{v['macs_per_token']:,} | max rank {ranks} | ORT check {ort_err:.1e}")
    save_json(CACHE / "prepare.json", out)
    log(f"wrote {(CACHE / 'prepare.json').relative_to(ROOT)}")


# --------------------------------------------------------------------------- run


def client_from_env():
    token = os.environ.get("QAI_HUB_API_TOKEN")
    if not token:
        sys.exit("QAI_HUB_API_TOKEN is not set in this shell. Export it here and re-run; "
                 "the script keeps it in memory only.")
    import qai_hub as hub

    return hub, hub.Client(config=hub.ClientConfig(api_token=token))


def cmd_run(args) -> None:
    prep = load_json(CACHE / "prepare.json", None)
    if prep is None:
        sys.exit("run `prepare` first")
    hub, client = client_from_env()
    for name in DEVICES:
        if not client.get_devices(name=name):
            elite = sorted({d.name for d in client.get_devices() if "Elite" in d.name})
            sys.exit(f"hosted device {name!r} not found; available: {elite}")
    mpath = CACHE / "manifest.json"
    man = load_json(mpath, {"models": {}, "jobs": {}, "qai_hub_version": hub.__version__})
    for kind in VARIANTS:
        if kind not in man["models"]:
            path = ROOT / prep["variants"][kind]["onnx"]
            log(f"upload {kind} ({path.stat().st_size / 1e6:.0f} MB)")
            man["models"][kind] = client.upload_model(str(path), name=f"isingedge-pilot-l{LAYER}-{kind}").model_id
            save_json(mpath, man)
    for kind in VARIANTS:
        model = client.get_model(man["models"][kind])
        for dev in DEVICES:
            for s in SEQ_LENS:
                key = f"{kind}|{dev}|s{s}"
                rec = man["jobs"].setdefault(key, {})
                if "compile" not in rec:
                    job = client.submit_compile_job(
                        model=model, device=hub.Device(dev), name=f"isingedge-pilot-{kind}-s{s}",
                        input_specs={"x": ((s, HIDDEN), "float32")}, options=COMPILE_OPTIONS)
                    rec["compile"] = job.job_id
                    save_json(mpath, man)
                    log(f"compile submitted {key}: {job.job_id}")
    deadline = time.time() + args.timeout_min * 60
    while True:
        pending = 0
        for key, rec in man["jobs"].items():
            kind, dev, _ = key.split("|")
            if rec.get("done"):
                continue
            pending += 1
            if "profile" not in rec:
                job = client.get_job(rec["compile"])
                st = job.get_status()
                if st.code != rec.get("compile_state"):
                    rec["compile_state"] = st.code
                    log(f"compile {key}: {st.code} {st.message or ''}".rstrip())
                if st.failure:
                    rec.update(done=True, error=f"compile: {st.message}")
                elif st.success:
                    pj = client.submit_inference_job(model=job.get_target_model(), device=hub.Device(dev),
                                                     name=f"isingedge-pilot-{kind}-{key[-4:].strip('|')}",
                                                     profile=True)
                    rec["profile"] = pj.job_id
                    log(f"profile submitted {key}: {pj.job_id}")
            else:
                pj = client.get_job(rec["profile"])
                st = pj.get_status()
                if st.code != rec.get("profile_state"):
                    rec["profile_state"] = st.code
                    log(f"profile {key}: {st.code} {st.message or ''}".rstrip())
                if st.failure:
                    rec.update(done=True, error=f"profile: {st.message}")
                elif st.success:
                    prof = pj.download_profile()
                    save_json(CACHE / "profiles" / (key.replace("|", "__").replace(" ", "_") + ".json"), prof)
                    rec["done"] = True
                    log(f"profile done {key}: {prof['execution_summary']['estimated_inference_time']} us")
            save_json(mpath, man)
        if pending == 0:
            break
        if time.time() > deadline:
            log(f"timeout with {pending} jobs pending; re-run `run` to resume")
            break
        time.sleep(args.poll_s)
    cmd_report(args)


# --------------------------------------------------------------------------- report


def summarise(prof: dict) -> dict:
    summ, detail = prof["execution_summary"], prof.get("execution_detail", [])
    times = summ.get("all_inference_times") or [summ["estimated_inference_time"]]
    units: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for d in detail:
        units[d["compute_unit"]] = units.get(d["compute_unit"], 0) + 1
        if "execution_time" in d:
            by_type[d["type"]] = by_type.get(d["type"], 0) + int(d["execution_time"])
    return {"estimated_us": int(summ["estimated_inference_time"]), "median_us": int(statistics.median(times)),
            "min_us": int(min(times)), "max_us": int(max(times)), "runs": len(times),
            "peak_memory": summ.get("estimated_inference_peak_memory"), "units": units, "time_by_type_us": by_type}


def _fmt_us(v: int | None) -> str:
    if v is None:
        return "not run"
    return f"{v / 1000:.2f} ms" if v >= 10_000 else f"{v:,} µs"


def cmd_report(args) -> None:
    prep = load_json(CACHE / "prepare.json", None)
    man = load_json(CACHE / "manifest.json", {"jobs": {}})
    if prep is None:
        sys.exit("run `prepare` first")
    cells, rows_detail = {}, []
    for key, rec in man["jobs"].items():
        p = CACHE / "profiles" / (key.replace("|", "__").replace(" ", "_") + ".json")
        s = summarise(json.loads(p.read_text())) if p.exists() else None
        cells[key] = s
        rows_detail.append((key, rec, s))
    cols = [(dev, s) for s in SEQ_LENS for dev in DEVICES]
    head = ["Variant", "Parameters", "Weight bytes at 4 bits", "Relative weight error"] + [
        f"{d.replace('Snapdragon ', '').replace(' CRD', '')}, {s} token{'s' if s > 1 else ''}" for d, s in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for kind in VARIANTS:
        v = prep["variants"][kind]
        row = [v["label"], f"{v['params'] / 1e6:.2f} M", f"{v['bytes_w4'] / 1e6:.1f} MB",
               f"{v['rel_weight_error']['all']:.3f}" if kind != "dense" else "0"]
        for dev, s in cols:
            c = cells.get(f"{kind}|{dev}|s{s}")
            txt = _fmt_us(c["median_us"]) if c else "not run"
            if c and set(c["units"]) - {"NPU"}:
                txt += " (" + ", ".join(f"{u} {n}" for u, n in sorted(c["units"].items())) + " ops)"
            row.append(txt)
        lines.append("| " + " | ".join(row) + " |")
    measured = sum(1 for c in cells.values() if c)
    src = prep["variants"]["dense"]
    note = (
        f"Qwen3-1.7B layer {LAYER} MLP (gate, up, SiLU, down), weights from `{prep['source']['repo']}` "
        f"at revision `{prep['source']['revision'][:12]}`. Latency is the median of AI Hub's profiled runs of the "
        f"compiled graph on the hosted device (`{COMPILE_OPTIONS}`, random calibration data, so the numbers are "
        f"latency only). Parameters and bytes are arithmetic; relative weight error is ||W - W_hat||_F / ||W||_F "
        f"over the three matrices, measured locally with plain truncated SVD (no whitening, no healing). "
        f"{measured} of {len(DEVICES) * len(SEQ_LENS) * len(VARIANTS)} device cells measured; job ids are in "
        f"`docs/pilot-mlp-layer14.md` and `docs/RESULTS.md`."
    )
    table = "\n".join(lines)
    detail = ["| Cell | Compile job | Profile job | Median | Min–max | Runs | Compute units | State |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, rec, s in sorted(rows_detail):
        state = rec.get("error") or rec.get("profile_state") or rec.get("compile_state") or "submitted"
        detail.append("| " + " | ".join([
            key.replace("|", " / "), rec.get("compile", "–"), rec.get("profile", "–"),
            _fmt_us(s["median_us"]) if s else "–", f"{_fmt_us(s['min_us'])}–{_fmt_us(s['max_us'])}" if s else "–",
            str(s["runs"]) if s else "–", ", ".join(f"{u} {n}" for u, n in sorted(s["units"].items())) if s else "–",
            state]) + " |")
    doc = "\n".join([
        f"# Optional AI Hub pilot: one Qwen3-1.7B MLP layer on hosted Snapdragon NPUs ({today()})", "",
        "Generated by `scripts/pilot_mlp_profile.py report` from cached AI Hub results. Do not edit by hand.", "",
        table, "", note, "",
        "## Local facts per variant", "",
        "| Variant | MACs per token | Max tensor rank (1 / 128 tokens) | ONNX ops | Rel. error gate / up / down |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {prep['variants'][k]['label']} | {prep['variants'][k]['macs_per_token']:,} | "
          f"{prep['variants'][k]['max_tensor_rank']['1']} / {prep['variants'][k]['max_tensor_rank']['128']} | "
          f"{', '.join(prep['variants'][k]['ops'])} | "
          + " / ".join(f"{prep['variants'][k]['rel_weight_error'][p]:.3f}" for p in ('gate', 'up', 'down')) + " |"
          for k in VARIANTS],
        "", "## AI Hub jobs", "", *detail, "",
        f"qai-hub client {man.get('qai_hub_version', 'unknown')}; dense ONNX {src['onnx_bytes'] / 1e6:.0f} MB.", ""])
    PILOT_DOC.write_text(doc)
    log(f"wrote {PILOT_DOC.relative_to(ROOT)} ({measured} measured cells)")
    if args.write_docs:
        block = f"{START}\n{table}\n\n{note}\n{END}"
        for path in (PROPOSAL, RESULTS):
            text = path.read_text()
            if START in text and END in text:
                a, b = text.index(START), text.index(END) + len(END)
                path.write_text(text[:a] + block + text[b:])
                log(f"updated marked block in {path.relative_to(ROOT)}")
            else:
                log(f"no pilot markers in {path.relative_to(ROOT)}; left unchanged")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--repo", help=f"weights repo (default {REPO}, public, Apache-2.0)")
    for name in ("run", "report"):
        q = sub.add_parser(name)
        q.add_argument("--no-write-docs", dest="write_docs", action="store_false")
        q.add_argument("--timeout-min", type=float, default=240)
        q.add_argument("--poll-s", type=float, default=30)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
