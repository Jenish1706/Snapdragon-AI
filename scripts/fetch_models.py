#!/usr/bin/env python3
"""Download the open-source model files used on device and verify them against their LFS sha256.

Qwen3 is Apache-2.0. The Q4_0 files are the exact files Qualcomm's GenieX benchmark matrix uses
(sdk/benchmark/qdc/bench-models.json): ggml-org for 0.6B, unsloth for 1.7B and 4B, so on-device
numbers are directly comparable with Qualcomm's published rows. Writes cache/models/manifest.json.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "models"
FILES = [
    ("ggml-org/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_0.gguf"),
    ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_0.gguf"),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_0.gguf"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-BF16.gguf"),
]


def lfs_oid(repo: str, path: str) -> tuple[str, int, str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main") as r:
        items = json.load(r)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}") as r:
        sha = json.load(r)["sha"]
    item = next(i for i in items if i["path"] == path)
    return item["lfs"]["oid"], int(item["size"]), sha


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for repo, path in FILES:
        if names and path not in names:
            continue
        dest = OUT / path
        oid, size, rev = lfs_oid(repo, path)
        if dest.exists() and manifest.get(path, {}).get("sha256") == oid and dest.stat().st_size == size:
            print(f"ok (cached) {path}", flush=True)
            continue
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
        print(f"downloading {path} ({size / 1e6:.0f} MB) from {repo}@{rev[:12]}", flush=True)
        h = hashlib.sha256()
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
                h.update(chunk)
        if h.hexdigest() != oid:
            tmp.unlink()
            sys.exit(f"sha256 mismatch for {path}")
        tmp.replace(dest)
        manifest[path] = {"repo": repo, "revision": rev, "sha256": oid, "bytes": size, "licence": "Apache-2.0"}
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"verified {path} sha256 {oid[:16]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
