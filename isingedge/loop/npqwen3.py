"""Qwen3 forward pass in numpy (fp32, CPU), for machines where PyTorch cannot run.

Reads the Hugging Face safetensors checkpoint directly (bf16 widened to fp32 exactly) and implements the
Qwen3 decoder: RMSNorm, per-head q/k RMSNorm, rotary embeddings, grouped-query causal attention, SiLU-gated
MLP, tied LM head. `NumpyQwen3` exposes the small surface Qwen3Oracle needs (hidden states for a batch of
chunks, and the MLP weights as writable arrays).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """All tensors of a .safetensors file as fp32 numpy arrays (bf16 and f16 are widened exactly)."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        buf = raw[base + a:base + b]
        dt = meta["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(buf, dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(meta["shape"])
    return out


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


class NumpyQwen3:
    def __init__(self, model_dir: Path):
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        w = load_safetensors(Path(model_dir) / "model.safetensors")
        self.n_layers, self.eps = cfg["num_hidden_layers"], cfg["rms_norm_eps"]
        self.n_heads, self.n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        self.embed = w["model.embed_tokens.weight"]
        self.lm_head = w.get("lm_head.weight", self.embed)
        self.norm = w["model.norm.weight"]
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append({k: w[p + k + ".weight"] for k in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")})
        inv = 1.0 / (cfg.get("rope_theta", 1e6) ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim))
        self._inv_freq = inv

    def mlp_weight(self, layer: int, name: str) -> np.ndarray:
        """The writable fp32 array of mlp.{gate,up,down}_proj ([out, in], like nn.Linear.weight)."""
        return self.layers[layer]["mlp." + name]

    def _rope(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        f = np.outer(np.arange(t, dtype=np.float64), self._inv_freq)
        emb = np.concatenate([f, f], -1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def hidden(self, ids: np.ndarray, hook=None) -> np.ndarray:
        """Final-norm hidden states [B, T, d] for token ids [B, T]; hook(layer, name, x) sees MLP inputs."""
        b, t = ids.shape
        hd, nh, nk = self.head_dim, self.n_heads, self.n_kv
        h = self.embed[ids]
        cos, sin = self._rope(t)
        mask = np.triu(np.full((t, t), -np.inf, dtype=np.float32), 1)
        rep = nh // nk
        for li, L in enumerate(self.layers):
            x = _rms(h, L["input_layernorm"], self.eps)
            q = (x @ L["self_attn.q_proj"].T).reshape(b, t, nh, hd)
            k = (x @ L["self_attn.k_proj"].T).reshape(b, t, nk, hd)
            v = (x @ L["self_attn.v_proj"].T).reshape(b, t, nk, hd)
            q = _rms(q, L["self_attn.q_norm"], self.eps).transpose(0, 2, 1, 3)
            k = _rms(k, L["self_attn.k_norm"], self.eps).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            half = hd // 2
            q = q * cos + np.concatenate([-q[..., half:], q[..., :half]], -1) * sin
            k = k * cos + np.concatenate([-k[..., half:], k[..., :half]], -1) * sin
            k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            s = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(hd)) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(0, 2, 1, 3).reshape(b, t, nh * hd)
            h = h + a @ L["self_attn.o_proj"].T
            x = _rms(h, L["post_attention_layernorm"], self.eps)
            if hook:
                hook(li, "gate_proj", x)
            g = x @ L["mlp.gate_proj"].T
            u = x @ L["mlp.up_proj"].T
            m = g / (1.0 + np.exp(-g)) * u
            if hook:
                hook(li, "down_proj", m)
            h = h + m @ L["mlp.down_proj"].T
        return _rms(h, self.norm, self.eps)

    def nll(self, ids: np.ndarray, first: int, npos: int) -> np.ndarray:
        """Token NLL [B, npos] of ids[:, first+1:first+1+npos] from logits at first .. first+npos-1."""
        hs = self.hidden(ids)[:, first:first + npos]
        logits = hs @ self.lm_head.T
        mx = logits.max(-1, keepdims=True)
        lse = mx[..., 0] + np.log(np.exp(logits - mx).sum(-1))
        tgt = np.take_along_axis(logits, ids[:, first + 1:first + 1 + npos, None], -1)[..., 0]
        return (lse - tgt).astype(np.float64)
