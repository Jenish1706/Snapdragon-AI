"""Measured objectives for the planning loop: Qwen3 perplexity in PyTorch, and mock oracles.

Qwen3Oracle replicates the Mac protocol (llama.cpp `llama-perplexity`, scripts/compress_ppl.py and
scripts/sensitivity_ppl.py) on a CUDA GPU, so rows are comparable in method but labelled by device:
  * WikiText-2 raw, tokenised whole without BOS, cut into chunks of `ctx` tokens; each chunk runs from
    scratch and only its second half is scored: logits at positions ctx/2 .. ctx-2 predict tokens
    ctx/2+1 .. ctx-1 (255 per 512-token chunk). PPL = exp(mean NLL); +/- = PPL * sqrt(var / (n - 1)).
  * Truncated SVD of each MLP matrix (gate, up, down) at keep_fraction_rank(d_in, d_out, keep).
    Activation-aware scaling factorises W diag(s), s_i = sqrt(mean x_i^2) + 1e-6 over 128 chunks of
    512 train tokens (the llama-imatrix statistic), then unscales. The reconstruction is rounded to
    fp16, as the Mac pipeline wrote it into an F16 GGUF; the model runs in fp32 (TF32 off).
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MLP = ("gate_proj", "up_proj", "down_proj")


def keep_fraction_rank(d_in: int, d_out: int, keep: float, multiple: int = 32) -> int:
    """`keep` of the two-factor break-even rank d_in*d_out/(d_in+d_out), rounded to a multiple of 32."""
    return max(multiple, round(keep * d_in * d_out / (d_in + d_out) / multiple) * multiple)


MPO_MODES = {1024: (16, 4, 16), 3072: (16, 12, 16)}  # shape "A" of scripts/compress_ppl.py


def mpo_params(in_modes, out_modes, chi: int) -> int:
    (i1, i2, i3), (o1, o2, o3) = in_modes, out_modes
    return o1 * i1 * chi + chi * o2 * i2 * chi + chi * o3 * i3


def match_mpo_bond(target: int, in_modes, out_modes) -> int:
    """Equal bond chi whose three-core MPO has the parameter count nearest to target."""
    (i1, i2, i3), (o1, o2, o3) = in_modes, out_modes
    a, b = o2 * i2, o1 * i1 + o3 * i3
    root = (-b + (b * b + 4 * a * target) ** 0.5) / (2 * a)
    cands = [c for c in (int(root) - 1, int(root), int(root) + 1, int(root) + 2) if c >= 1]
    return min(cands, key=lambda c: abs(mpo_params(in_modes, out_modes, c) - target))


def mpo_reconstruct(w, in_modes, out_modes, chi: int):
    """Dense reconstruction of a three-core TT-SVD (MPO) of w [d_out, d_in] with both bonds <= chi (torch)."""
    (i1, i2, i3), (o1, o2, o3) = in_modes, out_modes
    t = w.reshape(o1, o2, o3, i1, i2, i3).permute(0, 3, 1, 4, 2, 5).reshape(o1 * i1, o2 * i2 * o3 * i3)
    u, s, vt = _svd(t)
    c1 = min(chi, s.numel())
    rest = (s[:c1, None] * vt[:c1]).reshape(c1 * o2 * i2, o3 * i3)
    u2, s2, vt2 = _svd(rest)
    c2 = min(chi, s2.numel())
    m = (u2[:, :c2] @ (s2[:c2, None] * vt2[:c2])).reshape(c1, o2 * i2 * o3 * i3)
    t_hat = u[:, :c1] @ m
    return t_hat.reshape(o1, i1, o2, i2, o3, i3).permute(0, 2, 4, 1, 3, 5).reshape(o1 * o2 * o3, i1 * i2 * i3)


def _svd(t):
    import torch

    return torch.linalg.svd(t, full_matrices=False)


def scored_positions(ctx: int) -> tuple[int, int]:
    """(first, n): logits at first .. first+n-1 predict tokens first+1 .. ctx-1, as llama-perplexity."""
    first = ctx // 2
    return first, ctx - 1 - first


@dataclass
class PPL:
    """Perplexity over whole chunks, with per-chunk NLL sums kept for paired bootstrap."""

    nll: np.ndarray  # per-chunk sum of token NLL (nats)
    nll2: np.ndarray  # per-chunk sum of squared token NLL
    per_chunk: int
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def chunks(self) -> int:
        return len(self.nll)

    @property
    def ppl(self) -> float:
        return float(np.exp(self.nll.sum() / (self.per_chunk * self.chunks)))

    @property
    def err(self) -> float:
        n = self.per_chunk * self.chunks
        mean, m2 = self.nll.sum() / n, self.nll2.sum() / n
        var = m2 - mean * mean
        return float(self.ppl * np.sqrt(var / (n - 1))) if var > 0 else 0.0


class Qwen3Oracle:
    def __init__(self, model_dir: Path, wiki_dir: Path, cache_dir: Path, device: str = "cuda", ctx: int = 512,
                 batch: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device, self.ctx, self.batch = torch, device, ctx, batch
        self.model_dir, self.wiki_dir, self.cache_dir = Path(model_dir), Path(wiki_dir), Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=torch.float32)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32)
        self.model = model.float().to(device).eval()  # bf16 checkpoint -> fp32 is exact
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.dense = {(L, m): getattr(self.layers[L].mlp, m).weight.detach().to("cpu", copy=True)
                      for L in range(self.n_layers) for m in MLP}
        self._tokens: dict[str, np.ndarray] = {}
        self._scales: dict | None = None
        self._w: dict[tuple, dict] = {}  # (layer, keep, scaling, method) -> {matrix: fp16 tensor, "rel_error", ...}
        self._ref: dict[tuple, PPL] = {}

    # data --------------------------------------------------------------------------------------
    def tokens(self, split: str = "test") -> np.ndarray:
        if split not in self._tokens:
            raw = (self.wiki_dir / f"wiki.{split}.raw").read_bytes()
            path = self.cache_dir / f"tokens_{split}_{hashlib.sha256(raw).hexdigest()[:12]}.npy"
            if path.exists():
                ids = np.load(path)
            else:
                ids = np.asarray(self.tok(raw.decode("utf-8"), add_special_tokens=False)["input_ids"], dtype=np.int64)
                np.save(path, ids)
            self._tokens[split] = ids
        return self._tokens[split]

    def n_chunks(self, split: str = "test") -> int:
        return len(self.tokens(split)) // self.ctx

    # measurement -------------------------------------------------------------------------------
    def nll(self, n_chunks: int | None = None, split: str = "test") -> PPL:
        torch, ctx = self.torch, self.ctx
        ids = self.tokens(split)
        n = self.n_chunks(split) if n_chunks is None else min(n_chunks, self.n_chunks(split))
        first, npos = scored_positions(ctx)
        sums, sq = np.zeros(n), np.zeros(n)
        t0 = time.time()
        with torch.no_grad():
            for b0 in range(0, n, self.batch):
                b1 = min(n, b0 + self.batch)
                x = torch.from_numpy(ids[b0 * ctx:b1 * ctx].reshape(b1 - b0, ctx)).to(self.device)
                h = self.model.model(input_ids=x, use_cache=False).last_hidden_state[:, first:first + npos]
                logits = self.model.lm_head(h).float()
                nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                        x[:, first + 1:].reshape(-1), reduction="none")
                nll = nll.view(b1 - b0, npos).double()
                sums[b0:b1] = nll.sum(1).cpu().numpy()
                sq[b0:b1] = (nll * nll).sum(1).cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return PPL(sums, sq, npos, time.time() - t0)

    def dense_ref(self, n_chunks: int | None = None, split: str = "test") -> PPL:
        key = (n_chunks, split)
        if key not in self._ref:
            self._ref[key] = self.nll(n_chunks, split)
        return self._ref[key]

    def evaluate(self, plan: dict[int, float], n_chunks: int | None = 40, scaling: str = "imatrix",
                 method: str = "svd") -> PPL:
        with self.applied(plan, scaling, method):
            r = self.nll(n_chunks)
        d = self.dense_ref(n_chunks)
        r.extra = {"dense_ppl": d.ppl, "delta_pct": 100 * (r.ppl - d.ppl) / d.ppl,
                   "log_ratio": float(np.log(r.ppl / d.ppl))}
        return r

    # activation statistics and compression ----------------------------------------------------
    def scales(self, n_chunks: int = 128) -> dict[tuple[int, str], np.ndarray]:
        """RMS input activation per column of every MLP matrix over the first train chunks."""
        if self._scales is not None:
            return self._scales
        torch, ctx = self.torch, self.ctx
        path = self.cache_dir / f"imatrix_train_{n_chunks}x{ctx}.npz"
        if path.exists():
            z = np.load(path)
            self._scales = {(int(k.split("_", 1)[0][1:]), k.split("_", 1)[1]): z[k] for k in z.files}
            return self._scales
        acc: dict[tuple[int, str], torch.Tensor] = {}
        hooks = []

        def hook(key):
            def f(_mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).double()
                acc[key] = acc.get(key, 0) + (x * x).sum(0)
            return f

        for L in range(self.n_layers):
            hooks.append(self.layers[L].mlp.gate_proj.register_forward_pre_hook(hook((L, "gate_proj"))))
            hooks.append(self.layers[L].mlp.down_proj.register_forward_pre_hook(hook((L, "down_proj"))))
        ids = self.tokens("train")[:n_chunks * ctx].reshape(n_chunks, ctx)
        try:
            with torch.no_grad():
                for b0 in range(0, n_chunks, self.batch):
                    x = torch.from_numpy(ids[b0:b0 + self.batch]).to(self.device)
                    self.model.model(input_ids=x, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        out = {}
        for (L, m), v in acc.items():
            s = np.sqrt(v.cpu().numpy() / (n_chunks * ctx)) + 1e-6
            out[(L, m)] = s
            if m == "gate_proj":
                out[(L, "up_proj")] = s  # gate and up read the same input
        np.savez(path, **{f"L{L}_{m}": v for (L, m), v in out.items()})
        self._scales = out
        return out

    def prepare(self, layer: int, keeps: tuple[float, ...], scaling: str = "imatrix", method: str = "svd") -> None:
        """Factorise the layer's MLP once and cache the fp16 reconstruction at every keep.

        method "svd": truncated SVD at the keep-fraction rank; "mpo": three-core MPO (shape A) whose equal
        bond matches the SVD's parameter count, as scripts/compress_ppl.py.
        """
        torch = self.torch
        todo = [k for k in keeps if (layer, k, scaling, method) not in self._w]
        tag = "" if method == "svd" else f"_{method}"
        paths = {k: self.cache_dir / "svd" / f"L{layer:02d}_{scaling}_k{k}{tag}.pt" for k in todo}
        for k in list(todo):
            if paths[k].exists():
                self._w[(layer, k, scaling, method)] = torch.load(paths[k])
                todo.remove(k)
        if not todo:
            return
        sc = self.scales() if scaling == "imatrix" else None
        res = {k: {} for k in todo}
        errs = {k: [] for k in todo}
        for m in MLP:
            w = self.dense[(layer, m)].double()
            d_out, d_in = w.shape
            s = torch.from_numpy(sc[(layer, m)]) if sc is not None else None
            ws = w * s[None, :] if s is not None else w
            svd = torch.linalg.svd(ws, full_matrices=False) if method == "svd" else None
            for k in todo:
                r = keep_fraction_rank(d_in, d_out, k)
                if method == "svd":
                    u, sv, vt = svd
                    w_hat = (u[:, :r] * sv[:r]) @ vt[:r]
                else:
                    chi = match_mpo_bond(r * (d_in + d_out), MPO_MODES[d_in], MPO_MODES[d_out])
                    w_hat = mpo_reconstruct(ws, MPO_MODES[d_in], MPO_MODES[d_out], chi)
                    res[k]["bond"] = chi
                if s is not None:
                    w_hat = w_hat / s[None, :]
                w16 = w_hat.to(torch.float16)
                res[k][m] = w16
                errs[k].append(float(torch.linalg.norm(w - w16.double()) / torch.linalg.norm(w)))
        for k in todo:
            res[k]["rel_error"] = float(np.mean(errs[k]))
            res[k]["rank"] = keep_fraction_rank(*self.dense[(layer, "gate_proj")].shape[::-1], k)
            paths[k].parent.mkdir(parents=True, exist_ok=True)
            torch.save(res[k], paths[k])
            self._w[(layer, k, scaling, method)] = res[k]

    def info(self, layer: int, keep: float, scaling: str = "imatrix", method: str = "svd") -> dict:
        self.prepare(layer, (keep,), scaling, method)
        w = self._w[(layer, keep, scaling, method)]
        return {"rel_error": w["rel_error"], "rank": w["rank"], "bond": w.get("bond", "")}

    @contextlib.contextmanager
    def applied(self, plan: dict[int, float], scaling: str = "imatrix", method: str = "svd"):
        touched = []
        try:
            for L, keep in plan.items():
                if keep >= 1.0:
                    continue
                self.prepare(L, (keep,), scaling, method)
                w = self._w[(L, keep, scaling, method)]
                for m in MLP:
                    getattr(self.layers[L].mlp, m).weight.data.copy_(w[m].to(self.device, self.torch.float32))
                    touched.append((L, m))
            yield
        finally:
            for L, m in touched:
                getattr(self.layers[L].mlp, m).weight.data.copy_(self.dense[(L, m)].to(self.device))

    # forward-only proxy ------------------------------------------------------------------------
    def kl_many(self, plans: list[dict[int, float]], n_seq: int = 64, seq_len: int = 256, split: str = "valid",
                scaling: str = "imatrix") -> tuple[np.ndarray, np.ndarray]:
        """Per-sequence summed token KL in nats, both directions: (KL(dense || plan), KL(plan || dense)),
        each of shape [len(plans), n_seq]. Divide by seq_len for the per-token mean."""
        torch = self.torch
        ids = self.tokens(split)[:n_seq * seq_len].reshape(n_seq, seq_len)
        fwd, rev = np.zeros((len(plans), n_seq)), np.zeros((len(plans), n_seq))
        bs = max(1, self.batch // 2)  # two full-vocabulary log-prob tensors are alive at once
        with torch.no_grad():
            for b0 in range(0, n_seq, bs):
                x = torch.from_numpy(ids[b0:b0 + bs]).to(self.device)
                logp = torch.log_softmax(self.model(input_ids=x, use_cache=False).logits.float(), -1)
                for i, plan in enumerate(plans):
                    with self.applied(plan, scaling):
                        logq = torch.log_softmax(self.model(input_ids=x, use_cache=False).logits.float(), -1)
                    d = logp - logq
                    fwd[i, b0:b0 + bs] = (logp.exp() * d).sum((1, 2)).double().cpu().numpy()
                    rev[i, b0:b0 + bs] = (-(logq.exp() * d)).sum((1, 2)).double().cpu().numpy()
                    del logq, d
                del logp
        return fwd, rev


class MockOracle:
    """Synthetic objective with planted super-additive interactions, for tests and the synthetic path.

    y(x) = c.x + x'Wx + g * (c.x)^2 + noise, on the encoded binary vector x; y plays log(PPL/PPL_dense).
    """

    def __init__(self, n: int, seed: int = 0, gamma: float = 0.6, noise: float = 0.0):
        rng = np.random.default_rng(seed)
        self.c = rng.uniform(0.005, 0.05, n)
        w = rng.normal(0.0, 0.004, (n, n)) + (rng.uniform(size=(n, n)) < 0.1) * rng.uniform(0.0, 0.03, (n, n))
        w = np.triu(w, 1)
        self.w, self.gamma, self.noise, self.rng = w + w.T, gamma, noise, np.random.default_rng(seed + 1)
        self.calls = 0

    def true(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        lin = float(self.c @ x)
        return lin + 0.5 * float(x @ self.w @ x) + self.gamma * lin * lin

    def __call__(self, x: np.ndarray) -> float:
        self.calls += 1
        return self.true(x) + (self.rng.normal(0.0, self.noise) if self.noise else 0.0)
