"""Paired bootstrap over evaluation chunks, and rank correlation."""

from __future__ import annotations

import numpy as np


def ppl_from_chunks(nll: np.ndarray, per_chunk: int, idx: np.ndarray | None = None) -> np.ndarray | float:
    """Perplexity of the chunks selected by idx (rows of idx are bootstrap resamples)."""
    nll = np.asarray(nll, dtype=float)
    if idx is None:
        return float(np.exp(nll.sum() / (per_chunk * len(nll))))
    return np.exp(nll[idx].sum(-1) / (per_chunk * idx.shape[-1]))


def paired_bootstrap(nll_a, nll_b, per_chunk: int, dense_nll=None, reps: int = 4000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """PPL_a - PPL_b with a paired percentile interval, resampling chunks with replacement.

    With dense_nll the difference is in percentage points of the dense PPL on the same resample.
    """
    a, b = np.asarray(nll_a, dtype=float), np.asarray(nll_b, dtype=float)
    assert a.shape == b.shape
    idx = np.random.default_rng(seed).integers(0, len(a), size=(reps, len(a)))
    pa, pb = ppl_from_chunks(a, per_chunk, idx), ppl_from_chunks(b, per_chunk, idx)
    point = ppl_from_chunks(a, per_chunk) - ppl_from_chunks(b, per_chunk)
    diff = pa - pb
    if dense_nll is not None:
        d = np.asarray(dense_nll, dtype=float)
        diff = 100 * diff / ppl_from_chunks(d, per_chunk, idx)
        point = 100 * point / ppl_from_chunks(d, per_chunk)
    lo, hi = np.quantile(diff, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))
