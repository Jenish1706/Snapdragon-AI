"""SnapTwin performance model for prefill and autoregressive decode across compute engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import nnls

from isingedge.twin.model_facts import ModelFacts


@dataclass(frozen=True)
class Sample:
    model: str
    kind: str  # "pp" for prompt processing (prefill), "tg" for token generation (decode)
    prompt_tokens: int
    depth: int
    seconds: float


@dataclass(frozen=True)
class UnitFit:
    a_w: float = 0.0
    a_kv: float = 0.0
    t0: float = 0.0
    b_f: float = 0.0
    b_attn: float = 0.0
    b0: float = 0.0

    def prefill_s(self, m: ModelFacts, n: int) -> float:
        return self.b_f * m.flops_per_token * n + self.b_attn * m.attn_flops_per_pair * n * (n + 1) / 2.0 + self.b0

    def decode_s(self, m: ModelFacts, depth: float) -> float:
        return self.a_w * m.weight_bytes + self.a_kv * m.kv_bytes_per_token * depth + self.t0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def scaled(self, bw_ratio: float) -> UnitFit:
        factor = bw_ratio if bw_ratio > 0 else 1.0
        return UnitFit(
            a_w=self.a_w / factor,
            a_kv=self.a_kv / factor,
            t0=self.t0,
            b_f=self.b_f,
            b_attn=self.b_attn,
            b0=self.b0,
        )

    def effective_bandwidth_gbs(self) -> float:
        return 1.0 / (self.a_w * 1e9) if self.a_w > 0 else 0.0


def ape(pred: float, actual: float) -> float:
    return abs(pred - actual) / actual if actual > 0 else 0.0


def predict(unit: UnitFit, sample: Sample, facts: dict[str, ModelFacts]) -> float:
    m = facts[sample.model]
    if sample.kind == "pp":
        return unit.prefill_s(m, sample.prompt_tokens)
    eff_depth = sample.depth + (sample.prompt_tokens / 2.0 if sample.prompt_tokens > 0 else 0.0)
    return unit.decode_s(m, eff_depth)


def fit(samples: list[Sample], facts: dict[str, ModelFacts]) -> UnitFit:
    pp = [s for s in samples if s.kind == "pp"]
    tg = [s for s in samples if s.kind == "tg"]

    b_f, b_attn, b0 = 0.0, 0.0, 0.0
    if pp:
        A_pp = np.array([
            [facts[s.model].flops_per_token * s.prompt_tokens,
             facts[s.model].attn_flops_per_pair * s.prompt_tokens * (s.prompt_tokens + 1) / 2.0,
             1.0]
            for s in pp
        ])
        y_pp = np.array([s.seconds for s in pp])
        coef_pp, _ = nnls(A_pp, y_pp)
        b_f, b_attn, b0 = float(coef_pp[0]), float(coef_pp[1]), float(coef_pp[2])

    a_w, a_kv, t0 = 0.0, 0.0, 0.0
    if tg:
        A_tg = np.array([
            [facts[s.model].weight_bytes,
             facts[s.model].kv_bytes_per_token * (s.depth + s.prompt_tokens / 2.0),
             1.0]
            for s in tg
        ])
        y_tg = np.array([s.seconds for s in tg])
        coef_tg, _ = nnls(A_tg, y_tg)
        a_w, a_kv, t0 = float(coef_tg[0]), float(coef_tg[1]), float(coef_tg[2])

    return UnitFit(a_w=a_w, a_kv=a_kv, t0=t0, b_f=b_f, b_attn=b_attn, b0=b0)


def leave_one_model_out(samples: list[Sample], facts: dict[str, ModelFacts]) -> list[dict]:
    models = sorted({s.model for s in samples})
    records = []
    for held in models:
        train = [s for s in samples if s.model != held]
        test = [s for s in samples if s.model == held]
        if not train or not test:
            continue
        u = fit(train, facts)
        for s in test:
            pred = predict(u, s, facts)
            records.append({
                "model": s.model,
                "kind": s.kind,
                "pred": pred,
                "actual": s.seconds,
                "ape": ape(pred, s.seconds),
            })
    return records


def samples_from_llama_bench(rows: list[dict], model: str) -> list[Sample]:
    samples = []
    for r in rows:
        n_prompt = r.get("n_prompt", 0)
        n_gen = r.get("n_gen", 0)
        n_depth = r.get("n_depth", 0)
        avg_ns = r.get("avg_ns", 0.0)
        if n_prompt > 0 and n_gen == 0:
            samples.append(Sample(model, "pp", n_prompt, 0, avg_ns / 1e9))
        elif n_gen > 0:
            # seconds per token
            seconds_per_token = (avg_ns / 1e9) / n_gen
            samples.append(Sample(model, "tg", n_gen, n_depth, seconds_per_token))
    return samples
