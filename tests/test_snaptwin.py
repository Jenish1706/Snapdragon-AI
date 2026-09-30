import numpy as np

from isingedge.twin.model_facts import ModelFacts, load_facts
from isingedge.twin.snaptwin import (
    Sample,
    UnitFit,
    fit,
    leave_one_model_out,
    predict,
    samples_from_llama_bench,
)

FACTS = load_facts()
TRUE = UnitFit(a_w=1 / 60e9, a_kv=1 / 20e9, t0=2e-3, b_f=1 / 400e9, b_attn=1 / 150e9, b0=5e-3)


def _synthetic(unit: UnitFit, noise: float = 0.0) -> list[Sample]:
    rng = np.random.default_rng(1)
    out = []
    for name, m in FACTS.items():
        for n in (128, 256, 512, 1024, 2048, 4096):
            out.append(Sample(name, "pp", n, 0, unit.prefill_s(m, n) * (1 + noise * rng.standard_normal())))
        for d in (512, 1024, 2048, 4096, 8192):
            out.append(Sample(name, "tg", 128, d, unit.decode_s(m, d + 64) * (1 + noise * rng.standard_normal())))
    return out


def test_facts_are_the_three_qwen3_models():
    assert set(FACTS) == {"Qwen3-0.6B-Q4_0", "Qwen3-1.7B-Q4_0", "Qwen3-4B-Q4_0"}
    m = FACTS["Qwen3-1.7B-Q4_0"]
    assert m.kv_bytes_per_token == 2 * 28 * 8 * 128 * 2
    assert 0.9e9 < m.weight_bytes < 1.2e9


def test_fit_recovers_known_coefficients_without_noise():
    got = fit(_synthetic(TRUE), FACTS)
    for k, v in TRUE.as_dict().items():
        assert abs(got.as_dict()[k] - v) / v < 1e-3, k


def test_leave_one_model_out_error_is_small_on_consistent_data():
    recs = leave_one_model_out(_synthetic(TRUE, noise=0.02), FACTS)
    assert len(recs) == 33
    assert np.median([r["ape"] for r in recs]) < 0.05


def test_bandwidth_scaling_halves_bandwidth_bound_decode_time():
    m = FACTS["Qwen3-4B-Q4_0"]
    slow = UnitFit(1 / 50e9, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(slow.scaled(2.0).decode_s(m, 0) - slow.decode_s(m, 0) / 2) < 1e-12
    assert abs(slow.effective_bandwidth_gbs() - 50) < 1e-9


def test_llama_bench_rows_become_samples():
    rows = [{"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ns": 2e9},
            {"n_prompt": 0, "n_gen": 128, "n_depth": 4096, "avg_ns": 6.4e9}]
    s = samples_from_llama_bench(rows, "Qwen3-1.7B-Q4_0")
    assert s[0] == Sample("Qwen3-1.7B-Q4_0", "pp", 512, 0, 2.0)
    assert s[1].kind == "tg" and abs(s[1].seconds - 0.05) < 1e-12 and s[1].depth == 4096
    assert predict(TRUE, s[1], FACTS) > 0


def test_modelfacts_is_hashable_dataclass():
    m = ModelFacts("x", 1, 1, 1, 1, 1, 1, 1)
    assert m.flops_per_token == 2 and m.attn_flops_per_pair == 4
