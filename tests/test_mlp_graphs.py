import numpy as np
import pytest

from isingedge.profile.mlp_graphs import (
    MPO,
    Dense,
    LowRank,
    MPOSpec,
    keep_fraction_rank,
    max_tensor_rank,
    mlp_apply,
    mlp_model,
    mpo_cores,
    mpo_dense,
    relative_error,
    tensor_ranks,
)

RNG = np.random.default_rng(0)
HID, INTER = 24, 40  # 24 = 2*3*4, 40 = 2*5*4
UP_SPEC = MPOSpec(in_modes=(2, 3, 4), out_modes=(2, 5, 4), bonds=(3, 5))
DOWN_SPEC = MPOSpec(in_modes=(2, 5, 4), out_modes=(2, 3, 4), bonds=(3, 5))


def _w(d_out, d_in):
    return RNG.standard_normal((d_out, d_in)).astype(np.float32)


def test_keep_fraction_rank_matches_research_table():
    # tn-compression-recipes.md Table 3: Llama gate/up/down at keep 0.8 -> 1792, q/o 3072^2 -> 1216.
    assert keep_fraction_rank(3072, 8192, 0.8) == 1792
    assert keep_fraction_rank(3072, 3072, 0.8) == 1216


def test_svd_full_rank_is_exact():
    w = _w(INTER, HID)
    lin = LowRank.from_w(w, rank=HID)
    x = RNG.standard_normal((5, HID)).astype(np.float32)
    np.testing.assert_allclose(lin.apply(x), x @ w.T, rtol=1e-4, atol=1e-4)
    assert relative_error(w, lin.dense_w()) < 1e-6


def test_mpo_full_bonds_are_exact_and_truncation_is_consistent():
    w = _w(INTER, HID)
    full = MPOSpec(UP_SPEC.in_modes, UP_SPEC.out_modes, bonds=(10_000, 10_000))
    assert relative_error(w, mpo_dense(*mpo_cores(w, full))) < 1e-10
    lin = MPO.from_w(w, UP_SPEC)
    x = RNG.standard_normal((7, HID)).astype(np.float32)
    np.testing.assert_allclose(lin.apply(x), x @ lin.dense_w().T, rtol=1e-4, atol=1e-4)
    g1, g2, g3 = lin.cores()
    np.testing.assert_allclose(mpo_dense(g1, g2, g3), lin.dense_w(), rtol=1e-5, atol=1e-6)


def test_mpo_param_and_mac_counts():
    lin = MPO.from_w(_w(INTER, HID), UP_SPEC)
    (i1, i2, i3), (o1, o2, o3), (c1, c2) = lin.spec.in_modes, lin.spec.out_modes, lin.spec.bonds
    assert lin.params == o1 * i1 * c1 + c1 * o2 * i2 * c2 + c2 * o3 * i3
    assert lin.macs_per_token() == HID * c2 * o3 + i1 * o3 * i2 * c2 * o2 * c1 + o3 * o2 * i1 * c1 * o1


def _variants():
    wg, wu, wd = _w(INTER, HID), _w(INTER, HID), _w(HID, INTER)
    yield "dense", Dense.from_w(wg), Dense.from_w(wu), Dense.from_w(wd)
    yield "svd", LowRank.from_w(wg, 8), LowRank.from_w(wu, 8), LowRank.from_w(wd, 8)
    yield "mpo", MPO.from_w(wg, UP_SPEC), MPO.from_w(wu, UP_SPEC), MPO.from_w(wd, DOWN_SPEC)


@pytest.mark.parametrize("seq_len", [1, 128])
def test_every_tensor_has_rank_at_most_4(seq_len):
    for kind, gate, up, down in _variants():
        model = mlp_model(gate, up, down, HID)
        ranks = tensor_ranks(model, seq_len)
        assert len(ranks) == len(model.graph.node), kind
        assert max_tensor_rank(model, seq_len) <= 4, kind


def test_no_einsum_and_only_2d_matmul_weights():
    for kind, gate, up, down in _variants():
        model = mlp_model(gate, up, down, HID)
        assert {n.op_type for n in model.graph.node} <= {"MatMul", "Reshape", "Transpose", "Sigmoid", "Mul", "Identity"}
        weights = {i.name: i for i in model.graph.initializer}
        for n in model.graph.node:
            if n.op_type == "MatMul":
                assert len(weights[n.input[1]].dims) == 2, kind


@pytest.mark.parametrize("seq_len", [1, 5])
def test_onnx_runtime_matches_numpy_reference(seq_len):
    ort = pytest.importorskip("onnxruntime")
    x = RNG.standard_normal((seq_len, HID)).astype(np.float32)
    for kind, gate, up, down in _variants():
        model = mlp_model(gate, up, down, HID)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        (y,) = sess.run(None, {"x": x})
        np.testing.assert_allclose(y, mlp_apply(x, gate, up, down), rtol=1e-4, atol=1e-4, err_msg=kind)


def test_bf16_widening_and_header_parsing():
    import json
    import struct

    from isingedge.profile.hf_weights import bf16_to_float32, parse_header

    vals = np.array([1.0, -2.5, 3.140625, 0.0], dtype=np.float32)
    raw = (vals.view(np.uint32) >> 16).astype(np.uint16)  # exact for bf16-representable values
    np.testing.assert_array_equal(bf16_to_float32(raw), vals)
    hdr = json.dumps({"t": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    header, start = parse_header(struct.pack("<Q", len(hdr)) + hdr + raw.tobytes())
    assert header["t"]["shape"] == [2, 2] and start == 8 + len(hdr)


def test_equal_param_bond_matching():
    from isingedge.profile.mlp_graphs import match_mpo_bond, mpo_param_count

    # Qwen3-0.6B MLP (1024 <-> 3072), shape A modes; SVD keep 0.5 -> rank 384 -> 384 * 4096 params.
    target = 384 * (1024 + 3072)
    chi = match_mpo_bond(target, (16, 4, 16), (16, 12, 16))
    got = mpo_param_count((16, 4, 16), (16, 12, 16), (chi, chi))
    assert abs(got - target) / target < 0.01
    assert mpo_param_count((16, 12, 16), (16, 4, 16), (chi, chi)) == got  # down_proj mirrors gate/up
