import numpy as np

from isingedge.sensitivity.quant import q4_0_roundtrip, q8_0_roundtrip


def test_q4_0_levels_and_error_bound():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    r = q4_0_roundtrip(w)
    blocks, orig = r.reshape(-1, 32), w.reshape(-1, 32)
    for b, o in zip(blocks, orig):
        step = np.abs(o).max() / 8
        assert len(np.unique(np.round(b / step, 3))) <= 16  # at most 16 levels per block
        assert np.max(np.abs(b - o)) <= step * 0.5 + 1e-3 or np.max(np.abs(b - o)) <= step + 1e-3
    assert 0.05 < np.linalg.norm(r - w) / np.linalg.norm(w) < 0.2


def test_q8_0_is_nearly_lossless_and_exact_on_zero():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 96)).astype(np.float32)
    assert np.linalg.norm(q8_0_roundtrip(w) - w) / np.linalg.norm(w) < 0.01
    assert np.all(q4_0_roundtrip(np.zeros((1, 32))) == 0) and np.all(q8_0_roundtrip(np.zeros((1, 32))) == 0)
