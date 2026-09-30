import itertools

import numpy as np

from isingedge.solve.solvers import exact_subset, greedy_subset, sa_subset, sb_subset, subset_energy


def _instance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(-0.5, 1.0, n)
    w = rng.normal(0.3, 0.6, (n, n))
    return c, (w + w.T) / 2


def test_exact_subset_matches_naive_enumeration_including_the_split_point():
    c, w = _instance()
    naive = sorted((subset_energy(np.array(b), c, w), b) for b in itertools.product((0, 1), repeat=len(c)))
    for low_bits in (4, 12):
        got = exact_subset(c, w, top=3, low_bits=low_bits)
        assert [round(e, 9) for e, _ in got] == [round(e, 9) for e, _ in naive[:3]]
        assert tuple(got[0][1]) == naive[0][1]


def test_heuristics_are_valid_and_near_optimal_on_a_small_instance():
    c, w = _instance(n=14, seed=1)
    opt = exact_subset(c, w, top=1)[0][0]
    g = greedy_subset(c, w)
    x_sb, _ = sb_subset(c, w, seed=0, ballistic=True)
    x_sa, _ = sa_subset(c, w, reads=16, sweeps=2000, seed=0)
    for x in (g, x_sb, x_sa):
        assert subset_energy(x, c, w) >= opt - 1e-9
    assert subset_energy(x_sa, c, w) <= opt + 0.05 * abs(opt)
