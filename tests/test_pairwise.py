import itertools

import numpy as np

from isingedge.sensitivity.pairwise import Pair, choose, fit, loo


def _pairs(model_fn, n=40, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 25.0, size=28)
    allp = list(itertools.combinations(range(28), 2))
    picks = [allp[k] for k in rng.choice(len(allp), size=n, replace=False)]
    return [Pair(i, j, a[i], a[j], a[i] + a[j] + model_fn(i, j, a[i], a[j]) + noise * rng.standard_normal()) for i, j in picks]


def test_distance_model_is_recovered_and_chosen():
    true = lambda i, j, ai, aj: (0.5 + 3.0 * np.exp(-abs(i - j) / 2.0)) * ai * aj / 100
    pairs = _pairs(true)
    m = fit("distance", pairs)
    assert abs(m.g0 - 0.5) < 1e-6 and abs(m.g1 - 3.0) < 1e-6 and m.ell == 2.0
    best, table = choose(pairs)
    assert best.kind == "distance"
    assert min(table, key=lambda t: t["loo_mean_abs_pp"])["kind"] == "distance"


def test_additive_data_prefers_a_zero_interaction_model():
    pairs = _pairs(lambda i, j, ai, aj: 0.0, noise=0.05)
    add = loo("additive", pairs)
    mult = loo("multiplicative", pairs)
    assert add["loo_mean_abs_pp"] < mult["loo_mean_abs_pp"]
    assert abs(fit("scaled", pairs).g0) < 0.05
