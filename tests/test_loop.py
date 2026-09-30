import itertools

import numpy as np
import pytest

from isingedge.loop import acquire
from isingedge.loop.fm import fit_fm
from isingedge.loop.oracle import PPL, MockOracle, keep_fraction_rank, scored_positions
from isingedge.loop.pairmodels import PairModel, choose, fit, log_additive_pct
from isingedge.loop.run import run_local, run_loop, run_random
from isingedge.loop.space import Space, saved_units
from isingedge.loop.stats import paired_bootstrap, spearman


def _qubo(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.2, 0.5, n), np.triu(rng.normal(0.05, 0.2, (n, n)), 1)


def _small_space():
    return Space(layers=tuple(range(6)), keeps=(0.9, 0.8), units=(2, 5), budget=10)


def test_ranks_and_units_match_the_mac_pipeline():
    assert keep_fraction_rank(1024, 3072, 0.9) == 704 and keep_fraction_rank(1024, 3072, 0.8) == 608
    assert keep_fraction_rank(3072, 8192, 0.8) == 1792 and keep_fraction_rank(3072, 3072, 0.8) == 1216
    assert saved_units((0.9, 0.8)) == (2, 5)
    assert scored_positions(512) == (256, 255)


def test_mpo_bond_matching_and_exact_reconstruction():
    torch = pytest.importorskip("torch")
    from isingedge.loop.oracle import match_mpo_bond, mpo_params, mpo_reconstruct

    target = 384 * (1024 + 3072)  # SVD keep 0.5 on a Qwen3-0.6B MLP matrix
    chi = match_mpo_bond(target, (16, 4, 16), (16, 12, 16))
    assert abs(mpo_params((16, 4, 16), (16, 12, 16), chi) - target) / target < 0.01
    assert mpo_params((16, 12, 16), (16, 4, 16), chi) == mpo_params((16, 4, 16), (16, 12, 16), chi)
    w = torch.randn(40, 24, dtype=torch.float64)  # out modes (2, 5, 4), in modes (2, 3, 4)
    full = mpo_reconstruct(w, (2, 3, 4), (2, 5, 4), 10_000)
    assert torch.allclose(full, w, atol=1e-10)
    assert torch.linalg.norm(mpo_reconstruct(w, (2, 3, 4), (2, 5, 4), 2) - w) > 0


def test_ppl_matches_llama_perplexity_formula():
    rng = np.random.default_rng(0)
    tok = rng.gamma(2.0, 1.5, (3, 255))
    r = PPL(tok.sum(1), (tok * tok).sum(1), 255)
    flat = tok.ravel()
    assert r.ppl == pytest.approx(np.exp(flat.mean()))
    assert r.err == pytest.approx(np.exp(flat.mean()) * np.sqrt((np.mean(flat ** 2) - flat.mean() ** 2) / (flat.size - 1)))


def test_space_counts_enumeration_and_moves():
    s = _small_space()
    plans = list(s.enumerate())
    brute = [x for x in itertools.product((0, 1), repeat=s.n) if s.feasible(np.array(x))]
    assert len(plans) == len(brute) == s.count()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = s.random(rng)
        assert s.feasible(x)
        assert all(s.feasible(y) for y in s.neighbours(x))
        assert np.array_equal(s.encode(s.decode(x)), x)
    big = Space(tuple(range(2, 26)), (0.9, 0.8), (2, 5), 60)
    assert big.compositions == [(0, 12), (5, 10), (10, 8), (15, 6), (20, 4)]


def test_exact_cardinality_matches_brute_force_and_greedy_is_feasible():
    c, q = _qubo(10, 1)
    brute = sorted((acquire.energy(np.array(x), c, q), x) for x in itertools.product((0, 1), repeat=10) if sum(x) == 4)
    got = acquire.exact_cardinality(c, q, 4, top=3)
    assert [round(e, 9) for e, _ in got] == [round(e, 9) for e, _ in brute[:3]]
    g = acquire.greedy_cardinality(c, q, 4)
    assert g.sum() == 4 and acquire.energy(g, c, q) >= brute[0][0] - 1e-9


def test_space_solvers_are_feasible_and_exact_is_optimal():
    s = _small_space()
    c, q = _qubo(s.n, 2)
    best = acquire.exact(s, c, q, top=2)
    brute = sorted(acquire.energy(x, c, q) for x in s.enumerate())
    assert best[0][0] == pytest.approx(brute[0]) and best[1][0] == pytest.approx(brute[1])
    g = acquire.greedy(s, c, q)
    assert s.feasible(g) and acquire.energy(g, c, q) >= brute[0] - 1e-9
    div = acquire.exact(s, c, q, top=1, min_hamming=(best[0][1], 4))[0][1]
    assert s.feasible(div) and np.sum(div != best[0][1]) >= 4


def test_cpsat_matches_enumeration():
    pytest.importorskip("ortools")
    s = _small_space()
    c, q = _qubo(s.n, 3)
    r = acquire.cpsat(s, c, q, time_limit=10)
    assert r["status"] == "OPTIMAL" and r["energy"] == pytest.approx(acquire.exact(s, c, q)[0][0], abs=1e-5)
    r2 = acquire.cpsat(s, c, q, exclude=[r["x"]], time_limit=10)
    assert r2["energy"] == pytest.approx(acquire.exact(s, c, q, top=2)[1][0], abs=1e-5)


def test_annealers_return_feasible_plans_and_find_small_optima():
    s = _small_space()
    c, q = _qubo(s.n, 4)
    assert s.feasible(acquire.sa(s, c, q, seed=0, reads=4, sweeps=300)["x"])
    for ballistic in (True, False):
        assert s.feasible(acquire.sb(s, c, q, seed=0, ballistic=ballistic, agents=16, max_steps=300)["x"])
    lin, quad = _qubo(10, 8)  # unconstrained: both heuristics should reach the brute-force optimum
    best = min(acquire.energy(np.array(x), lin, quad) for x in itertools.product((0, 1), repeat=10))
    assert acquire.energy(acquire.anneal(lin, quad, seed=0, reads=8, sweeps=500), lin, quad) == pytest.approx(best)
    assert acquire.energy(acquire.bifurcate(lin, quad, seed=0, agents=64, steps=2000), lin, quad) <= best + 0.5 * abs(best)


def test_fm_recovers_a_planted_quadratic():
    rng = np.random.default_rng(5)
    n = 12
    v = rng.normal(0, 0.3, (n, 2))
    w = rng.normal(0, 1, n)
    x = (rng.uniform(size=(300, n)) < 0.4).astype(float)
    xv = x @ v
    y = 0.5 + x @ w + 0.5 * ((xv ** 2).sum(1) - (x @ (v ** 2)).sum(1))
    m = fit_fm(x[:250], y[:250], rank=2, l2_w=0.0, l2_v=0.0, seed=0)
    assert np.abs(m.predict(x[250:]) - y[250:]).max() < 1e-3
    c, qq, const = m.qubo()
    assert const + c @ x[0] + x[0] @ qq @ x[0] == pytest.approx(m.predict(x[0])[0])


def test_pair_models_and_log_additivity():
    assert log_additive_pct([10.0, 10.0]) == pytest.approx(21.0)
    assert PairModel("multiplicative").predict(0, 5, 10.0, 10.0) == pytest.approx(21.0)
    rng = np.random.default_rng(6)
    pairs = []
    for i, j in [(0, 1), (2, 7), (3, 4), (5, 9), (1, 8), (6, 7)]:
        ai, aj = rng.uniform(1, 5, 2)
        pairs.append((i, j, ai, aj, ai + aj + 3.0 * ai * aj / 100))
    assert fit("scaled", pairs).g0 == pytest.approx(3.0)
    model, table = choose(pairs)
    assert model.kind in ("scaled", "distance") and len(table) == 4


def test_bootstrap_and_spearman():
    rng = np.random.default_rng(7)
    a = rng.uniform(600, 900, 40)
    ci = paired_bootstrap(a, a, 255)
    assert ci["diff"] == 0 and ci["lo"] <= 0 <= ci["hi"] and not ci["excludes_zero"]
    worse = paired_bootstrap(a + 20, a, 255)
    assert worse["diff"] > 0 and worse["excludes_zero"]
    assert spearman([1, 2, 3, 4], [10, 20, 35, 90]) == pytest.approx(1.0)


def test_loop_and_baselines_on_a_mock_oracle():
    s = Space(tuple(range(8)), (0.9, 0.8), (2, 5), 15)
    oracle = MockOracle(s.n, seed=0)
    warm = [np.zeros(s.n, dtype=np.int8)]
    for i in range(s.n):
        x = np.zeros(s.n, dtype=np.int8)
        x[i] = 1
        warm.append(x)
    ys = [oracle.true(x) for x in warm]
    loop = run_loop(s, oracle, warm, ys, evals=6, seed=0, rank=3, compare_solvers=False, log=lambda *_: None)
    assert len(loop) == 6 and all(s.feasible(e.x) for e in loop)
    assert len({tuple(e.x) for e in loop}) == 6
    rnd = run_random(s, oracle, 6, seed=0, log=lambda *_: None)
    loc = run_local(s, oracle, rnd[0].x, 6, seed=0, log=lambda *_: None)
    assert all(s.feasible(e.x) for e in rnd + loc)
    opt = min(oracle.true(x) for x in s.enumerate())
    assert min(e.y for e in loop) <= opt + 0.25 * abs(opt)
