import itertools

import numpy as np

from isingedge.packer.core import (
    PrefillCost,
    f1_em,
    mmr,
    objective,
    pack,
    packing_qubo,
    qubo_pack,
    qubo_pack_lagrange,
    tfidf_scores,
    top_k,
)


def test_tfidf_ranks_the_matching_sentence_first():
    rel, sim = tfidf_scores("Who directed Ed Wood?", ["Ed Wood is a film directed by Tim Burton.",
                                                      "Scott Derrickson is an American director.",
                                                      "Bananas are yellow."])
    assert np.argmax(rel) == 0 and sim.shape == (3, 3) and abs(sim[0, 0] - 1) < 1e-9


def test_baselines_respect_the_budget():
    rel = np.array([0.9, 0.8, 0.7, 0.1])
    sim = np.eye(4)
    sim[0, 1] = sim[1, 0] = 0.95
    lens = np.array([10, 10, 10, 10])
    assert list(top_k(rel, lens, 20)) == [1, 1, 0, 0]
    assert list(mmr(rel, sim, lens, 20, lam=0.5)) == [1, 0, 1, 0]  # skips the near-duplicate


def test_packing_qubo_matches_objective_plus_budget_penalty_when_slack_is_exact():
    rng = np.random.default_rng(0)
    n = 6
    rel, sim, lens = rng.random(n), rng.random((n, n)), rng.integers(3, 9, n)
    sim = (sim + sim.T) / 2
    cost = PrefillCost(alpha=1e-3, beta=2e-6, n0=40)
    q, nu, off = packing_qubo(rel, sim, lens, budget=15, mu=0.4, lam=2.0, cost=cost)
    nb = q.shape[0] - nu
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits)
        used = int(lens @ x)
        if used > 15:
            continue
        slack = 15 - used
        y = np.array([(slack >> k) & 1 for k in range(nb)])
        z = np.concatenate([x, y]).astype(float)
        assert abs(z @ q @ z + off - objective(x, rel, sim, lens, 0.4, 2.0, cost)) < 1e-9


def test_qubo_pack_is_feasible_and_near_the_brute_force_optimum():
    rng = np.random.default_rng(1)
    n = 10
    rel, lens = rng.random(n), rng.integers(4, 12, n)
    sim = rng.random((n, n)) * 0.5
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1)
    x, sec = qubo_pack(rel, sim, lens, budget=30, mu=0.5, agents=32, steps=1000, seed=0)
    assert lens @ x <= 30 and sec > 0
    best = min(objective(np.array(b), rel, sim, lens, 0.5, 0.0, None)
               for b in itertools.product((0, 1), repeat=n) if lens @ np.array(b) <= 30)
    assert objective(x, rel, sim, lens, 0.5, 0.0, None) <= best + 0.1 * abs(best)


def test_hotpot_metrics():
    assert f1_em("The Eiffel Tower", "eiffel tower") == (1.0, 1.0)
    f1, em = f1_em("Tim Burton directed it", "Tim Burton")
    assert em == 0.0 and 0.5 < f1 < 1.0
    assert f1_em("yes", "no") == (0.0, 0.0)


def test_lagrangian_packer_is_feasible_and_fills_the_budget():
    rng = np.random.default_rng(2)
    n = 12
    rel, lens = rng.random(n), rng.integers(4, 12, n)
    sim = rng.random((n, n)) * 0.6
    sim = (sim + sim.T) / 2
    x, sec, info = qubo_pack_lagrange(rel, sim, lens, budget=40, mu=0.2, seed=0)
    assert lens @ x <= 40 and sec > 0 and info["core_units"] <= x.sum()
    x2, _, _ = pack(rel, sim, lens, budget=40)
    assert lens @ x2 <= 40 and lens @ x2 >= 40 - lens.max()  # greedy fill leaves less than one unit unused
    best = min(objective(np.array(b), rel, sim, lens, 0.2, 0.0, None)
               for b in itertools.product((0, 1), repeat=n) if lens @ np.array(b) <= 40)
    assert objective(x, rel, sim, lens, 0.2, 0.0, None) <= best + 0.15 * abs(best)
