import itertools

import numpy as np

from isingedge.qubo.planner import (
    Problem,
    UnitCost,
    Workload,
    default_options,
    groups_for_qwen3,
    qubo_energy,
)
from isingedge.solve.solvers import exact_chain, simulated_annealing, simulated_bifurcation


def _problem(n_layers=2, executable_only=False, seed=0):
    rng = np.random.default_rng(seed)
    groups = groups_for_qwen3(n_layers, hidden=64, inter=192, n_heads=4, n_kv=2, head_dim=16)
    cost = {"cpu": UnitCost(1 / 60e9, 1 / 20e9, 2e-3, 1 / 300e9, 1 / 80e9, 5e-3),
            "gpu": UnitCost(1 / 50e9, 1 / 40e9, 3e-3, 1 / 900e9, 1 / 300e9, 20e-3),
            "npu": UnitCost(1 / 45e9, 1 / 30e9, 6e-3, 1 / 3000e9, 1 / 900e9, 40e-3)}
    options = default_options(executable_only)
    dl = {}
    for g in groups:
        for o in options[g.part]:
            dl[(g.layer, g.part, o)] = (1 - o.keep) * 20 * rng.random() + (0.8 * rng.random() if o.bits == "q4_0" else 0)
    sw = {(u, v): 2e-4 for u in cost for v in cost if u != v}
    return Problem(groups, options, tuple(cost), cost, sw, {k: 10 * v for k, v in sw.items()}, dl,
                   kv_bytes_per_layer=2 * 2 * 16 * 2, attn_flops_per_pair_per_layer=4 * 4 * 16,
                   workload=Workload(prompt_tokens=512, gen_tokens=64), lam=1e-3)


def _all_plans(p):
    choices = [[(u, o) for u in p.units for o in p.options[g.part]] for g in p.groups]
    return itertools.product(*choices)


def test_qubo_energy_equals_objective_on_feasible_plans():
    p = _problem()
    q, idx, off = p.qubo()
    rng = np.random.default_rng(3)
    for _ in range(20):
        plan = [(rng.choice(p.units), p.options[g.part][rng.integers(len(p.options[g.part]))]) for g in p.groups]
        plan = [(str(u), o) for u, o in plan]
        x = p.encode(plan, idx)
        assert abs(qubo_energy(q, x, off) - p.objective(plan)) < 1e-9 * max(1.0, abs(p.objective(plan)))
        assert p.decode_x(x, idx) == plan


def test_infeasible_vectors_cost_more_and_decode_to_none():
    p = _problem()
    q, idx, off = p.qubo()
    plan, obj = exact_chain(p)
    x = p.encode(plan, idx)
    x2 = x.copy()
    x2[0] = 1 - x2[0]  # break the first one-hot
    assert p.decode_x(x2, idx) is None
    assert qubo_energy(q, x2, off) > obj


def test_exact_chain_matches_brute_force():
    p = _problem(n_layers=1)
    _, obj = exact_chain(p)
    brute = min(p.objective(list(c)) for c in _all_plans(p))
    assert abs(obj - brute) < 1e-12 * max(1.0, abs(brute))


def test_sb_and_sa_stay_close_to_the_exact_optimum():
    for layers, executable in ((2, True), (2, False)):
        p = _problem(n_layers=layers, executable_only=executable)
        _, opt = exact_chain(p)
        sb = simulated_bifurcation(p, agents=32, seed=0)
        sa = simulated_annealing(p, reads=16, sweeps=1000, seed=0)
        assert opt <= sb["objective"] <= opt * 1.01, (layers, executable)
        assert opt <= sa["objective"] <= opt * 1.10, (layers, executable)


def test_predicted_latency_is_positive_and_consistent():
    p = _problem()
    plan, _ = exact_chain(p)
    lat = p.predicted_latency(plan)
    assert lat["ttft_s"] > 0 and lat["decode_tps"] > 0


def test_joint_encoding_energy_and_solvers():
    from isingedge.solve.solvers import sa_joint, sb_joint

    p = _problem(n_layers=3, executable_only=False, seed=4)
    plan, opt = exact_chain(p)
    q, idx, off = p.qubo_joint()
    x = np.array([1 if (k[2], k[3]) == plan[k[1]] else 0 for k in idx])
    assert abs(qubo_energy(q, x, off) - opt) < 1e-9 * max(1.0, opt)
    assert p.decode_joint(x, idx) == plan
    x2 = x.copy()
    x2[0] = 1 - x2[0]
    assert p.decode_joint(x2, idx) is None and qubo_energy(q, x2, off) > opt
    sb = sb_joint(p, agents=32, seed=0)
    sa = sa_joint(p, reads=16, sweeps=20000, seed=0)
    assert opt <= sb["objective"] <= opt * 1.01
    assert opt <= sa["objective"] <= opt * 1.05


def test_pairwise_terms_enter_objective_qubo_and_solvers():
    from isingedge.solve.solvers import greedy, sa_joint, sb_joint

    p = _problem(n_layers=2, executable_only=False, seed=5)
    p.lam = 0.02  # make compression attractive so the pair term matters
    p.pairs = {(0, 1): 40.0}
    q, idx, off = p.qubo_joint()
    plans = [list(c) for c in _all_plans(p)]
    for trial in plans[::997]:
        x = np.array([1 if (k[2], k[3]) == trial[k[1]] else 0 for k in idx])
        assert abs(qubo_energy(q, x, off) - p.objective(trial)) < 1e-9 * max(1.0, abs(p.objective(trial)))
    opt = min(p.objective(t) for t in plans)
    dp_plan, _ = exact_chain(p)  # ignores the pair term: the "DP on the additive model" baseline
    assert p.objective(dp_plan) >= opt
    assert opt <= sb_joint(p, agents=32, seed=0)["objective"] <= opt * 1.01
    sa = sa_joint(p, reads=16, sweeps=20000, seed=0)
    assert opt <= sa["objective"] and len(sa["plan"]) == len(p.groups)  # SA quality is reported, not asserted
    g = greedy(p)
    assert opt <= g["objective"]
