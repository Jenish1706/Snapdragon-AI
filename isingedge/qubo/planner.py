"""Combinatorial planning problem formulation for placement, precision, and low-rank compression."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class UnitCost:
    a_w: float
    a_kv: float
    t0: float
    b_f: float
    b_attn: float
    b0: float


@dataclass(frozen=True)
class Workload:
    prompt_tokens: int
    gen_tokens: int


@dataclass(frozen=True)
class Option:
    keep: float
    bits: str = "q4_0"


@dataclass(frozen=True)
class Group:
    layer: int
    part: str  # "attn" or "mlp"
    weight_bytes: float
    flops_per_token: float


def default_options(executable_only: bool = False) -> dict[str, list[Option]]:
    if executable_only:
        return {
            "attn": [Option(1.0, "q4_0")],
            "mlp": [Option(1.0, "q4_0"), Option(0.8, "q4_0")],
        }
    return {
        "attn": [Option(1.0, "q4_0")],
        "mlp": [
            Option(1.0, "q4_0"),
            Option(0.9, "q4_0"),
            Option(0.8, "q4_0"),
            Option(0.7, "q4_0"),
            Option(0.6, "q4_0"),
        ],
    }


def groups_for_qwen3(
    n_layers: int,
    hidden: int = 64,
    inter: int = 192,
    n_heads: int = 4,
    n_kv: int = 2,
    head_dim: int = 16,
) -> list[Group]:
    groups = []
    attn_bytes = 2.0 * (hidden * (n_heads * head_dim + 2 * n_kv * head_dim + hidden)) * 0.5  # q4_0
    mlp_bytes = 3.0 * (hidden * inter) * 0.5
    attn_flops = 2.0 * (hidden * (n_heads * head_dim + 2 * n_kv * head_dim + hidden))
    mlp_flops = 2.0 * (3.0 * hidden * inter)
    
    for l in range(n_layers):
        groups.append(Group(layer=l, part="attn", weight_bytes=attn_bytes, flops_per_token=attn_flops))
        groups.append(Group(layer=l, part="mlp", weight_bytes=mlp_bytes, flops_per_token=mlp_flops))
    return groups


def qubo_energy(q: np.ndarray, x: np.ndarray, off: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ q @ x + off)


class Problem:
    def __init__(
        self,
        groups: list[Group],
        options: dict[str, list[Option]],
        units: tuple[str, ...],
        cost: dict[str, UnitCost],
        sw: dict[tuple[str, str], float],
        sw_act: dict[tuple[str, str], float],
        dl: dict[tuple[int, str, Option], float],
        kv_bytes_per_layer: float,
        attn_flops_per_pair_per_layer: float,
        workload: Workload,
        lam: float = 1e-3,
    ):
        self.groups = groups
        self.options = options
        self.units = units
        self.cost = cost
        self.sw = sw
        self.sw_act = sw_act
        self.dl = dl
        self.kv_bytes_per_layer = kv_bytes_per_layer
        self.attn_flops_per_pair_per_layer = attn_flops_per_pair_per_layer
        self.workload = workload
        self.lam = lam
        self.pairs: dict[tuple[int, int], float] = {}

    def objective(self, plan: list[tuple[str, Option]]) -> float:
        tot_time = 0.0
        tot_dl = 0.0
        n_p = self.workload.prompt_tokens
        n_g = self.workload.gen_tokens

        for i, ((u, o), g) in enumerate(zip(plan, self.groups)):
            c = self.cost[u]
            wb = g.weight_bytes * o.keep * (0.5 if o.bits == "q4_0" else 1.0)
            fl = g.flops_per_token * o.keep

            # prefill
            t_prefill = c.b_f * fl * n_p + (c.b_attn * self.attn_flops_per_pair_per_layer * n_p * (n_p + 1) / 2.0 if g.part == "attn" else 0.0) + c.b0
            # decode per token
            t_decode_tok = c.a_w * wb + (c.a_kv * self.kv_bytes_per_layer * n_p if g.part == "attn" else 0.0) + c.t0
            tot_time += t_prefill + n_g * t_decode_tok

            # switch cost
            if i > 0:
                prev_u = plan[i - 1][0]
                if prev_u != u:
                    tot_time += self.sw.get((prev_u, u), 0.0)

            # accuracy loss
            tot_dl += self.dl.get((g.layer, g.part, o), 0.0)

        # pair interactions
        for (i, j), inter in self.pairs.items():
            oi, oj = plan[i][1], plan[j][1]
            if oi.keep < 1.0 and oj.keep < 1.0:
                tot_dl += inter

        return float(tot_time + self.lam * tot_dl)

    def predicted_latency(self, plan: list[tuple[str, Option]]) -> dict[str, float]:
        n_p = self.workload.prompt_tokens
        n_g = self.workload.gen_tokens
        ttft = 0.0
        decode_tok = 0.0
        for i, ((u, o), g) in enumerate(zip(plan, self.groups)):
            c = self.cost[u]
            wb = g.weight_bytes * o.keep
            fl = g.flops_per_token * o.keep
            ttft += c.b_f * fl * n_p + (c.b_attn * self.attn_flops_per_pair_per_layer * n_p * (n_p + 1) / 2.0 if g.part == "attn" else 0.0) + c.b0
            decode_tok += c.a_w * wb + (c.a_kv * self.kv_bytes_per_layer * n_p if g.part == "attn" else 0.0) + c.t0
            if i > 0 and plan[i - 1][0] != u:
                ttft += self.sw.get((plan[i - 1][0], u), 0.0)
                decode_tok += self.sw.get((plan[i - 1][0], u), 0.0)
        return {"ttft_s": ttft, "decode_tps": 1.0 / decode_tok if decode_tok > 0 else 0.0}

    def qubo(self) -> tuple[np.ndarray, list, float]:
        return self.qubo_joint()

    def encode(self, plan: list[tuple[str, Option]], idx: list) -> np.ndarray:
        x = np.zeros(len(idx), dtype=np.int8)
        for var_idx, (_, g_idx, u, o) in enumerate(idx):
            if plan[g_idx] == (u, o):
                x[var_idx] = 1
        return x

    def decode_x(self, x: np.ndarray, idx: list) -> list[tuple[str, Option]] | None:
        return self.decode_joint(x, idx)

    def qubo_joint(self) -> tuple[np.ndarray, list, float]:
        idx = []
        for g_idx, g in enumerate(self.groups):
            for u in self.units:
                for o in self.options[g.part]:
                    idx.append(("joint", g_idx, u, o))

        n_vars = len(idx)
        Q = np.zeros((n_vars, n_vars), dtype=float)
        c_diag = np.zeros(n_vars, dtype=float)

        penalty = 1000.0
        n_p = self.workload.prompt_tokens
        n_g = self.workload.gen_tokens

        for i, (_, g_idx, u, o) in enumerate(idx):
            g = self.groups[g_idx]
            cost_u = self.cost[u]
            wb = g.weight_bytes * o.keep * (0.5 if o.bits == "q4_0" else 1.0)
            fl = g.flops_per_token * o.keep
            t_prefill = cost_u.b_f * fl * n_p + (cost_u.b_attn * self.attn_flops_per_pair_per_layer * n_p * (n_p + 1) / 2.0 if g.part == "attn" else 0.0) + cost_u.b0
            t_decode_tok = cost_u.a_w * wb + (cost_u.a_kv * self.kv_bytes_per_layer * n_p if g.part == "attn" else 0.0) + cost_u.t0
            t_iso = t_prefill + n_g * t_decode_tok
            c_diag[i] = t_iso + self.lam * self.dl.get((g.layer, g.part, o), 0.0)

        # one-hot penalty: penalty * (sum_i x_i - 1)^2 = penalty * (-sum_i x_i + 2 sum_{i < j} x_i x_j) + penalty
        offset = 0.0
        for g_idx in range(len(self.groups)):
            group_vars = [i for i, item in enumerate(idx) if item[1] == g_idx]
            for i in group_vars:
                c_diag[i] -= penalty
            for a in range(len(group_vars)):
                for b in range(a + 1, len(group_vars)):
                    va, vb = group_vars[a], group_vars[b]
                    Q[va, vb] += 2.0 * penalty
            offset += penalty

        # switch terms between adjacent groups
        for g_idx in range(len(self.groups) - 1):
            curr_vars = [i for i, item in enumerate(idx) if item[1] == g_idx]
            next_vars = [j for j, item in enumerate(idx) if item[1] == g_idx + 1]
            for vi in curr_vars:
                u_curr = idx[vi][2]
                for vj in next_vars:
                    u_next = idx[vj][2]
                    if u_curr != u_next:
                        sw_cost = self.sw.get((u_curr, u_next), 0.0)
                        if vi < vj:
                            Q[vi, vj] += sw_cost
                        else:
                            Q[vj, vi] += sw_cost

        # pair interactions
        for (g1, g2), inter in self.pairs.items():
            g1_vars = [i for i, item in enumerate(idx) if item[1] == g1 and item[3].keep < 1.0]
            g2_vars = [j for j, item in enumerate(idx) if item[1] == g2 and item[3].keep < 1.0]
            for vi in g1_vars:
                for vj in g2_vars:
                    pair_cost = self.lam * inter
                    if vi < vj:
                        Q[vi, vj] += pair_cost
                    else:
                        Q[vj, vi] += pair_cost

        np.fill_diagonal(Q, c_diag)
        return Q, idx, offset

    def decode_joint(self, x: np.ndarray, idx: list) -> list[tuple[str, Option]] | None:
        plan = []
        for g_idx in range(len(self.groups)):
            group_vars = [i for i, item in enumerate(idx) if item[1] == g_idx]
            active = [i for i in group_vars if x[i] == 1]
            if len(active) != 1:
                return None
            chosen = idx[active[0]]
            plan.append((chosen[2], chosen[3]))
        return plan
