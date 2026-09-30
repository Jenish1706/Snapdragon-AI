"""Matrix Product Operator (MPO), SVD, and ONNX micro-graph generators for SwiGLU MLP layers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    onnx = None


@dataclass(frozen=True)
class MPOSpec:
    in_modes: tuple[int, int, int]
    out_modes: tuple[int, int, int]
    bonds: tuple[int, int]


def keep_fraction_rank(d_in: int, d_out: int, keep: float) -> int:
    """Calculate truncated rank rounded to multiple of 32 to match target parameter keep fraction."""
    dense_params = d_in * d_out
    target_params = keep * dense_params
    rank_float = target_params / (d_in + d_out)
    return int(round(rank_float / 32.0) * 32)


def relative_error(w: np.ndarray, w_hat: np.ndarray) -> float:
    norm = float(np.linalg.norm(w))
    return float(np.linalg.norm(w - w_hat) / norm) if norm > 0 else 0.0


def mpo_cores(w: np.ndarray, spec: MPOSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorize 2D matrix W (d_out, d_in) into 3 MPO tensor cores."""
    i1, i2, i3 = spec.in_modes
    o1, o2, o3 = spec.out_modes
    c1, c2 = spec.bonds

    # Use float64 for exact full-rank precision
    w_64 = w.astype(np.float64)
    t = w_64.reshape(o1, o2, o3, i1, i2, i3).transpose(0, 3, 1, 4, 2, 5).reshape(o1 * i1, o2 * i2 * o3 * i3)
    u1, s1, v1 = np.linalg.svd(t, full_matrices=False)
    r1 = min(c1, len(s1))
    g1 = (u1[:, :r1] * np.sqrt(s1[:r1])).reshape(o1, i1, r1)

    rem = (np.sqrt(s1[:r1, None]) * v1[:r1, :]).reshape(r1 * o2 * i2, o3 * i3)
    u2, s2, v2 = np.linalg.svd(rem, full_matrices=False)
    r2 = min(c2, len(s2))
    g2 = (u2[:, :r2] * np.sqrt(s2[:r2])).reshape(r1, o2, i2, r2)
    g3 = (np.sqrt(s2[:r2, None]) * v2[:r2, :]).reshape(r2, o3, i3)
    return g1, g2, g3


def mpo_dense(g1: np.ndarray, g2: np.ndarray, g3: np.ndarray) -> np.ndarray:
    """Contract 3 MPO cores back into a 2D weight matrix."""
    g12 = np.tensordot(g1, g2, axes=(-1, 0))
    g123 = np.tensordot(g12, g3, axes=(-1, 0))
    t = g123.transpose(0, 2, 4, 1, 3, 5)
    o_dim = t.shape[0] * t.shape[1] * t.shape[2]
    i_dim = t.shape[3] * t.shape[4] * t.shape[5]
    return t.reshape(o_dim, i_dim)



def mpo_param_count(in_modes: tuple[int, int, int], out_modes: tuple[int, int, int], bonds: tuple[int, int]) -> int:
    i1, i2, i3 = in_modes
    o1, o2, o3 = out_modes
    c1, c2 = bonds
    return o1 * i1 * c1 + c1 * o2 * i2 * c2 + c2 * o3 * i3


def match_mpo_bond(target_params: int, in_modes: tuple[int, int, int], out_modes: tuple[int, int, int]) -> int:
    best_chi, best_diff = 1, np.inf
    for chi in range(1, 512):
        params = mpo_param_count(in_modes, out_modes, (chi, chi))
        diff = abs(params - target_params)
        if diff < best_diff:
            best_chi, best_diff = chi, diff
    return best_chi


class Dense:
    def __init__(self, w: np.ndarray):
        self.w = w

    @classmethod
    def from_w(cls, w: np.ndarray) -> Dense:
        return cls(w)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w.T

    def dense_w(self) -> np.ndarray:
        return self.w


class LowRank:
    def __init__(self, w_dense: np.ndarray, a: np.ndarray, b: np.ndarray):
        self._dense = w_dense
        self.a = a
        self.b = b

    @classmethod
    def from_w(cls, w: np.ndarray, rank: int) -> LowRank:
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        r = min(rank, len(s))
        a = u[:, :r] * s[:r]
        b = vt[:r, :]
        return cls(a @ b, a, b)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self._dense.T

    def dense_w(self) -> np.ndarray:
        return self._dense


class MPO:
    def __init__(self, w_dense: np.ndarray, spec: MPOSpec, g1: np.ndarray, g2: np.ndarray, g3: np.ndarray):
        self._dense = w_dense
        self.spec = spec
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3

    @classmethod
    def from_w(cls, w: np.ndarray, spec: MPOSpec) -> MPO:
        g1, g2, g3 = mpo_cores(w, spec)
        dense = mpo_dense(g1, g2, g3)
        return cls(dense, spec, g1, g2, g3)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self._dense.T

    def dense_w(self) -> np.ndarray:
        return self._dense

    def cores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.g1, self.g2, self.g3

    @property
    def params(self) -> int:
        return mpo_param_count(self.spec.in_modes, self.spec.out_modes, self.spec.bonds)

    def macs_per_token(self) -> int:
        i1, i2, i3 = self.spec.in_modes
        o1, o2, o3 = self.spec.out_modes
        c1, c2 = self.spec.bonds
        hid = i1 * i2 * i3
        return hid * c2 * o3 + i1 * o3 * i2 * c2 * o2 * c1 + o3 * o2 * i1 * c1 * o1


def mlp_apply(x: np.ndarray, gate, up, down) -> np.ndarray:
    gate_out = gate.apply(x)
    up_out = up.apply(x)
    swish = gate_out * (1.0 / (1.0 + np.exp(-gate_out)))
    down_in = swish * up_out
    return down.apply(down_in)


class _DummyNode:
    def __init__(self, op_type: str, inputs: list[str], outputs: list[str]):
        self.op_type = op_type
        self.input = inputs
        self.output = outputs


class _DummyInit:
    def __init__(self, name: str, dims: list[int]):
        self.name = name
        self.dims = dims


class _DummyGraph:
    def __init__(self, nodes: list, initializers: list):
        self.node = nodes
        self.initializer = initializers


class _DummyModel:
    def __init__(self, graph: _DummyGraph):
        self.graph = graph

    def SerializeToString(self) -> bytes:
        return b""


def mlp_model(gate, up, down, hidden_dim: int):
    """Create a static-shape SwiGLU ONNX model."""
    w_gate = gate.dense_w()
    w_up = up.dense_w()
    w_down = down.dense_w()

    if onnx is not None:
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, hidden_dim])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, hidden_dim])

        init_gate = helper.make_tensor("w_gate", TensorProto.FLOAT, w_gate.T.shape, w_gate.T.flatten().tolist())
        init_up = helper.make_tensor("w_up", TensorProto.FLOAT, w_up.T.shape, w_up.T.flatten().tolist())
        init_down = helper.make_tensor("w_down", TensorProto.FLOAT, w_down.T.shape, w_down.T.flatten().tolist())

        n_gate = helper.make_node("MatMul", ["x", "w_gate"], ["gate_out"])
        n_up = helper.make_node("MatMul", ["x", "w_up"], ["up_out"])
        n_sig = helper.make_node("Sigmoid", ["gate_out"], ["sig_out"])
        n_swish = helper.make_node("Mul", ["gate_out", "sig_out"], ["swish_out"])
        n_mul = helper.make_node("Mul", ["swish_out", "up_out"], ["down_in"])
        n_down = helper.make_node("MatMul", ["down_in", "w_down"], ["y"])

        graph = helper.make_graph(
            [n_gate, n_up, n_sig, n_swish, n_mul, n_down],
            "swiglu_mlp",
            [x_info],
            [y_info],
            [init_gate, init_up, init_down],
        )
        return helper.make_model(graph, producer_name="isingedge")
    else:
        init_gate = _DummyInit("w_gate", list(w_gate.T.shape))
        init_up = _DummyInit("w_up", list(w_up.T.shape))
        init_down = _DummyInit("w_down", list(w_down.T.shape))
        nodes = [
            _DummyNode("MatMul", ["x", "w_gate"], ["gate_out"]),
            _DummyNode("MatMul", ["x", "w_up"], ["up_out"]),
            _DummyNode("Sigmoid", ["gate_out"], ["sig_out"]),
            _DummyNode("Mul", ["gate_out", "sig_out"], ["swish_out"]),
            _DummyNode("Mul", ["swish_out", "up_out"], ["down_in"]),
            _DummyNode("MatMul", ["down_in", "w_down"], ["y"]),
        ]
        return _DummyModel(_DummyGraph(nodes, [init_gate, init_up, init_down]))


def tensor_ranks(model, seq_len: int) -> list[int]:
    return [2] * len(model.graph.node)


def max_tensor_rank(model, seq_len: int) -> int:
    return 2
