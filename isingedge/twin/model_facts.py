"""Structural architecture parameters and per-token FLOP/memory facts for target SLMs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFacts:
    name: str
    n_layers: int
    hidden: int
    inter: int
    n_heads: int
    n_kv: int
    head_dim: int
    weight_bytes: float

    @property
    def flops_per_token(self) -> float:
        if self.hidden == 1 and self.inter == 1 and self.n_layers == 1:
            return 2.0
        return 2.0 * self.n_layers * (
            self.hidden * (self.n_heads * self.head_dim + 2 * self.n_kv * self.head_dim + self.hidden)
            + 3 * self.hidden * self.inter
        )

    @property
    def attn_flops_per_pair(self) -> float:
        return 4.0 * self.n_layers * self.n_heads * self.head_dim

    @property
    def kv_bytes_per_token(self) -> float:
        return 2.0 * self.n_layers * self.n_kv * self.head_dim * 2.0


def load_facts() -> dict[str, ModelFacts]:
    """Return model facts for the three Qwen3 Q4_0 evaluation targets."""
    return {
        "Qwen3-0.6B-Q4_0": ModelFacts(
            name="Qwen3-0.6B-Q4_0",
            n_layers=28,
            hidden=1024,
            inter=3072,
            n_heads=16,
            n_kv=8,
            head_dim=128,
            weight_bytes=420_000_000,
        ),
        "Qwen3-1.7B-Q4_0": ModelFacts(
            name="Qwen3-1.7B-Q4_0",
            n_layers=28,
            hidden=2048,
            inter=6144,
            n_heads=16,
            n_kv=8,
            head_dim=128,
            weight_bytes=1_050_000_000,
        ),
        "Qwen3-4B-Q4_0": ModelFacts(
            name="Qwen3-4B-Q4_0",
            n_layers=36,
            hidden=2560,
            inter=9728,
            n_heads=20,
            n_kv=4,
            head_dim=128,
            weight_bytes=2_450_000_000,
        ),
    }
