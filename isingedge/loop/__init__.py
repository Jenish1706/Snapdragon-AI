"""Closed measure -> fit -> solve planning loop (S4).

space:   one-hot encoding of per-layer compression choices and the exact-bytes budget.
fm:      second-order factorization-machine surrogate on log(PPL / PPL_dense).
acquire: surrogate minimisation under the budget (exact enumeration, CP-SAT, greedy, SA, SB).
stats:   paired bootstrap intervals and rank correlation.
oracle:  the measured objective (Qwen3 perplexity in PyTorch); mock oracles for the synthetic path.
"""
