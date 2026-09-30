"""Context prompt packing via QUBO, MMR, and greedy baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import time
import numpy as np


@dataclass(frozen=True)
class PrefillCost:
    alpha: float
    beta: float
    n0: int


def tfidf_scores(query: str, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute TF-IDF relevance and sentence-sentence similarity."""
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    docs = [tokenize(query)] + [tokenize(s) for s in sentences]
    vocab = sorted({w for doc in docs for w in doc})
    v2i = {w: i for i, w in enumerate(vocab)}
    
    # DF
    df = np.zeros(len(vocab))
    for doc in docs:
        for w in set(doc):
            df[v2i[w]] += 1
    idf = np.log((len(docs) + 1.0) / (df + 1.0)) + 1.0

    mat = np.zeros((len(sentences), len(vocab)))
    for i, s in enumerate(sentences):
        tokens = tokenize(s)
        for t in tokens:
            mat[i, v2i[t]] += 1
        mat[i] *= idf
        norm = np.linalg.norm(mat[i])
        if norm > 0:
            mat[i] /= norm

    q_vec = np.zeros(len(vocab))
    for t in docs[0]:
        q_vec[v2i[t]] += 1
    q_vec *= idf
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec /= q_norm

    rel = mat @ q_vec
    sim = mat @ mat.T
    np.fill_diagonal(sim, 1.0)
    return rel, sim


def top_k(rel: np.ndarray, lens: np.ndarray, budget: int) -> np.ndarray:
    n = len(rel)
    x = np.zeros(n, dtype=np.int8)
    order = np.argsort(-rel)
    used = 0
    for idx in order:
        if used + lens[idx] <= budget:
            x[idx] = 1
            used += lens[idx]
    return x


def mmr(rel: np.ndarray, sim: np.ndarray, lens: np.ndarray, budget: int, lam: float = 0.5) -> np.ndarray:
    n = len(rel)
    x = np.zeros(n, dtype=np.int8)
    used = 0
    selected = []
    
    while True:
        best_cand, best_score = None, -np.inf
        for i in range(n):
            if x[i] == 0 and used + lens[i] <= budget:
                div = max([sim[i, j] for j in selected]) if selected else 0.0
                score = lam * rel[i] - (1.0 - lam) * div
                if score > best_score:
                    best_cand, best_score = i, score
        if best_cand is None:
            break
        x[best_cand] = 1
        used += lens[best_cand]
        selected.append(best_cand)
    return x


def objective(
    x: np.ndarray,
    rel: np.ndarray,
    sim: np.ndarray,
    lens: np.ndarray,
    mu: float,
    lam: float,
    cost: PrefillCost | None,
) -> float:
    x = np.asarray(x, dtype=float)
    # Relevance is negative (minimization)
    val = -float(rel @ x)
    # Redundancy between distinct chosen pairs
    val += mu * float(np.sum(np.triu(sim, 1) * np.outer(x, x)))
    
    if cost is not None and lam > 0:
        tot_len = float(lens @ x)
        n = cost.n0 + tot_len
        val += lam * (cost.alpha * n + cost.beta * n * (n + 1.0) / 2.0)
    return float(val)


def packing_qubo(
    rel: np.ndarray,
    sim: np.ndarray,
    lens: np.ndarray,
    budget: int,
    mu: float,
    lam: float,
    cost: PrefillCost | None,
    penalty: float = 10.0,
) -> tuple[np.ndarray, int, float]:
    n = len(rel)
    nb = int(math.ceil(math.log2(budget + 1)))
    n_vars = n + nb
    Q = np.zeros((n_vars, n_vars), dtype=float)
    off = 0.0

    # 1. Base objective terms on x
    for i in range(n):
        Q[i, i] -= rel[i]
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += mu * sim[i, j]

    if cost is not None and lam > 0:
        n0 = cost.n0
        a, b = cost.alpha, cost.beta
        # off term: lam * (a * n0 + b * n0 * (n0 + 1) / 2)
        off += lam * (a * n0 + b * n0 * (n0 + 1.0) / 2.0)
        # linear term: lam * [ a * L_i + b / 2 * ((2 n0 + 1) L_i + L_i^2) ]
        for i in range(n):
            L = lens[i]
            Q[i, i] += lam * (a * L + (b / 2.0) * ((2 * n0 + 1) * L + L * L))
        # cross term: lam * b * L_i * L_j
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += lam * b * lens[i] * lens[j]

    # 2. Penalty on budget slack: penalty * (sum L_i x_i + sum 2^k y_k - budget)^2
    # Expanding penalty * (sum_v w_v z_v - B)^2:
    # = penalty * [ sum_v w_v^2 z_v + 2 sum_{u < v} w_u w_v z_u z_v - 2 B sum_v w_v z_v + B^2 ]
    weights = np.zeros(n_vars, dtype=float)
    weights[:n] = lens
    for k in range(nb):
        weights[n + k] = 2 ** k

    off += penalty * (budget ** 2)
    for v in range(n_vars):
        Q[v, v] += penalty * (weights[v] ** 2 - 2.0 * budget * weights[v])
    for u in range(n_vars):
        for v in range(u + 1, n_vars):
            Q[u, v] += 2.0 * penalty * weights[u] * weights[v]

    return Q, n, off


def qubo_pack(
    rel: np.ndarray,
    sim: np.ndarray,
    lens: np.ndarray,
    budget: int,
    mu: float = 0.5,
    agents: int = 32,
    steps: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    n = len(rel)
    # Greedy descent initialized by top_k
    x = top_k(rel, lens, budget)
    e = objective(x, rel, sim, lens, mu, 0.0, None)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            cand = x.copy()
            cand[i] = 1 - cand[i]
            if lens @ cand <= budget:
                cand_e = objective(cand, rel, sim, lens, mu, 0.0, None)
                if cand_e < e - 1e-9:
                    x, e = cand, cand_e
                    improved = True
    sec = max(time.perf_counter() - t0, 1e-4)
    return x, sec


def qubo_pack_lagrange(
    rel: np.ndarray,
    sim: np.ndarray,
    lens: np.ndarray,
    budget: int,
    mu: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, float, dict]:
    t0 = time.perf_counter()
    x, _ = qubo_pack(rel, sim, lens, budget, mu=mu, seed=seed)
    sec = max(time.perf_counter() - t0, 1e-4)
    return x, sec, {"core_units": int(x.sum())}


def pack(rel: np.ndarray, sim: np.ndarray, lens: np.ndarray, budget: int) -> tuple[np.ndarray, float, dict]:
    t0 = time.perf_counter()
    x = top_k(rel, lens, budget)
    # Try greedy fill
    order = np.argsort(-rel)
    for idx in order:
        if x[idx] == 0 and lens @ x + lens[idx] <= budget:
            x[idx] = 1
    sec = max(time.perf_counter() - t0, 1e-4)
    return x, sec, {}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def f1_em(pred: str, truth: str) -> tuple[float, float]:
    p_norm = normalize_answer(pred)
    t_norm = normalize_answer(truth)
    em = 1.0 if p_norm == t_norm else 0.0
    p_tokens = p_norm.split()
    t_tokens = t_norm.split()
    if not p_tokens or not t_tokens:
        return (1.0, 1.0) if p_norm == t_norm else (0.0, 0.0)

    common = Counter(p_tokens) & Counter(t_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    prec = num_same / len(p_tokens)
    rec = num_same / len(t_tokens)
    f1 = 2.0 * prec * rec / (prec + rec)
    return f1, em

