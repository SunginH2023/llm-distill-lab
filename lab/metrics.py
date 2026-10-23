"""Set-level metrics + paired significance tests.

In multi-label settings, looking only at flat accuracy is almost guaranteed to mislead.
It is common for overall exact match to stay flat while multi-label rows drop by half.
So overall EM / single-label-row EM / multi-label-row EM are always reported together.
"""

from __future__ import annotations

import math

import numpy as np


def exact_match(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return (pred == gold).all(axis=1)


def micro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = float((pred & gold).sum())
    fp = float((pred & ~gold).sum())
    fn = float((~pred & gold).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def summary(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    em = exact_match(pred, gold)
    multi = gold.sum(axis=1) >= 2
    return {
        "em": float(em.mean()),
        "micro_f1": micro_f1(pred, gold),
        "em_single": float(em[~multi].mean()) if (~multi).any() else float("nan"),
        "em_multi": float(em[multi].mean()) if multi.any() else float("nan"),
        "pred_multi_rate": float((pred.sum(axis=1) >= 2).mean()),
        "gold_multi_rate": float(multi.mean()),
    }


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float]:
    """EM comparison of two paired systems. Normal approximation with continuity correction (no scipy needed)."""
    b01 = int((~a_correct & b_correct).sum())   # a wrong, b right
    b10 = int((a_correct & ~b_correct).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "z": 0.0, "p": 1.0}
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    z = max(z, 0.0)
    p = math.erfc(z / math.sqrt(2.0))
    return {"b01": b01, "b10": b10, "z": z, "p": p}


def oracle_ceiling(gold: np.ndarray) -> float:
    """Upper bound when the label count is given - a reference line that shows where the remaining score is."""
    return 1.0


def rank_of(scores: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """Agreement between two rankings. +1 identical, -1 fully reversed."""
    items = list(rank_a)
    pos_a = {k: i for i, k in enumerate(rank_a)}
    pos_b = {k: i for i, k in enumerate(rank_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 0.0
