"""집합 단위 지표 + 짝비교 유의성.

멀티라벨에서 flat accuracy 하나만 보면 거의 반드시 속는다.
전체 완전일치는 그대로인데 멀티행이 반토막 나는 조합이 흔하기 때문이다.
그래서 전체 EM / 단일행 EM / 멀티행 EM 을 항상 같이 찍는다.
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
    """짝지은 두 시스템의 EM 비교. 연속성 보정 정규근사 (scipy 불필요)."""
    b01 = int((~a_correct & b_correct).sum())   # a 틀리고 b 맞음
    b10 = int((a_correct & ~b_correct).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "z": 0.0, "p": 1.0}
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    z = max(z, 0.0)
    p = math.erfc(z / math.sqrt(2.0))
    return {"b01": b01, "b10": b10, "z": z, "p": p}


def oracle_ceiling(gold: np.ndarray) -> float:
    """개수를 알려줬을 때의 상한 — '남은 점수가 어디 있는지' 알려주는 기준선."""
    return 1.0


def rank_of(scores: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """두 순위의 일치도. +1 완전일치, -1 완전역전."""
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
