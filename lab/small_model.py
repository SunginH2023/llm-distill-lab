"""작은 모델 — numpy 소프트맥스 회귀 (의존성 0).

이 실험실이 재는 것은 모델 용량이 아니라 라벨 품질과 채점표 설계다.
그래서 학습기는 일부러 가장 단순한 것으로 고정했다. 라벨링 LLM 라벨을 바꿨을 때
움직이는 점수가 전부 라벨 쪽 효과가 되도록.

소프트 타깃(투표 분포)을 그대로 받는다 = 지식증류의 최소 형태.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit(X: np.ndarray, T: np.ndarray, *, l2: float = 3e-4,
        steps: int = 200, lr: float = 0.5, momentum: float = 0.9,
        sample_weight: np.ndarray | None = None) -> np.ndarray:
    """타깃 분포 T (n, C) 에 대한 교차엔트로피 최소화. 전배치 GD + 모멘텀.

    T 가 one-hot 이면 보통의 분류, 투표 분포면 KD.
    """
    # dtype 을 float32 로 통일한다. float32 X 와 float64 W 를 섞으면 numpy 가 매 스텝
    # 큰 쪽을 통째로 업캐스트한다 — 실측 14ms → 4ms.
    n, d = X.shape
    C = T.shape[1]
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    if sample_weight is None:
        w = np.ones((n, 1), dtype=np.float32)
    else:
        w = sample_weight.reshape(-1, 1).astype(np.float32)
    w = w / w.mean()

    W = np.zeros((d + 1, C), dtype=np.float32)
    V = np.zeros_like(W)
    for _ in range(steps):
        P = softmax(Xb @ W)
        G = Xb.T @ (w * (P - T)) / n + l2 * W
        V = momentum * V - lr * G
        W += V
    return W


def predict_proba(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return softmax(Xb @ W)


def decode(proba: np.ndarray, ratio_threshold: float = 0.5) -> np.ndarray:
    """top-1 은 항상, top-2 는 p2/p1 >= t 일 때만 채택 → (n, C) bool.

    실무에서 '개수를 따로 맞히는 판정기'를 여러 층위로 시도했지만 전부
    이 한 줄(확률비 게이트)을 이기지 못했다. e5 에서 그 재현을 다룬다.
    """
    n, C = proba.shape
    rows = np.arange(n)
    order = np.argsort(-proba, axis=1)
    p1, p2 = order[:, 0], order[:, 1]
    out = np.zeros((n, C), dtype=bool)
    out[rows, p1] = True
    take = proba[rows, p2] >= ratio_threshold * proba[rows, p1]
    out[rows[take], p2[take]] = True
    return out


def train_and_predict(feats_tr: np.ndarray, targets: np.ndarray,
                      feats_ev: np.ndarray, **kw) -> np.ndarray:
    W = fit(feats_tr, targets, **kw)
    return predict_proba(W, feats_ev)
