"""The small model - numpy softmax regression (zero dependencies).

What this lab measures is label quality and scorecard design, not model capacity.
So the learner is deliberately fixed to the simplest possible one, so that any score
movement when the labeling LLM's labels change is entirely a label-side effect.

It accepts soft targets (vote distributions) as-is = the minimal form of knowledge distillation.
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
    """Minimize cross-entropy against the target distribution T (n, C). Full-batch GD + momentum.

    If T is one-hot this is ordinary classification; if it is a vote distribution, it is KD.
    """
    # Unify dtype to float32. Mixing a float32 X with a float64 W makes numpy upcast the
    # larger operand wholesale on every step - measured 14ms -> 4ms.
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
    """Always take top-1; take top-2 only when p2/p1 >= t -> (n, C) bool.

    In production, dedicated "label count predictors" were tried at several levels, but
    none beat this one line (a probability-ratio gate). e5 reproduces that.
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
