"""A synthetic world in which the ground truth exists.

In a production pipeline the "true label" cannot be observed. All that is observable is the
labels stamped by the labeling LLM, so there is no way to prove "our decision was right".

This module moves the same structure into a world where the answer is known. Because we hold
the oracle (the true labels) here, we can score after the fact whether the decisions of a
pipeline that does not use the oracle were actually right. The oracle is used only for
evaluation and never for training or label refinement.

Every component of the world mimics a phenomenon encountered repeatedly in production:

| Mechanism                          | Phenomenon it mimics                                                   |
|------------------------------------|------------------------------------------------------------------------|
| Confusable pairs (nearby centroids) | Type pairs whose definitions overlap, so even humans disagree          |
| Magnet class                       | A catch-all ("other") top-level class that absorbs every ambiguous item |
| True multi-label                   | Items where two types genuinely hold at once (not an error)            |
| Reduced small-model representation | The small model's capacity gap (a smaller model than the labeling LLM) |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pairs with overlapping definition boundaries - built by pulling centroids toward each other.
CONFUSABLE_PAIRS: tuple[tuple[int, int], ...] = ((3, 4), (5, 6), (7, 8))
# Pairs that can genuinely hold at the same time (combinations where true multi-label is allowed).
COOCCUR_PAIRS: tuple[tuple[int, int], ...] = ((2, 9), (10, 11))
MULTI_ALLOWED: frozenset[frozenset[int]] = frozenset(
    frozenset(p) for p in CONFUSABLE_PAIRS + COOCCUR_PAIRS
)


@dataclass(frozen=True)
class WorldConfig:
    n: int = 60_000
    n_classes: int = 12
    dim: int = 24
    student_dim: int = 23      # reduced representation seen by the small model (capacity gap)
    feat_noise: float = 0.25
    sigma: float = 1.0
    centroid_scale: float = 3.5
    pair_gap: float = 0.30     # smaller means confusable pairs sit closer together
    prior_decay: float = 0.9   # long-tailed class frequencies
    magnet_class: int = 0
    magnet_prior_boost: float = 1.5
    multi_ratio: float = 0.40  # both are true if post2 >= multi_ratio * post1
    seed: int = 7


@dataclass
class World:
    cfg: WorldConfig
    mu: np.ndarray            # (C, d) class centroids
    prior: np.ndarray         # (C,)
    x: np.ndarray             # (n, d) observations
    log_post: np.ndarray      # (n, C) true log posterior - never given directly to the labeling LLM / small model
    post: np.ndarray          # (n, C)
    truth: np.ndarray         # (n, C) bool - Bayes-optimal label set = the oracle
    feats: np.ndarray         # (n, student_dim) small-model input (standardized)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def n_classes(self) -> int:
        return self.mu.shape[0]

    @property
    def primary(self) -> np.ndarray:
        """The true top label. Not the argmax of the truth matrix - a set holds no order, so
        on multi-label rows argmax just picks the smaller index. The top label is defined by the posterior."""
        return np.argmax(self.post, axis=1)

    @property
    def ambiguity(self) -> np.ndarray:
        """1 - max posterior. Larger means closer to a decision boundary."""
        return 1.0 - self.post.max(axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_world(cfg: WorldConfig | None = None) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    C, d = cfg.n_classes, cfg.dim

    # --- class centroids ---------------------------------------------------
    mu = rng.normal(size=(C, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    mu *= cfg.centroid_scale

    # Confusable pairs: pull the two centroids toward each other -> definition boundaries overlap
    for a, b in CONFUSABLE_PAIRS:
        mid = 0.5 * (mu[a] + mu[b])
        mu[a] = mid + (mu[a] - mid) * cfg.pair_gap
        mu[b] = mid + (mu[b] - mid) * cfg.pair_gap

    # Magnet class: placed near the global center so it absorbs ambiguous items
    mu[cfg.magnet_class] = mu.mean(axis=0) + 0.15 * mu[cfg.magnet_class]

    # --- prior (long tail + the magnet class is common) -----------------------
    prior = cfg.prior_decay ** np.arange(C)
    prior[cfg.magnet_class] *= cfg.magnet_prior_boost
    prior /= prior.sum()

    # --- samples -----------------------------------------------------------
    y_gen = rng.choice(C, size=cfg.n, p=prior)
    x = mu[y_gen] + cfg.sigma * rng.normal(size=(cfg.n, d))

    # Expand ||x - mu||^2 so that only an (n, C) array is built (avoids an n×C×d tensor)
    sq = (x * x).sum(1)[:, None] - 2.0 * x @ mu.T + (mu * mu).sum(1)[None, :]
    log_post = -0.5 * sq / cfg.sigma**2 + np.log(prior)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = _softmax(log_post)

    # --- oracle label set ------------------------------------------------
    # The true label is defined as the Bayes-optimal set, not "the class used for generation".
    # That is the upper bound reachable from x alone, and it is what makes the "ceiling" well-defined.
    order = np.argsort(-post, axis=1)
    p1, p2 = order[:, 0], order[:, 1]
    rows = np.arange(cfg.n)
    take_second = post[rows, p2] >= cfg.multi_ratio * post[rows, p1]
    allowed = np.array(
        [frozenset((a, b)) in MULTI_ALLOWED for a, b in zip(p1, p2)]
    )
    take_second &= allowed

    truth = np.zeros((cfg.n, C), dtype=bool)
    truth[rows, p1] = True
    truth[rows[take_second], p2[take_second]] = True

    # --- small-model input: orthogonal projection + noise (capacity gap) --------------------------
    # Why draw it orthogonally (QR): a plain Gaussian projection is ill-conditioned and creates
    # optimization difficulty rather than information loss. Scores would then drop not because
    # "the small model is weak" but because "it fails to converge", and the label-quality effect
    # gets buried in that noise. (This actually fooled us once early on.)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    proj = q[:, :cfg.student_dim]
    feats = x @ proj + cfg.feat_noise * rng.normal(size=(cfg.n, cfg.student_dim))
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    feats = feats.astype(np.float32)

    return World(cfg=cfg, mu=mu, prior=prior, x=x, log_post=log_post,
                 post=post, truth=truth, feats=feats)


def split_indices(n: int, sizes: dict[str, int], seed: int = 13) -> dict[str, np.ndarray]:
    """Non-overlapping index split. Order is fixed (reproducibility)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out, cur = {}, 0
    for name, size in sizes.items():
        out[name] = np.sort(perm[cur:cur + size])
        cur += size
    if cur > n:
        raise ValueError(f"sum of split sizes ({cur}) exceeds n ({n})")
    return out
