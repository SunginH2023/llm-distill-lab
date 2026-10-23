"""The labeling LLM (a simulated stand-in) + consensus gate + LLM-as-a-judge.

This module mirrors a situation that is common today.

    There is no budget for human labeling, but a capable LLM is available.
    -> Ask the LLM to label. Those labels are pseudo-labels.
    -> But nobody knows how trustworthy these pseudo-labels are.
    -> So people stack methods on top: call it several times, have another
      LLM act as judge, send only the split votes to humans, and so on.

This module simulates that "LLM that produces labels". One key design choice:
the LLM's error is planted as two separate components.

    Random component (temperature)  - fluctuates on every call. Averages out over repeated calls.
    Systematic component (bias / skew) - errs in the same direction every time. Never averages out.

Pseudo-labels inherit the systematic component unchanged. If an evaluation set built from
those pseudo-labels is used to score models, a model that also imitates the systematic
component scores high. This is the effect measured in e2.

The systematic component is planted in two ways:
  * `beta_magnet` : a tendency to push ambiguous items into the catch-all ("other") class (the magnet class)
  * `pair_skew`   : a tendency to always pick the same side of a confusable pair
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import CONFUSABLE_PAIRS, World

# Grade: routes rows by how split the votes are. Not converted into a probability.
AGREE, PARTIAL, REVIEW = 0, 1, 2
GRADE_NAMES = {AGREE: "agree", PARTIAL: "partial", REVIEW: "review"}


@dataclass(frozen=True)
class LabelerConfig:
    gamma: float = 0.8         # sharpness. 1.0 is theoretically optimal; lower is blurrier
    beta_magnet: float = 1.1   # systematic bias: preference for the catch-all class
    pair_skew: float = 0.8     # systematic bias: preference for one side of a confusable pair
    temperature: float = 0.7   # call-to-call fluctuation (random component)
    seed: int = 11


# Two presets: a capable main LLM / a cheap auxiliary LLM
MAIN = LabelerConfig()
CHEAP = LabelerConfig(gamma=0.35, beta_magnet=1.9, pair_skew=1.4, temperature=1.1, seed=23)


class LabelerLLM:
    """The LLM that assigns labels. `vote(k)` corresponds to k calls."""

    def __init__(self, world: World, cfg: LabelerConfig = MAIN):
        self.world, self.cfg = world, cfg
        bias = np.zeros(world.n_classes)
        bias[world.cfg.magnet_class] += cfg.beta_magnet
        for a, _b in CONFUSABLE_PAIRS:      # prefers the first code of each pair
            bias[a] += cfg.pair_skew
        self.bias = bias
        self._logits = cfg.gamma * world.log_post + bias[None, :]

    def vote(self, idx: np.ndarray, k: int, seed: int) -> np.ndarray:
        """k calls. Returns integer vote counts of shape (len(idx), C).

        Softmax sampling via the Gumbel-max trick = an LLM call with temperature.
        """
        rng = np.random.default_rng(seed)
        logits = self._logits[idx]
        counts = np.zeros(logits.shape, dtype=np.int32)
        for _ in range(k):
            g = rng.gumbel(size=logits.shape)
            pick = np.argmax(logits / max(self.cfg.temperature, 1e-6) + g, axis=1)
            counts[np.arange(len(idx)), pick] += 1
        return counts


class JudgeLLM:
    """LLM-as-a-judge. Instead of choosing a new label, it reviews a proposed label.

    Kept separate because it is a common pattern in practice. The usual expectation is
    "asking another LLM once more improves quality", but in reality the outcome depends
    on what biases the judge has (measured in e1).

    The judge answers "is this label the top choice?" stochastically. The judge has its
    own biases, and when those biases overlap with the main LLM's, neither filters out
    the other's errors.
    """

    def __init__(self, world: World, cfg: LabelerConfig = CHEAP, strictness: float = 0.0):
        self.world, self.cfg, self.strictness = world, cfg, strictness
        bias = np.zeros(world.n_classes)
        bias[world.cfg.magnet_class] += cfg.beta_magnet
        for a, _b in CONFUSABLE_PAIRS:
            bias[a] += cfg.pair_skew
        self._logits = cfg.gamma * world.log_post + bias[None, :]

    def accept(self, idx: np.ndarray, proposed: np.ndarray, seed: int) -> np.ndarray:
        """Answers whether to accept each proposed label, as an (n,) bool array.

        Raising `strictness` makes the judge stricter (= lower acceptance rate).
        """
        rng = np.random.default_rng(seed)
        lg = self._logits[idx]
        rows = np.arange(len(idx))
        masked = np.where(np.arange(lg.shape[1])[None, :] == proposed[:, None], -np.inf, lg)
        margin = lg[rows, proposed] - masked.max(axis=1) - self.strictness
        p_accept = 1.0 / (1.0 + np.exp(-margin / max(self.cfg.temperature, 1e-6)))
        return rng.random(len(idx)) < p_accept


@dataclass
class PseudoLabels:
    """A bundle of LLM-assigned labels.

    Carries not only the label set (`mat`) but also which label is first (`primary`).
    An early version stored only the set; during label refinement the top label changed
    silently and the metrics got tangled. A set representation cannot hold order.
    """
    mat: np.ndarray       # (n, C) bool - label set
    primary: np.ndarray   # (n,) top label
    frac: np.ndarray      # (n, C) vote fractions
    grade: np.ndarray     # (n,) AGREE / PARTIAL / REVIEW
    k: np.ndarray         # (n,) number of calls may differ per row (re-voting ambiguous rows)

    @property
    def multi_rate(self) -> float:
        return float((self.mat.sum(axis=1) >= 2).mean())

    def copy(self) -> "PseudoLabels":
        return PseudoLabels(self.mat.copy(), self.primary.copy(),
                            self.frac, self.grade, self.k)


def labels_from_votes(counts: np.ndarray, secondary_frac: float = 0.34) -> PseudoLabels:
    """Vote counts -> pseudo-labels + grade.

    Key point: vote counts are NOT converted into a "confidence probability"; they are used
    only as a gate that routes rows. (An earlier attempt to use confidence as a quality
    gate was dropped because it could not be calibrated.)

    Rule for adopting the second label: the runner-up gets more than 1/3 of all votes.
    With k=3 this rule can never be satisfied
    -> pseudo-labels built from 3 calls structurally cannot express multi-label. This is where e4 starts.
    """
    k = counts.sum(axis=1)
    frac = counts / np.maximum(k, 1)[:, None]
    n, C = counts.shape
    rows = np.arange(n)

    order = np.argsort(-counts, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]

    mat = np.zeros((n, C), dtype=bool)
    mat[rows, p1] = True
    second = frac[rows, p2] >= secondary_frac
    mat[rows[second], p2[second]] = True

    top = frac[rows, p1]
    grade = np.where(top >= 1.0 - 1e-9, AGREE, np.where(top > 0.5, PARTIAL, REVIEW))
    return PseudoLabels(mat=mat, primary=p1, frac=frac, grade=grade, k=k)


def vote_targets(labels: PseudoLabels) -> np.ndarray:
    """Use the vote distribution directly as the training target (the minimal form of knowledge distillation).

    Using only the final label throws away "by how many votes the LLM picked that answer".
    If the vote distribution is preserved, this experiment is possible without re-querying the LLM.
    """
    s = labels.frac.sum(axis=1, keepdims=True)
    return (labels.frac / np.maximum(s, 1e-9)).astype(np.float32)


def set_targets(labels: PseudoLabels, secondary_weight: float = 0.25) -> np.ndarray:
    """Label set -> probability target. One-hot if single, (1-w, w) if multi."""
    n, C = labels.mat.shape
    rows = np.arange(n)
    t = np.zeros((n, C), dtype=np.float32)
    t[rows, labels.primary] = 1.0

    extra = labels.mat.copy()
    extra[rows, labels.primary] = False
    has_extra = extra.any(axis=1)
    if has_extra.any():
        e = extra[has_extra].astype(np.float32)
        e /= e.sum(axis=1, keepdims=True)
        idx = np.where(has_extra)[0]
        t[idx] = e * secondary_weight
        t[idx, labels.primary[idx]] = 1.0 - secondary_weight
    return t
