"""Signals from outside the pipeline (weak external axes).

Scoring with labels assigned by the labeling LLM cannot escape the "same-family effect".
Escaping it requires observations that come from outside the LLM pipeline. Two were used
in production:

  * the coarse category the user picked themselves at submission time
  * the routing target a staff member actually sent the item to

Both are (a) much coarser than the label taxonomy and (b) noisy, but (c) statistically
independent of the labeling LLM. So absolute scores come out low and only relative comparisons are valid.

Here we build two axes with the same properties. One is used for label refinement and the
other is kept for evaluation only - evaluating with the axis used for refinement would also be circular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeakAxis:
    """Many-to-one map from classes to coarse groups + observation noise."""
    name: str
    group_of: np.ndarray     # (C,) class -> group
    observed: np.ndarray     # (n,) observed group (noisy)
    noise: float

    @property
    def n_groups(self) -> int:
        return int(self.group_of.max()) + 1

    def compatible(self, pred: np.ndarray) -> np.ndarray:
        """Is the predicted set compatible with the observed group? -> (n,) bool.

        Measures consistency, not "accuracy": does the model's answer not contradict this observation?
        """
        pred_groups = np.where(pred, self.group_of[None, :], -1)
        return (pred_groups == self.observed[:, None]).any(axis=1)

    def score(self, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
        ok = self.compatible(pred)
        return float(ok.mean() if mask is None else ok[mask].mean())

    def score_top1(self, proba: np.ndarray, mask: np.ndarray | None = None) -> float:
        """The outside signal scores only the top label.

        Scoring the whole set favors models that emit many labels (more chances to be compatible).
        In other words, the outside signal cannot score the label "count" - a structural limit of
        this axis, which is why count/decoding is measured separately in its own experiment (e5).
        """
        top1 = np.argmax(proba, axis=1)
        ok = self.group_of[top1] == self.observed
        return float(ok.mean() if mask is None else ok[mask].mean())

    def correct_top1(self, proba: np.ndarray) -> np.ndarray:
        top1 = np.argmax(proba, axis=1)
        return self.group_of[top1] == self.observed


def make_axis(name: str, primary: np.ndarray, n_classes: int, n_groups: int,
              noise: float, seed: int) -> WeakAxis:
    """Derived from the true labels, with noise mixed in. The labeling LLM plays no part in this process."""
    rng = np.random.default_rng(seed)
    group_of = rng.integers(0, n_groups, size=n_classes)
    # make sure no group is empty
    for g in range(n_groups):
        if not (group_of == g).any():
            group_of[rng.integers(0, n_classes)] = g

    obs = group_of[primary].copy()
    flip = rng.random(len(obs)) < noise
    obs[flip] = rng.integers(0, n_groups, size=int(flip.sum()))
    return WeakAxis(name=name, group_of=group_of, observed=obs, noise=noise)


def contaminate(axis: WeakAxis, attractor_class: int, strength: float,
                seed: int) -> WeakAxis:
    """Contaminate an outside signal "in the same direction as the labeling LLM".

    Coming from outside the loop is not enough on its own. If the axis shares the labeling
    LLM's bias (e.g. the labeling LLM has also learned an operational auto-routing rule),
    the axis gets pulled back into the loop. Here the observation noise is not uniform but
    skewed toward the group the labeling LLM favors.
    """
    rng = np.random.default_rng(seed)
    obs = axis.observed.copy()
    target = axis.group_of[attractor_class]
    pull = rng.random(len(obs)) < strength
    obs[pull] = target
    return WeakAxis(f"{axis.name} (contaminated)", axis.group_of, obs, axis.noise)


def discriminative_mask(axis: WeakAxis, proba: np.ndarray) -> np.ndarray:
    """Keep only rows where the axis actually distinguishes the top two candidates.

    If both candidates fall in the same group, the axis says nothing about that row.
    This mirrors excluding "non-discriminative catch-all categories" in production.
    """
    order = np.argsort(-proba, axis=1)
    return axis.group_of[order[:, 0]] != axis.group_of[order[:, 1]]


def tiebreak(key, axis: WeakAxis, only: np.ndarray, secondary_frac: float = 0.34):
    """For ambiguous rows only, flip the primary label using an axis from outside the loop.

    Condition: the current top label contradicts the observed group and the runner-up is
    compatible -> swap them. The oracle is not used. Restricting targets to ambiguous rows
    via `only` matters - touching unanimous rows too lets a noisy axis erode correct labels.

    Returns (new key, number of swapped rows).
    """
    out = key.copy()
    n = out.mat.shape[0]
    rows = np.arange(n)
    order = np.argsort(-key.frac, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]
    g = axis.group_of
    swap = only & (g[p1] != axis.observed) & (g[p2] == axis.observed)

    s = rows[swap]
    out.primary[s] = p2[s]
    out.mat[s] = False
    out.mat[s, p2[s]] = True
    keep_old = key.frac[s, p1[s]] >= secondary_frac      # the old top label is demoted to secondary
    out.mat[s[keep_old], p1[s[keep_old]]] = True
    return out, int(swap.sum())
