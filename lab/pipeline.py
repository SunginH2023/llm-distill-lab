"""The assembly line shared by all experiments.

    build world -> labeling-LLM votes -> consensus gate -> pseudo-labels -> train small model -> score with three scorecards

Having three scorecards is the point of this repo.

    circular : pseudo-labels produced by the labeling LLM      - the only thing observable in production
    weak     : signals from outside the pipeline (coarse, noisy) - what can be built in production
    oracle   : the true labels                                 - exists only in the synthetic world; used solely to validate the two above
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import labeler, metrics, signals, small_model
from .world import World, WorldConfig, make_world, split_indices

# Kept as fractions - must run unchanged on small worlds too (for tests).
SPLIT_FRACTIONS = {"train": 0.50, "val": 0.133, "test": 0.333}


@dataclass
class Lab:
    world: World
    idx: dict[str, np.ndarray]
    axis_fix: signals.WeakAxis     # axis used only for label refinement
    axis_eval: signals.WeakAxis    # axis used only for evaluation - never used for refinement
    _teachers: dict = field(default_factory=dict)

    # --- slicing ---------------------------------------------------------
    def feats(self, split: str) -> np.ndarray:
        return self.world.feats[self.idx[split]]

    def truth(self, split: str) -> np.ndarray:
        return self.world.truth[self.idx[split]]

    def truth_top1(self, split: str) -> np.ndarray:
        return self.world.primary[self.idx[split]]

    def axis(self, which: str, split: str) -> signals.WeakAxis:
        a = self.axis_fix if which == "fix" else self.axis_eval
        i = self.idx[split]
        return signals.WeakAxis(a.name, a.group_of, a.observed[i], a.noise)

    def get_teacher(self, cfg: labeler.LabelerConfig = labeler.MAIN) -> labeler.LabelerLLM:
        key = (cfg.gamma, cfg.beta_magnet, cfg.pair_skew, cfg.temperature, cfg.seed)
        if key not in self._teachers:
            self._teachers[key] = labeler.LabelerLLM(self.world, cfg)
        return self._teachers[key]

    # --- LLM calls -------------------------------------------------------
    def vote(self, split: str, k: int, cfg: labeler.LabelerConfig = labeler.MAIN,
             seed: int = 0) -> np.ndarray:
        return self.get_teacher(cfg).vote(self.idx[split], k=k, seed=7000 + 13 * seed + k)

    def key(self, split: str, k: int, cfg: labeler.LabelerConfig = labeler.MAIN,
            seed: int = 0, **kw) -> labeler.PseudoLabels:
        return labeler.labels_from_votes(self.vote(split, k, cfg, seed), **kw)


def build(world_cfg: WorldConfig | None = None, seed: int = 13) -> Lab:
    w = make_world(world_cfg)
    sizes = {k: int(w.n * f) for k, f in SPLIT_FRACTIONS.items()}
    idx = split_indices(w.n, sizes, seed=seed)
    C = w.n_classes
    return Lab(
        world=w, idx=idx,
        axis_fix=signals.make_axis("user-selected category", w.primary, C,
                                   n_groups=5, noise=0.30, seed=101),
        axis_eval=signals.make_axis("routing-target assignment", w.primary, C,
                                    n_groups=7, noise=0.25, seed=202),
    )


# --- small model -----------------------------------------------------------------

def train_small_model(lab: Lab, targets: np.ndarray, train_split: str = "train",
                  eval_split: str = "test", weight: np.ndarray | None = None,
                  **kw) -> np.ndarray:
    return small_model.train_and_predict(lab.feats(train_split), targets,
                                     lab.feats(eval_split), sample_weight=weight, **kw)


# --- scoring -----------------------------------------------------------------

def key_quality(lab: Lab, key: labeler.PseudoLabels, split: str) -> dict[str, float]:
    """Score the pseudo-labels themselves against the oracle (label quality only, independent of the small model)."""
    truth = lab.truth(split)
    s = metrics.summary(key.mat, truth)
    return {
        "key_em": s["em"],
        "key_micro_f1": s["micro_f1"],
        "key_top1": float((key.primary == lab.truth_top1(split)).mean()),
        "key_multi_rate": s["pred_multi_rate"],
    }


def score_all(lab: Lab, proba: np.ndarray, split: str, circular: labeler.PseudoLabels,
              ratio_threshold: float = 0.5) -> dict[str, float]:
    """Score the same predictions with all three scorecards."""
    pred = small_model.decode(proba, ratio_threshold)
    truth = lab.truth(split)
    ax = lab.axis("eval", split)
    mask = signals.discriminative_mask(ax, proba)
    return {
        "same_source_em": float(metrics.exact_match(pred, circular.mat).mean()),
        "same_source_top1": float((np.argmax(proba, axis=1) == circular.primary).mean()),
        "outside_signal": ax.score_top1(proba, mask),
        "truth_em": float(metrics.exact_match(pred, truth).mean()),
        "truth_top1_acc": float((np.argmax(proba, axis=1) == lab.truth_top1(split)).mean()),
        "truth_multi_em": metrics.summary(pred, truth)["em_multi"],
    }
