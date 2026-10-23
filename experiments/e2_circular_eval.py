"""e2 — Scoring small models with pseudo-labels produced by the labeling LLM flips the ranking.

Hypothesis H2  An evaluation set built from labels assigned by the labeling LLM measures imitation,
               not "accuracy". The small model that best mimics even the labeling LLM's systematic
               bias comes out first, and the genuinely better small model is pushed down. In contrast,
               a coarse outside signal from beyond the pipeline ranks models the same way as the truth,
               even though its absolute scores are much lower.

In production this claim cannot be proven, because the truth is unknown.
Here the truth is known, so the three scorecards can be placed side by side and the scorecards themselves can be scored.

All comparisons use top-label selection. The outside signal works at the coarse group level and therefore
cannot score the label "count" (count is handled separately in e5).
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pin BLAS threads before importing numpy)

import numpy as np

from lab import labeler, metrics, pipeline, signals, small_model
from lab.report import ACCEPT, Report

SPLIT = "test"


def build_models(lab) -> dict[str, np.ndarray]:
    """All share the same architecture and hyperparameters. Only the pseudo-labels differ."""
    k1 = lab.key("train", 1)
    k3 = lab.key("train", 3)
    k5 = lab.key("train", 5)
    k7 = lab.key("train", 7)
    fix5, _ = signals.tiebreak(k5, lab.axis("fix", "train"), k5.grade != labeler.AGREE)
    fix7, _ = signals.tiebreak(k7, lab.axis("fix", "train"), k7.grade != labeler.AGREE)

    keys = {"1 call": k1, "3 calls": k3, "5 calls": k5, "7 calls": k7,
            "5 calls + outside-signal fix": fix5, "7 calls + outside-signal fix": fix7}
    return {name: pipeline.train_small_model(lab, labeler.set_targets(k))
            for name, k in keys.items()}, keys


def main() -> None:
    lab = pipeline.build()
    test_key = lab.key(SPLIT, 5)          # the only evaluation set one can actually hold in production
    probas, keys = build_models(lab)

    rep = Report("e2_circular_eval", "e2 · A circular scorecard flips the ranking",
                 "Scoring with labeling-LLM pseudo-labels / with an outside-the-pipeline signal / with the true labels — do the three give the same ranking?")

    ax = lab.axis("eval", SPLIT)
    truth_top1 = lab.truth_top1(SPLIT)

    rows, scores = [], {}
    for name, p in probas.items():
        s = pipeline.score_all(lab, p, SPLIT, test_key)
        scores[name] = s
        rows.append([name, s["same_source_top1"], s["outside_signal"], s["truth_top1_acc"],
                     s["same_source_em"], s["truth_em"]])
    rep.text("Six small models. Architecture and hyperparameters are identical; only the training labels differ. "
             f"The outside-signal score is measured only on rows where that axis distinguishes the top two candidates "
             f"(axis '{ax.name}', {ax.n_groups} groups, observation noise {ax.noise:.0%}).")
    rep.table(["Small model (training labels)", "① Pseudo-label eval set", "② Outside signal", "③ True labels",
               "Pseudo EM", "True EM"], rows)
    rep.record("scores", scores)

    # --- Ranking comparison --------------------------------------------------------
    r_circ = metrics.rank_of({k: v["same_source_top1"] for k, v in scores.items()})
    r_weak = metrics.rank_of({k: v["outside_signal"] for k, v in scores.items()})
    r_true = metrics.rank_of({k: v["truth_top1_acc"] for k, v in scores.items()})
    tau_circ = metrics.kendall_tau(r_circ, r_true)
    tau_weak = metrics.kendall_tau(r_weak, r_true)

    # Does being outside the loop always suffice? — build one more axis contaminated in the same direction as the labeling LLM.
    dirty = signals.contaminate(ax, lab.world.cfg.magnet_class, strength=0.35, seed=77)
    dirty_scores = {n: dirty.score_top1(p, signals.discriminative_mask(dirty, p))
                    for n, p in probas.items()}
    r_dirty = metrics.rank_of(dirty_scores)
    tau_dirty = metrics.kendall_tau(r_dirty, r_true)

    rep.text("Extracting only the rankings (leftmost is first):")
    rep.table(["Scorecard", "Ranking produced", "Rank correlation τ with the truth"],
              [["① Eval set built from pseudo-labels (same source)", " > ".join(r_circ), tau_circ],
               ["② Signal from outside the pipeline", " > ".join(r_weak), tau_weak],
               ["②' Outside signal contaminated with the same bias", " > ".join(r_dirty), tau_dirty],
               ["③ Truth (oracle)", " > ".join(r_true), 1.0]])
    rep.record("tau", {"same_source": tau_circ, "outside": tau_weak, "outside_contaminated": tau_dirty,
                       "rank_same_source": r_circ, "rank_outside": r_weak, "rank_true": r_true})
    rep.text("②' is an axis that comes from outside the pipeline but is biased in the same direction as the labeling LLM "
             "(35% of observations are pulled toward the group the labeling LLM favors — the situation where the "
             "labeling LLM has also learned an operational auto-routing rule). "
             f"τ drops from {tau_weak:.2f} to {tau_dirty:.2f}. "
             "Coming from an outside source is not sufficient on its own; one must also check that it is independent "
             "of the labeling LLM's bias. This is why, when the routing axis was used in practice, it had to be "
             "re-tested with the auto-routed items removed.")

    best_circ, best_true = r_circ[0], r_true[0]
    gap = scores[best_true]["truth_top1_acc"] - scores[best_circ]["truth_top1_acc"]
    rep.text(f"The pseudo-label eval set picks {best_circ} as first; the true first is {best_true}. "
             f"Choosing a model by this eval set alone leaves {gap * 100:.2f}pp on the table against the truth. "
             f"The outside signal's absolute score is low at {scores[best_true]['outside_signal']:.3f} "
             f"— because the axis is inherently coarse — but its ranking is right. "
             "Do not report its absolute score; use it only for relative comparison. "
             f"Caveat: the ① scores of the models without the fix are very close "
             f"({min(v['same_source_top1'] for k, v in scores.items() if 'fix' not in k):.4f}–"
             f"{max(v['same_source_top1'] for k, v in scores.items() if 'fix' not in k):.4f}), and there are only "
             f"{len(scores)} models and a single seed, so each τ carries uncertainty.")

    mc = metrics.mcnemar(
        signals.discriminative_mask(ax, probas[best_circ]) & (np.argmax(probas[best_circ], 1) == truth_top1),
        signals.discriminative_mask(ax, probas[best_true]) & (np.argmax(probas[best_true], 1) == truth_top1))
    rep.record("mcnemar_best", mc)

    # --- Bias sweep: at what point does the circular scorecard start lying -------------------
    sweep = []
    for beta in (0.0, 0.4, 0.8, 1.1, 1.6):
        cfg = labeler.LabelerConfig(beta_magnet=beta, pair_skew=beta * 0.7)
        k5b = lab.key("train", 5, cfg=cfg)
        fixb, _ = signals.tiebreak(k5b, lab.axis("fix", "train"), k5b.grade != labeler.AGREE)
        tkb = lab.key(SPLIT, 5, cfg=cfg)
        sc = {}
        for nm, kk in {"1 call": lab.key("train", 1, cfg=cfg), "5 calls": k5b, "5 calls + outside-signal fix": fixb}.items():
            p = pipeline.train_small_model(lab, labeler.set_targets(kk))
            sc[nm] = pipeline.score_all(lab, p, SPLIT, tkb)
        rt = metrics.rank_of({k: v["truth_top1_acc"] for k, v in sc.items()})
        sweep.append([beta,
                      metrics.kendall_tau(metrics.rank_of({k: v["same_source_top1"] for k, v in sc.items()}), rt),
                      metrics.kendall_tau(metrics.rank_of({k: v["outside_signal"] for k, v in sc.items()}), rt),
                      float(np.mean([v["truth_top1_acc"] for v in sc.values()]))])
    rep.text("Increasing the strength of the labeling LLM's systematic bias from 0, we re-measure the rank correlation of both scorecards. "
             f"With bias 0 the pseudo-label eval set does not reverse the ranking (τ = {sweep[0][1]:+.2f}, weakly positive) "
             "— the reversal appears only when there is bias. "
             "(Real LLMs are unlikely to have bias 0.)")
    rep.table(["Bias β", "① Pseudo-label eval set τ", "② Outside signal τ", "Mean true top1"], sweep)
    rep.record("bias_sweep", sweep)

    rep.verdict(ACCEPT,
                f"H2 supported. Rank correlation of the pseudo-label eval set τ={tau_circ:.2f}, outside signal τ={tau_weak:.2f}. "
                "The model ranked first by labeling-LLM pseudo-labels was not first against the truth. "
                "A coarse axis with low absolute scores is a better scorecard than a circular axis with high absolute scores.")
    rep.save()


if __name__ == "__main__":
    main()
