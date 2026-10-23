"""e3 — Do not discard the labeling LLM's "wavering"; use it as a training signal (the minimal form of knowledge distillation).

Hypothesis H3  Using only the labeling LLM's final label throws away "by how many votes the labeling LLM
               picked that answer". Supplying the vote distribution as a soft target as well makes the
               small model better. The gain should concentrate where the labeling LLM wavered.

What matters here is not the average gain but where it was earned. An overall +0.5pp
disappears in rounding, but "+2pp only in the split-vote stratum" tells you what to do next.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pin BLAS threads before importing numpy)

import numpy as np

from lab import labeler, metrics, pipeline
from lab.report import PARTIAL, Report

SPLIT = "test"
K = 5


def main() -> None:
    lab = pipeline.build()
    train_key = lab.key("train", K)
    test_key = lab.key(SPLIT, K)
    truth_top1 = lab.truth_top1(SPLIT)

    rep = Report("e3_soft_labels", "e3 · Vote distribution as soft targets (KD)",
                 "Does training on the vote distribution instead of the labeling LLM's final label improve the small model?")

    n = len(train_key.primary)
    onehot = np.zeros((n, lab.world.n_classes), dtype=np.float32)
    onehot[np.arange(n), train_key.primary] = 1.0

    targets = {
        "A hard (primary one-hot)": onehot,
        "B set (primary .75 / secondary .25)": labeler.set_targets(train_key),
        "C soft (raw vote distribution)": labeler.vote_targets(train_key),
    }
    probas = {name: pipeline.train_small_model(lab, t) for name, t in targets.items()}

    rows = []
    for name, p in probas.items():
        s = pipeline.score_all(lab, p, SPLIT, test_key)
        rows.append([name, s["truth_top1_acc"], s["truth_em"], s["truth_multi_em"],
                     s["same_source_top1"]])
    rep.text(f"A single set of pseudo-labels from {K} labeling-LLM calls is trained in three target forms only. "
             "Data, architecture, and number of steps are all identical.")
    rep.table(["Training target", "True top1", "True EM", "True multi-row EM", "Circular top1"], rows)
    rep.record("targets", rows)
    rep.text("The first thing that stands out is that soft targets lower overall EM"
             f" (B {rows[1][2]:.4f} → C {rows[2][2]:.4f}), "
             "even though top1 and multi-row EM went up in the same table. "
             "Soft targets leave mass on the runner-up label, so top-2 is opened more often, "
             "and overall EM loses when multi-label predictions increase (→ the Pareto front in e5). "
             "This is the first place where 'better/worse' depends on the metric.")

    # --- Where was it earned: strata where the labeling LLM wavered vs unanimous strata --------------------
    # Strata are defined by the labeling LLM's votes on the test side (oracle-free stratification that is also possible in production).
    strata = {
        "Unanimous stratum (agree)": test_key.grade == labeler.AGREE,
        "Split-vote stratum (partial+review)": test_key.grade != labeler.AGREE,
    }
    base = np.argmax(probas["B set (primary .75 / secondary .25)"], axis=1) == truth_top1
    kd = np.argmax(probas["C soft (raw vote distribution)"], axis=1) == truth_top1

    strat_rows = []
    for name, m in strata.items():
        mc = metrics.mcnemar(base[m], kd[m])
        strat_rows.append([name, int(m.sum()), float(base[m].mean()), float(kd[m].mean()),
                           (kd[m].mean() - base[m].mean()) * 100, mc["p"]])
    mc_all = metrics.mcnemar(base, kd)
    strat_rows.append(["All", len(base), float(base.mean()), float(kd.mean()),
                       (kd.mean() - base.mean()) * 100, mc_all["p"]])
    rep.text("Where did true top1 move when switching from B (set) to C (soft)? "
             "Stratification uses only the labeling LLM's votes on the test side — stratification must be "
             "possible without the oracle to transfer directly to production.")
    rep.table(["Stratum", "n", "B set", "C soft", "Δ(pp)", "McNemar p"], strat_rows)
    rep.record("strata", strat_rows)
    rep.record("mcnemar_all", mc_all)

    gain_split = strat_rows[1][4]
    gain_all = strat_rows[-1][4]
    rep.text(f"Overall gain {gain_all:+.2f}pp, split-vote stratum {gain_split:+.2f}pp. "
             "The direction is as predicted — the gain comes only where the labeling LLM wavered. "
             "But the absolute size of the overall gain is small, so it hardly justifies the cost of retraining on its own. "
             "The real value of soft targets lies not in the score but in the fact that "
             "preserving the vote distribution makes post-hoc experiments like this possible without re-querying the LLM.")

    rep.verdict(PARTIAL,
                f"H3 partially supported. The direction and location of the gain match the hypothesis (split stratum {gain_split:+.2f}pp), "
                f"but the overall {gain_all:+.2f}pp is not large. "
                "Adopt it, but do not claim 'this is what delivers the performance'.")
    rep.save()


if __name__ == "__main__":
    main()
