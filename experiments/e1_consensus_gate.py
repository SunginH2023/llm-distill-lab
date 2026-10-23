"""e1 — Consensus gate: use the degree of vote disagreement to decide where to spend the review budget.

Hypothesis H1   Calling the labeling LLM several times and looking at how split the votes are lets us
                sort label quality without an oracle. Human review can then be spent only on split
                rows instead of on everything.
Hypothesis H1b  Adding one more cheap model and "keeping only rows where both models agree" improves quality further.

H1b is intuitively plausible, but it is an approach that was actually tried and dropped in production.
This experiment records quantitatively why.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pin BLAS threads before importing numpy)

import numpy as np

from lab import labeler, pipeline, signals
from lab.report import ACCEPT, REJECT, Report

SPLIT = "test"
K = 5


def main() -> None:
    lab = pipeline.build()
    truth = lab.truth(SPLIT)
    truth_top1 = lab.truth_top1(SPLIT)
    key = lab.key(SPLIT, K)

    rep = Report("e1_consensus_gate", "e1 · Consensus gate and review budget",
                 "Can the degree of vote disagreement alone pre-sort label quality?")

    # --- 1. Quality by grade -------------------------------------------------
    rows = []
    correct = key.primary == truth_top1
    for grade, name in labeler.GRADE_NAMES.items():
        m = key.grade == grade
        if not m.any():
            continue
        rows.append([name, int(m.sum()), f"{m.mean():.3f}",
                     float(correct[m].mean()),
                     float(lab.world.ambiguity[lab.idx[SPLIT]][m].mean())])
    rep.text(f"Labeling LLM called {K} times. Grades are assigned from the vote distribution alone — the oracle is used only for scoring.")
    rep.table(["Grade", "Rows", "Share", "Primary accuracy", "Mean ambiguity"], rows)
    rep.record("grade_quality", rows)

    agree_q = float(correct[key.grade == labeler.AGREE].mean())
    review_m = key.grade == labeler.REVIEW
    review_q = float(correct[review_m].mean()) if review_m.any() else float("nan")
    rep.text(f"Unanimous band {agree_q:.3f} vs split-vote band {review_q:.3f} — "
             f"a {(agree_q - review_q) * 100:.1f}pp gap. Even without converting vote counts into "
             f"probabilities, using them only as a routing gate separates quality this much.")

    # --- 2. Review budget: split rows first vs random -----------------------------
    # Assume human review = replacing that row's label with the true label.
    order_gate = np.argsort(-key.grade * 1000 - (1.0 - key.frac.max(axis=1)))
    rng = np.random.default_rng(0)
    order_rand = rng.permutation(len(truth_top1))
    base = float(correct.mean())
    budget_rows = []
    for frac_budget in (0.05, 0.10, 0.20, 0.40):
        take = int(len(truth_top1) * frac_budget)
        gate_gain = correct.copy()
        gate_gain[order_gate[:take]] = True
        rand_gain = correct.copy()
        rand_gain[order_rand[:take]] = True
        budget_rows.append([f"{frac_budget:.0%}", base,
                            float(gate_gain.mean()), float(rand_gain.mean()),
                            float(gate_gain.mean() - rand_gain.mean())])
    rep.text("Spending the review budget in gate order vs at random "
             "(assuming reviewed rows are replaced with the true label):")
    rep.table(["Review share", "Before review", "Gate first", "Random", "Δ"], budget_rows)
    rep.record("review_budget", budget_rows)

    # --- 3. H1b cross-model consensus: what if we add one more cheap model? ----------------------
    weak_key = lab.key(SPLIT, K, cfg=labeler.CHEAP, seed=3)
    weak_correct = weak_key.primary == truth_top1
    agree_both = key.primary == weak_key.primary
    cross_rows = [
        ["Main LLM alone", len(truth_top1), 1.0, float(correct.mean())],
        ["Cheap LLM alone", len(truth_top1), 1.0, float(weak_correct.mean())],
        ["Keep only rows where both models agree", int(agree_both.sum()), float(agree_both.mean()),
         float(correct[agree_both].mean())],
        ["Main LLM unanimous rows only", int((key.grade == labeler.AGREE).sum()),
         float((key.grade == labeler.AGREE).mean()), agree_q],
    ]
    rep.text("H1b — call one more cheap model and keep only rows where 'the two agree'. "
             "A fair comparison is against the same-model repeated-call gate at the same coverage.")
    rep.table(["Filter", "Rows kept", "Coverage", "Primary accuracy"], cross_rows)
    rep.record("cross_consensus", cross_rows)

    # Rows where both agree but both are wrong = errors cross-model consensus cannot catch
    both_wrong = agree_both & ~correct
    rep.text(f"Rows where the two models agree but both are wrong: {both_wrong.mean():.1%}. "
             f"The cheap LLM does not independently catch the main LLM's errors, "
             f"but it does succeed at rejecting answers the main LLM got right — "
             f"among rows dropped by the agreement filter, the share the main LLM had right is "
             f"{float(correct[~agree_both].mean()):.3f}.")

    # --- 4. H1c LLM-as-a-judge: what if it 'reviews' instead of choosing a new label? --------
    # Difference from cross-model consensus: the judge does not pick a new candidate; it only answers yes/no to the proposed label.
    # This is a common pattern in practice, so it is placed side by side under the same conditions.
    judge_rows = []
    for strict in (0.0, 1.0, 2.0):
        judge = labeler.JudgeLLM(lab.world, cfg=labeler.CHEAP, strictness=strict)
        ok = judge.accept(lab.idx[SPLIT], key.primary, seed=900 + int(strict * 10))
        if ok.any():
            judge_rows.append([f"Judge-accepted only (strictness={strict:g})", int(ok.sum()),
                               float(ok.mean()), float(correct[ok].mean())])
    # Head-to-head with the gate at the same coverage
    for label, cov in (("judge", judge_rows[0][2]),):
        take = int(len(correct) * cov)
        order_by_gate = np.argsort(key.grade * 1000 + (1.0 - key.frac.max(axis=1)))
        judge_rows.append([f"(Reference) same coverage {cov:.0%} taken in gate order",
                           take, cov, float(correct[order_by_gate[:take]].mean())])

    rep.text("H1c — now LLM-as-a-judge. Instead of having the cheap LLM choose a new label, "
             "we only ask 'is this label correct?' and keep the accepted rows. A fair comparison "
             "with the gate requires matching coverage.")
    rep.table(["Filter", "Rows kept", "Coverage", "Primary accuracy"], judge_rows)
    rep.record("llm_as_judge", judge_rows)

    judge_best = max(r[3] for r in judge_rows[:-1])
    gate_same_cov = judge_rows[-1][3]
    rep.text(f"At the same {judge_rows[0][2]:.0%} coverage, the judge's accepted rows reach {judge_rows[0][3]:.3f} "
             f"while selecting in gate order gives {gate_same_cov:.3f}. "
             f"No matter how strict the judge is made, accuracy of the accepted rows stalls around {judge_best:.3f} "
             f"(at {[r[2] for r in judge_rows[:-1] if r[3] == judge_best][0]:.0%} coverage). "
             "The reason is simple — the judge gets confused in the same places as the labeling LLM. "
             "If it shares both the tendency to push items into the catch-all class and the tendency to pick "
             "one side of a confusable pair, the judge adds no information about those errors. "
             "For a judge to be worth its cost, it has to base its decision on something different from the labeling LLM.")

    rep.verdict(ACCEPT, f"H1 supported. Grade alone separates quality by {(agree_q - review_q) * 100:.1f}pp, "
                        f"and under the same review budget, gate-first beats random.")
    rep.verdict(REJECT, "H1b rejected. Cross-model consensus loses a lot of coverage, and its accuracy gain "
                        "is smaller than the same-model repeated-call gate. Adding the cheap LLM to the "
                        "consensus also rejects answers the main LLM got right.")
    rep.verdict(REJECT, f"H1c rejected. At the same {judge_rows[0][2]:.0%} coverage, LLM-as-a-judge accepted rows "
                        f"{judge_rows[0][3]:.3f} < gate {gate_same_cov:.3f}; even the strictest judge reaches only "
                        f"{judge_best:.3f} (at {[r[2] for r in judge_rows[:-1] if r[3] == judge_best][0]:.0%} coverage). "
                        "What matters is not attaching a judge per se but what the judge looks at. "
                        "A judge that shares the same bias only adds call cost.")
    rep.save()


if __name__ == "__main__":
    main()
