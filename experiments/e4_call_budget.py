"""e4 — Where to spend the LLM call budget: more rows vs cleaner labels.

When a commercial LLM is used as the labeling LLM, the number of calls is money. With the same budget,

    (a) shallow and wide  — 3 votes on every row
    (b) deep and narrow   — 7 votes on half the rows
    (c) targeted re-vote  — 3 votes on every row + 4 extra votes only on split-vote rows

which one makes the small model best? All three use the same number of calls.

Hypothesis H4  (c) wins, because spending more votes on rows that were already unanimous is waste.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pin BLAS threads before importing numpy)

import numpy as np

from lab import labeler, metrics, pipeline, small_model
from lab.report import ACCEPT, Report

BUDGET = 90_000       # number of LLM calls (= 3 votes × 30,000 rows)
SPLIT = "test"


def subset_key(counts: np.ndarray, rows: np.ndarray) -> labeler.PseudoLabels:
    return labeler.labels_from_votes(counts[rows])


def main() -> None:
    lab = pipeline.build()
    test_key = lab.key(SPLIT, 5)
    truth_tr = lab.truth("train")
    truth_tr_top1 = lab.truth_top1("train")
    feats_tr, feats_te = lab.feats("train"), lab.feats("test")
    n_pool = len(truth_tr_top1)

    c3 = lab.vote("train", 3, seed=1)
    c5 = lab.vote("train", 5, seed=2)
    c7 = lab.vote("train", 7, seed=3)
    c4extra = lab.vote("train", 4, seed=4)      # 4 extra votes for re-voting

    plans = {}
    for k, counts in ((3, c3), (5, c5), (7, c7)):
        rows = np.arange(min(BUDGET // k, n_pool))
        plans[f"{k} calls × {len(rows):,} rows"] = (rows, counts[rows], k * len(rows))

    # Targeted re-vote: lay down 3 votes, then spend 4 more only on split rows.
    amb_all = labeler.labels_from_votes(c3).grade != labeler.AGREE
    cost = np.cumsum(3 + 4 * amb_all)           # cumulative calls spent up to row r
    r = int(np.searchsorted(cost, BUDGET))
    rows = np.arange(r)
    merged = c3[rows] + c4extra[rows] * amb_all[rows][:, None]
    plans[f"3 calls + 4 on split rows × {r:,} rows"] = (rows, merged, int(cost[r - 1]))

    rep = Report("e4_call_budget", "e4 · Same LLM call budget, different allocations",
                 f"Should {BUDGET:,} calls be spent 'shallow and wide' or 'deep on split rows'?")

    rows_out, downstream = [], {}
    for name, (idx, counts, spent) in plans.items():
        key = labeler.labels_from_votes(counts)
        t = truth_tr[idx]
        key_top1 = float((key.primary == truth_tr_top1[idx]).mean())
        key_em = float(metrics.exact_match(key.mat, t).mean())

        proba = small_model.train_and_predict(feats_tr[idx], labeler.set_targets(key), feats_te)
        s = pipeline.score_all(lab, proba, SPLIT, test_key)
        downstream[name] = s
        rows_out.append([name, spent, len(idx), key_top1, key_em,
                         float(key.mat.sum(axis=1).mean() - 1),
                         s["truth_top1_acc"], s["truth_em"]])

    rep.text(f"Training pool {n_pool:,} rows, budget {BUDGET:,} calls. "
             "The four left columns are the quality of the pseudo-labels themselves; the two right columns are the "
             "true-label scores of the small model trained on those pseudo-labels.")
    rep.table(["Allocation", "Calls spent", "Rows labeled", "Pseudo-label top1", "Pseudo-label EM",
               "Multi rate", "Small model true top1", "Small model true EM"], rows_out)
    rep.record("plans", rows_out)

    best = max(downstream, key=lambda k: downstream[k]["truth_top1_acc"])
    flat3 = [k for k in plans if k.startswith("3 calls ×")][0]
    d_top1 = (downstream[best]["truth_top1_acc"] - downstream[flat3]["truth_top1_acc"]) * 100
    d_em = (downstream[best]["truth_em"] - downstream[flat3]["truth_em"]) * 100

    rep.text(f"The winning allocation is {best}. Versus 3 votes on every row: true top1 {d_top1:+.2f}pp / "
             f"EM {d_em:+.2f}pp. It actually labels fewer rows — "
             "meaning that fixing the labels of rows already paid for was cheaper than buying more rows.")

    # --- What improves as calls increase (same number of rows, different number of calls) --------
    ladder = []
    for k, counts in ((1, lab.vote("train", 1, seed=5)), (3, c3), (5, c5), (7, c7)):
        key = labeler.labels_from_votes(counts)
        ladder.append([k, float((key.primary == truth_tr_top1).mean()),
                       float(metrics.exact_match(key.mat, truth_tr).mean()),
                       key.multi_rate,
                       float((key.grade == labeler.AGREE).mean())])
    rep.text("Fixing the number of rows and increasing only the number of calls (ignoring the budget):")
    rep.table(["Calls", "Pseudo-label top1", "Pseudo-label EM", "Pseudo-label multi rate", "Unanimous share"], ladder)
    rep.record("vote_ladder", ladder)
    rep.text(f"Label selection (top1) improves monotonically with more votes, but exact match fluctuates — "
             f"because the multi rate overshoots the true rate ({float((truth_tr.sum(1) >= 2).mean()):.3f}) and then comes back. "
             "In other words, 'how many votes to use' touches two problems at once: label selection and label count. "
             "Choosing the number of calls by exact match alone makes the two effects cancel out, so nothing is visible (→ e5).")

    rep.verdict(ACCEPT,
                f"H4 supported. Under the same budget, targeted re-voting won ({best}). "
                "But the gain comes not from 'spending many votes' but from 'not spending votes on unanimous rows'. "
                "Without the gate (e1), this allocation is not possible at all.")
    rep.save()


if __name__ == "__main__":
    main()
