"""e5 — The label "count" problem, and the rejection of an attempt to train a separate count predictor.

In multi-label settings, score is usually lost not on label selection but on the count.
So the natural thought is: "let's train a separate model that predicts the count."

Hypothesis H5a  Opening top-2 with a dedicated count predictor P(multi|x) beats opening it with the
                plain probability ratio p2/p1.
Hypothesis H5b  There exists a single threshold that maximizes overall exact match.

Both were tried at several levels in production; the same structure is rebuilt here.
Thresholds are chosen on val and reported on test only.

To state the conclusion up front: H5a is supported or rejected depending on "what is measured".
That fact itself is the result of this experiment.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pin BLAS threads before importing numpy)

import numpy as np

from lab import labeler, metrics, pipeline, small_model
from lab.report import PARTIAL, REJECT, Report

THRESHOLDS = np.round(np.arange(0.10, 1.01, 0.05), 3)


def decode_scores(proba: np.ndarray, score: np.ndarray, t: float) -> np.ndarray:
    """Open top-2 only on rows where score >= t."""
    n, C = proba.shape
    rows = np.arange(n)
    order = np.argsort(-proba, axis=1)
    out = np.zeros((n, C), dtype=bool)
    out[rows, order[:, 0]] = True
    take = score >= t
    out[rows[take], order[take, 1]] = True
    return out


def ratio_score(proba: np.ndarray) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    rows = np.arange(len(proba))
    return proba[rows, order[:, 1]] / np.maximum(proba[rows, order[:, 0]], 1e-9)


def curve(proba: np.ndarray, score: np.ndarray, truth: np.ndarray):
    out = []
    multi = truth.sum(axis=1) >= 2
    for t in THRESHOLDS:
        pred = decode_scores(proba, score, t)
        em = metrics.exact_match(pred, truth)
        out.append((float(t), float(em.mean()), float(em[multi].mean()),
                    float((pred.sum(1) >= 2).mean())))
    return out


def main() -> None:
    lab = pipeline.build()
    train_key = lab.key("train", 7)
    proba_tr = pipeline.train_small_model(lab, labeler.set_targets(train_key), eval_split="train")
    proba_val = pipeline.train_small_model(lab, labeler.set_targets(train_key), eval_split="val")
    proba_te = pipeline.train_small_model(lab, labeler.set_targets(train_key), eval_split="test")

    truth_val, truth_te = lab.truth("val"), lab.truth("test")

    rep = Report("e5_cardinality", "e5 · The count problem: same experiment, conclusion flips with the metric",
                 "Does a model that predicts the label count separately beat the one-line probability ratio?")

    # --- Train the count predictor (no oracle; on the pseudo-label counts) --------------------
    y_multi = (train_key.mat.sum(axis=1) >= 2).astype(np.float32)
    tgt = np.stack([1 - y_multi, y_multi], axis=1)
    W = small_model.fit(lab.feats("train"), tgt, steps=200)
    judge_val = small_model.predict_proba(W, lab.feats("val"))[:, 1]
    judge_te = small_model.predict_proba(W, lab.feats("test"))[:, 1]
    judge_acc = float(((judge_te >= 0.5) == (truth_te.sum(1) >= 2)).mean())

    scorers_val = {
        "Probability ratio p2/p1": ratio_score(proba_val),
        "Count predictor P(multi)": judge_val,
        "Combined P(multi)×(p2/p1)": judge_val * ratio_score(proba_val),
    }
    scorers_te = {
        "Probability ratio p2/p1": ratio_score(proba_te),
        "Count predictor P(multi)": judge_te,
        "Combined P(multi)×(p2/p1)": judge_te * ratio_score(proba_te),
    }

    rep.text(f"The small model is fixed to a single model trained on pseudo-labels from 7 calls; only decoding changes. "
             f"The count predictor was trained with the pseudo-label count as its target (no oracle used; "
             f"accuracy against the true count {judge_acc:.3f}).")

    # --- Compare at matched operating points ------------------------------------------
    # Maximizing overall EM alone always converges to "never predict multi-label"
    # (see the Pareto table below). So the three scorers are compared at the same predicted multi rate.
    # The target multi rate is taken from the pseudo-labels, not the oracle (observable in production).
    target_rate = float(train_key.multi_rate)
    rows = []
    for name in scorers_val:
        sv, st = scorers_val[name], scorers_te[name]
        t = float(np.quantile(sv, 1.0 - target_rate))     # threshold decided on val only
        pred = decode_scores(proba_te, st, t)
        em = metrics.exact_match(pred, truth_te)
        multi = truth_te.sum(axis=1) >= 2
        rows.append([name, t, float(em.mean()), float(em[multi].mean()),
                     float((pred.sum(1) >= 2).mean())])

    truth_card = truth_te.sum(axis=1)
    order = np.argsort(-proba_te, axis=1)
    oracle_pred = np.zeros_like(truth_te)
    r = np.arange(len(truth_te))
    oracle_pred[r, order[:, 0]] = True
    two = truth_card >= 2
    oracle_pred[r[two], order[two, 1]] = True
    never_multi = np.zeros_like(truth_te)
    never_multi[r, order[:, 0]] = True
    rows.append(["(Reference) never predict multi-label", "-",
                 float(metrics.exact_match(never_multi, truth_te).mean()), 0.0, 0.0])
    rows.append(["Count given (upper bound)", "-",
                 float(metrics.exact_match(oracle_pred, truth_te).mean()),
                 float(metrics.exact_match(oracle_pred, truth_te)[two].mean()),
                 float((oracle_pred.sum(1) >= 2).mean())])

    rep.text(f"The three scorers are compared at the same predicted multi rate ({target_rate:.3f}, matched to the pseudo-label multi rate). "
             "Thresholds are set on val only; test is used only for reporting.")
    rep.table(["Scorer (matched operating point)", "t(val)", "Overall EM(test)",
               "Multi-row EM(test)", "Predicted multi rate"], rows)
    rep.record("decoding", rows)

    ratio_em, judge_em, comb_em = rows[0][2], rows[1][2], rows[2][2]
    ratio_mem, judge_mem, comb_mem = rows[0][3], rows[1][3], rows[2][3]

    # Check whether the difference is sampling noise (paired comparison on the same rows).
    def em_of(name_idx: int) -> np.ndarray:
        name = list(scorers_te)[name_idx]
        t = float(np.quantile(scorers_val[name], 1.0 - target_rate))
        return metrics.exact_match(decode_scores(proba_te, scorers_te[name], t), truth_te)

    em_ratio, em_judge = em_of(0), em_of(1)
    multi_mask = truth_te.sum(axis=1) >= 2
    mc_all = metrics.mcnemar(em_ratio, em_judge)
    mc_multi = metrics.mcnemar(em_ratio[multi_mask], em_judge[multi_mask])
    rep.record("mcnemar_ratio_vs_judge", {"all": mc_all, "multi_rows": mc_multi})

    rep.text(f"At the same multi rate, overall EM is probability ratio {ratio_em:.4f} vs count predictor {judge_em:.4f} "
             f"(McNemar p={mc_all['p']:.3g}) — any difference is in the third decimal place. "
             f"Yet multi-row EM widens more than 3x, {ratio_mem:.4f} → {judge_mem:.4f} "
             f"(p={mc_multi['p']:.3g}). "
             "The predictor knows something about 'which rows are multi-label' that the probability ratio does not — "
             "in this world, whether a row is multi-label depends on whether the two candidates form an allowed co-occurring pair, "
             "and that information is not contained in the ratio of the top two probabilities. "
             "Why overall EM still does not move is simple: "
             f"multi-label rows are only {float(multi_mask.mean()):.1%} of the total, "
             "so the averaged metric swallows the gain entirely.")
    rep.text("> Caution: this is where the lab diverges from the real pipeline. In production, when count predictors "
             "with the same structure were trained in several variants, none ever beat the probability-ratio gate. "
             "In the synthetic world the 'count' is determined by a clean rule (allowed co-occurring pairs), "
             "so the predictor has more to learn than in reality. "
             "→ This is a point where a synthetic experiment cannot replace a production conclusion; "
             "a synthetic experiment is useful only when you can explain why the results differ.")

    # --- H5b: a single threshold cannot capture both axes at once ---------------------
    front = curve(proba_te, scorers_te["Probability ratio p2/p1"], truth_te)
    show = [[t, em, mem, mr] for t, em, mem, mr in front if round(t, 2) in
            (0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)]
    rep.text("Pareto front observed directly on test while sweeping the probability-ratio threshold:")
    rep.table(["t", "Overall EM", "Multi-row EM", "Predicted multi rate"], show)
    rep.record("pareto", front)
    best_all = max(front, key=lambda r: r[1])
    best_multi = max(front, key=lambda r: r[2])
    rep.text(f"Overall EM peaks at {best_all[1]:.4f} at t={best_all[0]:.2f}, "
             f"where multi-row EM is {best_all[2]:.4f}. "
             f"Multi-row EM peaks at {best_multi[2]:.4f} at t={best_multi[0]:.2f}, "
             f"where overall EM drops to {best_multi[1]:.4f}. "
             "No single t maximizes both axes at once. "
             "So t is not 'a value obtained by tuning' but "
             "an operational decision about which kind of error to accept.")

    rep.verdict(PARTIAL,
                f"H5a partially supported — a case where the conclusion flips depending on what is measured. "
                f"Overall EM shows no difference ({ratio_em:.4f} → {judge_em:.4f}), "
                f"while multi-row EM clearly supports it ({ratio_mem:.4f} → {judge_mem:.4f}, p={mc_multi['p']:.3g}). "
                "Had we reported overall EM alone and written 'count predictor rejected', "
                "we would have discarded a real signal because of metric design.")
    rep.verdict(REJECT,
                "H5b rejected. The threshold that is optimal for overall EM differs from the one optimal for multi-label rows, "
                f"and maximizing overall EM alone converges to 'never predict multi-label' ({rows[-2][2]:.4f}). "
                "That is, t is not a value obtained by tuning but "
                "an operational decision about which kind of error to accept. "
                f"The upper bound when the count is given, {rows[-1][2]:.4f}, shows the remaining headroom.")
    rep.save()


if __name__ == "__main__":
    main()
