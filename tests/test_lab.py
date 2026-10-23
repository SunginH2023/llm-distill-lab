"""Things that must hold before any experiment can make sense.

These tests do not measure performance. They only check that the experimental design does not undermine itself.
The most important ones are the `test_oracle_is_never_used_for_training` family —
if the oracle leaks into the training path of a synthetic experiment, every result becomes meaningless.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401

import numpy as np

from lab import labeler, metrics, pipeline, signals, small_model
from lab.world import WorldConfig, make_world


def test_world_is_deterministic():
    a, b = make_world(WorldConfig(n=2000)), make_world(WorldConfig(n=2000))
    assert np.array_equal(a.truth, b.truth)
    assert np.allclose(a.feats, b.feats)


def test_truth_is_bayes_optimal_top1():
    w = make_world(WorldConfig(n=5000))
    # The posterior top-1 must always be inside the true label set
    # (argmax over the set follows index order, so it must not be used for this comparison — this fooled us once)
    top1 = np.argmax(w.post, axis=1)
    assert w.truth[np.arange(w.n), top1].all()
    assert w.truth.sum(axis=1).min() >= 1
    assert w.truth.sum(axis=1).max() <= 2


def test_multi_label_rate_is_realistic():
    w = make_world(WorldConfig(n=20000))
    rate = (w.truth.sum(axis=1) >= 2).mean()
    assert 0.05 < rate < 0.20, rate


def test_more_votes_improve_primary_label():
    """More votes must monotonically improve label 'selection' (the count is a separate matter — e4/e5)."""
    lab = pipeline.build(WorldConfig(n=20000))
    truth_top1 = lab.truth_top1("train")
    acc = [float((lab.key("train", k).primary == truth_top1).mean()) for k in (1, 3, 5, 7)]
    assert acc == sorted(acc), acc


def test_gate_separates_quality():
    lab = pipeline.build(WorldConfig(n=20000))
    key = lab.key("train", 5)
    ok = key.primary == lab.truth_top1("train")
    q_agree = ok[key.grade == labeler.AGREE].mean()
    q_review = ok[key.grade == labeler.REVIEW].mean()
    assert q_agree - q_review > 0.3, (q_agree, q_review)


def test_weak_axis_is_independent_of_teacher():
    """The outside signal must be derived directly from the truth, without passing through the labeling LLM."""
    lab = pipeline.build(WorldConfig(n=20000))
    ax = lab.axis_eval
    truth_group = ax.group_of[lab.world.primary]
    agree = (ax.observed == truth_group).mean()
    assert abs(agree - (1 - ax.noise * (1 - 1 / ax.n_groups))) < 0.02


def test_axes_for_fixing_and_scoring_are_disjoint():
    """Evaluating with the axis used for refinement is also circular. The two axes must differ."""
    lab = pipeline.build(WorldConfig(n=2000))
    assert not np.array_equal(lab.axis_fix.group_of, lab.axis_eval.group_of)
    assert not np.array_equal(lab.axis_fix.observed, lab.axis_eval.observed)


def test_tiebreak_only_touches_ambiguous_rows():
    lab = pipeline.build(WorldConfig(n=20000))
    key = lab.key("train", 5)
    only = key.grade != labeler.AGREE
    fixed, n_swap = signals.tiebreak(key, lab.axis("fix", "train"), only)
    changed = (fixed.primary != key.primary)
    assert n_swap > 0
    assert not changed[~only].any()


def test_student_learns_and_beats_chance():
    lab = pipeline.build(WorldConfig(n=20000))
    key = lab.key("train", 5)
    proba = pipeline.train_small_model(lab, labeler.set_targets(key))
    acc = (np.argmax(proba, axis=1) == lab.truth_top1("test")).mean()
    assert acc > 0.6, acc


def test_metrics_edge_cases():
    gold = np.array([[True, False], [True, True]])
    assert metrics.exact_match(gold, gold).all()
    assert metrics.micro_f1(gold, gold) == 1.0
    assert metrics.mcnemar(np.ones(10, bool), np.ones(10, bool))["p"] == 1.0
    assert metrics.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_decode_never_drops_top1():
    proba = np.array([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05]])
    for t in (0.1, 0.5, 0.9):
        pred = small_model.decode(proba, t)
        assert pred[np.arange(2), np.argmax(proba, axis=1)].all()
        assert pred.sum(axis=1).max() <= 2
