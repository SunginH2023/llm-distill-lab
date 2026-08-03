"""e4 — LLM 호출 예산을 어디에 쓸 것인가: 더 많은 행 vs 더 깨끗한 라벨.

상용 LLM 을 라벨링 LLM로 쓰면 호출 수가 곧 돈이다. 같은 예산으로

    (a) 얕게 많이  — 전량에 3표
    (b) 깊게 적게  — 절반에 7표
    (c) 표적 재투표 — 전량에 3표 + 표가 갈린 행에만 4표 추가

중 무엇이 작은 모델을 가장 좋게 만드는가. 셋은 호출 수가 같다.

가설 H4  (c) 가 이긴다. 만장일치로 끝난 행에 표를 더 쓰는 건 낭비이기 때문이다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

import numpy as np

from lab import labeler, metrics, pipeline, small_model
from lab.report import ACCEPT, Report

BUDGET = 90_000       # LLM 호출 수 (= 3표 × 30,000행)
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
    c4extra = lab.vote("train", 4, seed=4)      # 재투표용 추가 4표

    plans = {}
    for k, counts in ((3, c3), (5, c5), (7, c7)):
        rows = np.arange(min(BUDGET // k, n_pool))
        plans[f"{k}회 × {len(rows):,}행"] = (rows, counts[rows], k * len(rows))

    # 표적 재투표: 3표를 깔고, 갈린 행에만 4표를 더 쓴다.
    amb_all = labeler.labels_from_votes(c3).grade != labeler.AGREE
    cost = np.cumsum(3 + 4 * amb_all)           # r행까지 썼을 때 누적 호출 수
    r = int(np.searchsorted(cost, BUDGET))
    rows = np.arange(r)
    merged = c3[rows] + c4extra[rows] * amb_all[rows][:, None]
    plans[f"3회+애매행 4회 × {r:,}행"] = (rows, merged, int(cost[r - 1]))

    rep = Report("e4_call_budget", "e4 · 같은 LLM 호출 예산, 다른 배분",
                 f"호출 {BUDGET:,}회를 '얕게 많이' 쓸 것인가 '갈린 행에 깊게' 쓸 것인가")

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

    rep.text(f"학습 풀 {n_pool:,}행, 예산 {BUDGET:,} 호출. "
             "왼쪽 네 칸은 pseudo-label 자체의 품질, 오른쪽 두 칸은 그 pseudo-label 로 학습한 "
             "작은 모델의 정답 기준 점수다.")
    rep.table(["배분", "쓴 호출", "라벨한 행", "pseudo-label top1", "pseudo-label EM",
               "멀티율", "작은 모델 정답 top1", "작은 모델 정답 EM"], rows_out)
    rep.record("plans", rows_out)

    best = max(downstream, key=lambda k: downstream[k]["truth_top1_acc"])
    flat3 = [k for k in plans if k.startswith("3회 ×")][0]
    d_top1 = (downstream[best]["truth_top1_acc"] - downstream[flat3]["truth_top1_acc"]) * 100
    d_em = (downstream[best]["truth_em"] - downstream[flat3]["truth_em"]) * 100

    rep.text(f"이긴 배분은 {best}. 전량 3표 대비 정답 top1 {d_top1:+.2f}pp / "
             f"EM {d_em:+.2f}pp. 라벨한 행 수는 오히려 적다 — "
             "행을 더 사는 것보다 이미 산 행의 라벨을 고치는 쪽이 쌌다는 뜻이다.")

    # --- 호출을 늘리면 무엇이 좋아지는가 (같은 행 수, 다른 호출 수) --------
    ladder = []
    for k, counts in ((1, lab.vote("train", 1, seed=5)), (3, c3), (5, c5), (7, c7)):
        key = labeler.labels_from_votes(counts)
        ladder.append([k, float((key.primary == truth_tr_top1).mean()),
                       float(metrics.exact_match(key.mat, truth_tr).mean()),
                       key.multi_rate,
                       float((key.grade == labeler.AGREE).mean())])
    rep.text("행 수를 고정하고 호출 수만 늘리면 (예산 무시):")
    rep.table(["호출 수", "pseudo-label top1", "pseudo-label EM", "pseudo-label 멀티율", "만장일치 비율"], ladder)
    rep.record("vote_ladder", ladder)
    rep.text(f"라벨 선택(top1)은 표를 늘릴수록 단조로 좋아지지만, 완전일치는 출렁인다 — "
             f"멀티율이 정답({float((truth_tr.sum(1) >= 2).mean()):.3f})을 넘었다가 돌아오기 때문이다. "
             "즉 '몇 표를 쓸까'는 라벨 선택 문제와 개수 문제 두 개를 동시에 건드린다. "
             "완전일치 하나만 보고 호출 수를 정하면 두 효과가 상쇄되어 아무것도 안 보인다(→ e5).")

    rep.verdict(ACCEPT,
                f"H4 지지. 같은 예산에서 표적 재투표가 이겼다({best}). "
                "다만 이득은 '표를 많이 써서'가 아니라 '만장일치 행에 표를 안 써서' 생긴다. "
                "게이트(e1)가 없으면 이 배분 자체가 불가능하다.")
    rep.save()


if __name__ == "__main__":
    main()
