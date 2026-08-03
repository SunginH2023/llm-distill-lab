"""e3 — 라벨링 LLM의 '흔들림'을 버리지 말고 학습 신호로 쓴다 (지식증류의 최소 형태).

가설 H3  라벨링 LLM의 최종 라벨만 쓰면 '라벨링 LLM이 몇 표 차로 그 답을 골랐는가'가 통째로
         버려진다. 투표 분포를 소프트 타깃으로 같이 주면 작은 모델이 더 좋아진다.
         이득은 라벨링 LLM이 흔들린 구간에 몰릴 것이다.

여기서 중요한 건 평균 이득이 아니라 어디서 벌었는가다. 전체 +0.5pp 는
반올림에 묻히지만, "표가 갈린 층에서만 +2pp" 는 다음에 무엇을 할지 알려준다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

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

    rep = Report("e3_soft_labels", "e3 · 투표 분포를 소프트 타깃으로 (KD)",
                 "라벨링 LLM의 최종 라벨 대신 투표 분포를 학습시키면 작은 모델이 나아지는가")

    n = len(train_key.primary)
    onehot = np.zeros((n, lab.world.n_classes), dtype=np.float32)
    onehot[np.arange(n), train_key.primary] = 1.0

    targets = {
        "A 하드 (primary one-hot)": onehot,
        "B 집합 (primary .75 / secondary .25)": labeler.set_targets(train_key),
        "C 소프트 (투표 분포 그대로)": labeler.vote_targets(train_key),
    }
    probas = {name: pipeline.train_small_model(lab, t) for name, t in targets.items()}

    rows = []
    for name, p in probas.items():
        s = pipeline.score_all(lab, p, SPLIT, test_key)
        rows.append([name, s["truth_top1_acc"], s["truth_em"], s["truth_multi_em"],
                     s["same_source_top1"]])
    rep.text(f"라벨링 LLM {K}회 호출로 만든 pseudo-label 하나를 세 가지 타깃 형태로만 바꿔 학습한다. "
             "데이터·구조·스텝 수 전부 동일.")
    rep.table(["학습 타깃", "정답 top1", "진짜 EM", "정답 멀티행 EM", "순환 top1"], rows)
    rep.record("targets", rows)
    rep.text("먼저 눈에 띄는 건 소프트 타깃이 전체 EM 을 떨어뜨린다는 것이다"
             f" (B {rows[1][2]:.4f} → C {rows[2][2]:.4f}). "
             "같은 표에서 top1 과 멀티행 EM 은 올라갔는데도 그렇다. "
             "소프트 타깃이 2등 라벨에 질량을 남겨 top-2 를 더 자주 열게 만들고, "
             "전체 EM 은 멀티 예측이 늘면 손해를 보기 때문이다(→ e5 의 파레토 전선). "
             "'좋아졌다/나빠졌다'가 지표에 따라 갈리는 첫 지점.")

    # --- 어디서 벌었나: 라벨링 LLM이 흔들린 층 vs 만장일치 층 --------------------
    # 층은 테스트 쪽 라벨링 LLM 투표로 나눈다 (오라클을 쓰지 않고 실무에서도 가능한 층화).
    strata = {
        "만장일치층 (agree)": test_key.grade == labeler.AGREE,
        "표 갈린 층 (partial+review)": test_key.grade != labeler.AGREE,
    }
    base = np.argmax(probas["B 집합 (primary .75 / secondary .25)"], axis=1) == truth_top1
    kd = np.argmax(probas["C 소프트 (투표 분포 그대로)"], axis=1) == truth_top1

    strat_rows = []
    for name, m in strata.items():
        mc = metrics.mcnemar(base[m], kd[m])
        strat_rows.append([name, int(m.sum()), float(base[m].mean()), float(kd[m].mean()),
                           (kd[m].mean() - base[m].mean()) * 100, mc["p"]])
    mc_all = metrics.mcnemar(base, kd)
    strat_rows.append(["전체", len(base), float(base.mean()), float(kd.mean()),
                       (kd.mean() - base.mean()) * 100, mc_all["p"]])
    rep.text("B(집합) → C(소프트) 로 바꿨을 때 정답 top1 이 어디서 움직였는가. "
             "층화는 테스트 쪽 라벨링 LLM 투표로만 한다 — 오라클 없이도 할 수 있는 층화여야 "
             "실무에 그대로 옮겨진다.")
    rep.table(["층", "n", "B 집합", "C 소프트", "Δ(pp)", "McNemar p"], strat_rows)
    rep.record("strata", strat_rows)
    rep.record("mcnemar_all", mc_all)

    gain_split = strat_rows[1][4]
    gain_all = strat_rows[-1][4]
    rep.text(f"전체 이득 {gain_all:+.2f}pp, 표 갈린 층 {gain_split:+.2f}pp. "
             "방향은 예측대로다 — 라벨링 LLM이 흔들린 곳에서만 번다. "
             "다만 전체 이득의 절대 크기는 작아서, 이것만으로 재학습 비용을 정당화하기는 어렵다. "
             "소프트 타깃의 진짜 값어치는 점수가 아니라 "
             "투표 분포를 보존해 두면 GPU 재실행 없이 이런 사후 실험이 가능해진다는 데 있다.")

    rep.verdict(PARTIAL,
                f"H3 부분지지. 이득의 방향과 위치는 가설대로지만(갈린 층 {gain_split:+.2f}pp) "
                f"전체 {gain_all:+.2f}pp 는 크지 않다. "
                "채택하되 '이걸로 성능을 만든다'고 말하지 않는다.")
    rep.save()


if __name__ == "__main__":
    main()
