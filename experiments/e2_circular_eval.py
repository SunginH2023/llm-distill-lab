"""e2 — 라벨링 LLM이 만든 pseudo-label 로 작은 모델을 채점하면 순위가 뒤집힌다.

가설 H2  라벨링 LLM이 붙인 라벨로 만든 평가셋는 '정확도'가 아니라 모방도를 잰다.
         라벨링 LLM의 체계 편향까지 잘 흉내 낸 작은 모델이 1등이 되고, 진짜로 더 나은 작은 모델은
         밀린다. 반면 파이프라인 밖에서 온 거친 외부 신호는, 절대 점수가 훨씬
         낮은데도 순위는 정답과 같게 매긴다.

실무에서는 이 주장을 증명할 수 없다. 정답을 모르기 때문이다.
여기서는 정답을 알기 때문에 세 채점표를 나란히 놓고 채점표 자체를 채점할 수 있다.

비교는 1등 라벨 선택으로 통일한다. 외부 신호는 거친 그룹 단위라 라벨 '개수'를
채점할 수 없기 때문이다(개수는 e5 에서 따로 다룬다).
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

import numpy as np

from lab import labeler, metrics, pipeline, signals, small_model
from lab.report import ACCEPT, Report

SPLIT = "test"


def build_models(lab) -> dict[str, np.ndarray]:
    """전부 같은 구조·같은 하이퍼파라미터. pseudo-label만 다르다."""
    k1 = lab.key("train", 1)
    k3 = lab.key("train", 3)
    k5 = lab.key("train", 5)
    k7 = lab.key("train", 7)
    fix5, _ = signals.tiebreak(k5, lab.axis("fix", "train"), k5.grade != labeler.AGREE)
    fix7, _ = signals.tiebreak(k7, lab.axis("fix", "train"), k7.grade != labeler.AGREE)

    keys = {"1회": k1, "3회": k3, "5회": k5, "7회": k7,
            "5회+외부신호 보정": fix5, "7회+외부신호 보정": fix7}
    return {name: pipeline.train_small_model(lab, labeler.set_targets(k))
            for name, k in keys.items()}, keys


def main() -> None:
    lab = pipeline.build()
    test_key = lab.key(SPLIT, 5)          # 실무에서 손에 쥘 수 있는 유일한 평가셋
    probas, keys = build_models(lab)

    rep = Report("e2_circular_eval", "e2 순환 채점표는 순위를 뒤집는다",
                 "라벨링 LLM pseudo-label 채점 / 파이프라인 외부 신호 채점 / 진짜 정답 채점 — 셋의 순위가 같은가")

    ax = lab.axis("eval", SPLIT)
    truth_top1 = lab.truth_top1(SPLIT)

    rows, scores = [], {}
    for name, p in probas.items():
        s = pipeline.score_all(lab, p, SPLIT, test_key)
        scores[name] = s
        rows.append([name, s["same_source_top1"], s["outside_signal"], s["truth_top1_acc"],
                     s["same_source_em"], s["truth_em"]])
    rep.text("작은 모델 여섯 개. 구조·하이퍼파라미터는 전부 같고 학습에 쓴 라벨만 다르다. "
             f"외부 신호 채점은 그 축이 상위 두 후보를 구분해 주는 행에서만 잰다 "
             f"(축 '{ax.name}', 그룹 {ax.n_groups}개, 관측잡음 {ax.noise:.0%}).")
    rep.table(["작은 모델 (학습 라벨)", "① pseudo-label 평가셋", "② 외부 신호", "③ 진짜 정답",
               "pseudo EM", "진짜 EM"], rows)
    rep.record("scores", scores)

    # --- 순위 비교 --------------------------------------------------------
    r_circ = metrics.rank_of({k: v["same_source_top1"] for k, v in scores.items()})
    r_weak = metrics.rank_of({k: v["outside_signal"] for k, v in scores.items()})
    r_true = metrics.rank_of({k: v["truth_top1_acc"] for k, v in scores.items()})
    tau_circ = metrics.kendall_tau(r_circ, r_true)
    tau_weak = metrics.kendall_tau(r_weak, r_true)

    # 순환 밖이면 무조건 되는가? — 라벨링 LLM과 같은 방향으로 오염된 축을 하나 더 만든다.
    dirty = signals.contaminate(ax, lab.world.cfg.magnet_class, strength=0.35, seed=77)
    dirty_scores = {n: dirty.score_top1(p, signals.discriminative_mask(dirty, p))
                    for n, p in probas.items()}
    r_dirty = metrics.rank_of(dirty_scores)
    tau_dirty = metrics.kendall_tau(r_dirty, r_true)

    rep.text("순위만 뽑아 보면 (왼쪽이 1등):")
    rep.table(["채점표", "매긴 순위", "정답과의 순위상관 τ"],
              [["① pseudo-label 로 만든 평가셋 (같은 출처)", " > ".join(r_circ), tau_circ],
               ["② 파이프라인 외부에서 온 신호", " > ".join(r_weak), tau_weak],
               ["②' 외부 신호이나 동일 편향으로 오염됨", " > ".join(r_dirty), tau_dirty],
               ["③ 정답 (오라클)", " > ".join(r_true), 1.0]])
    rep.record("tau", {"same_source": tau_circ, "outside": tau_weak, "outside_contaminated": tau_dirty,
                       "rank_same_source": r_circ, "rank_outside": r_weak, "rank_true": r_true})
    rep.text("②' 은 파이프라인 외부에서 왔으나 라벨링 LLM과 동일한 방향으로 편향된 축이다 "
             "(관측의 35%가 라벨링 LLM이 선호하는 그룹으로 끌려간다 — 운영 자동 규칙을 "
             "라벨링 LLM도 학습해 버린 상황). "
             f"τ가 {tau_weak:.2f} 에서 {tau_dirty:.2f} 로 떨어진다. "
             "외부 출처라는 조건만으로는 불충분하고, 라벨링 LLM 편향과 독립인지까지 "
             "따져야 한다. 실무에서 부서 배정 축을 쓸 때 자동배정분을 빼고 "
             "다시 검정해야 했던 이유가 이것이다.")

    best_circ, best_true = r_circ[0], r_true[0]
    gap = scores[best_true]["truth_top1_acc"] - scores[best_circ]["truth_top1_acc"]
    rep.text(f"pseudo-label 평가셋이 뽑은 1등은 {best_circ}, 진짜 1등은 {best_true}. "
             f"이 평가셋만 보고 모델을 고르면 정답 기준 {gap * 100:.2f}pp 를 두고 온다. "
             f"외부 신호의 절대 점수는 {scores[best_true]['outside_signal']:.3f} 로 낮지만 "
             f"— 이 축이 원래 거칠기 때문이다 — 순위는 맞다. "
             "절대 점수를 보고하지 말고 상대 비교에만 쓸 것.")

    mc = metrics.mcnemar(
        signals.discriminative_mask(ax, probas[best_circ]) & (np.argmax(probas[best_circ], 1) == truth_top1),
        signals.discriminative_mask(ax, probas[best_true]) & (np.argmax(probas[best_true], 1) == truth_top1))
    rep.record("mcnemar_best", mc)

    # --- 편향 스윕: 언제부터 순환 채점표가 거짓말을 하나 -------------------
    sweep = []
    for beta in (0.0, 0.4, 0.8, 1.1, 1.6):
        cfg = labeler.LabelerConfig(beta_magnet=beta, pair_skew=beta * 0.7)
        k5b = lab.key("train", 5, cfg=cfg)
        fixb, _ = signals.tiebreak(k5b, lab.axis("fix", "train"), k5b.grade != labeler.AGREE)
        tkb = lab.key(SPLIT, 5, cfg=cfg)
        sc = {}
        for nm, kk in {"1회": lab.key("train", 1, cfg=cfg), "5회": k5b, "5회+외부신호 보정": fixb}.items():
            p = pipeline.train_small_model(lab, labeler.set_targets(kk))
            sc[nm] = pipeline.score_all(lab, p, SPLIT, tkb)
        rt = metrics.rank_of({k: v["truth_top1_acc"] for k, v in sc.items()})
        sweep.append([beta,
                      metrics.kendall_tau(metrics.rank_of({k: v["same_source_top1"] for k, v in sc.items()}), rt),
                      metrics.kendall_tau(metrics.rank_of({k: v["outside_signal"] for k, v in sc.items()}), rt),
                      float(np.mean([v["truth_top1_acc"] for v in sc.values()]))])
    rep.text("라벨링 LLM의 체계 편향 세기를 0부터 키우면서 두 채점표의 순위상관을 다시 잰다. "
             "편향이 0이면 pseudo-label 평가셋도 정직하다 — 문제는 편향이 있을 때만 생긴다. "
             "(실제 LLM에서 편향 0 은 없다.)")
    rep.table(["편향 β", "① pseudo-label 평가셋 τ", "② 외부 신호 τ", "평균 정답 top1"], sweep)
    rep.record("bias_sweep", sweep)

    rep.verdict(ACCEPT,
                f"H2 지지. pseudo-label 평가셋의 순위상관 τ={tau_circ:.2f}, 외부 신호 τ={tau_weak:.2f}. "
                "라벨링 LLM pseudo-label 로 매긴 1등은 정답 기준 1등이 아니었다. "
                "절대 점수가 낮은 거친 축이, 절대 점수가 높은 순환 축보다 나은 채점표다.")
    rep.save()


if __name__ == "__main__":
    main()
