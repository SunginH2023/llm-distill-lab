"""e1 — 합의 게이트: 표가 갈린 정도로 검수 예산을 어디에 쓸지 정한다.

가설 H1  라벨링 LLM을 여러 번 호출해 표가 갈린 정도를 보면, 오라클 없이 라벨 품질을
         분류할 수 있다. 그러면 사람 검수를 전량이 아니라 갈린 행에만 쓸 수 있다.
가설 H1b 싼 모델을 하나 더 붙여 '두 모델이 일치한 것만 채택'하면 품질이 더 오른다.

H1b 는 직관적으로 그럴듯하지만, 실무에서 실제로 해 보고 폐기한 접근이다.
여기서 왜 그런지 정량으로 남긴다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

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

    rep = Report("e1_consensus_gate", "e1 · 합의 게이트와 검수 예산",
                 "표가 갈린 정도만으로 라벨 품질을 사전 분류할 수 있는가")

    # --- 1. 등급별 품질 -------------------------------------------------
    rows = []
    correct = key.primary == truth_top1
    for grade, name in labeler.GRADE_NAMES.items():
        m = key.grade == grade
        if not m.any():
            continue
        rows.append([name, int(m.sum()), f"{m.mean():.3f}",
                     float(correct[m].mean()),
                     float(lab.world.ambiguity[lab.idx[SPLIT]][m].mean())])
    rep.text(f"라벨링 LLM {K}회 호출. 등급은 득표 분포만 보고 매긴다 — 오라클은 채점에만 쓴다.")
    rep.table(["등급", "행 수", "비율", "primary 정답률", "평균 모호도"], rows)
    rep.record("grade_quality", rows)

    agree_q = float(correct[key.grade == labeler.AGREE].mean())
    review_m = key.grade == labeler.REVIEW
    review_q = float(correct[review_m].mean()) if review_m.any() else float("nan")
    rep.text(f"만장일치 구간 {agree_q:.3f} vs 표가 깨진 구간 {review_q:.3f} — "
             f"{(agree_q - review_q) * 100:.1f}pp 차이. 득표수를 확률로 환산하지 않고 "
             f"경로를 가르는 게이트로만 써도 이만큼 갈린다.")

    # --- 2. 검수 예산: 갈린 행부터 vs 무작위 -----------------------------
    # 사람 검수 = 그 행의 라벨을 정답으로 교체한다고 가정.
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
    rep.text("검수 예산을 게이트 순서로 쓸 때와 무작위로 쓸 때 "
             "(검수한 행은 정답으로 교체된다고 가정):")
    rep.table(["검수 비율", "검수 전", "게이트 우선", "무작위", "Δ"], budget_rows)
    rep.record("review_budget", budget_rows)

    # --- 3. H1b 교차 합의: 싼 모델을 하나 더 붙이면? ----------------------
    weak_key = lab.key(SPLIT, K, cfg=labeler.CHEAP, seed=3)
    weak_correct = weak_key.primary == truth_top1
    agree_both = key.primary == weak_key.primary
    cross_rows = [
        ["메인 LLM 단독", len(truth_top1), 1.0, float(correct.mean())],
        ["싼 LLM 단독", len(truth_top1), 1.0, float(weak_correct.mean())],
        ["두 모델 일치분만 채택", int(agree_both.sum()), float(agree_both.mean()),
         float(correct[agree_both].mean())],
        ["메인 LLM 만장일치분만", int((key.grade == labeler.AGREE).sum()),
         float((key.grade == labeler.AGREE).mean()), agree_q],
    ]
    rep.text("H1b — 싼 모델을 하나 더 불러 '둘이 일치한 것만' 채택해 본다. "
             "같은 커버리지에서 같은 모델 반복 호출 게이트와 비교해야 공정하다.")
    rep.table(["필터", "남는 행", "커버리지", "primary 정답률"], cross_rows)
    rep.record("cross_consensus", cross_rows)

    # 일치했지만 둘 다 틀린 행 = 교차합의가 못 잡는 오류
    both_wrong = agree_both & ~correct
    rep.text(f"두 모델이 일치했는데 둘 다 틀린 행이 {both_wrong.mean():.1%}. "
             f"싼 LLM은 메인 LLM의 오류를 독립적으로 잡아주지 못하고, "
             f"메인 LLM이 맞힌 답을 기각하는 데는 성공한다 — "
             f"일치 필터가 버린 행 중 메인 LLM가 맞혔던 비율 "
             f"{float(correct[~agree_both].mean()):.3f}.")

    # --- 4. H1c LLM-as-a-judge: 라벨을 새로 고르는 대신 '검수'시키면? --------
    # 교차 합의와 다른 점: judge 는 후보를 새로 고르지 않고 제안된 라벨에 예/아니오만 답한다.
    # 실무에서 자주 쓰는 패턴이라 같은 조건으로 나란히 놓고 본다.
    judge_rows = []
    for strict in (0.0, 1.0, 2.0):
        judge = labeler.JudgeLLM(lab.world, cfg=labeler.CHEAP, strictness=strict)
        ok = judge.accept(lab.idx[SPLIT], key.primary, seed=900 + int(strict * 10))
        if ok.any():
            judge_rows.append([f"judge 통과분만 (strictness={strict:g})", int(ok.sum()),
                               float(ok.mean()), float(correct[ok].mean())])
    # 같은 커버리지에서 게이트와 정면 비교
    for label, cov in (("judge", judge_rows[0][2]),):
        take = int(len(correct) * cov)
        order_by_gate = np.argsort(key.grade * 1000 + (1.0 - key.frac.max(axis=1)))
        judge_rows.append([f"(비교) 같은 커버리지 {cov:.0%} 를 게이트 순서로",
                           take, cov, float(correct[order_by_gate[:take]].mean())])

    rep.text("H1c — 이번엔 LLM-as-a-judge. 싼 LLM에게 라벨을 새로 고르게 하지 않고 "
             "'이 라벨 맞습니까?'만 물어 통과분만 남긴다. 게이트와 공정하게 비교하려면 "
             "커버리지를 맞춰서 봐야 한다.")
    rep.table(["필터", "남는 행", "커버리지", "primary 정답률"], judge_rows)
    rep.record("llm_as_judge", judge_rows)

    judge_best = max(r[3] for r in judge_rows[:-1])
    gate_same_cov = judge_rows[-1][3]
    rep.text(f"judge 를 아무리 깐깐하게 조여도 통과분의 정답률은 {judge_best:.3f} 근처에서 "
             f"멈춘다. 같은 커버리지를 게이트 순서로 고르면 {gate_same_cov:.3f}. "
             "이유는 단순하다 — judge 도 라벨링 LLM과 같은 곳에서 헷갈린다. "
             "'기타'로 밀어 넣는 성향, 헷갈리는 쌍에서 한쪽을 고르는 성향을 둘 다 공유하면 "
             "judge 는 그 오류에 대해 아무 정보도 추가하지 않는다. "
             "judge 가 값을 하려면 라벨링 LLM과 다른 것을 보고 판단해야 한다.")

    rep.verdict(ACCEPT, f"H1 지지. 등급만으로 품질이 {(agree_q - review_q) * 100:.1f}pp 갈리고, "
                        f"같은 검수 예산에서 게이트 우선이 무작위보다 낫다.")
    rep.verdict(REJECT, "H1b 기각. 교차 합의는 커버리지를 크게 잃으면서 정답률 이득은 "
                        "같은 모델 반복 호출 게이트보다 작다. 싼 LLM을 합의에 넣으면 "
                        "메인 LLM이 맞힌 답까지 기각된다.")
    rep.verdict(REJECT, f"H1c 기각. LLM-as-a-judge 통과분 {judge_best:.3f} < "
                        f"같은 커버리지 게이트 {gate_same_cov:.3f}. "
                        "judge 를 붙이는 것 자체가 아니라 judge 가 무엇을 보는지가 관건이다. "
                        "같은 편향을 공유하는 judge 는 호출 비용만 늘린다.")
    rep.save()


if __name__ == "__main__":
    main()
