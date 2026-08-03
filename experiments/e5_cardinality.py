"""e5 — 라벨 '개수' 문제. 그리고 개수 판정기를 따로 학습하려던 시도의 기각.

멀티라벨에서 점수를 잃는 곳은 대개 라벨 선택이 아니라 개수다.
그래서 자연스럽게 이런 생각을 한다: "개수를 맞히는 모델을 따로 학습하자."

가설 H5a  개수 전용 판정기 P(멀티|x) 로 top-2 를 열면, 그냥 확률비 p2/p1 로
          여는 것보다 낫다.
가설 H5b  전체 완전일치를 최대화하는 임계값 하나가 존재한다.

둘 다 실무에서 여러 층위로 시도해 본 것이고, 여기서 같은 구조를 다시 세운다.
임계값은 val 에서 고르고 test 에서만 보고한다.

결론을 미리 적어 두면: H5a 는 '무엇을 재느냐'에 따라 채택도 기각도 된다.
그 사실 자체가 이 실험의 결과다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

import numpy as np

from lab import labeler, metrics, pipeline, small_model
from lab.report import PARTIAL, REJECT, Report

THRESHOLDS = np.round(np.arange(0.10, 1.01, 0.05), 3)


def decode_scores(proba: np.ndarray, score: np.ndarray, t: float) -> np.ndarray:
    """score >= t 인 행만 top-2 를 연다."""
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

    rep = Report("e5_cardinality", "e5 · 개수 문제: 같은 실험, 지표에 따라 뒤집히는 결론",
                 "라벨 개수를 따로 맞히는 모델이 확률비 한 줄을 이기는가")

    # --- 개수 판정기 학습 (오라클 없이, pseudo-label의 개수로) --------------------
    y_multi = (train_key.mat.sum(axis=1) >= 2).astype(np.float32)
    tgt = np.stack([1 - y_multi, y_multi], axis=1)
    W = small_model.fit(lab.feats("train"), tgt, steps=200)
    judge_val = small_model.predict_proba(W, lab.feats("val"))[:, 1]
    judge_te = small_model.predict_proba(W, lab.feats("test"))[:, 1]
    judge_acc = float(((judge_te >= 0.5) == (truth_te.sum(1) >= 2)).mean())

    scorers_val = {
        "확률비 p2/p1": ratio_score(proba_val),
        "개수 판정기 P(멀티)": judge_val,
        "결합 P(멀티)×(p2/p1)": judge_val * ratio_score(proba_val),
    }
    scorers_te = {
        "확률비 p2/p1": ratio_score(proba_te),
        "개수 판정기 P(멀티)": judge_te,
        "결합 P(멀티)×(p2/p1)": judge_te * ratio_score(proba_te),
    }

    rep.text(f"작은 모델은 7회 호출로 만든 pseudo-label 로 학습한 것 하나로 고정하고 디코딩만 바꾼다. "
             f"개수 판정기는 pseudo-label의 개수를 타깃으로 학습했다(오라클 미사용, "
             f"정답 개수 기준 정확도 {judge_acc:.3f}).")

    # --- 운영점을 맞춰서 비교한다 ------------------------------------------
    # 전체 EM 만 최대화하면 답은 언제나 "멀티라벨을 아예 예측하지 마라"로 수렴한다
    # (아래 파레토 표 참조). 그래서 세 점수기를 같은 예측 멀티율에서 비교한다.
    # 목표 멀티율은 오라클이 아니라 pseudo-label의 멀티율로 잡는다(실무에서 관측 가능).
    target_rate = float(train_key.multi_rate)
    rows = []
    for name in scorers_val:
        sv, st = scorers_val[name], scorers_te[name]
        t = float(np.quantile(sv, 1.0 - target_rate))     # val 에서만 임계값 결정
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
    rows.append(["(참고) 멀티라벨을 아예 안 예측", "-",
                 float(metrics.exact_match(never_multi, truth_te).mean()), 0.0, 0.0])
    rows.append(["개수를 알려줬을 때 (상한)", "-",
                 float(metrics.exact_match(oracle_pred, truth_te).mean()),
                 float(metrics.exact_match(oracle_pred, truth_te)[two].mean()),
                 float((oracle_pred.sum(1) >= 2).mean())])

    rep.text(f"세 점수기를 같은 예측 멀티율({target_rate:.3f}, pseudo-label 멀티율에 맞춤)에서 비교한다. "
             "임계값은 val 에서만 정하고 test 는 보고에만 쓴다.")
    rep.table(["점수기 (운영점 일치)", "t(val)", "전체 EM(test)",
               "멀티행 EM(test)", "예측 멀티율"], rows)
    rep.record("decoding", rows)

    ratio_em, judge_em, comb_em = rows[0][2], rows[1][2], rows[2][2]
    ratio_mem, judge_mem, comb_mem = rows[0][3], rows[1][3], rows[2][3]

    # 차이가 표본잡음인지 확인한다 (같은 행을 짝지어 비교).
    def em_of(name_idx: int) -> np.ndarray:
        name = list(scorers_te)[name_idx]
        t = float(np.quantile(scorers_val[name], 1.0 - target_rate))
        return metrics.exact_match(decode_scores(proba_te, scorers_te[name], t), truth_te)

    em_ratio, em_judge = em_of(0), em_of(1)
    multi_mask = truth_te.sum(axis=1) >= 2
    mc_all = metrics.mcnemar(em_ratio, em_judge)
    mc_multi = metrics.mcnemar(em_ratio[multi_mask], em_judge[multi_mask])
    rep.record("mcnemar_ratio_vs_judge", {"all": mc_all, "multi_rows": mc_multi})

    rep.text(f"같은 멀티율에서 전체 EM 은 확률비 {ratio_em:.4f} vs 판정기 {judge_em:.4f} "
             f"(McNemar p={mc_all['p']:.3g}) — 차이가 있어도 소수점 셋째 자리다. "
             f"그런데 멀티행 EM 은 {ratio_mem:.4f} → {judge_mem:.4f} 로 3배 넘게 벌어진다 "
             f"(p={mc_multi['p']:.3g}). "
             "판정기는 '어느 행이 멀티인가'에 대해 확률비가 모르는 것을 안다 — "
             "이 세계에서 멀티 여부는 두 후보가 공존 허용 쌍인지에 달려 있고, "
             "그건 상위 두 확률의 비율에 들어 있지 않은 정보이기 때문이다. "
             "그런데도 전체 EM 이 안 움직이는 이유는 단순하다: "
             f"멀티행이 전체의 {float(multi_mask.mean()):.1%} 뿐이라 "
             "평균 지표가 그 이득을 통째로 삼킨다.")
    rep.text("> 주의: 여기서 이 실험실이 실제 파이프라인과 갈린다. 실무에서 같은 구조의 "
             "개수 판정기를 여러 변형으로 학습시켰을 때는 확률비 게이트를 끝내 넘지 못했다. "
             "합성 세계에서는 '개수'가 공존 허용 쌍이라는 깔끔한 규칙으로 정해지므로 "
             "판정기가 배울 것이 실제보다 많다. "
             "→ 합성 실험이 실무 결론을 대체하지 못하는 지점이고, "
             "차이가 난 이유를 설명할 수 있을 때에만 합성 실험이 쓸모 있다.")

    # --- H5b: 임계값 하나로는 두 축을 동시에 못 잡는다 ---------------------
    front = curve(proba_te, scorers_te["확률비 p2/p1"], truth_te)
    show = [[t, em, mem, mr] for t, em, mem, mr in front if round(t, 2) in
            (0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)]
    rep.text("확률비 임계값을 움직이며 test 에서 직접 본 파레토 전선:")
    rep.table(["t", "전체 EM", "멀티행 EM", "예측 멀티율"], show)
    rep.record("pareto", front)
    best_all = max(front, key=lambda r: r[1])
    best_multi = max(front, key=lambda r: r[2])
    rep.text(f"전체 EM 은 t={best_all[0]:.2f} 에서 {best_all[1]:.4f} 로 최대인데, "
             f"그때 멀티행 EM 은 {best_all[2]:.4f} 다. "
             f"멀티행 최대는 t={best_multi[0]:.2f} 의 {best_multi[2]:.4f} 이고 "
             f"이때 전체는 {best_multi[1]:.4f} 로 떨어진다. "
             "하나의 t 로 두 축을 동시에 최대화할 수 없다. "
             "그러니 t 는 '튜닝해서 얻는 값'이 아니라 "
             "어느 쪽 오류를 감당할지 정하는 운영 결정이다.")

    rep.verdict(PARTIAL,
                f"H5a 부분지지 — 무엇을 재느냐에 따라 결론이 뒤집히는 사례. "
                f"전체 EM 으로는 차이가 없고({ratio_em:.4f} → {judge_em:.4f}) "
                f"멀티행 EM 으로는 명백한 지지({ratio_mem:.4f} → {judge_mem:.4f}, p={mc_multi['p']:.3g}). "
                "전체 EM 하나만 찍고 '판정기 기각'이라고 적었다면 "
                "실재하는 신호를 지표 설계 때문에 버렸을 것이다.")
    rep.verdict(REJECT,
                "H5b 기각. 전체 EM 최적 임계값과 멀티행 최적 임계값이 다르고, "
                f"전체 EM 만 최대화하면 답은 '멀티라벨을 아예 예측하지 마라'({rows[-2][2]:.4f})로 "
                "수렴한다. 즉 t 는 튜닝해서 얻는 값이 아니라 "
                "어느 쪽 오류를 감당할지 정하는 운영 결정이다. "
                f"개수를 알려줬을 때의 상한 {rows[-1][2]:.4f} 이 남은 여지를 알려준다.")
    rep.save()


if __name__ == "__main__":
    main()
