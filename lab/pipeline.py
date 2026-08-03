"""실험들이 공유하는 조립 라인.

    세계 생성 → 라벨링 LLM 투표 → 합의 게이트 → pseudo-label → 작은 모델 학습 → 세 채점표로 채점

채점표가 셋인 것이 이 repo 의 요점이다.

    circular : 라벨링 LLM이 만든 pseudo-label            — 실무에서 유일하게 관측 가능한 것
    weak     : 파이프라인 외부 신호 (거칠고 잡음 많다) — 실무에서 만들 수 있는 것
    oracle   : 진짜 정답                     — 합성 세계에만 존재. 위 둘을 검증하는 데만 쓴다
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import labeler, metrics, signals, small_model
from .world import World, WorldConfig, make_world, split_indices

# 비율로 둔다 — 작은 세계로도 (테스트용) 그대로 돌아가야 한다.
SPLIT_FRACTIONS = {"train": 0.50, "val": 0.133, "test": 0.333}


@dataclass
class Lab:
    world: World
    idx: dict[str, np.ndarray]
    axis_fix: signals.WeakAxis     # 라벨 정제 전용 축
    axis_eval: signals.WeakAxis    # 평가 전용 축 — 정제에 절대 쓰지 않는다
    _teachers: dict = field(default_factory=dict)

    # --- 조각 꺼내기 ---------------------------------------------------
    def feats(self, split: str) -> np.ndarray:
        return self.world.feats[self.idx[split]]

    def truth(self, split: str) -> np.ndarray:
        return self.world.truth[self.idx[split]]

    def truth_top1(self, split: str) -> np.ndarray:
        return self.world.primary[self.idx[split]]

    def axis(self, which: str, split: str) -> signals.WeakAxis:
        a = self.axis_fix if which == "fix" else self.axis_eval
        i = self.idx[split]
        return signals.WeakAxis(a.name, a.group_of, a.observed[i], a.noise)

    def get_teacher(self, cfg: labeler.LabelerConfig = labeler.MAIN) -> labeler.LabelerLLM:
        key = (cfg.gamma, cfg.beta_magnet, cfg.pair_skew, cfg.temperature, cfg.seed)
        if key not in self._teachers:
            self._teachers[key] = labeler.LabelerLLM(self.world, cfg)
        return self._teachers[key]

    # --- LLM 호출 -------------------------------------------------------
    def vote(self, split: str, k: int, cfg: labeler.LabelerConfig = labeler.MAIN,
             seed: int = 0) -> np.ndarray:
        return self.get_teacher(cfg).vote(self.idx[split], k=k, seed=7000 + 13 * seed + k)

    def key(self, split: str, k: int, cfg: labeler.LabelerConfig = labeler.MAIN,
            seed: int = 0, **kw) -> labeler.PseudoLabels:
        return labeler.labels_from_votes(self.vote(split, k, cfg, seed), **kw)


def build(world_cfg: WorldConfig | None = None, seed: int = 13) -> Lab:
    w = make_world(world_cfg)
    sizes = {k: int(w.n * f) for k, f in SPLIT_FRACTIONS.items()}
    idx = split_indices(w.n, sizes, seed=seed)
    C = w.n_classes
    return Lab(
        world=w, idx=idx,
        axis_fix=signals.make_axis("사용자 선택 카테고리", w.primary, C,
                                   n_groups=5, noise=0.30, seed=101),
        axis_eval=signals.make_axis("처리 조직 배정", w.primary, C,
                                    n_groups=7, noise=0.25, seed=202),
    )


# --- 작은 모델 -----------------------------------------------------------------

def train_small_model(lab: Lab, targets: np.ndarray, train_split: str = "train",
                  eval_split: str = "test", weight: np.ndarray | None = None,
                  **kw) -> np.ndarray:
    return small_model.train_and_predict(lab.feats(train_split), targets,
                                     lab.feats(eval_split), sample_weight=weight, **kw)


# --- 채점 -----------------------------------------------------------------

def key_quality(lab: Lab, key: labeler.PseudoLabels, split: str) -> dict[str, float]:
    """pseudo-label 자체를 오라클로 채점한다 (작은 모델과 무관하게 라벨 품질만)."""
    truth = lab.truth(split)
    s = metrics.summary(key.mat, truth)
    return {
        "key_em": s["em"],
        "key_micro_f1": s["micro_f1"],
        "key_top1": float((key.primary == lab.truth_top1(split)).mean()),
        "key_multi_rate": s["pred_multi_rate"],
    }


def score_all(lab: Lab, proba: np.ndarray, split: str, circular: labeler.PseudoLabels,
              ratio_threshold: float = 0.5) -> dict[str, float]:
    """같은 예측을 세 채점표로 채점한다."""
    pred = small_model.decode(proba, ratio_threshold)
    truth = lab.truth(split)
    ax = lab.axis("eval", split)
    mask = signals.discriminative_mask(ax, proba)
    return {
        "same_source_em": float(metrics.exact_match(pred, circular.mat).mean()),
        "same_source_top1": float((np.argmax(proba, axis=1) == circular.primary).mean()),
        "outside_signal": ax.score_top1(proba, mask),
        "truth_em": float(metrics.exact_match(pred, truth).mean()),
        "truth_top1_acc": float((np.argmax(proba, axis=1) == lab.truth_top1(split)).mean()),
        "truth_multi_em": metrics.summary(pred, truth)["em_multi"],
    }
