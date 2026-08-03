"""라벨을 만들어 주는 LLM(대리자) + 합의 게이트 + LLM-as-a-judge.

요즘 흔한 상황을 그대로 옮긴 모듈.

    사람이 라벨링할 여력은 없고, 성능 좋은 LLM은 있다.
    → LLM에게 라벨을 시킨다. 그게 pseudo-label 이다.
    → 그런데 이 pseudo-label 이 얼마나 믿을 만한지는 아무도 모른다.
    → 그래서 여러 번 호출해 보고, 다른 LLM에게 judge 를 맡기고,
      표가 갈린 것만 사람이 보고... 여러 방법을 얹게 된다.

이 모듈은 그 '라벨을 만드는 LLM' 을 시뮬레이션한다. 중요한 설계 하나:
LLM의 오차를 두 성분으로 나눠서 심는다.

    무작위 성분 (temperature)  — 호출할 때마다 흔들린다. 여러 번 부르면 지워진다.
    체계 성분 (bias / skew)    — 매번 같은 방향으로 틀린다. 몇 번을 불러도 안 지워진다.

pseudo-label 은 체계 성분을 그대로 물려받는다. 그 pseudo-label 로 만든 평가셋으로
모델을 채점하면 체계 성분까지 잘 따라 한 모델이 높은 점수를 받는다.
e2 에서 재는 것이 이 현상이다.

체계 성분은 두 가지로 심는다.
  * `beta_magnet` : 애매하면 '기타'류(자석 클래스)로 밀어 넣는 성향
  * `pair_skew`   : 헷갈리는 쌍에서 늘 같은 쪽을 고르는 성향
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import CONFUSABLE_PAIRS, World

# 등급: 표가 갈린 정도로 경로를 가른다. 확률로 환산하지 않는다.
AGREE, PARTIAL, REVIEW = 0, 1, 2
GRADE_NAMES = {AGREE: "agree", PARTIAL: "partial", REVIEW: "review"}


@dataclass(frozen=True)
class LabelerConfig:
    gamma: float = 0.8         # 날카로움. 1.0 이면 이론상 최적, 낮을수록 흐릿하다
    beta_magnet: float = 1.1   # 체계 편향: '기타'류 선호
    pair_skew: float = 0.8     # 체계 편향: 헷갈리는 쌍에서 한쪽 선호
    temperature: float = 0.7   # 호출 간 흔들림 (무작위 성분)
    seed: int = 11


# 성능 좋은 메인 LLM / 값싼 보조 LLM 두 프리셋
MAIN = LabelerConfig()
CHEAP = LabelerConfig(gamma=0.35, beta_magnet=1.9, pair_skew=1.4, temperature=1.1, seed=23)


class LabelerLLM:
    """라벨을 붙여 주는 LLM. `vote(k)` 가 k 번 호출에 해당한다."""

    def __init__(self, world: World, cfg: LabelerConfig = MAIN):
        self.world, self.cfg = world, cfg
        bias = np.zeros(world.n_classes)
        bias[world.cfg.magnet_class] += cfg.beta_magnet
        for a, _b in CONFUSABLE_PAIRS:      # 쌍마다 앞쪽 코드를 선호한다
            bias[a] += cfg.pair_skew
        self.bias = bias
        self._logits = cfg.gamma * world.log_post + bias[None, :]

    def vote(self, idx: np.ndarray, k: int, seed: int) -> np.ndarray:
        """k 번 호출. 반환 (len(idx), C) 정수 득표수.

        Gumbel-max 트릭으로 softmax 샘플링 = temperature 를 준 LLM 호출.
        """
        rng = np.random.default_rng(seed)
        logits = self._logits[idx]
        counts = np.zeros(logits.shape, dtype=np.int32)
        for _ in range(k):
            g = rng.gumbel(size=logits.shape)
            pick = np.argmax(logits / max(self.cfg.temperature, 1e-6) + g, axis=1)
            counts[np.arange(len(idx)), pick] += 1
        return counts


class JudgeLLM:
    """LLM-as-a-judge. 라벨을 새로 고르는 게 아니라 제안된 라벨을 검수한다.

    현장에서 자주 쓰는 패턴이라 따로 뒀다. 흔한 기대는 "다른 LLM에게 한 번 더
    물어보면 품질이 오른다" 인데, 실제로는 judge 가 어떤 편향을 갖고 있는지에
    따라 결과가 갈린다(e1 에서 측정).

    judge 는 "이 라벨이 1등감인가"에 확률적으로 답한다. judge 자신도 편향이 있고,
    그 편향이 메인 LLM과 겹치면 서로의 오류를 걸러 주지 못한다.
    """

    def __init__(self, world: World, cfg: LabelerConfig = CHEAP, strictness: float = 0.0):
        self.world, self.cfg, self.strictness = world, cfg, strictness
        bias = np.zeros(world.n_classes)
        bias[world.cfg.magnet_class] += cfg.beta_magnet
        for a, _b in CONFUSABLE_PAIRS:
            bias[a] += cfg.pair_skew
        self._logits = cfg.gamma * world.log_post + bias[None, :]

    def accept(self, idx: np.ndarray, proposed: np.ndarray, seed: int) -> np.ndarray:
        """제안된 라벨을 받아들일지 (n,) bool 로 답한다.

        `strictness` 를 올리면 더 깐깐해진다(= 통과율이 떨어진다).
        """
        rng = np.random.default_rng(seed)
        lg = self._logits[idx]
        rows = np.arange(len(idx))
        masked = np.where(np.arange(lg.shape[1])[None, :] == proposed[:, None], -np.inf, lg)
        margin = lg[rows, proposed] - masked.max(axis=1) - self.strictness
        p_accept = 1.0 / (1.0 + np.exp(-margin / max(self.cfg.temperature, 1e-6)))
        return rng.random(len(idx)) < p_accept


@dataclass
class PseudoLabels:
    """LLM이 붙인 라벨 묶음.

    집합(`mat`)만이 아니라 1등이 누구인지(`primary`)를 같이 들고 다닌다.
    처음엔 집합만 저장했다가 라벨을 정제할 때 1등이 소리 없이 바뀌어 지표가 엉켰다.
    집합 표현은 순서를 담지 못한다.
    """
    mat: np.ndarray       # (n, C) bool — 라벨 집합
    primary: np.ndarray   # (n,) 1등 라벨
    frac: np.ndarray      # (n, C) 득표 비율
    grade: np.ndarray     # (n,) AGREE / PARTIAL / REVIEW
    k: np.ndarray         # (n,) 행마다 호출 횟수가 다를 수 있다 (애매행 재호출)

    @property
    def multi_rate(self) -> float:
        return float((self.mat.sum(axis=1) >= 2).mean())

    def copy(self) -> "PseudoLabels":
        return PseudoLabels(self.mat.copy(), self.primary.copy(),
                            self.frac, self.grade, self.k)


def labels_from_votes(counts: np.ndarray, secondary_frac: float = 0.34) -> PseudoLabels:
    """득표수 → pseudo-label + 등급.

    핵심: 득표수를 '신뢰도 확률'로 환산하지 않는다. 경로를 가르는 게이트로만 쓴다.
    (confidence 를 품질 게이트로 쓰려던 시도는 교정이 안 돼 접은 적이 있다.)

    2등 라벨 채택 규칙 = 2등이 전체 표의 1/3 초과.
    k=3 에서는 이 규칙이 만족될 수 없다
    → 3번 호출해 만든 pseudo-label 은 구조적으로 멀티라벨을 표현하지 못한다. e4 의 출발점.
    """
    k = counts.sum(axis=1)
    frac = counts / np.maximum(k, 1)[:, None]
    n, C = counts.shape
    rows = np.arange(n)

    order = np.argsort(-counts, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]

    mat = np.zeros((n, C), dtype=bool)
    mat[rows, p1] = True
    second = frac[rows, p2] >= secondary_frac
    mat[rows[second], p2[second]] = True

    top = frac[rows, p1]
    grade = np.where(top >= 1.0 - 1e-9, AGREE, np.where(top > 0.5, PARTIAL, REVIEW))
    return PseudoLabels(mat=mat, primary=p1, frac=frac, grade=grade, k=k)


def vote_targets(labels: PseudoLabels) -> np.ndarray:
    """투표 분포를 그대로 학습 타깃으로 (지식증류의 최소 형태).

    최종 라벨만 쓰면 'LLM이 몇 표 차로 그 답을 골랐는가'가 통째로 버려진다.
    투표 분포를 보존해 두면 GPU를 다시 돌리지 않고도 이 실험이 가능하다.
    """
    s = labels.frac.sum(axis=1, keepdims=True)
    return (labels.frac / np.maximum(s, 1e-9)).astype(np.float32)


def set_targets(labels: PseudoLabels, secondary_weight: float = 0.25) -> np.ndarray:
    """라벨 집합을 확률 타깃으로. 단일이면 one-hot, 멀티면 (1-w, w)."""
    n, C = labels.mat.shape
    rows = np.arange(n)
    t = np.zeros((n, C), dtype=np.float32)
    t[rows, labels.primary] = 1.0

    extra = labels.mat.copy()
    extra[rows, labels.primary] = False
    has_extra = extra.any(axis=1)
    if has_extra.any():
        e = extra[has_extra].astype(np.float32)
        e /= e.sum(axis=1, keepdims=True)
        idx = np.where(has_extra)[0]
        t[idx] = e * secondary_weight
        t[idx, labels.primary[idx]] = 1.0 - secondary_weight
    return t
