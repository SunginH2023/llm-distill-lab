"""파이프라인 외부 신호 (weak external axes).

라벨링 LLM이 붙인 라벨로 채점하면 '같은 집안 효과'를 못 벗어난다. 벗어나려면 LLM 파이프라인
외부에서 온 관측이 필요하다. 실무에서 쓴 것은 두 개였다:

  * 사용자가 접수 시점에 직접 고른 굵은 카테고리
  * 담당자가 실제로 그 건을 보낸 처리 조직

둘 다 (a) 라벨 체계보다 훨씬 거칠고 (b) 잡음이 많지만 (c) 라벨링 LLM과 통계적으로
독립이다. 그래서 절대 점수는 낮게 나오고 상대 비교만 유효하다.

여기서는 같은 성질의 축을 두 개 만든다. 하나는 라벨 정제용, 다른 하나는
평가 전용으로 분리한다 — 정제에 쓴 축으로 평가하면 그것도 순환이기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeakAxis:
    """클래스 → 굵은 그룹의 다대일 사상 + 관측 잡음."""
    name: str
    group_of: np.ndarray     # (C,) 클래스 → 그룹
    observed: np.ndarray     # (n,) 관측된 그룹 (잡음 포함)
    noise: float

    @property
    def n_groups(self) -> int:
        return int(self.group_of.max()) + 1

    def compatible(self, pred: np.ndarray) -> np.ndarray:
        """예측 집합이 관측된 그룹과 양립하는가 → (n,) bool.

        '정확도'가 아니라 일관성을 잰다: 모델의 답이 이 관측과 모순되지 않는가.
        """
        pred_groups = np.where(pred, self.group_of[None, :], -1)
        return (pred_groups == self.observed[:, None]).any(axis=1)

    def score(self, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
        ok = self.compatible(pred)
        return float(ok.mean() if mask is None else ok[mask].mean())

    def score_top1(self, proba: np.ndarray, mask: np.ndarray | None = None) -> float:
        """외부 신호는 1등 라벨만 채점한다.

        집합 전체를 채점하면 라벨을 많이 뱉는 모델이 유리해진다(양립할 기회가 늘어나므로).
        즉 외부 신호는 '개수'를 채점할 수 없다 — 이 축의 구조적 한계이고,
        그래서 개수/디코딩은 별도 실험(e5)에서 따로 잰다.
        """
        top1 = np.argmax(proba, axis=1)
        ok = self.group_of[top1] == self.observed
        return float(ok.mean() if mask is None else ok[mask].mean())

    def correct_top1(self, proba: np.ndarray) -> np.ndarray:
        top1 = np.argmax(proba, axis=1)
        return self.group_of[top1] == self.observed


def make_axis(name: str, primary: np.ndarray, n_classes: int, n_groups: int,
              noise: float, seed: int) -> WeakAxis:
    """진짜 라벨에서 파생시키되 잡음을 섞는다. 라벨링 LLM은 이 과정에 관여하지 않는다."""
    rng = np.random.default_rng(seed)
    group_of = rng.integers(0, n_groups, size=n_classes)
    # 빈 그룹이 없도록 보정
    for g in range(n_groups):
        if not (group_of == g).any():
            group_of[rng.integers(0, n_classes)] = g

    obs = group_of[primary].copy()
    flip = rng.random(len(obs)) < noise
    obs[flip] = rng.integers(0, n_groups, size=int(flip.sum()))
    return WeakAxis(name=name, group_of=group_of, observed=obs, noise=noise)


def contaminate(axis: WeakAxis, attractor_class: int, strength: float,
                seed: int) -> WeakAxis:
    """외부 신호를 '라벨링 LLM과 같은 방향으로' 오염시킨다.

    순환 밖에서 왔다는 것만으로는 부족하다. 그 축이 라벨링 LLM과 같은 편향을 공유하면
    (예: 운영 자동배정 규칙을 라벨링 LLM도 학습해 버린 경우) 축은 다시 순환에 빨려든다.
    여기서는 관측 잡음이 균등이 아니라 라벨링 LLM이 좋아하는 그룹으로 쏠리게 만든다.
    """
    rng = np.random.default_rng(seed)
    obs = axis.observed.copy()
    target = axis.group_of[attractor_class]
    pull = rng.random(len(obs)) < strength
    obs[pull] = target
    return WeakAxis(f"{axis.name} (오염됨)", axis.group_of, obs, axis.noise)


def discriminative_mask(axis: WeakAxis, proba: np.ndarray) -> np.ndarray:
    """축이 상위 두 후보를 실제로 구분해 주는 행만 남긴다.

    두 후보가 같은 그룹이면 이 축은 그 행에 대해 아무 말도 못 한다.
    실무에서 '판별력 없는 잡동사니 카테고리'를 제외한 것과 같은 처리다.
    """
    order = np.argsort(-proba, axis=1)
    return axis.group_of[order[:, 0]] != axis.group_of[order[:, 1]]


def tiebreak(key, axis: WeakAxis, only: np.ndarray, secondary_frac: float = 0.34):
    """애매한 행에 한해, 순환 밖 축으로 primary 를 뒤집는다.

    조건: 지금 1등이 관측 그룹과 모순이고 2등은 양립 → 둘을 교체.
    오라클은 쓰지 않는다. `only` 로 대상을 애매행에 한정하는 게 중요하다 —
    만장일치 행까지 건드리면 잡음 많은 축이 멀쩡한 라벨을 깎아먹는다.

    반환 (새 Key, 교체된 행 수).
    """
    out = key.copy()
    n = out.mat.shape[0]
    rows = np.arange(n)
    order = np.argsort(-key.frac, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]
    g = axis.group_of
    swap = only & (g[p1] != axis.observed) & (g[p2] == axis.observed)

    s = rows[swap]
    out.primary[s] = p2[s]
    out.mat[s] = False
    out.mat[s, p2[s]] = True
    keep_old = key.frac[s, p1[s]] >= secondary_frac      # 원래 1등은 secondary 로 강등
    out.mat[s[keep_old], p1[s[keep_old]]] = True
    return out, int(swap.sum())
