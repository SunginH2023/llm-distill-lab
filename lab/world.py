"""정답이 존재하는 합성 세계.

실무 파이프라인에서는 '진짜 정답'을 관측할 수 없다. 관측 가능한 건 라벨링 LLM이
찍어준 라벨뿐이고, 그래서 "우리 판단이 옳았는가"를 증명할 방법이 없다.

이 모듈은 같은 구조를 정답을 아는 세계로 옮긴다. 여기서는 오라클(진짜 정답)을
가지고 있으므로, 오라클을 쓰지 않는 파이프라인의 판단들이 실제로 옳았는지
사후에 채점할 수 있다. 오라클은 평가에만 쓰고 학습·라벨 정제에는 절대 쓰지 않는다.

세계의 구성물은 전부 실무에서 반복해서 마주친 현상을 모사한다:

| 장치                | 모사하는 현상                                            |
|---------------------|----------------------------------------------------------|
| 헷갈리는 쌍 (근접 중심)  | 정의 경계가 겹쳐 사람도 갈리는 유형 쌍                    |
| 자석 클래스         | 애매한 건이 전부 흘러드는 '기타'류 대분류                 |
| 진짜 멀티라벨       | 두 유형이 실제로 동시에 성립하는 건 (오답이 아님)         |
| 축소된 작은 모델 표현    | 작은 모델의 표현력 격차 (라벨링 LLM보다 작은 모델)              |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 정의 경계가 겹치는 쌍 — 중심을 서로 끌어당겨 만든다.
CONFUSABLE_PAIRS: tuple[tuple[int, int], ...] = ((3, 4), (5, 6), (7, 8))
# 실제로 동시에 성립할 수 있는 쌍 (진짜 멀티라벨이 허용되는 조합).
COOCCUR_PAIRS: tuple[tuple[int, int], ...] = ((2, 9), (10, 11))
MULTI_ALLOWED: frozenset[frozenset[int]] = frozenset(
    frozenset(p) for p in CONFUSABLE_PAIRS + COOCCUR_PAIRS
)


@dataclass(frozen=True)
class WorldConfig:
    n: int = 60_000
    n_classes: int = 12
    dim: int = 24
    student_dim: int = 23      # 작은 모델이 보는 축소 표현 (표현력 격차)
    feat_noise: float = 0.25
    sigma: float = 1.0
    centroid_scale: float = 3.5
    pair_gap: float = 0.30     # 작을수록 헷갈리는 쌍이 붙는다
    prior_decay: float = 0.9   # 클래스 빈도 롱테일
    magnet_class: int = 0
    magnet_prior_boost: float = 1.5
    multi_ratio: float = 0.40  # post2 >= multi_ratio * post1 이면 둘 다 참
    seed: int = 7


@dataclass
class World:
    cfg: WorldConfig
    mu: np.ndarray            # (C, d) 클래스 중심
    prior: np.ndarray         # (C,)
    x: np.ndarray             # (n, d) 관측
    log_post: np.ndarray      # (n, C) 진짜 로그 사후확률 — 라벨링 LLM/작은 모델에게 직접 주지 않는다
    post: np.ndarray          # (n, C)
    truth: np.ndarray         # (n, C) bool — 베이즈 최적 라벨 집합 = 오라클
    feats: np.ndarray         # (n, student_dim) 작은 모델 입력 (표준화됨)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def n_classes(self) -> int:
        return self.mu.shape[0]

    @property
    def primary(self) -> np.ndarray:
        """진짜 1등 라벨. truth 행렬의 argmax 가 아니다 — 집합은 순서를 담지 않아서
        멀티행에서 argmax 는 그냥 인덱스가 작은 쪽을 고른다. 1등은 사후확률로 정의한다."""
        return np.argmax(self.post, axis=1)

    @property
    def ambiguity(self) -> np.ndarray:
        """1 - max posterior. 클수록 경계에 가깝다."""
        return 1.0 - self.post.max(axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_world(cfg: WorldConfig | None = None) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    C, d = cfg.n_classes, cfg.dim

    # --- 클래스 중심 ---------------------------------------------------
    mu = rng.normal(size=(C, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    mu *= cfg.centroid_scale

    # 헷갈리는 쌍: 두 중심을 서로 끌어당긴다 → 정의 경계가 겹친다
    for a, b in CONFUSABLE_PAIRS:
        mid = 0.5 * (mu[a] + mu[b])
        mu[a] = mid + (mu[a] - mid) * cfg.pair_gap
        mu[b] = mid + (mu[b] - mid) * cfg.pair_gap

    # 자석 클래스: 전체 중심 근처에 앉혀 애매한 건을 빨아들이게 한다
    mu[cfg.magnet_class] = mu.mean(axis=0) + 0.15 * mu[cfg.magnet_class]

    # --- 사전분포 (롱테일 + 자석 클래스가 흔하다) -----------------------
    prior = cfg.prior_decay ** np.arange(C)
    prior[cfg.magnet_class] *= cfg.magnet_prior_boost
    prior /= prior.sum()

    # --- 표본 -----------------------------------------------------------
    y_gen = rng.choice(C, size=cfg.n, p=prior)
    x = mu[y_gen] + cfg.sigma * rng.normal(size=(cfg.n, d))

    # ||x - mu||^2 를 전개해서 (n, C) 만 만든다 (n×C×d 텐서 회피)
    sq = (x * x).sum(1)[:, None] - 2.0 * x @ mu.T + (mu * mu).sum(1)[None, :]
    log_post = -0.5 * sq / cfg.sigma**2 + np.log(prior)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = _softmax(log_post)

    # --- 오라클 라벨 집합 ------------------------------------------------
    # 진짜 정답은 '생성에 쓰인 클래스'가 아니라 베이즈 최적 집합으로 둔다.
    # x만 보고 도달 가능한 상한이 그것이고, 그래야 '천장'이 정의된다.
    order = np.argsort(-post, axis=1)
    p1, p2 = order[:, 0], order[:, 1]
    rows = np.arange(cfg.n)
    take_second = post[rows, p2] >= cfg.multi_ratio * post[rows, p1]
    allowed = np.array(
        [frozenset((a, b)) in MULTI_ALLOWED for a, b in zip(p1, p2)]
    )
    take_second &= allowed

    truth = np.zeros((cfg.n, C), dtype=bool)
    truth[rows, p1] = True
    truth[rows[take_second], p2[take_second]] = True

    # --- 작은 모델 입력: 직라벨링 LLM영 + 잡음 (표현력 격차) --------------------------
    # 직교(QR)로 뽑는 이유: 일반 가우시안 사영은 조건수가 나빠 정보 손실이 아니라
    # 최적화 난이도를 만든다. 그러면 "작은 모델이 약해서"가 아니라 "수렴을 못 해서"
    # 점수가 떨어지고, 라벨 품질 효과가 그 잡음에 묻힌다. (실제로 처음에 이걸로 한 번 속았다.)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    proj = q[:, :cfg.student_dim]
    feats = x @ proj + cfg.feat_noise * rng.normal(size=(cfg.n, cfg.student_dim))
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    feats = feats.astype(np.float32)

    return World(cfg=cfg, mu=mu, prior=prior, x=x, log_post=log_post,
                 post=post, truth=truth, feats=feats)


def split_indices(n: int, sizes: dict[str, int], seed: int = 13) -> dict[str, np.ndarray]:
    """겹치지 않는 인덱스 분할. 순서는 고정(재현성)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out, cur = {}, 0
    for name, size in sizes.items():
        out[name] = np.sort(perm[cur:cur + size])
        cur += size
    if cur > n:
        raise ValueError(f"분할 크기 합({cur})이 n({n})보다 큽니다")
    return out
