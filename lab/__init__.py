"""LLM이 붙인 pseudo-label 로 작은 모델을 학습시키는 사이클을,
정답을 아는 합성 세계에서 처음부터 끝까지 돌려 보는 실험실.

주의: numpy 를 임포트하기 전에 BLAS 스레드 수를 고정한다.

이 실험실의 행렬곱은 (n=30000, k=24, m=12) 처럼 극단적으로 납작하다.
OpenBLAS 는 코어가 많은 기계에서 이런 모양을 32스레드로 쪼개려다 스레드
오버헤드가 연산보다 커진다 — 실측으로 33ms → 2ms (16배) 차이가 났다.
"학습이 느리다"고 판단해 스텝 수를 줄이는 대신 프로파일링부터 한 결과다.
"""

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__all__ = ["world", "labeler", "small_model", "signals", "metrics", "report", "pipeline"]
