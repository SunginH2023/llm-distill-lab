"""전 실험을 순서대로 돌리고 results/RESULTS.md 를 다시 만든다.

    python experiments/run_all.py      (또는 make all)

전부 결정적이다 — 같은 커밋에서 몇 번을 돌려도 같은 수를 낸다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

import importlib
import io
import time
from contextlib import redirect_stdout

from lab.report import RESULTS

ORDER = [
    "e1_consensus_gate",
    "e2_circular_eval",
    "e3_soft_labels",
    "e4_call_budget",
    "e5_cardinality",
]

HEADER = """# 실험 결과 (자동 생성)

`python experiments/run_all.py` 로 다시 만들어진다. 손으로 고치지 말 것.
모든 수는 고정 시드의 합성 데이터에서 나온다 — 실무 데이터는 한 줄도 들어 있지 않다.

"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    parts, t_start = [HEADER], time.time()
    for name in ORDER:
        mod = importlib.import_module(f"experiments.{name}")
        t0 = time.time()
        buf = io.StringIO()
        with redirect_stdout(buf):          # 각 실험은 자기 md 를 stdout 으로도 찍는다
            mod.main()
        print(f"[{name}] {time.time() - t0:.1f}s")
        parts.append((RESULTS / f"{name}.md").read_text(encoding="utf-8"))

    (RESULTS / "RESULTS.md").write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    print(f"→ results/RESULTS.md  (총 {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
