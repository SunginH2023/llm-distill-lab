"""의존성 없는 테스트 러너. `pytest tests/` 로도 그대로 돌아간다.

    python tests/run_tests.py
"""

from __future__ import annotations

import pathlib as _p
import sys as _s

_HERE = _p.Path(__file__).resolve().parent
_s.path.insert(0, str(_HERE.parent))
_s.path.insert(0, str(_HERE))
import lab as _lab  # noqa: F401,E402  (numpy 임포트 전에 BLAS 스레드 고정)

import time  # noqa: E402
import traceback  # noqa: E402

import test_lab  # noqa: E402


def main() -> int:
    fns = [(n, f) for n, f in vars(test_lab).items()
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"  ok   {name}  ({time.time() - t0:.2f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
