"""실험 결과를 마크다운 표 + JSON 으로 남긴다.

모든 실험은 (1) 가설 (2) 수치 (3) 판정 세 줄로 끝난다.
판정이 없는 실험 결과는 다음 주에 같은 실험을 또 하게 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

ACCEPT, REJECT, PARTIAL = "✅ 지지", "❌ 기각", "△ 부분지지"


def fmt(v, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(headers: list[str], rows: list[list], align_right_from: int = 1) -> str:
    sep = []
    for i in range(len(headers)):
        sep.append("---:" if i >= align_right_from else "---")
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


class Report:
    def __init__(self, slug: str, title: str, hypothesis: str):
        self.slug, self.title = slug, title
        self.parts: list[str] = [f"## {title}", "", f"가설 — {hypothesis}", ""]
        self.data: dict = {"slug": slug, "title": title, "hypothesis": hypothesis}

    def text(self, s: str) -> "Report":
        self.parts += [s, ""]
        return self

    def table(self, headers, rows, **kw) -> "Report":
        self.parts += [table(headers, rows, **kw), ""]
        return self

    def record(self, key: str, value) -> "Report":
        self.data[key] = value
        return self

    def verdict(self, mark: str, why: str) -> "Report":
        self.parts += [f"판정 — {mark}. {why}", ""]
        self.data["verdict"] = f"{mark}: {why}"
        return self

    def save(self) -> None:
        RESULTS.mkdir(exist_ok=True)
        md = "\n".join(self.parts).rstrip() + "\n"
        (RESULTS / f"{self.slug}.md").write_text(md, encoding="utf-8")
        (RESULTS / f"{self.slug}.json").write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
        print(md)
