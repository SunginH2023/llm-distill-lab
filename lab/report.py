"""Write experiment results as Markdown tables + JSON.

Every experiment ends with three lines: (1) hypothesis (2) numbers (3) verdict.
An experiment result without a verdict means running the same experiment again next week.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

ACCEPT, REJECT, PARTIAL = "✅ Supported", "❌ Rejected", "△ Partially supported"


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
        self.parts: list[str] = [f"## {title}", "", f"Hypothesis — {hypothesis}", ""]
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
        self.parts += [f"Verdict — {mark}. {why}", ""]
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
