"""Synthetic world + pseudo-label pipeline -> SFT training data (JSONL).

Carries the conclusions the earlier experiments reached on vectors over to the
text / LLM side:

    e1  -> rows with split votes are marked via `grade` (review queue / re-query targets)
    e3  -> the vote distribution is stored as `vote_dist` (for soft-target experiments)
    e4  -> the default pseudo-label spends extra calls only on ambiguous rows
    e2  -> evaluation rows also carry an outside signal (`outside_signal`) and the true label

The `truth` field is for evaluation only; the training script never reads it
(in a synthetic experiment, letting the oracle leak into training would make
every result meaningless).

The rendered texts and the strings written to meta.json / summary.json are
intentionally Korean (see render_text.py). They are dataset content and are
kept byte-identical; only comments and console messages are in English.

    python finetune/build_sft_data.py --out finetune/data --n-train 4000
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (pins BLAS threads before numpy is imported)

import argparse
import json

import numpy as np

from finetune import render_text as rt
from lab import labeler, pipeline


def targeted_pseudo_labels(lab, split: str, base_k: int = 3, extra_k: int = 4):
    """Apply the e4 conclusion: base_k calls for every row, plus extra_k more only for split-vote rows."""
    base = lab.vote(split, base_k, seed=11)
    amb = labeler.labels_from_votes(base).grade != labeler.AGREE
    extra = lab.vote(split, extra_k, seed=12)
    merged = base + extra * amb[:, None]
    return labeler.labels_from_votes(merged), amb


def dump(path: _p.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = _p.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pipeline.build()
    caps = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    ax_eval = lab.axis_eval
    meta = {"classes": rt.CLASS_NAMES, "prompt_template": rt.PROMPT,
            # Outside-signal crosswalk (evaluation only): category -> coarse group
            "outside_signal_group_of": [int(g) for g in ax_eval.group_of],
            "outside_signal_noise": ax_eval.noise,
            # Korean note kept verbatim (dataset content). Gloss: "All categories
            # and phrases are fictional, produced by the synthetic generator."
            "note": "카테고리·문구는 전부 합성 생성기가 만든 가상의 것."}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    summary = {}
    for split, cap in caps.items():
        idx = lab.idx[split][:cap]
        post = lab.world.post[idx]
        # NOTE: hash() of a str is salted per process unless PYTHONHASHSEED is set,
        # so the rendered texts (not the labels) vary between runs by default.
        # Run with a fixed PYTHONHASHSEED for byte-reproducible output.
        texts = rt.render(post, seed=args.seed + hash(split) % 1000)

        labels, amb = targeted_pseudo_labels(lab, split)
        sel = np.arange(len(lab.idx[split]))[:cap]

        truth = lab.world.truth[idx]
        axis = lab.axis("eval", split)

        rows = []
        for j, i in enumerate(sel):
            pseudo = np.where(labels.mat[i])[0].tolist()
            # Put the top-voted label first (a set carries no order, so sort explicitly).
            pseudo.sort(key=lambda c: (c != labels.primary[i], c))
            rows.append({
                "id": int(idx[j]),
                "text": texts[j],
                # The prompt is not stored; it is always rebuilt from `text`.
                # (Copying the full category list into every row would inflate the file about 3x.)
                "completion": rt.build_completion(pseudo),
                "pseudo_labels": [rt.code(c) for c in pseudo],
                "grade": labeler.GRADE_NAMES[int(labels.grade[i])],
                "n_calls": int(labels.k[i]),
                "vote_dist": [round(float(v), 4) for v in labels.frac[i]],
                # --- The two fields below are evaluation only; the training script never reads them ---
                "truth": [rt.code(c) for c in np.where(truth[j])[0].tolist()],
                "outside_signal": int(axis.observed[sel[j]]),
            })
        dump(out / f"{split}.jsonl", rows)
        # summary.json keys are kept in Korean so the committed file stays byte-identical.
        # Glosses: multi-label rate / split-vote rate / mean number of calls /
        # pseudo-label accuracy against the true label.
        summary[split] = {
            "n": len(rows),
            "멀티라벨 비율": round(float(np.mean([len(r["pseudo_labels"]) >= 2 for r in rows])), 4),
            "표 갈린 비율": round(float(np.mean([r["grade"] != "agree" for r in rows])), 4),
            "평균 호출 수": round(float(np.mean([r["n_calls"] for r in rows])), 2),
            "pseudo-label 정확도(진짜 정답 대비)": round(
                float(np.mean([set(r["pseudo_labels"]) == set(r["truth"]) for r in rows])), 4),
        }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n→ {out}/{{train,val,test}}.jsonl")
    print("Example -----------------------------------------------")
    first = json.loads((out / "train.jsonl").read_text(encoding="utf-8").split("\n")[0])
    print(first["text"])
    print(" pseudo:", first["pseudo_labels"], "| grade:", first["grade"],
          "| truth:", first["truth"])


if __name__ == "__main__":
    main()
