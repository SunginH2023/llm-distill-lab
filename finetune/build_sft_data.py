"""합성 세계 + pseudo-label 파이프라인 → SFT 학습 데이터(JSONL).

앞의 실험들이 벡터 위에서 내린 결론을 그대로 텍스트/LLM 쪽으로 옮긴다.

    e1  → 표가 갈린 행은 `grade` 로 표시해 둔다 (검수 큐 / 재호출 대상)
    e3  → 투표 분포를 `vote_dist` 로 같이 저장한다 (소프트 타깃 실험용)
    e4  → 애매한 행에만 호출을 더 쓴 pseudo-label 을 기본값으로 쓴다
    e2  → 평가셋에는 외부 신호(outside_signal)와 진짜 정답을 같이 넣어 둔다

`truth` 필드는 평가 전용. 학습 스크립트는 이 필드를 읽지 않는다
(합성 실험에서 오라클이 학습 경로로 새면 전부 무의미해지므로).

    python finetune/build_sft_data.py --out finetune/data --n-train 4000
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
import lab as _lab  # noqa: F401  (numpy 임포트 전에 BLAS 스레드 고정)

import argparse
import json

import numpy as np

from finetune import render_text as rt
from lab import labeler, pipeline


def targeted_pseudo_labels(lab, split: str, base_k: int = 3, extra_k: int = 4):
    """e4 의 결론을 그대로: 전량 base_k 회 호출 + 표가 갈린 행에만 extra_k 회 더."""
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
            # 외부 신호(평가 전용) 크로스워크: 카테고리 → 굵은 그룹
            "outside_signal_group_of": [int(g) for g in ax_eval.group_of],
            "outside_signal_noise": ax_eval.noise,
            "note": "카테고리·문구는 전부 합성 생성기가 만든 가상의 것."}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    summary = {}
    for split, cap in caps.items():
        idx = lab.idx[split][:cap]
        post = lab.world.post[idx]
        texts = rt.render(post, seed=args.seed + hash(split) % 1000)

        labels, amb = targeted_pseudo_labels(lab, split)
        sel = np.arange(len(lab.idx[split]))[:cap]

        truth = lab.world.truth[idx]
        axis = lab.axis("eval", split)

        rows = []
        for j, i in enumerate(sel):
            pseudo = np.where(labels.mat[i])[0].tolist()
            # 1등이 항상 앞에 오게 (집합은 순서를 담지 않는다 → 명시적으로 정렬)
            pseudo.sort(key=lambda c: (c != labels.primary[i], c))
            rows.append({
                "id": int(idx[j]),
                "text": texts[j],
                # 프롬프트는 저장하지 않는다 — text 로부터 항상 재생성된다.
                # (행마다 카테고리 목록을 통째로 복사하면 파일이 6배로 불어난다.)
                "completion": rt.build_completion(pseudo),
                "pseudo_labels": [rt.code(c) for c in pseudo],
                "grade": labeler.GRADE_NAMES[int(labels.grade[i])],
                "n_calls": int(labels.k[i]),
                "vote_dist": [round(float(v), 4) for v in labels.frac[i]],
                # --- 아래 둘은 평가 전용. 학습 스크립트는 읽지 않는다 ---
                "truth": [rt.code(c) for c in np.where(truth[j])[0].tolist()],
                "outside_signal": int(axis.observed[sel[j]]),
            })
        dump(out / f"{split}.jsonl", rows)
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
    print("예시 --------------------------------------------------")
    first = json.loads((out / "train.jsonl").read_text(encoding="utf-8").split("\n")[0])
    print(first["text"])
    print(" pseudo:", first["pseudo_labels"], "| grade:", first["grade"],
          "| 진짜:", first["truth"])


if __name__ == "__main__":
    main()
