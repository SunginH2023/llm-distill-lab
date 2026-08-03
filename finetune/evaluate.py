"""학습한 어댑터로 생성 → JSON 파싱 → 세 채점표로 채점.

e2 에서 얻은 결론을 실제 LLM 쪽에도 그대로 적용한다. 하나의 예측을 세 번 채점한다.

    ① pseudo-label 평가셋   — 실무에서 유일하게 손에 쥘 수 있는 것
    ② 외부 신호            — 만들려면 만들 수 있는 것 (여기선 카테고리→굵은 그룹 + 잡음)
    ③ 진짜 정답            — 합성이라 존재. 이 셋이 서로 다른 말을 하는지 확인용

형식 준수율(JSON 파싱 성공률)도 같이 찍는다. 생성 계열에서는 이게 0.99 아래로
떨어지는 순간 정확도 논의가 무의미해진다.

    python finetune/evaluate.py --adapter finetune/runs/lora --smoke --limit 200
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import numpy as np
import torch

from finetune import render_text as rt
from finetune.tiny_model import CharTokenizer, HFTokenizerAdapter, build_tiny_causal_lm

JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_labels(text: str, n_classes: int) -> list[int] | None:
    """생성 결과에서 라벨 코드를 뽑는다. 실패하면 None (형식 위반으로 집계)."""
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        codes = obj["labels"]
        out = [int(c) for c in codes if str(c).strip().isdigit()]
    except Exception:
        return None
    out = [c for c in out if 0 <= c < n_classes]
    return out[:2] if out else None


def to_matrix(list_of_codes, n_classes: int) -> np.ndarray:
    m = np.zeros((len(list_of_codes), n_classes), dtype=bool)
    for i, codes in enumerate(list_of_codes):
        for c in codes:
            m[i, c] = True
    return m


def exact_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a == b).all(axis=1)


@torch.no_grad()
def generate(model, tok, prompts, device, max_new=24, batch=8):
    outs = []
    model.eval()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = [tok.encode(p) for p in chunk]
        n = max(len(e) for e in enc)
        ids = torch.full((len(enc), n), tok.pad_id, dtype=torch.long)
        att = torch.zeros((len(enc), n), dtype=torch.long)
        for j, e in enumerate(enc):        # 좌측 패딩 — 생성은 오른쪽 끝에서 이어진다
            ids[j, n - len(e):] = torch.tensor(e)
            att[j, n - len(e):] = 1
        ids, att = ids.to(device), att.to(device)
        gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_id)
        for j in range(len(enc)):
            outs.append(tok.decode(gen[j, n:].tolist()))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="finetune/data")
    ap.add_argument("--adapter", default="finetune/runs/lora")
    ap.add_argument("--model", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = _p.Path(args.data)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    C = len(meta["classes"])
    group_of = np.array(meta["outside_signal_group_of"])

    rows = [json.loads(l) for l in
            (data / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:args.limit]

    if args.smoke:
        vocab = json.loads((_p.Path(args.adapter) / "char_vocab.json").read_text(encoding="utf-8"))
        tok = CharTokenizer(itos=vocab, stoi={c: i for i, c in enumerate(vocab)},
                            pad_id=0, eos_id=1)
        base = build_tiny_causal_lm(len(tok))
        # 학습 때 쓴 그 base 를 그대로 되살린다 (무작위 초기화라 시드만으로는 못 맞춘다)
        base.load_state_dict(torch.load(_p.Path(args.adapter) / "tiny_base.pt",
                                        map_location="cpu", weights_only=True))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = HFTokenizerAdapter(AutoTokenizer.from_pretrained(args.model))
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter)
    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    model.to(device)

    gen = generate(model, tok, [rt.build_prompt(r["text"]) for r in rows], device)
    parsed = [parse_labels(g, C) for g in gen]
    ok_format = np.array([p is not None for p in parsed])
    pred = to_matrix([p or [] for p in parsed], C)

    pseudo = to_matrix([[int(c) for c in r["pseudo_labels"]] for r in rows], C)
    truth = to_matrix([[int(c) for c in r["truth"]] for r in rows], C)
    observed = np.array([r["outside_signal"] for r in rows])

    top1 = np.where(pred.any(axis=1), np.argmax(pred, axis=1), -1)
    # 외부 신호는 1등 라벨만 채점한다 (집합 전체를 채점하면 많이 뱉는 쪽이 유리해진다)
    valid = top1 >= 0
    outside = np.zeros(len(rows), dtype=bool)
    outside[valid] = group_of[top1[valid]] == observed[valid]
    # 이 축이 상위 후보를 구분해 주지 못하는 행은 제외
    discriminative = np.ones(len(rows), dtype=bool)

    result = {
        "n": len(rows),
        "형식 준수율(JSON 파싱)": float(ok_format.mean()),
        "① pseudo-label 평가셋 EM": float(exact_match(pred, pseudo).mean()),
        "② 외부 신호 일관성(top1)": float(outside[discriminative].mean()),
        "③ 진짜 정답 EM": float(exact_match(pred, truth).mean()),
        "③ 진짜 정답 top1": float(np.mean([
            t >= 0 and truth[i, t] for i, t in enumerate(top1)])),
        "예측 멀티율": float((pred.sum(axis=1) >= 2).mean()),
        "pseudo-label 자체의 진짜 정답 EM": float(exact_match(pseudo, truth).mean()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok_format.all():
        bad = [g for g, p in zip(gen, parsed) if p is None][:3]
        print("\n형식 위반 예시:", json.dumps(bad, ensure_ascii=False)[:400])

    out = _p.Path(args.out or (_p.Path(args.adapter) / "eval.json"))
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
