"""Generate with the trained adapter -> parse JSON -> score against three references.

Applies the e2 conclusion to the real LLM path: each prediction is scored three times.

    (1) pseudo-label eval set -- the only reference available in practice
    (2) outside signal        -- one that can be built if needed (here: category -> coarse group + noise)
    (3) true label            -- exists only because the data are synthetic; used to check
                                 whether the three references disagree

The format compliance rate (JSON parse success) is reported too. For generative
models, accuracy figures become meaningless once this drops below 0.99.

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
    """Extract label codes from a generation. Returns None on failure (counted as a format violation)."""
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
        for j, e in enumerate(enc):        # left padding: generation continues from the right end
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
        # Restore the exact base used in training (random init cannot be matched by seed alone).
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
    # Score the outside signal on the top-1 label only (scoring the whole set would favor over-prediction).
    valid = top1 >= 0
    outside = np.zeros(len(rows), dtype=bool)
    outside[valid] = group_of[top1[valid]] == observed[valid]
    # Placeholder for excluding rows where this axis cannot separate the top candidates
    # (currently all rows are kept).
    discriminative = np.ones(len(rows), dtype=bool)

    result = {
        "n": len(rows),
        "format_ok (JSON parse rate)": float(ok_format.mean()),
        "(1) pseudo-label eval set EM": float(exact_match(pred, pseudo).mean()),
        "(2) outside-signal agreement (top1)": float(outside[discriminative].mean()),
        "(3) true-label EM": float(exact_match(pred, truth).mean()),
        "(3) true-label top1": float(np.mean([
            t >= 0 and truth[i, t] for i, t in enumerate(top1)])),
        "predicted multi-label rate": float((pred.sum(axis=1) >= 2).mean()),
        "pseudo-label EM vs true label": float(exact_match(pseudo, truth).mean()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok_format.all():
        bad = [g for g, p in zip(gen, parsed) if p is None][:3]
        print("\nformat violation examples:", json.dumps(bad, ensure_ascii=False)[:400])

    out = _p.Path(args.out or (_p.Path(args.adapter) / "eval.json"))
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
