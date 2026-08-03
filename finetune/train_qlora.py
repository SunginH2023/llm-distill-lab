"""pseudo-label 로 작은 모델을 LoRA / QLoRA SFT 한다.

    # 다운로드·GPU 없이 전 경로 검증 (CPU, 수십 초)
    python finetune/train_qlora.py --smoke

    # 실제 모델로 4bit QLoRA
    python finetune/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --load-4bit \
        --epochs 1 --bs 4 --grad-accum 4 --lr 2e-4 --lora-r 32

학습 대상은 정답 구간뿐. 프롬프트 토큰은 라벨을 -100 으로 마스킹한다.
이걸 빠뜨리면 모델이 프롬프트(카테고리 목록·규칙)를 외우는 데 용량을 쓴다.
손실은 예쁘게 떨어지는데 정작 분류는 안 되는, 눈치채기 어려운 실패.

이 스크립트는 `truth` 필드를 읽지 않는다. 학습에 쓰는 건 pseudo-label 뿐이다.
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from finetune import render_text as rt
from finetune.tiny_model import CharTokenizer, HFTokenizerAdapter, build_tiny_causal_lm

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SFTDataset(Dataset):
    """프롬프트는 마스킹, 정답만 학습."""

    def __init__(self, rows, tok, max_len: int, soft_weight: bool = False):
        self.ex = []
        for r in rows:
            p = tok.encode(rt.build_prompt(r["text"]))
            c = tok.encode(" " + r["completion"]) + [tok.eos_id]
            ids = (p + c)[:max_len]
            labels = ([-100] * len(p) + c)[:max_len]
            # e1 의 등급을 학습 가중치로 쓸 수 있게 남겨 둔다 (기본은 균등).
            w = 1.0
            if soft_weight:
                w = {"agree": 1.0, "partial": 0.7, "review": 0.3}[r["grade"]]
            self.ex.append((ids, labels, w))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    w = torch.tensor([x[2] for x in batch], dtype=torch.float)
    for i, (a, b, _) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return ids, lab, att, w


def weighted_causal_loss(logits, labels, weights):
    """행별 가중치를 적용한 causal LM 손실.

    HF 기본 손실은 배치 전체를 한 번에 평균 내서 행 가중치를 걸 수 없다.
    e1 의 등급(agree/partial/review)을 학습 가중치로 실험하려면 직접 써야 한다.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    ls = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="none").view(labels.shape)
    mask = (labels != -100).float()
    per_row = (ls * mask).sum(1) / mask.sum(1).clamp(min=1)
    return (per_row * weights).sum() / weights.sum().clamp(min=1e-6)


def load_jsonl(path: _p.Path, limit: int | None = None):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def build_model(args, tok):
    if args.smoke:
        model = build_tiny_causal_lm(len(tok))
        print(f"  [smoke] 무작위 초기화 tiny-Llama, "
              f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        return model

    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if args.load_4bit:
        # QLoRA: 4bit NF4 + double quant, 계산은 bf16 으로.
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpoint)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="finetune/data")
    ap.add_argument("--out", default="finetune/runs/lora")
    ap.add_argument("--model", default=None, help="HF 모델 id (미지정 시 --smoke 필요)")
    ap.add_argument("--smoke", action="store_true", help="다운로드·GPU 없이 전 경로 검증")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (CUDA 필요)")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grade-weight", action="store_true",
                    help="표가 갈린 행의 학습 가중치를 낮춘다 (e1 등급 활용)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.smoke and not args.model:
        ap.error("--model 을 주거나 --smoke 를 쓰세요")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data = _p.Path(args.data)
    train_rows = load_jsonl(data / "train.jsonl", args.limit or (192 if args.smoke else None))

    if args.smoke:
        corpus = [rt.build_prompt(r["text"]) + " " + r["completion"] for r in train_rows]
        tok = CharTokenizer.from_texts(corpus)
        args.max_len = min(args.max_len, 1024)
    else:
        from transformers import AutoTokenizer
        tok = HFTokenizerAdapter(AutoTokenizer.from_pretrained(args.model))

    ds = SFTDataset(train_rows, tok, args.max_len, soft_weight=args.grade_weight)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_id))
    print(f"  학습 {len(ds)}건 · 배치 {args.bs} × 누적 {args.grad_accum} "
          f"= 유효배치 {args.bs * args.grad_accum}")

    model = build_model(args, tok)
    # 스모크의 base 는 무작위 초기화라 어딘가 저장해 두지 않으면 평가 때 재현이 안 된다.
    # (처음에 이걸 빠뜨려서, 학습 손실은 0.13까지 떨어졌는데 생성은 전부 깨졌다.
    #  어댑터를 '다른 난수로 초기화된 base' 위에 얹고 있었던 것.)
    base_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                  if args.smoke else None)

    from peft import LoraConfig, get_peft_model
    # 스모크에서는 lm_head 도 같이 학습한다. base 가 무작위 초기화라 어댑터만으로는
    # 출력 토큰 분포 자체를 배울 수 없어서, 형식(JSON)조차 못 만든다.
    # 사전학습된 실제 모델에서는 필요 없다 (그래서 스모크에서만 켠다).
    extra = ["lm_head"] if args.smoke else None
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, modules_to_save=extra,
        bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={args.lora_r} · 학습 파라미터 {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    model.to(device)
    model.train()
    if args.grad_checkpoint:
        # 주의: DDP 와 같이 쓰면 reduction 훅과 충돌한다. 단일 프로세스에서만 켤 것.
        model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total_steps = (args.max_steps if args.max_steps > 0
                   else math.ceil(len(dl) / args.grad_accum) * args.epochs)
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    step, first_loss, last_loss, t0 = 0, None, None, time.time()
    done = False
    for ep in range(args.epochs):
        if done:
            break
        for i, (ids, lab, att, w) in enumerate(dl):
            ids, lab, att, w = ids.to(device), lab.to(device), att.to(device), w.to(device)
            out = model(input_ids=ids, attention_mask=att)
            loss = weighted_causal_loss(out.logits.float(), lab, w) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                cur = loss.detach().item() * args.grad_accum
                first_loss = cur if first_loss is None else first_loss
                last_loss = cur
                if step % 10 == 0 or step == 1:
                    print(f"  step {step:4d}/{total_steps}  loss {cur:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}")
                if args.max_steps > 0 and step >= args.max_steps:
                    done = True
                    break

    out_dir = _p.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    meta = {"model": args.model or "tiny-smoke", "smoke": args.smoke,
            "load_4bit": args.load_4bit, "lora_r": args.lora_r,
            "steps": step, "first_loss": first_loss, "last_loss": last_loss,
            "seconds": round(time.time() - t0, 1), "grade_weight": args.grade_weight,
            "n_train": len(ds), "device": device}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    if args.smoke:
        (out_dir / "char_vocab.json").write_text(json.dumps(tok.itos, ensure_ascii=False),
                                                 encoding="utf-8")
        torch.save(base_state, out_dir / "tiny_base.pt")
    print(f"\n  손실 {first_loss:.4f} → {last_loss:.4f}  ({time.time() - t0:.1f}s, {device})")
    print(f"  → {out_dir}")


if __name__ == "__main__":
    main()
