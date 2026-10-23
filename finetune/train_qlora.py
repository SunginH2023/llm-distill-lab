"""LoRA / QLoRA SFT of a small model on pseudo-labels.

    # Verify the full path without downloads or a GPU (CPU, tens of seconds)
    python finetune/train_qlora.py --smoke

    # 4-bit QLoRA on a real model
    python finetune/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --load-4bit \
        --epochs 1 --bs 4 --grad-accum 4 --lr 2e-4 --lora-r 32

Only the answer span is trained on; prompt tokens have their labels masked to -100.
Without this mask the model spends capacity memorizing the prompt (category list
and rules). The loss still drops nicely while classification does not improve,
a failure that is hard to notice.

This script never reads the `truth` field. Only pseudo-labels are used for training.
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
    """Mask the prompt; train only on the answer."""

    def __init__(self, rows, tok, max_len: int, soft_weight: bool = False):
        self.ex = []
        for r in rows:
            p = tok.encode(rt.build_prompt(r["text"]))
            c = tok.encode(" " + r["completion"]) + [tok.eos_id]
            ids = (p + c)[:max_len]
            labels = ([-100] * len(p) + c)[:max_len]
            # Optionally use the e1 grade as a training weight (uniform by default).
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
    """Causal LM loss with per-row weights.

    The default HF loss averages over the whole batch at once, so per-row weights
    cannot be applied. Using the e1 grade (agree/partial/review) as a training
    weight therefore requires a custom loss.
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
        print(f"  [smoke] randomly initialized tiny-Llama, "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        return model

    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if args.load_4bit:
        # QLoRA: 4-bit NF4 + double quantization, compute in bf16.
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
    ap.add_argument("--model", default=None, help="HF model id (required unless --smoke)")
    ap.add_argument("--smoke", action="store_true", help="verify the full path without downloads or a GPU")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (requires CUDA)")
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
                    help="down-weight rows with split votes (uses the e1 grade)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.smoke and not args.model:
        ap.error("pass --model or use --smoke")

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
    print(f"  train {len(ds)} rows · batch {args.bs} x accum {args.grad_accum} "
          f"= effective batch {args.bs * args.grad_accum}")

    model = build_model(args, tok)
    # The smoke base is randomly initialized, so it must be saved or it cannot be
    # reproduced at evaluation time. (An early version missed this: training loss
    # fell to 0.13 but every generation was broken, because the adapter was being
    # applied to a base initialized with different random numbers.)
    base_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                  if args.smoke else None)

    from peft import LoraConfig, get_peft_model
    # In smoke mode lm_head is trained as well. With a randomly initialized base the
    # adapter alone cannot learn the output token distribution, so the model fails
    # to produce even the JSON format. A real pretrained model does not need this,
    # hence it is enabled only for --smoke.
    extra = ["lm_head"] if args.smoke else None
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, modules_to_save=extra,
        bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={args.lora_r} · trainable params {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    model.to(device)
    model.train()
    if args.grad_checkpoint:
        # Caution: conflicts with DDP reduction hooks. Enable only in a single process.
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
    print(f"\n  loss {first_loss:.4f} → {last_loss:.4f}  ({time.time() - t0:.1f}s, {device})")
    print(f"  → {out_dir}")


if __name__ == "__main__":
    main()
