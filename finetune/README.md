**English** | [한국어](README.ko.md)

# finetune: LoRA / QLoRA training on pseudo-labels

Experiments e1-e5 run on vector representations. Whether the same conclusions
hold on an actual LLM training path needs a separate check. This folder is that
bridge.

```
synthetic world -> Korean text -> labeling LLM (votes + gate) -> pseudo-labels
                                                                    |
                                                                    v
         scored against 3 references <- generation + JSON parsing <- LoRA / QLoRA SFT
```

The labeling LLM is the simulated labeler in `lab/labeler.py`; no external API is called.

## Quick start

```bash
pip install -r finetune/requirements.txt

# build the data (~1 s)
PYTHONHASHSEED=0 python finetune/build_sft_data.py
# verify the full path without downloads or a GPU (CPU, train + eval about 26 s on an Apple M1 laptop)
python finetune/train_qlora.py --smoke \
    --epochs 8 --bs 8 --grad-accum 1 --lr 8e-3 --limit 256
python finetune/evaluate.py --smoke --limit 64
```

`make smoke` runs the last two commands, and `make data` runs the first one.

With a real model:

```bash
python finetune/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --load-4bit \
    --epochs 1 --bs 4 --grad-accum 4 --lr 2e-4 --lora-r 32 --grad-checkpoint
python finetune/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter finetune/runs/lora
```

## Why the data are in Korean

The synthetic inquiry texts, the 12 category names, and the classification prompt
are intentionally Korean. They are dataset content, not documentation: the strings
in `render_text.py` are rendered into `finetune/data/*.jsonl`, stored in `meta.json`,
and fed to the model verbatim. They are therefore kept byte-identical, with English
glosses in the source comments. Code, comments, and console output are in English.

## What `--smoke` verifies

Training code that is never executed reveals its bugs only at real run time.
Downloading a multi-GB checkpoint for every check is wasteful, though.

`--smoke` builds a randomly initialized two-layer Llama (about 0.25M parameters; the exact count depends on the character vocabulary) and a
character-level tokenizer on the fly, and runs the full path on CPU.

| Verified | Not verified |
|---|---|
| JSONL loading -> prompt/answer tokenization | Classification performance of a real model |
| -100 masking of the prompt span | 4-bit quantization path (needs CUDA) |
| LoRA injection / trainable-parameter ratio | Large-batch and distributed training |
| Training loop, loss decrease, scheduler | Format collapse on long contexts |
| Adapter save -> reload -> generation | |
| JSON parsing and scoring against three references | |

Measured on the committed data in `finetune/data` (CPU, 256 training rows, 8 epochs, 64 test rows;
`make smoke` train + eval about 26 s on an Apple M1 laptop; torch 2.14, transformers 5.20, peft 0.21):

| Metric | Value | Note |
|---|---:|---|
| Training loss | 5.54 -> 0.12 | first vs. last optimizer step (single-batch values) |
| Format compliance (JSON parse) | 1.000 | |
| (1) EM on the pseudo-label eval set | 0.234 | |
| (2) Outside-signal agreement (top-1) | 0.266 | |
| (3) True-label EM | 0.156 | random guessing: 1/12 ≈ 0.083 |
| EM of the pseudo-labels themselves | 0.672 | reference ceiling set by the training labels |

The true-label EM of a ~0.25M randomly initialized model is about twice the
random-guess level. These numbers confirm that the pipeline works; they are not
a measure of classification performance. The pseudo-label EM (0.672) is reported
alongside as a reference ceiling imposed by the training labels.

## Implementation notes

- If prompt masking is omitted, the model spends capacity memorizing the category
  list and rules. The loss still falls normally but classification does not improve.
  Because this is hard to detect, it is documented in a comment on `SFTDataset`.
- An adapter is valid only together with its base weights. The smoke base is
  randomly initialized, so if it is not saved but rebuilt at evaluation time, the
  adapter ends up on a base drawn from different random numbers. Training loss then
  still drops to 0.13 while every generation is invalid (see the comment in
  `train_qlora.py`). This cannot happen with a pretrained model, but it is a
  candidate cause whenever training looks healthy and generation does not.
- Category codes have a fixed width of two digits. Many LLM tokenizers split
  numbers digit by digit, so mixing `3` and `11` makes the generation length vary
  and raises the parse-failure rate.
- The per-row weighted loss is implemented by hand. The default HF loss averages
  over the batch, so per-row weights cannot be applied. Using the e1 grade
  (agree/partial/review) as a training weight (`--grade-weight`) requires a custom loss.
- Gradient checkpointing conflicts with DDP reduction hooks; it is enabled only in
  a single process. In multimodal LoRA setups, parameters of unused towers are
  flagged as unused, which requires `ddp_find_unused_parameters=True`.

## Data format

One line of `finetune/data/*.jsonl`:

```json
{"id": 1, "text": "확인 부탁드려요. 아이디를 입력해도 넘어가지 않아요 로그인이 안 됩니다 ...",
 "completion": "{\"labels\": [\"05\"]}",
 "pseudo_labels": ["05"], "grade": "partial", "n_calls": 7,
 "vote_dist": [0.0, ...],
 "truth": ["05"], "outside_signal": 3}
```

- `pseudo_labels` / `grade` / `n_calls` / `vote_dist`: used for training; produced
  only by the (simulated) labeling-LLM calls.
- `truth` / `outside_signal`: evaluation only. The training script never reads
  these fields. If the true labels leaked into the training path of a synthetic
  experiment, the results could not be interpreted.
- `prompt` is not stored; it is rebuilt from `text` (storing the full category list
  in every row would roughly triple the file size).
- `summary.json` keeps Korean keys (multi-label rate, split-vote rate, mean number
  of calls, pseudo-label accuracy against the true label) so the committed file
  stays byte-identical to the generator output.

### Why some rows look mislabeled

Some rows read as if `text` and `truth` disagree, for example an obvious stock-out
question whose true label is something else, or a unanimous pseudo-label that is
plainly wrong for the wording. This is expected. The labels, votes and truth are
all decided in the vector world (`lab/world.py`, `lab/labeler.py`); the simulated
labeling LLM never reads the text. The text is rendered afterwards by sampling
phrases in proportion to the posterior, so phrases from other categories mix in.
The text is therefore a vehicle for exercising the LoRA / QLoRA path end to end,
not an input from which the labels can be inferred, and the smoke numbers are
pipeline checks rather than classification results.

### Reproducibility

`build_sft_data.py` seeds the text renderer with `hash(split)`, and Python salts
string hashes per process. Without a fixed `PYTHONHASHSEED`, the labels and all
other fields are reproduced exactly, but the wording of the rendered `text` differs
between runs. Set `PYTHONHASHSEED` (e.g. `0`) for byte-identical output. The
committed `finetune/data` was generated before the seed was pinned, so data
regenerated with `PYTHONHASHSEED=0` (as `make data` does) has different `text`
wording than the committed files. The smoke numbers above were measured on the
committed data; on regenerated data the smoke loss can differ slightly (in our run
5.55 -> 0.18 instead of 5.54 -> 0.12).

## Note on categories and phrases

The 12 categories and all phrases in `render_text.py` are fictional strings produced
by the synthetic generator and are unrelated to any real service, organization, or
dataset. They only attach names to the structure of `world.py` (one absorbing class,
three boundary-overlap pairs, two co-occurrence pairs).
