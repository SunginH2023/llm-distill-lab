**English** | [한국어](README.ko.md)

# llm-distill-lab

A record of experiments that reproduce, end to end, the cycle of training a small model on
LLM-generated pseudo-labels: from data construction to evaluation, in a synthetic world where
the ground truth is defined.

```
pip install numpy                  # numpy is the only dependency of lab/
python experiments/run_all.py      # under a minute on a laptop CPU (about 17 s on an Apple M1 laptop); output in results/RESULTS.md
```

## Related papers

- *Borrowed Accuracy: How Teacher-Labeled Test Sets Inflate Reported Performance in LLM-Labeled Pipelines*.
  IEEE BigData 2026 (Industry & Government Track), under review.
  e2 studies the same mechanism, a pseudo-labeled eval set that rewards imitation of the labeler,
  in a controlled synthetic world. The paper itself uses a public benchmark (GoEmotions);
  this repository does not reproduce the paper's numbers.
- *How Often Must You Call the Verifier? Budget Saturation in Multi-Label Cascades*. ICTC 2026, accepted.
  e4 asks a related question: how to allocate a fixed budget of LLM calls.

---

## 1. Background

In a classification task on a new domain, the label taxonomy and its boundary definitions are not
established in advance. Human labeling presupposes labeling guidelines, and writing the guidelines
presupposes looking at the data. At the start, neither is available.

LLM labeling breaks this dependency. Even when its accuracy is low, every item receives an initial
label, and the resulting distribution makes it possible to discuss splitting and merging categories.
From this point of view, a pseudo-label is not a final answer but an initial state.

Once a pseudo-label baseline exists, small amounts of verified labels can later be added as
incremental updates, without rebuilding from scratch. A baseline is also needed to measure where
verified labels give the largest gain.

---

## 2. Problem

The quality of pseudo-labels is not directly observed. The following techniques are used to compensate:

- calling the same model several times and checking agreement (self-consistency)
- reviewing labels with a separate model (LLM-as-a-judge)
- human review of items where agreement breaks
- using the vote distribution as a soft target
- extra calls restricted to ambiguous items

Verifying the effect of each technique requires ground truth. However, the eval-set labels are also
produced by the same LLM. The observed metric is therefore not accuracy against the truth but the
degree to which the model reproduces the labeling model. These two quantities can select different
models as the best.

---

## 3. Approach

We build a synthetic world with a defined ground truth (oracle), and score after the fact whether
decisions made without the oracle were correct. The oracle is used only for evaluation. It is
excluded from the training and label-refinement paths.

No real-world data is included. All numbers come from synthetic data generated with fixed seeds.

---

## 4. Experiments

| Exp. | Question | Result | Verdict |
|---|---|---|---|
| [e1](results/e1_consensus_gate.md) | Can the degree of agreement across repeated calls sort label quality in advance, without ground truth? | Unanimous band 0.972 vs. split-vote band 0.270 (70.2pp) | Supported |
| e1b | Does adding a cheap LLM and keeping only items where the two models agree improve quality? | Coverage drops to 54%; accuracy falls 0.792 → 0.784 | Rejected |
| e1c | Does reviewing labels with an LLM-as-a-judge improve quality? | Judge 0.794 vs. vote gate 0.935 at the same ~59% coverage; even the strictest judge reaches only 0.809 (at 22% coverage) | Rejected |
| [e2](results/e2_circular_eval.md) | Does an eval set built from pseudo-labels preserve the ranking of models? | That eval set τ = −0.73; signal from outside the pipeline τ = +1.00 | Supported |
| [e3](results/e3_soft_labels.md) | Does training on the vote distribution instead of the final label help? | Split-vote stratum top-1 +1.62pp, overall top-1 +1.03pp. Overall EM falls | Partially supported |
| [e4](results/e4_call_budget.md) | Under the same call budget, which allocation is better? | Targeted re-calls on 18,886 items > 3 calls on all 30,000 items (top-1 +1.81pp) | Supported |
| [e5](results/e5_cardinality.md) | Does a dedicated label-count predictor beat a probability-ratio gate? | Overall EM tied (0.686 → 0.688); multi-label-row EM 0.054 → 0.181 | Partially supported |

Verdicts: 3 Supported, 2 Partially supported, 3 Rejected (e1b, e1c, and H5b in e5).
Rejected attempts are recorded in the same format as the others.

---

## 5. Main result: rank reversal depending on where the eval labels come from

We train six small models with identical architecture and hyperparameters; only the training labels
differ. The same predictions are then scored with three scorecards.

| Scorecard | In practice | In the synthetic world |
|---|---|---|
| ① Eval set built from pseudo-labels | Available | Available |
| ② Signal from outside the pipeline (coarse, noisy) | Available | Available |
| ③ Ground truth | Not observable | Available |

Model names give the number of labeler calls used to build the training labels.
"+ fix" (short for "+ outside-signal fix") means that the labels of split-vote items were additionally
corrected with a separate outside axis, the category the user selected at input time. This is not the
axis that scorecard ② uses: ② scores against the routing-target assignment, so the correction and the
evaluation never share an axis (see `lab/pipeline.py` and [METHOD.md §3.4](METHOD.md#34-separate-the-correction-axis-from-the-evaluation-axis)).
Rankings produced by the three scorecards:

| Scorecard | 1st | 2nd | 3rd | 4th | 5th | 6th | τ vs. truth |
|---|---|---|---|---|---|---|---:|
| ① Pseudo-label eval set | 5 calls | 7 calls | 3 calls | 1 call | 5 calls + fix | 7 calls + fix | −0.73 |
| ② Outside signal | 7 calls + fix | 5 calls + fix | 1 call | 7 calls | 5 calls | 3 calls | +1.00 |
| ②' Outside signal (contaminated by the same bias) | 7 calls + fix | 5 calls + fix | 3 calls | 5 calls | 1 call | 7 calls | +0.33 |
| ③ Ground truth | 7 calls + fix | 5 calls + fix | 1 call | 7 calls | 5 calls | 3 calls | +1.00 |

- ① is negatively correlated with the ground-truth ranking (τ = −0.73). Choosing the deployed
  model by ① alone loses 3.28pp against the ground truth.
- ② has a low absolute score (0.67), but its ranking matches the ground truth.
  We do not report absolute scores from the outside signal; we use it only for relative comparison
  between models.
- The ① scores of the top models are very close (0.8279–0.8303 for the four models without the fix),
  and there are only six models and a single seed, so each τ carries uncertainty.
- ②' is a signal that comes from outside but is biased in the same direction as the labeling LLM.
  τ falls from 1.00 to 0.33. Being external is not sufficient; independence of bias must also be checked.

The cause becomes clear when the LLM error is split into two components.

| Error component | Example | Under repeated calls | Remains in pseudo-labels | Learned by small model |
|---|---|---|---|---|
| Random | Boundary decisions that vary between calls | Cancels out | No | No (averaged out) |
| Systematic | Assigning ambiguous items to 'Other'; favoring one side of a specific pair | Does not cancel | Yes | Yes |

The small model removes the random component by averaging and inherits the systematic component.
When scored against those pseudo-labels, the model that also reproduces the systematic component
therefore ranks first.

Rank correlation as a function of the systematic-bias strength β:

| Bias β | 0.0 | 0.4 | 0.8 | 1.1 | 1.6 |
|---|---:|---:|---:|---:|---:|
| ① Pseudo-label eval set τ | +0.33 | −0.33 | −1.00 | −1.00 | −1.00 |
| ② Outside signal τ | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |

At β = 0, the ranking from ① is not reversed (τ = +0.33, weakly positive). The reversal occurs only
when systematic bias is present, and real LLMs are unlikely to have β = 0.

---

## 6. Extension to the text domain (LoRA / QLoRA)

To keep compute cost low, e1–e5 run on vector representations.
Real LLMs take text as input. We therefore render the same world as Korean inquiry texts,
build SFT data with the same pseudo-label pipeline, and connect it to LoRA / QLoRA training.
→ [`finetune/`](finetune/README.md)

```
synthetic world → Korean text → LLM calls (votes + gate) → pseudo-labels
                                                              ↓
          score with three scorecards ← generation, JSON parsing ← LoRA / QLoRA SFT
```

Phrases are sampled in proportion to the posterior, not from the true label. For boundary cases,
expressions from two categories are mixed, so the text itself becomes ambiguous.

```bash
pip install -r finetune/requirements.txt
make data     # synthetic world → SFT JSONL
make smoke    # checks the full train / generate / score path (CPU, no downloads, about 26 s on an Apple M1 laptop)
```

`--smoke` builds, on the fly, a randomly initialized 2-layer Llama (about 0.25M parameters) and a
character-level tokenizer, so the full path runs without model downloads or a GPU:
tokenization → prompt masking → LoRA injection → training → adapter save and reload → generation →
JSON parsing → scoring.

Measured on the committed data in `finetune/data` (`make smoke`, train + eval about 26 s on an Apple M1 laptop):
loss 5.54 → 0.12, format compliance 1.000, ground-truth EM 0.156 (random guessing 0.083).
These numbers check that the pipeline runs; they are not classification performance.

---

## 7. Status

| Stage | Status |
|---|---|
| Validate pipeline and evaluation design in the synthetic world | Done (e1–e5) |
| LoRA / QLoRA training code and generation-based evaluation path | Done (verified with a CPU smoke run) |
| QLoRA (SFT + PEFT) training of real open-weight models | Done outside this repository (see below) |
| Search for the best point at which to add a small set of verified labels | Planned |

Training of real open-weight models was carried out in a separate, non-public environment.
Its data and results are not included in this repository.
If anything from that work is added here, it will be limited to reproducible settings and metric definitions.

### Next steps (planned, not yet done)

We plan to extend the same evaluation-reliability question to embodied AI: for example, scoring robot
trajectories in simulation (LIBERO, SimplerEnv) with LLM/VLM judges, and measuring when such scores
preserve the ranking of policies. This has not been carried out, and there are no results.

---

## 8. Origin of the experiments

The experiments target problems observed repeatedly while operating a large-scale Korean text
multi-class classification pipeline. The data, labels, and metrics of that pipeline cannot be
published, so only its structure was extracted and moved into a synthetic environment.

| Stage | Experiments | Question |
|---|---|---|
| Data construction | e1, e4, `finetune/build_sft_data.py` | How labels are generated; how to split the budget between human review and LLM calls |
| Model training | e3, `finetune/train_qlora.py` | Which LLM output to use as the training signal (hard / set / vote distribution) |
| Evaluation design | e2, e5, `finetune/evaluate.py` | How to demonstrate an improvement; which misjudgments a metric induces |

Every experiment is recorded as hypothesis → numbers → verdict (Supported / Partially supported / Rejected).

---

## 9. Observations

1. If the eval-set labels come from the same source as the model being evaluated, the resulting
   metric is closer to reproduction than to accuracy.
2. A noisy outside signal preserves the ranking better than a precise signal from the same source.
   This assumes the outside signal's bias is independent.
3. In the original production pipeline that motivated this lab, most of the improvement came from
   label quality rather than from changing the model. The lab fixes the learner, so it cannot test
   that claim itself; what it does show is that, under the same call budget, correcting the labels of
   existing data beat adding data (e4).
4. Average metrics absorb improvements in small subsets. Looking only at overall EM, the signal in
   e5 is not detected.
5. Three techniques with high prior expectations (cheap-LLM agreement e1b, LLM-as-a-judge e1c,
   count predictor e5) gave results different from what was expected.

---

## 10. Scope and limitations

- The learner in `lab/` is linear softmax regression. The conclusions apply to label quality and
  evaluation design, not to model architecture or training technique. The real LoRA / QLoRA path is
  separate, in `finetune/`, and what it verifies is that the pipeline runs, not classification performance.
- `lab/` contains no text. Prompt sensitivity, output-format collapse, tokenization, and long context
  are covered only partially, in `finetune/`.
- The outside signal is defined as a noisy function of the ground truth. The τ = 1.00 in e2 is
  therefore a value under idealized conditions. Real outside signals have their own bias; that
  condition was measured as well, as ②' (contaminated signal), with τ = 0.33.
- In e5, the conclusion in the synthetic environment differed from the conclusion in practice.
  In the synthetic environment, the number of labels is determined by a rule on which pairs may
  co-occur, so the predictor has more signal to learn than in reality.
  The difference and its cause are documented in [the e5 report](results/e5_cardinality.md).

---

## 11. Layout

| Path | Contents |
|---|---|
| `lab/world.py` | Synthetic world with defined ground truth (overlapping boundary pairs, an 'Other' absorbing class, genuine multi-label items) |
| `lab/labeler.py` | Labeling LLM, consensus gate, LLM-as-a-judge. Errors are injected as separate random and systematic components |
| `lab/small_model.py` | numpy softmax regression (supports soft targets). No external dependencies |
| `lab/signals.py` | Signals from outside the pipeline. Separate axes for label correction and for evaluation |
| `lab/metrics.py` | Set EM, multi-label-row EM, micro-F1, McNemar |
| `lab/pipeline.py` | world → calls → gate → pseudo-labels → training → three scorecards |
| `experiments/` | e1–e5, run_all.py |
| `finetune/` | Korean text rendering, LoRA / QLoRA SFT, generation-based evaluation |
| `results/` | Outputs (md, json). RESULTS.md is generated automatically |
| `tests/` | 11 integrity checks on the experimental design |
| `METHOD.md` | Procedure for building a non-circular scorecard (a procedure document, not results) |

```bash
make all      # all experiments (about 17 s on an Apple M1 laptop)
make test     # tests (runs without pytest)
make e2       # a single experiment
make data     # generate SFT data
make smoke    # check the full LoRA path (about 26 s on an Apple M1 laptop)
```

The small model is fixed to linear softmax regression. Keeping the learner simple means that changes
in the metrics can be attributed to label quality, with less confounding from model variance.

---

## 12. Reproducibility

- `lab/` and `experiments/` depend only on numpy. No GPU, network access, or data download is needed.
- All randomness uses fixed seeds. Repeated runs on the same commit produce identical results in the
  same numpy/BLAS environment (`results/RESULTS.md` verified byte-identical on a fresh clone). Across
  environments, last-digit floating-point differences can appear; the rounded numbers in the reports
  are unchanged.
- Raw numbers are in `results/*.json` and interpretation in `*.md`. Every number in this document is
  taken from those files, except the smoke-run numbers in §6, which come from the `finetune/` smoke run
  (`finetune/runs/lora/eval.json`).

MIT License.
