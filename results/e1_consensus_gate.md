## e1 · Consensus gate and review budget

Hypothesis — Can the degree of vote disagreement alone pre-sort label quality?

Labeling LLM called 5 times. Grades are assigned from the vote distribution alone — the oracle is used only for scoring.

| Grade | Rows | Share | Primary accuracy | Mean ambiguity |
|---|---:|---:|---:|---:|
| agree | 8771 | 0.439 | 0.9723 | 0.1316 |
| partial | 9555 | 0.478 | 0.7168 | 0.3750 |
| review | 1654 | 0.083 | 0.2703 | 0.4766 |

Unanimous band 0.972 vs split-vote band 0.270 — a 70.2pp gap. Even without converting vote counts into probabilities, using them only as a routing gate separates quality this much.

Spending the review budget in gate order vs at random (assuming reviewed rows are replaced with the true label):

| Review share | Before review | Gate first | Random | Δ |
|---|---:|---:|---:|---:|
| 5% | 0.7920 | 0.8282 | 0.8025 | 0.0257 |
| 10% | 0.7920 | 0.8592 | 0.8132 | 0.0460 |
| 20% | 0.7920 | 0.8994 | 0.8346 | 0.0648 |
| 40% | 0.7920 | 0.9595 | 0.8772 | 0.0823 |

H1b — call one more cheap model and keep only rows where 'the two agree'. A fair comparison is against the same-model repeated-call gate at the same coverage.

| Filter | Rows kept | Coverage | Primary accuracy |
|---|---:|---:|---:|
| Main LLM alone | 19980 | 1.0000 | 0.7920 |
| Cheap LLM alone | 19980 | 1.0000 | 0.4445 |
| Keep only rows where both models agree | 10720 | 0.5365 | 0.7840 |
| Main LLM unanimous rows only | 8771 | 0.4390 | 0.9723 |

Rows where the two models agree but both are wrong: 11.6%. The cheap LLM does not independently catch the main LLM's errors, but it does succeed at rejecting answers the main LLM got right — among rows dropped by the agreement filter, the share the main LLM had right is 0.801.

H1c — now LLM-as-a-judge. Instead of having the cheap LLM choose a new label, we only ask 'is this label correct?' and keep the accepted rows. A fair comparison with the gate requires matching coverage.

| Filter | Rows kept | Coverage | Primary accuracy |
|---|---:|---:|---:|
| Judge-accepted only (strictness=0) | 11691 | 0.5851 | 0.7936 |
| Judge-accepted only (strictness=1) | 7747 | 0.3877 | 0.8003 |
| Judge-accepted only (strictness=2) | 4315 | 0.2160 | 0.8090 |
| (Reference) same coverage 59% taken in gate order | 11691 | 0.5851 | 0.9351 |

At the same 59% coverage, the judge's accepted rows reach 0.794 while selecting in gate order gives 0.935. No matter how strict the judge is made, accuracy of the accepted rows stalls around 0.809 (at 22% coverage). The reason is simple — the judge gets confused in the same places as the labeling LLM. If it shares both the tendency to push items into the catch-all class and the tendency to pick one side of a confusable pair, the judge adds no information about those errors. For a judge to be worth its cost, it has to base its decision on something different from the labeling LLM.

Verdict — ✅ Supported. H1 supported. Grade alone separates quality by 70.2pp, and under the same review budget, gate-first beats random.

Verdict — ❌ Rejected. H1b rejected. Cross-model consensus loses a lot of coverage, and its accuracy gain is smaller than the same-model repeated-call gate. Adding the cheap LLM to the consensus also rejects answers the main LLM got right.

Verdict — ❌ Rejected. H1c rejected. At the same 59% coverage, LLM-as-a-judge accepted rows 0.794 < gate 0.935; even the strictest judge reaches only 0.809 (at 22% coverage). What matters is not attaching a judge per se but what the judge looks at. A judge that shares the same bias only adds call cost.
