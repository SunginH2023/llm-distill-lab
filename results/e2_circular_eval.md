## e2 · A circular scorecard flips the ranking

Hypothesis — Scoring with labeling-LLM pseudo-labels / with an outside-the-pipeline signal / with the true labels — do the three give the same ranking?

Six small models. Architecture and hyperparameters are identical; only the training labels differ. The outside-signal score is measured only on rows where that axis distinguishes the top two candidates (axis 'routing-target assignment', 7 groups, observation noise 25%).

| Small model (training labels) | ① Pseudo-label eval set | ② Outside signal | ③ True labels | Pseudo EM | True EM |
|---|---:|---:|---:|---:|---:|
| 1 call | 0.8279 | 0.6542 | 0.7888 | 0.6906 | 0.6514 |
| 3 calls | 0.8285 | 0.6452 | 0.7661 | 0.6934 | 0.6509 |
| 5 calls | 0.8303 | 0.6507 | 0.7796 | 0.6990 | 0.6597 |
| 7 calls | 0.8287 | 0.6515 | 0.7806 | 0.7021 | 0.6671 |
| 5 calls + outside-signal fix | 0.8260 | 0.6663 | 0.8045 | 0.6945 | 0.6877 |
| 7 calls + outside-signal fix | 0.8238 | 0.6708 | 0.8124 | 0.6965 | 0.6992 |

Extracting only the rankings (leftmost is first):

| Scorecard | Ranking produced | Rank correlation τ with the truth |
|---|---:|---:|
| ① Eval set built from pseudo-labels (same source) | 5 calls > 7 calls > 3 calls > 1 call > 5 calls + outside-signal fix > 7 calls + outside-signal fix | -0.7333 |
| ② Signal from outside the pipeline | 7 calls + outside-signal fix > 5 calls + outside-signal fix > 1 call > 7 calls > 5 calls > 3 calls | 1.0000 |
| ②' Outside signal contaminated with the same bias | 7 calls + outside-signal fix > 5 calls + outside-signal fix > 3 calls > 5 calls > 1 call > 7 calls | 0.3333 |
| ③ Truth (oracle) | 7 calls + outside-signal fix > 5 calls + outside-signal fix > 1 call > 7 calls > 5 calls > 3 calls | 1.0000 |

②' is an axis that comes from outside the pipeline but is biased in the same direction as the labeling LLM (35% of observations are pulled toward the group the labeling LLM favors — the situation where the labeling LLM has also learned an operational auto-routing rule). τ drops from 1.00 to 0.33. Coming from an outside source is not sufficient on its own; one must also check that it is independent of the labeling LLM's bias. This is why, when the routing axis was used in practice, it had to be re-tested with the auto-routed items removed.

The pseudo-label eval set picks 5 calls as first; the true first is 7 calls + outside-signal fix. Choosing a model by this eval set alone leaves 3.28pp on the table against the truth. The outside signal's absolute score is low at 0.671 — because the axis is inherently coarse — but its ranking is right. Do not report its absolute score; use it only for relative comparison. Caveat: the ① scores of the models without the fix are very close (0.8279–0.8303), and there are only 6 models and a single seed, so each τ carries uncertainty.

Increasing the strength of the labeling LLM's systematic bias from 0, we re-measure the rank correlation of both scorecards. With bias 0 the pseudo-label eval set does not reverse the ranking (τ = +0.33, weakly positive) — the reversal appears only when there is bias. (Real LLMs are unlikely to have bias 0.)

| Bias β | ① Pseudo-label eval set τ | ② Outside signal τ | Mean true top1 |
|---|---:|---:|---:|
| 0.0000 | 0.3333 | 1.0000 | 0.8715 |
| 0.4000 | -0.3333 | 1.0000 | 0.8535 |
| 0.8000 | -1.0000 | 1.0000 | 0.8206 |
| 1.1000 | -1.0000 | 1.0000 | 0.7911 |
| 1.6000 | -1.0000 | 1.0000 | 0.7443 |

Verdict — ✅ Supported. H2 supported. Rank correlation of the pseudo-label eval set τ=-0.73, outside signal τ=1.00. The model ranked first by labeling-LLM pseudo-labels was not first against the truth. A coarse axis with low absolute scores is a better scorecard than a circular axis with high absolute scores.
