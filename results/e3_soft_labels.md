## e3 · Vote distribution as soft targets (KD)

Hypothesis — Does training on the vote distribution instead of the labeling LLM's final label improve the small model?

A single set of pseudo-labels from 5 labeling-LLM calls is trained in three target forms only. Data, architecture, and number of steps are all identical.

| Training target | True top1 | True EM | True multi-row EM | Circular top1 |
|---|---:|---:|---:|---:|
| A hard (primary one-hot) | 0.7704 | 0.6555 | 0.0645 | 0.8287 |
| B set (primary .75 / secondary .25) | 0.7796 | 0.6597 | 0.0894 | 0.8303 |
| C soft (raw vote distribution) | 0.7899 | 0.6468 | 0.1753 | 0.8310 |

The first thing that stands out is that soft targets lower overall EM (B 0.6597 → C 0.6468), even though top1 and multi-row EM went up in the same table. Soft targets leave mass on the runner-up label, so top-2 is opened more often, and overall EM loses when multi-label predictions increase (→ the Pareto front in e5). This is the first place where 'better/worse' depends on the metric.

Where did true top1 move when switching from B (set) to C (soft)? Stratification uses only the labeling LLM's votes on the test side — stratification must be possible without the oracle to transfer directly to production.

| Stratum | n | B set | C soft | Δ(pp) | McNemar p |
|---|---:|---:|---:|---:|---:|
| Unanimous stratum (agree) | 8771 | 0.9566 | 0.9593 | 0.2736 | 0.0009 |
| Split-vote stratum (partial+review) | 11209 | 0.6412 | 0.6574 | 1.6237 | 0.0000 |
| All | 19980 | 0.7796 | 0.7899 | 1.0310 | 0.0000 |

Overall gain +1.03pp, split-vote stratum +1.62pp. The direction is as predicted — the gain comes only where the labeling LLM wavered. But the absolute size of the overall gain is small, so it hardly justifies the cost of retraining on its own. The real value of soft targets lies not in the score but in the fact that preserving the vote distribution makes post-hoc experiments like this possible without re-querying the LLM.

Verdict — △ Partially supported. H3 partially supported. The direction and location of the gain match the hypothesis (split stratum +1.62pp), but the overall +1.03pp is not large. Adopt it, but do not claim 'this is what delivers the performance'.
