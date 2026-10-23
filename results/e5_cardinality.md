## e5 · The count problem: same experiment, conclusion flips with the metric

Hypothesis — Does a model that predicts the label count separately beat the one-line probability ratio?

The small model is fixed to a single model trained on pseudo-labels from 7 calls; only decoding changes. The count predictor was trained with the pseudo-label count as its target (no oracle used; accuracy against the true count 0.885).

The three scorers are compared at the same predicted multi rate (0.113, matched to the pseudo-label multi rate). Thresholds are set on val only; test is used only for reporting.

| Scorer (matched operating point) | t(val) | Overall EM(test) | Multi-row EM(test) | Predicted multi rate |
|---|---:|---:|---:|---:|
| Probability ratio p2/p1 | 0.6155 | 0.6861 | 0.0536 | 0.1218 |
| Count predictor P(multi) | 0.1636 | 0.6883 | 0.1810 | 0.1120 |
| Combined P(multi)×(p2/p1) | 0.0734 | 0.6924 | 0.0746 | 0.1192 |
| (Reference) never predict multi-label | - | 0.7279 | 0.0000 | 0.0000 |
| Count given (upper bound) | - | 0.7836 | 0.4854 | 0.1148 |

At the same multi rate, overall EM is probability ratio 0.6861 vs count predictor 0.6883 (McNemar p=0.376) — any difference is in the third decimal place. Yet multi-row EM widens more than 3x, 0.0536 → 0.1810 (p=6.2e-42). The predictor knows something about 'which rows are multi-label' that the probability ratio does not — in this world, whether a row is multi-label depends on whether the two candidates form an allowed co-occurring pair, and that information is not contained in the ratio of the top two probabilities. Why overall EM still does not move is simple: multi-label rows are only 11.5% of the total, so the averaged metric swallows the gain entirely.

> Caution: this is where the lab diverges from the real pipeline. In production, when count predictors with the same structure were trained in several variants, none ever beat the probability-ratio gate. In the synthetic world the 'count' is determined by a clean rule (allowed co-occurring pairs), so the predictor has more to learn than in reality. → This is a point where a synthetic experiment cannot replace a production conclusion; a synthetic experiment is useful only when you can explain why the results differ.

Pareto front observed directly on test while sweeping the probability-ratio threshold:

| t | Overall EM | Multi-row EM | Predicted multi rate |
|---|---:|---:|---:|
| 0.1000 | 0.5063 | 0.3223 | 0.4911 |
| 0.2000 | 0.5781 | 0.2229 | 0.3671 |
| 0.3000 | 0.6209 | 0.1627 | 0.2828 |
| 0.4000 | 0.6453 | 0.1143 | 0.2229 |
| 0.5000 | 0.6671 | 0.0785 | 0.1702 |
| 0.7000 | 0.6974 | 0.0397 | 0.0910 |
| 0.9000 | 0.7182 | 0.0100 | 0.0285 |

Overall EM peaks at 0.7279 at t=1.00, where multi-row EM is 0.0000. Multi-row EM peaks at 0.3223 at t=0.10, where overall EM drops to 0.5063. No single t maximizes both axes at once. So t is not 'a value obtained by tuning' but an operational decision about which kind of error to accept.

Verdict — △ Partially supported. H5a partially supported — a case where the conclusion flips depending on what is measured. Overall EM shows no difference (0.6861 → 0.6883), while multi-row EM clearly supports it (0.0536 → 0.1810, p=6.2e-42). Had we reported overall EM alone and written 'count predictor rejected', we would have discarded a real signal because of metric design.

Verdict — ❌ Rejected. H5b rejected. The threshold that is optimal for overall EM differs from the one optimal for multi-label rows, and maximizing overall EM alone converges to 'never predict multi-label' (0.7279). That is, t is not a value obtained by tuning but an operational decision about which kind of error to accept. The upper bound when the count is given, 0.7836, shows the remaining headroom.
