**English** | [한국어](METHOD.ko.md)

# Building a non-circular scorecard

This document describes a procedure, not experimental results.
It covers how to build a scorecard outside the LLM pipeline without new human labeling.

---

## 1. Identifying what is being measured

Suppose labels are generated with an LLM (pseudo-labels), a small model is trained on them, and the
model is scored on an eval set built the same way. The resulting number then means the following.

> flat accuracy 0.92 = reproduces the output of that LLM at about the 92% level

This distinction has practical consequences, because the two quantities select different models as
the best ([e2](results/e2_circular_eval.md): rank correlation on the same-source eval set τ = −0.73).

Splitting the LLM error into two components explains why.

| Component | Example | Under repeated calls | Remains in pseudo-labels | Learned by small model |
|---|---|---|---|---|
| Random | Boundary decisions that vary between calls | Cancels out | No | No |
| Systematic | Assigning ambiguous items to 'Other'; favoring one side of a specific pair | Does not cancel | Yes | Yes |

The small model removes the random component by averaging and inherits the systematic component.
When scored against those pseudo-labels, the model that also reproduces the systematic component
ranks first, and its performance against the ground truth can be lower.

This is not an argument against self-consistency or LLM-as-a-judge.
Repeated calls do not cancel systematic bias, so an eval set built with these techniques must also be
treated as coming from the same source.

---

## 2. Requirements for an outside observation

Instead of new human labeling, we use observations that already exist in the system and did not
pass through the LLM. There are three requirements.

1. It must be causally independent of the LLM pipeline. LLM outputs must not affect its value.
2. It must have little missingness. If observations concentrate in certain regions, only those
   regions are measured.
3. It must partially correspond to the label taxonomy. A coarse or many-to-one correspondence is acceptable.

Example candidates:

- the top-level category the user selected at input time
- the team or queue to which the item was actually assigned and by which it was handled
- behavior logs recorded afterwards (repeat inquiries, churn, conversion, etc.)

These axes do not measure accuracy. They measure consistency, that is, whether the model output
contradicts the observation. Their absolute scores are therefore low (0.65–0.67 in this lab).

---

## 3. Procedure

### 3.1 Define the crosswalk from meaning

Write the mapping between top-level groups and label codes from definitions and names, not from
LLM label statistics. A mapping built from statistics brings the circularity back.

### 3.2 Exclude non-discriminative rows

If the axis maps the top two candidates to the same group, the axis carries no information for that
row. Including such rows only adds noise (`signals.discriminative_mask`).
Non-specific groups such as 'Other' or 'Unclassified' are also excluded.

### 3.3 Score the top-1 label, not the count

Scoring the whole set as correct if any label matches favors models that output many labels.
An outside signal structurally cannot score the label count, so the count is handled separately
([e5](results/e5_cardinality.md)).

### 3.4 Separate the correction axis from the evaluation axis

An outside signal can be used both for label correction (breaking ties on ambiguous rows) and for
scoring models. Using the same axis for both creates circularity.
If there is only one axis, split by axis rather than by row.

### 3.5 Check independence of bias

Being external is not sufficient. If the axis is biased in the same direction as the LLM, the
circularity returns. Example: values produced by automatic operational rules that the LLM has also learned.

Re-test after excluding automatically generated values, and check that the same conclusion holds on
the subset where humans were involved.
②' in [e2](results/e2_circular_eval.md) is such a case: τ falls from 1.00 to 0.33.

### 3.6 Do not report absolute scores

The absolute value of an outside signal reflects the coarseness of the axis, not model quality.
Report only pairwise comparisons between models, together with a paired test such as McNemar.
If the sample is too small, state that the difference is not significant.

---

## 4. Additional rules

### Report three or more metrics

A single overall-EM metric leads to misjudgments. Cases observed in this lab:

- Soft targets raise top-1 and multi-label-row EM while lowering overall EM ([e3](results/e3_soft_labels.md)).
- Maximizing overall EM alone leads to the optimum of never predicting multiple labels ([e5](results/e5_cardinality.md): 0.7279).
- The count predictor ties on overall EM but is more than three times better on multi-label-row EM.

Minimum set: overall EM, single-label-row EM, multi-label-row EM, top-1 accuracy.
For generative models, add format compliance (JSON parse success rate). If this value is below 0.99,
the premise for comparing accuracy does not hold.

### The threshold is an operational decision

The threshold that maximizes overall EM and the threshold that maximizes multi-label-row EM differ.
This is a choice, on the Pareto front, of which error type to accept; there is no single optimum.
Choose the threshold on val and use test only for reporting.

### Compare at the same operating point

Comparing two methods each at its own best threshold cannot be interpreted.
Match an observable quantity, such as the predicted multi-label rate, before comparing.

### Keep the raw votes

Storing only the final label loses the vote distribution.
Keeping it makes soft-target experiments, stratified analysis, and selection of items for re-calls
possible without retraining.

### Use the gate for routing, not as a probability

Converting vote counts into confidence probabilities does not give calibrated values.
Restricting the gate to routing into three paths, `agree / partial / review`, already gives enough separation
([e1](results/e1_consensus_gate.md): quality difference between grades 70.2pp).

### Bias independence of the judge

Adding an LLM-as-a-judge is neutral in itself. If the judge shares the bias of the labeling LLM,
it adds no information
([e1c](results/e1_consensus_gate.md): at the same ~59% coverage, judge-accepted items 0.794 < vote gate 0.935;
even the strictest judge reaches only 0.809, at 22% coverage).
For the judge to contribute, its input, its evidence, or its model family must differ.

---

## 5. Checklist

Check before reporting a performance improvement:

- [ ] Who produced the eval-set labels? Do they share a source with the model being evaluated?
- [ ] If so, does the document state whether the number is accuracy or reproduction?
- [ ] Is there a signal from outside the pipeline? Is that axis independent of the LLM's bias?
- [ ] Are three or more metrics reported? Are multi-label rows observed separately? Is format compliance included?
- [ ] Was the threshold chosen without using test?
- [ ] Is a paired test reported? Is non-significance stated where it applies?
- [ ] Are rejected attempts recorded as well?
