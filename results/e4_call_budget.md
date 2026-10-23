## e4 · Same LLM call budget, different allocations

Hypothesis — Should 90,000 calls be spent 'shallow and wide' or 'deep on split rows'?

Training pool 30,000 rows, budget 90,000 calls. The four left columns are the quality of the pseudo-labels themselves; the two right columns are the true-label scores of the small model trained on those pseudo-labels.

| Allocation | Calls spent | Rows labeled | Pseudo-label top1 | Pseudo-label EM | Multi rate | Small model true top1 | Small model true EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 calls × 30,000 rows | 90000 | 30000 | 0.7711 | 0.7148 | 0.0000 | 0.7650 | 0.6490 |
| 5 calls × 18,000 rows | 90000 | 18000 | 0.7874 | 0.6737 | 0.2059 | 0.7816 | 0.6611 |
| 7 calls × 12,857 rows | 89999 | 12857 | 0.8026 | 0.7175 | 0.1085 | 0.7805 | 0.6668 |
| 3 calls + 4 on split rows × 18,886 rows | 89994 | 18886 | 0.8013 | 0.7177 | 0.1004 | 0.7831 | 0.6668 |

The winning allocation is 3 calls + 4 on split rows × 18,886 rows. Versus 3 votes on every row: true top1 +1.81pp / EM +1.78pp. It actually labels fewer rows — meaning that fixing the labels of rows already paid for was cheaper than buying more rows.

Fixing the number of rows and increasing only the number of calls (ignoring the budget):

| Calls | Pseudo-label top1 | Pseudo-label EM | Pseudo-label multi rate | Unanimous share |
|---|---:|---:|---:|---:|
| 1 | 0.7284 | 0.6725 | 0.0000 | 1.0000 |
| 3 | 0.7711 | 0.7148 | 0.0000 | 0.5570 |
| 5 | 0.7871 | 0.6714 | 0.2065 | 0.4377 |
| 7 | 0.8025 | 0.7151 | 0.1109 | 0.3787 |

Label selection (top1) improves monotonically with more votes, but exact match fluctuates — because the multi rate overshoots the true rate (0.113) and then comes back. In other words, 'how many votes to use' touches two problems at once: label selection and label count. Choosing the number of calls by exact match alone makes the two effects cancel out, so nothing is visible (→ e5).

Verdict — ✅ Supported. H4 supported. Under the same budget, targeted re-voting won (3 calls + 4 on split rows × 18,886 rows). But the gain comes not from 'spending many votes' but from 'not spending votes on unanimous rows'. Without the gate (e1), this allocation is not possible at all.
