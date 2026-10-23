"""A lab that runs the full cycle of training a small model on LLM-assigned
pseudo-labels, end to end, inside a synthetic world where the ground truth is known.

Note: the BLAS thread count is pinned before numpy is imported.

Matrix products in this lab are extremely flat, e.g. (n=30000, k=24, m=12).
On many-core machines OpenBLAS splits such shapes across 32 threads, and the
threading overhead exceeds the compute itself; measured: 33ms -> 2ms (16x).
This came from profiling first, instead of concluding "training is slow" and cutting steps.
"""

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__all__ = ["world", "labeler", "small_model", "signals", "metrics", "report", "pipeline"]
