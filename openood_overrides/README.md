# OpenOOD postprocessor overrides

The formal experiments use OpenOOD commit `3c35632ee91b54b09d1f085d04f94744cece7d0b` plus three local setup/scoring overrides.

The changes were introduced because the direct implementations retain large feature tensors or evaluate class scores in Python loops, which is impractical for full ImageNet-scale reproduction on a 24 GB rental GPU.

- `dice_postprocessor.py` computes the same global feature mean by streaming batch sums.
- `she_postprocessor.py` computes correctly classified per-class feature means by streaming sums and counts.
- `rmds_postprocessor.py` computes biased covariance scatter matrices by parallel streaming updates and evaluates relative Mahalanobis class scores in vectorized form.

`tests/test_rmds_vectorization.py` checks vectorized RMDS scores and streaming statistics against direct tensor computations. `audit_extended_baseline_reproduction.py` records the configuration and source hashes used by the paper.

These files remain derivative components of OpenOOD and are subject to the upstream project’s license.

