# Reproducibility protocol

This document records the experimental boundary used by the manuscript. It is intended to prevent accidental test-set tuning when reproducing or extending the detector.

## 1. Three distinct data roles

1. **Setup data** construct anchors or class centroids from ID training data.
2. **Validation data** select a score rule and fusion weight. For ImageNet-1K, the 5,000-image ID validation subset is split class-wise into two tuning and three holdout images per class. The 1,763-image OpenImage-O validation set is deterministically divided into tuning and holdout halves.
3. **Test data** are read only after the selected configuration is serialized, hashed, and accepted by the validation gate.

ID-only mean and standard deviation estimates calibrate the score components. OOD validation scores select the rule and weight but do not set the affine normalization.

## 2. Locked endpoint

The selected endpoint is maximum class-centroid cosine similarity fused with MSP. The validation grid is

```text
alpha in {0.00, 0.05, ..., 1.00}
score = alpha * z(centroid cosine) + (1 - alpha) * z(MSP)
```

Candidates are ordered by validation AUROC, then FPR@95, then the smaller geometry weight. The selected rule uses `alpha = 0.8`. The decision SHA-256 recorded during the formal experiments is stored in `configs/locked_protocol.json`.

## 3. Randomness and reported variation

- CIFAR-10, CIFAR-100, and ImageNet-200 use three official OpenOOD checkpoint seeds.
- ImageNet-1K uses one fixed pretrained backbone; the three reported detector seeds vary the sampled centroid bank. They are not three independently trained ImageNet-1K classifiers.
- Mean and sample standard deviation are descriptive summaries. They are not confidence intervals or claims of statistical significance.

## 4. Primary metrics

AUROC and FPR@95 are primary. AUPR-IN and AUPR-OUT are retained in every formal run and are reported in detailed tables.

## 5. Exact and approximate endpoints

The trajectory score at `T = 0` is exactly the empirical static anchor-affinity field. Replacing that field by a class-centroid cosine score is an approximation controlled by within-class concentration and common-scale assumptions; it is not an algebraic identity for arbitrary class-dependent bandwidths or weights.

`run_static_reduction_bridge.py` audits this reduction in six pre-specified
stages. It reads only the ImageNet-1K ID validation split and OpenImage-O
validation split. The Near-OOD and Far-OOD test loaders are not constructed by
this runner. Stage S1 evaluates the exact all-class static field
(`candidate_k = 0`); it is intentionally distinct from the candidate-pruned
deployment score.

## 6. OpenOOD provenance

The evaluation code is checked out at:

```text
https://github.com/Jingkang50/OpenOOD.git
commit 3c35632ee91b54b09d1f085d04f94744cece7d0b
```

Three postprocessors are overridden to reduce setup memory without changing their intended statistics:

- DICE: streaming feature mean;
- SHE: streaming correctly classified class means;
- RMDS: streaming scatter matrices and vectorized relative Mahalanobis scoring.

The repository includes equivalence tests and a local audit script. This is an engineering-equivalence claim for the checked implementation, not a published-number reproduction claim.

## 7. Hardware timing

Efficiency comparisons use the same GPU, process, precision, batch size, warm-up policy, and number of repeats. Data loading is excluded from timed inference. Batch-one latency is the main deployment-oriented comparison.

## 8. CTM control and setup accounting

`run_ctm_baseline.py` reproduces the CTM class-mean cosine statistic under the
same local OpenOOD data and metric protocol. Formal CTM centroids use all ID
training features. Fixed ImageNet-1K backbones are evaluated once rather than
being relabelled as multiple independent seeds.

The paired efficiency runner separately records (i) the number of images
processed by the backbone during setup and (ii) the number of samples entering
the final centroid estimate. This distinction matters on CIFAR, where the
current implementation extracts the full training feature set before retaining
the class-balanced bank, and on ImageNet, where it processes only the bank.

## 9. CADRef and LogitGap controls

`run_cadref_logitgap.py` keeps the local OpenOOD v1.5 loaders, preprocessing,
backbones, checkpoints, and metric implementation fixed while changing only the
post-hoc score. The runner records the inspected upstream repository commits
and source hashes in every completion manifest.

- Formal CADRef-Energy estimates its raw class means and global mean training
  energy from the complete ID training split.
- Fixed LogitGap is training-free; its comparison-logit count follows the
  released fixed convention recorded in the runner.
- Neither method reads OOD test samples during setup or hyperparameter
  selection.
- CIFAR-10, CIFAR-100, and ImageNet-200 use three independently trained
  checkpoint seeds. Each fixed ImageNet-1K backbone is evaluated once.

Smoke runs cap both evaluation and CADRef setup samples and are compatibility
checks only. `summarize_cadref_logitgap.py` rejects incomplete or smoke-derived
formal matrices before producing the paper tables.
