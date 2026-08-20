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

