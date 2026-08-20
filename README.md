# From Hamiltonian Flows to Class-Centroid Geometry

Official research code for **“From Hamiltonian Flows to Class-Centroid Geometry: A Mechanistic Study of Efficient Out-of-Distribution Detection.”**

This repository contains the Hamiltonian detector, its exact zero-step endpoint, the locked Centroid–MSP detector, OpenOOD benchmark runners, ablations, baseline reproductions, efficiency measurements, and table-generation scripts used in the manuscript.

## What the code tests

The project starts from a Hamiltonian hypothesis in normalized feature space and then subjects it to controlled falsification experiments.

- `hamiltonian_detector.py` implements five radial potentials, source weighting, bandwidth fitting, and projected leapfrog trajectories.
- `ood_experiment.py` runs CIFAR-10/CIFAR-100 experiments.
- `Imagenet_ood_experiment.py` runs ImageNet-200/ImageNet-1K OpenOOD experiments.
- `run_locked_imagenet1k_fusion.py` and `run_locked_centroid_transfer.py` implement the validation-locked Centroid–MSP endpoint.
- `run_journal_experiments.sh` reproduces the journal experiment matrix and efficiency audits.

The central empirical finding is that nonzero Hamiltonian propagation is not a scale-stable source of OOD separation in the tested construction. Its exact zero-step endpoint survives, admits a bounded class-centroid surrogate, and yields a low-cost detector when combined with MSP under a locked validation protocol.

## Repository status

The code is prepared for anonymous or public paper review. Datasets, pretrained checkpoints, feature caches, and raw server outputs are intentionally excluded. Download and verification utilities are included.

## Environment

The reported experiments used:

- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- torchvision 0.16.2
- OpenOOD evaluation code pinned to commit `3c35632ee91b54b09d1f085d04f94744cece7d0b`

Install a CUDA-compatible PyTorch build first, then install the remaining dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install -r requirements.txt
python -m pip install libmr==0.1.9 --no-build-isolation
bash scripts/setup_openood.sh
export PYTHONPATH="$PWD/OpenOOD:${PYTHONPATH:-}"
```

The setup script checks out the recorded OpenOOD commit and installs the three memory-equivalent postprocessor overrides used for DICE, SHE, and RMDS. See [`openood_overrides/README.md`](openood_overrides/README.md) for the audit boundary.

## Quick verification

```bash
python -m py_compile \
  hamiltonian_detector.py \
  ood_experiment.py \
  Imagenet_ood_experiment.py \
  openood_hamiltonian_postprocessor.py \
  prepare_openood.py

python -m pytest -q tests
bash -n run_journal_experiments.sh
```

## Data and checkpoints

Create persistent directories on the server:

```bash
export PROJECT="$PWD"
export DATA_ROOT="$PROJECT/data"
export OPENOOD_CKPT_ROOT="$PROJECT/openood_pretrained"
export OUTPUT_ROOT="$PROJECT/results_openood"
export LOG_ROOT="$PROJECT/logs"
export TORCH_HOME="$PROJECT/cache/torch"
mkdir -p "$DATA_ROOT" "$OPENOOD_CKPT_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$TORCH_HOME"
```

Prepare all four reported benchmarks with resumable downloads:

```bash
python -u prepare_openood.py \
  --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
  --data-root "$DATA_ROOT" \
  --results-root "$OPENOOD_CKPT_ROOT" \
  --archive-dir "$PROJECT/archives" \
  --download-backend auto \
  --keep-archives
```

ImageNet-1K itself must be obtained under the ImageNet terms. The preparation script does not redistribute it. AutoDL extraction and path-layout instructions are provided in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Reproduce the main studies

Run the smoke test first:

```bash
bash run_journal_experiments.sh smoke
```

Then run the formal stages independently so interrupted rental-server jobs can be resumed:

```bash
bash run_journal_experiments.sh trajectory_cifar
bash run_journal_experiments.sh mass_cifar
bash run_journal_experiments.sh trajectory_imagenet200
bash run_journal_experiments.sh mass_imagenet200
bash run_journal_experiments.sh trajectory_imagenet1k
bash run_journal_experiments.sh baselines_cheap
bash run_journal_experiments.sh baselines_heavy
bash run_journal_experiments.sh baselines_extended
bash run_journal_experiments.sh efficiency_b1
bash run_journal_experiments.sh efficiency_extended
bash run_journal_experiments.sh efficiency_scale
bash run_journal_experiments.sh required_summaries
```

Every stage writes explicit completion markers and upserts CSV rows, allowing safe restart without duplicating completed runs.

## Locked detector protocol

The paper endpoint is recorded in [`configs/locked_protocol.json`](configs/locked_protocol.json). In brief:

- geometry score: maximum cosine similarity to normalized class centroids;
- confidence score: MSP;
- fusion weight: `0.8` on geometry and `0.2` on MSP after ID-only standardization;
- centroid bank: 12 correctly indexed ID setup samples per class unless an ablation states otherwise;
- test access: the score rule and weight are hashed and locked before Near-/Far-OOD test evaluation.

The validation procedure and the distinction between tuning, ID calibration, and final test access are described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Output layout

The main scripts create:

```text
results/
results_openood/
results_openood_baselines/
logs/
```

These runtime directories are ignored by Git. Summary scripts convert the formal CSV files into mean±standard-deviation tables and LaTeX fragments used by the manuscript.

## Scope of the baseline claims

Baseline numbers are local engineering reproductions under a common OpenOOD loader, backbone, checkpoint, and metric implementation. They are not presented as independent replications of every published number in the original baseline papers. The RMDS vectorization and streaming-statistics equivalence tests are included in `tests/test_rmds_vectorization.py`.

## Citation

If this code is useful, please cite the accompanying manuscript. A machine-readable record is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@article{wang2026hamiltonian,
  title   = {From Hamiltonian Flows to Class-Centroid Geometry: A Mechanistic Study of Efficient Out-of-Distribution Detection},
  author  = {Wang, Zilin and Xiao, Lianghai},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## Third-party software and data

OpenOOD remains governed by its upstream license. CIFAR, ImageNet, and the OOD benchmark datasets remain governed by their respective terms. No third-party dataset or pretrained checkpoint is redistributed here.

## License

The original code in this repository is released under the [MIT License](LICENSE). The files under `openood_overrides/` remain derivative components of OpenOOD and are also subject to the upstream project license and notices.
