#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   nohup bash run_server_experiments.sh > server_experiments.log 2>&1 &
# Environment overrides:
#   PYTHON=python3 DATA_ROOT=/mnt/data RESULTS_ROOT=/mnt/results
#   RUN_IMAGENET200=1 RUN_IMAGENET1K=1 SEEDS="0 1 2"
#   ARCHIVE_ROOT=/mnt/archives RUN_METHOD_ABLATIONS=1

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results_openood}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${SCRIPT_DIR}/archives}"
SEEDS="${SEEDS:-${SEED:-0}}"
RUN_IMAGENET200="${RUN_IMAGENET200:-1}"
RUN_IMAGENET1K="${RUN_IMAGENET1K:-1}"
RUN_METHOD_ABLATIONS="${RUN_METHOD_ABLATIONS:-0}"
N_ANCHORS="${N_ANCHORS:-5}"
SETUP_SAMPLES_PER_CLASS="${SETUP_SAMPLES_PER_CLASS:-12}"
HAM_EPOCHS="${HAM_EPOCHS:-10}"
N_STEPS="${N_STEPS:-10}"
DT="${DT:-0.05}"
CANDIDATE_K="${CANDIDATE_K:-20}"
SIM_BATCH="${SIM_BATCH:-32}"
HAM_BATCH_SIZE="${HAM_BATCH_SIZE:-16}"
WEIGHT_DOWNLOAD_BACKEND="${WEIGHT_DOWNLOAD_BACKEND:-auto}"
POTENTIALS=(gaussian laplacian cauchy imq matern32)
read -r -a SEED_LIST <<< "${SEEDS}"
BENCHMARKS=()
if [[ "${RUN_IMAGENET200}" == "1" ]]; then
  BENCHMARKS+=(imagenet200)
fi
if [[ "${RUN_IMAGENET1K}" == "1" ]]; then
  BENCHMARKS+=(imagenet1k)
fi
if [[ "${#BENCHMARKS[@]}" -eq 0 ]]; then
  echo "Nothing to run: enable RUN_IMAGENET200 and/or RUN_IMAGENET1K" >&2
  exit 2
fi

cd "${SCRIPT_DIR}"
mkdir -p "${ARCHIVE_ROOT}" "${OUTPUT_ROOT}"

"${PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), "A CUDA-enabled PyTorch installation is required"
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PYTHON}" -m pip install "numpy>=1.24,<2" "Cython>=0.29.30,<3"
"${PYTHON}" -m pip install -r requirements_server.txt
"${PYTHON}" -m pip install --no-build-isolation "libmr>=0.1.9"

# Download official OpenOOD v1.5 imglists and all ImageNet OOD datasets once.
"${PYTHON}" prepare_openood.py \
  --benchmarks "${BENCHMARKS[@]}" \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --archive-dir "${ARCHIVE_ROOT}" \
  --no-checkpoints

# Only ImageNet-200 needs the OpenOOD checkpoint archive.  ImageNet-1K uses
# torchvision's official ResNet-50 V1 weights, downloaded on first use.
if [[ "${RUN_IMAGENET200}" == "1" ]]; then
  "${PYTHON}" prepare_openood.py \
    --benchmarks imagenet200 \
    --data-root "${DATA_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --archive-dir "${ARCHIVE_ROOT}" \
    --no-datasets
fi

run_benchmark() {
  local benchmark="$1"
  local seed="$2"
  local common_args=(
    --n-anchors "${N_ANCHORS}"
    --setup-samples-per-class "${SETUP_SAMPLES_PER_CLASS}"
    --ham-epochs "${HAM_EPOCHS}"
    --n-steps "${N_STEPS}"
    --dt "${DT}"
    --candidate-k "${CANDIDATE_K}"
    --sim-batch "${SIM_BATCH}"
    --ham-batch-size "${HAM_BATCH_SIZE}"
  )
  "${PYTHON}" Imagenet_ood_experiment.py \
    --id-data "${benchmark}" \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${RESULTS_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --seed "${seed}" \
    --weight-download-backend "${WEIGHT_DOWNLOAD_BACKEND}" \
    "${common_args[@]}" \
    --skip-download \
    --mass-mode uniform \
    --bandwidth-loss static \
    --potentials "${POTENTIALS[@]}"

  if [[ "${RUN_METHOD_ABLATIONS}" == "1" ]]; then
    # Isolate each paper/code difference with Gaussian, then run the combined
    # method-faithful setting.  Outputs use separate configuration directories.
    "${PYTHON}" Imagenet_ood_experiment.py \
      --id-data "${benchmark}" \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${RESULTS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --seed "${seed}" --skip-download \
      "${common_args[@]}" \
      --potential gaussian \
      --mass-mode effective_rank --mass-normalization none \
      --bandwidth-loss static

    "${PYTHON}" Imagenet_ood_experiment.py \
      --id-data "${benchmark}" \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${RESULTS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --seed "${seed}" --skip-download \
      "${common_args[@]}" \
      --potential gaussian \
      --mass-mode uniform \
      --bandwidth-loss trajectory

    "${PYTHON}" Imagenet_ood_experiment.py \
      --id-data "${benchmark}" \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${RESULTS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --seed "${seed}" --skip-download \
      "${common_args[@]}" \
      --potential gaussian \
      --mass-mode effective_rank --mass-normalization none \
      --bandwidth-loss trajectory
  fi
}

for seed in "${SEED_LIST[@]}"; do
  if [[ "${RUN_IMAGENET200}" == "1" ]]; then
    run_benchmark imagenet200 "${seed}"
  fi
  if [[ "${RUN_IMAGENET1K}" == "1" ]]; then
    run_benchmark imagenet1k "${seed}"
  fi
done

echo "All requested OpenOOD experiments completed. Results: ${OUTPUT_ROOT}"
