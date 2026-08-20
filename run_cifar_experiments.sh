#!/usr/bin/env bash
# Staged, resumable CIFAR-10/CIFAR-100 experiments for AutoDL.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-${SCRIPT_DIR}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT}/data}"
OPENOOD_CKPT_ROOT="${OPENOOD_CKPT_ROOT:-${PROJECT}/openood_pretrained}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT}/archives}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT}/results/cifar}"
LOG_ROOT="${LOG_ROOT:-${PROJECT}/logs/cifar}"
TORCH_HOME="${TORCH_HOME:-${PROJECT}/cache/torch}"

SEEDS=(${SEEDS:-0 1 2})
N_ANCHORS="${N_ANCHORS:-80}"
N_STEPS="${N_STEPS:-120}"
DT="${DT:-0.05}"
HAM_EPOCHS="${HAM_EPOCHS:-25}"
SIM_BATCH="${SIM_BATCH:-64}"
ANCHOR_CHUNK="${ANCHOR_CHUNK:-10}"
CANDIDATE_K="${CANDIDATE_K:-0}"
SIGMA_INIT="${SIGMA_INIT:-1.0}"
TRAJECTORY_TRAIN_STEPS="${TRAJECTORY_TRAIN_STEPS:-120}"
HAM_TRAIN_SAMPLES_PER_CLASS="${HAM_TRAIN_SAMPLES_PER_CLASS:-0}"

export PROJECT DATA_ROOT OPENOOD_CKPT_ROOT ARCHIVE_ROOT RESULT_ROOT LOG_ROOT TORCH_HOME
export PYTHONPATH="${PROJECT}/OpenOOD:${PYTHONPATH:-}"

cd "${PROJECT}"
mkdir -p "${DATA_ROOT}" "${OPENOOD_CKPT_ROOT}" "${ARCHIVE_ROOT}" \
  "${RESULT_ROOT}" "${LOG_ROOT}" "${RESULT_ROOT}/completed" "${TORCH_HOME}"

COMMON=(
  --protocol openood
  --encoder_source official
  --model resnet18
  --data_root "${DATA_ROOT}"
  --results_dir "${RESULT_ROOT}"
  --openood_ckpt_root "${OPENOOD_CKPT_ROOT}"
  --n_anchors "${N_ANCHORS}"
  --n_steps "${N_STEPS}"
  --dt "${DT}"
  --ham_epochs "${HAM_EPOCHS}"
  --sim_batch "${SIM_BATCH}"
  --anchor_chunk "${ANCHOR_CHUNK}"
  --candidate_k "${CANDIDATE_K}"
  --sigma_init "${SIGMA_INIT}"
  --ham_train_samples_per_class "${HAM_TRAIN_SAMPLES_PER_CLASS}"
)

run_once() {
  local tag="$1"
  shift
  local config_key
  config_key="$(printf '%s\0' "$@" | sha256sum | cut -c1-16)"
  local marker="${RESULT_ROOT}/completed/${tag}_${config_key}.done"
  local logfile="${LOG_ROOT}/${tag}_${config_key}.log"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${tag} 已完成"
    return 0
  fi
  echo "[run] ${tag}"
  python -u ood_experiment.py "$@" 2>&1 | tee "${logfile}"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${marker}"
  echo "[done] ${tag}"
}

prepare_data() {
  if [[ -f /etc/network_turbo ]]; then
    # AutoDL acceleration is useful for the Hugging Face dataset mirrors.
    # shellcheck disable=SC1091
    source /etc/network_turbo
  fi
  python -u prepare_openood.py \
    --benchmarks cifar10 cifar100 \
    --data-root "${DATA_ROOT}" \
    --results-root "${OPENOOD_CKPT_ROOT}" \
    --archive-dir "${ARCHIVE_ROOT}" \
    --download-backend auto \
    --keep-archives \
    --no-checkpoints

  # Official CIFAR checkpoints currently come from OpenOOD's Google Drive.
  # A pre-uploaded cifar10_checkpoint.zip/cifar100_checkpoint.zip in
  # ARCHIVE_ROOT is selected automatically if Google Drive is unavailable.
  python -u prepare_openood.py \
    --benchmarks cifar10 cifar100 \
    --data-root "${DATA_ROOT}" \
    --results-root "${OPENOOD_CKPT_ROOT}" \
    --archive-dir "${ARCHIVE_ROOT}" \
    --download-backend auto \
    --keep-archives \
    --no-datasets
}

verify_data() {
  python -u verify_cifar_openood.py --data-root "${DATA_ROOT}"
  python - <<'PY'
import os
from pathlib import Path
from openood_cifar import discover_cifar_checkpoint

root = Path(os.environ["OPENOOD_CKPT_ROOT"])
for dataset in ("cifar10", "cifar100"):
    for seed in (0, 1, 2):
        path = discover_cifar_checkpoint(root, dataset, seed)
        print(f"{dataset} seed {seed}: {path} ({path.stat().st_size / 1024**2:.2f} MB)")
PY
}

run_smoke() {
  local smoke_root="${PROJECT}/results/cifar_smoke"
  local smoke_logs="${PROJECT}/logs/cifar_smoke"
  mkdir -p "${smoke_root}" "${smoke_logs}"
  for dataset in cifar10 cifar100; do
    local tag="smoke_${dataset}_s0"
    local marker="${smoke_root}/${tag}.done"
    if [[ -f "${marker}" ]]; then
      echo "[skip] ${tag} 已完成"
      continue
    fi
    python -u ood_experiment.py \
      --protocol openood --encoder_source official --model resnet18 \
      --data_root "${DATA_ROOT}" --results_dir "${smoke_root}" \
      --openood_ckpt_root "${OPENOOD_CKPT_ROOT}" \
      --in_dist "${dataset}" --seed 0 --potential gaussian \
      --mass_mode uniform --mass_normalization none --bandwidth_loss static \
      --n_anchors 5 --n_steps 3 --dt "${DT}" --ham_epochs 1 \
      --sim_batch 64 --anchor_chunk 5 --candidate_k 0 \
      --ham_train_samples_per_class 8 --max_eval_samples 128 \
      2>&1 | tee "${smoke_logs}/${tag}.log"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${marker}"
  done
}

run_reproduction() {
  for seed in "${SEEDS[@]}"; do
    for dataset in cifar10 cifar100; do
      run_once \
        "reproduce_${dataset}_s${seed}_a${N_ANCHORS}_t${N_STEPS}" \
        "${COMMON[@]}" --in_dist "${dataset}" --seed "${seed}" \
        --potential gaussian --mass_mode uniform --mass_normalization none \
        --bandwidth_loss static
    done
  done
}

run_components() {
  for seed in "${SEEDS[@]}"; do
    for dataset in cifar10 cifar100; do
      run_once \
        "component_mass_${dataset}_s${seed}_a${N_ANCHORS}_t${N_STEPS}" \
        "${COMMON[@]}" --in_dist "${dataset}" --seed "${seed}" \
        --potential gaussian --mass_mode effective_rank \
        --mass_normalization none --bandwidth_loss static

      run_once \
        "component_trajectory_${dataset}_s${seed}_a${N_ANCHORS}_t${N_STEPS}_tt${TRAJECTORY_TRAIN_STEPS}" \
        "${COMMON[@]}" --in_dist "${dataset}" --seed "${seed}" \
        --potential gaussian --mass_mode uniform --mass_normalization none \
        --bandwidth_loss trajectory \
        --trajectory_train_steps "${TRAJECTORY_TRAIN_STEPS}"

      run_once \
        "component_full_${dataset}_s${seed}_a${N_ANCHORS}_t${N_STEPS}_tt${TRAJECTORY_TRAIN_STEPS}" \
        "${COMMON[@]}" --in_dist "${dataset}" --seed "${seed}" \
        --potential gaussian --mass_mode effective_rank \
        --mass_normalization none --bandwidth_loss trajectory \
        --trajectory_train_steps "${TRAJECTORY_TRAIN_STEPS}"
    done
  done
}

run_potentials() {
  for seed in "${SEEDS[@]}"; do
    for dataset in cifar10 cifar100; do
      run_once \
        "potentials_full_${dataset}_s${seed}_a${N_ANCHORS}_t${N_STEPS}_tt${TRAJECTORY_TRAIN_STEPS}" \
        "${COMMON[@]}" --in_dist "${dataset}" --seed "${seed}" \
        --potentials gaussian laplacian cauchy imq matern32 \
        --mass_mode effective_rank --mass_normalization none \
        --bandwidth_loss trajectory \
        --trajectory_train_steps "${TRAJECTORY_TRAIN_STEPS}"
    done
  done
}

run_selftrained_backbones() {
  local appendix_root="${PROJECT}/results/cifar_backbones"
  local appendix_logs="${PROJECT}/logs/cifar_backbones"
  mkdir -p "${appendix_root}" "${appendix_logs}" "${appendix_root}/completed"
  for seed in "${SEEDS[@]}"; do
    for dataset in cifar10 cifar100; do
      for backbone in resnet18 densenet100; do
        local tag="backbone_${dataset}_${backbone}_s${seed}"
        local marker="${appendix_root}/completed/${tag}.done"
        if [[ -f "${marker}" ]]; then
          echo "[skip] ${tag} 已完成"
          continue
        fi
        python -u ood_experiment.py \
          --protocol openood --encoder_source self_trained \
          --model "${backbone}" --epochs 100 \
          --data_root "${DATA_ROOT}" --results_dir "${appendix_root}" \
          --in_dist "${dataset}" --seed "${seed}" \
          --potentials gaussian laplacian cauchy imq matern32 \
          --mass_mode effective_rank --mass_normalization none \
          --bandwidth_loss trajectory \
          --trajectory_train_steps "${TRAJECTORY_TRAIN_STEPS}" \
          --n_anchors "${N_ANCHORS}" --n_steps "${N_STEPS}" --dt "${DT}" \
          --ham_epochs "${HAM_EPOCHS}" --sim_batch "${SIM_BATCH}" \
          --anchor_chunk "${ANCHOR_CHUNK}" --candidate_k "${CANDIDATE_K}" \
          --sigma_init "${SIGMA_INIT}" \
          --ham_train_samples_per_class "${HAM_TRAIN_SAMPLES_PER_CLASS}" \
          2>&1 | tee "${appendix_logs}/${tag}.log"
        printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${marker}"
      done
    done
  done
}

summarize() {
  python -u summarize_cifar_results.py \
    --input "${RESULT_ROOT}/ood_results_cifar_v3.csv" \
    --output-dir "${RESULT_ROOT}/summary" \
    --expected-seeds 0 1 2 --strict-seeds
}

usage() {
  cat <<'EOF'
用法: bash run_cifar_experiments.sh <阶段>

阶段：
  prepare       下载 CIFAR OpenOOD 数据与官方三种子检查点
  verify        严格检查列表、所有图片和六个官方检查点
  smoke         每个数据集仅 128 个样本的端到端冒烟测试
  reproduce     Gaussian + uniform/static，三种子完整基线
  components    mass、trajectory 与 full method 三组消融
  potentials    full method 下五种势能函数
  backbones     自训练 ResNet-18/DenseNet-100 的补充稳健性实验
  summarize     生成按数据集和 Near/Far/All 的 mean±std CSV
  main          reproduce + components + potentials + summarize
EOF
}

stage="${1:-help}"
case "${stage}" in
  prepare) prepare_data ;;
  verify) verify_data ;;
  smoke) run_smoke ;;
  reproduce) run_reproduction ;;
  components) run_components ;;
  potentials) run_potentials ;;
  backbones) run_selftrained_backbones ;;
  summarize) summarize ;;
  main)
    run_reproduction
    run_components
    run_potentials
    summarize
    ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
