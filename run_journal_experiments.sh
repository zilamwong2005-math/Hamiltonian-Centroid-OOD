#!/usr/bin/env bash
# Resumable experiment stages required for the Neural Networks submission.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-${SCRIPT_DIR}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT}/data}"
OPENOOD_CKPT_ROOT="${OPENOOD_CKPT_ROOT:-${PROJECT}/openood_pretrained}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT}/results_openood}"
JOURNAL_ROOT="${JOURNAL_ROOT:-${PROJECT}/results/journal}"
LOG_ROOT="${LOG_ROOT:-${PROJECT}/logs/journal}"
TORCH_HOME="${TORCH_HOME:-${PROJECT}/cache/torch}"
PYTHON="${PYTHON:-python}"

read -r -a SEED_LIST <<< "${SEEDS:-0 1 2}"
read -r -a CIFAR_T_LIST <<< "${CIFAR_T_VALUES:-0 1 3 10}"
read -r -a IMAGENET200_T_LIST <<< "${IMAGENET200_T_VALUES:-0 1 3}"
read -r -a IMAGENET1K_T_LIST <<< "${IMAGENET1K_T_VALUES:-0 3}"

export PROJECT DATA_ROOT OPENOOD_CKPT_ROOT OUTPUT_ROOT JOURNAL_ROOT LOG_ROOT TORCH_HOME
export PYTHONPATH="${PROJECT}/OpenOOD:${PYTHONPATH:-}"

cd "${PROJECT}"
mkdir -p "${JOURNAL_ROOT}" "${LOG_ROOT}" "${JOURNAL_ROOT}/completed"

if [[ ! -f ood_experiment.py || ! -f Imagenet_ood_experiment.py ]]; then
  echo "Run this script from a complete Round6 project" >&2
  exit 2
fi

WEIGHT_ARGS=()
if "${PYTHON}" Imagenet_ood_experiment.py --help 2>&1 | \
    grep -q -- "--weight-download-backend"; then
  WEIGHT_ARGS=(--weight-download-backend "${WEIGHT_DOWNLOAD_BACKEND:-auto}")
fi

run_once() {
  local tag="$1"
  shift
  local key marker logfile
  key="$(printf '%s\0' "$@" | sha256sum | cut -c1-16)"
  marker="${JOURNAL_ROOT}/completed/${tag}_${key}.done"
  logfile="${LOG_ROOT}/${tag}_${key}.log"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "[run] ${tag}"
  "$@" 2>&1 | tee "${logfile}"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${marker}"
  echo "[done] ${tag}"
}

run_smoke() {
  local cifar_root="${JOURNAL_ROOT}/smoke_cifar"
  local image_root="${JOURNAL_ROOT}/smoke_openood"
  mkdir -p "${cifar_root}" "${image_root}"
  for dataset in cifar10 cifar100; do
    run_once "smoke_${dataset}_t0" \
      "${PYTHON}" -u ood_experiment.py \
      --protocol openood --encoder_source official --model resnet18 \
      --data_root "${DATA_ROOT}" --results_dir "${cifar_root}" \
      --openood_ckpt_root "${OPENOOD_CKPT_ROOT}" \
      --in_dist "${dataset}" --seed 0 --potential gaussian \
      --mass_mode uniform --mass_normalization none \
      --bandwidth_loss static --trajectory_train_steps 0 \
      --n_anchors 5 --n_steps 0 --dt 0.05 --ham_epochs 1 \
      --sim_batch 64 --anchor_chunk 5 --candidate_k 0 --sigma_init 1.0 \
      --ham_train_samples_per_class 8 --max_eval_samples 128
  done
  for benchmark in imagenet200 imagenet1k; do
    run_once "smoke_${benchmark}_t0" \
      "${PYTHON}" -u Imagenet_ood_experiment.py \
      --id-data "${benchmark}" --openood-root "${PROJECT}/OpenOOD" \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${OPENOOD_CKPT_ROOT}" \
      --output-root "${image_root}" --seed 0 "${WEIGHT_ARGS[@]}" \
      --potential gaussian --n-anchors 5 --setup-samples-per-class 12 \
      --ham-epochs 1 --ham-lr 0.001 --ham-batch-size 32 \
      --bandwidth-loss static --trajectory-train-steps 0 \
      --n-steps 0 --dt 0.05 --candidate-k 20 --sim-batch 32 \
      --sigma-init 0.5 --sigma-min 0.05 --sigma-max 4.0 \
      --mass-mode uniform --mass-normalization none \
      --prediction-source backbone --batch-size 64 \
      --setup-batch-size 128 --num-workers 8 --max-eval-samples 128 \
      --skip-download
  done
  echo "SMOKE ONLY: these capped results must not be reported in the paper."
}

run_cifar_trajectory() {
  local result_root="${JOURNAL_ROOT}/cifar_trajectory"
  mkdir -p "${result_root}"
  for seed in "${SEED_LIST[@]}"; do
    for dataset in cifar10 cifar100; do
      for steps in "${CIFAR_T_LIST[@]}"; do
        run_once "trajectory_${dataset}_s${seed}_t${steps}" \
          "${PYTHON}" -u ood_experiment.py \
          --protocol openood --encoder_source official --model resnet18 \
          --data_root "${DATA_ROOT}" --results_dir "${result_root}" \
          --openood_ckpt_root "${OPENOOD_CKPT_ROOT}" \
          --in_dist "${dataset}" --seed "${seed}" \
          --potentials gaussian imq \
          --mass_mode uniform --mass_normalization none \
          --bandwidth_loss static --trajectory_train_steps 0 \
          --n_anchors 80 --n_steps "${steps}" --dt 0.05 \
          --ham_epochs 25 --sim_batch 64 --anchor_chunk 10 \
          --candidate_k 0 --sigma_init 1.0 \
          --ham_train_samples_per_class 0 --max_eval_samples 0
      done
    done
  done
}

run_cifar_mass() {
  local result_root="${JOURNAL_ROOT}/cifar_mass"
  mkdir -p "${result_root}"
  for seed in "${SEED_LIST[@]}"; do
    for dataset in cifar10 cifar100; do
      for normalization in none class_mean global_mean; do
        run_once "mass_${dataset}_s${seed}_${normalization}" \
          "${PYTHON}" -u ood_experiment.py \
          --protocol openood --encoder_source official --model resnet18 \
          --data_root "${DATA_ROOT}" --results_dir "${result_root}" \
          --openood_ckpt_root "${OPENOOD_CKPT_ROOT}" \
          --in_dist "${dataset}" --seed "${seed}" \
          --potential gaussian \
          --mass_mode effective_rank --mass_normalization "${normalization}" \
          --bandwidth_loss static --trajectory_train_steps 0 \
          --n_anchors 80 --n_steps 0 --dt 0.05 \
          --ham_epochs 25 --sim_batch 64 --anchor_chunk 10 \
          --candidate_k 0 --sigma_init 1.0 \
          --ham_train_samples_per_class 0 --max_eval_samples 0
      done
    done
  done
}

run_imagenet_trajectory_one() {
  local benchmark="$1" seed="$2" steps="$3"
  local batch_size=64
  if [[ "${benchmark}" == "imagenet200" ]]; then
    batch_size=128
  fi
  run_once "trajectory_${benchmark}_s${seed}_t${steps}" \
    "${PYTHON}" -u Imagenet_ood_experiment.py \
    --id-data "${benchmark}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --seed "${seed}" "${WEIGHT_ARGS[@]}" \
    --potentials gaussian imq \
    --n-anchors 5 --setup-samples-per-class 12 \
    --ham-epochs 10 --ham-lr 0.001 --ham-batch-size 32 \
    --bandwidth-loss static --trajectory-train-steps 0 \
    --n-steps "${steps}" --dt 0.05 --candidate-k 20 \
    --sim-batch 32 --sigma-init 0.5 --sigma-min 0.05 --sigma-max 4.0 \
    --mass-mode uniform --mass-normalization none \
    --prediction-source backbone --batch-size "${batch_size}" \
    --setup-batch-size 128 --num-workers 8 --max-eval-samples 0 \
    --skip-download
}

run_imagenet200_trajectory() {
  for seed in "${SEED_LIST[@]}"; do
    for steps in "${IMAGENET200_T_LIST[@]}"; do
      run_imagenet_trajectory_one imagenet200 "${seed}" "${steps}"
    done
  done
}

run_imagenet1k_trajectory() {
  for seed in "${SEED_LIST[@]}"; do
    for steps in "${IMAGENET1K_T_LIST[@]}"; do
      run_imagenet_trajectory_one imagenet1k "${seed}" "${steps}"
    done
  done
}

# Complete the previously exploratory T=10 condition with all detector seeds.
# This is deliberately separate from trajectory_imagenet1k so that an AutoDL
# restart does not re-run the already complete T=0/3 matrix.
run_imagenet1k_t10() {
  for seed in "${SEED_LIST[@]}"; do
    run_imagenet_trajectory_one imagenet1k "${seed}" 10
  done
}

run_ctm_smoke() {
  run_once "ctm_smoke" \
    "${PYTHON}" -u run_ctm_baseline.py \
    --stage smoke \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --hamiltonian-root "${OUTPUT_ROOT}" \
    --journal-root "${JOURNAL_ROOT}" \
    --output-root "${JOURNAL_ROOT}/ctm" \
    --cache-root "${PROJECT}/cache/pretrained" \
    --seeds "${SEED_LIST[@]}" \
    --num-workers 8 --max-eval-samples 128 \
    --smoke-setup-samples-per-class 2 --no-progress
}

run_ctm_full() {
  run_once "ctm_full" \
    "${PYTHON}" -u run_ctm_baseline.py \
    --stage full \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --hamiltonian-root "${OUTPUT_ROOT}" \
    --journal-root "${JOURNAL_ROOT}" \
    --output-root "${JOURNAL_ROOT}/ctm" \
    --cache-root "${PROJECT}/cache/pretrained" \
    --seeds "${SEED_LIST[@]}" --num-workers 8
}

run_static_reduction_bridge_smoke() {
  run_once "static_reduction_bridge_smoke" \
    "${PYTHON}" -u run_static_reduction_bridge.py \
    --stage smoke --openood-root "${PROJECT}/OpenOOD" \
    --hamiltonian-root "${OUTPUT_ROOT}" \
    --output-root "${JOURNAL_ROOT}/static_reduction_bridge" \
    --seeds "${SEED_LIST[@]}" --batch-size 64 --max-eval-samples 128
}

run_static_reduction_bridge_full() {
  run_once "static_reduction_bridge_full" \
    "${PYTHON}" -u run_static_reduction_bridge.py \
    --stage full --openood-root "${PROJECT}/OpenOOD" \
    --hamiltonian-root "${OUTPUT_ROOT}" \
    --output-root "${JOURNAL_ROOT}/static_reduction_bridge" \
    --seeds "${SEED_LIST[@]}" --batch-size 64
}

run_ctm_summary() {
  run_once "ctm_summary" \
    "${PYTHON}" -u summarize_ctm_comparison.py \
    --stage full --ctm-root "${JOURNAL_ROOT}/ctm" \
    --journal-root "${JOURNAL_ROOT}" \
    --output-root "${JOURNAL_ROOT}/summary_ctm_comparison"
}

run_required_additions_audit() {
  run_once "required_additions_audit" \
    "${PYTHON}" -u audit_required_additions.py \
    --output-root "${OUTPUT_ROOT}" --journal-root "${JOURNAL_ROOT}"
}

run_imagenet200_mass() {
  for seed in "${SEED_LIST[@]}"; do
    for normalization in none class_mean global_mean; do
      run_once "mass_imagenet200_s${seed}_${normalization}" \
        "${PYTHON}" -u Imagenet_ood_experiment.py \
        --id-data imagenet200 --openood-root "${PROJECT}/OpenOOD" \
        --data-root "${DATA_ROOT}" \
        --openood-results-root "${OPENOOD_CKPT_ROOT}" \
        --output-root "${OUTPUT_ROOT}" --seed "${seed}" "${WEIGHT_ARGS[@]}" \
        --potential gaussian \
        --n-anchors 5 --setup-samples-per-class 12 \
        --ham-epochs 10 --ham-lr 0.001 --ham-batch-size 16 \
        --bandwidth-loss static --trajectory-train-steps 0 \
        --n-steps 0 --dt 0.05 --candidate-k 20 --sim-batch 32 \
        --sigma-init 0.5 --sigma-min 0.05 --sigma-max 4.0 \
        --mass-mode effective_rank --mass-normalization "${normalization}" \
        --mass-resolution 0 --prediction-source backbone \
        --batch-size 128 --setup-batch-size 128 --num-workers 8 \
        --max-eval-samples 0 --skip-download
    done
  done
}

run_cheap_baselines() {
  run_once "baselines_cheap" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
    --methods ebo mls gen react scale \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${PROJECT}/results_openood_baselines" \
    --seeds "${SEED_LIST[@]}" --num-workers 8 --continue-on-error
}

run_heavy_baselines() {
  run_once "baselines_heavy" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
    --methods knn vim \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${PROJECT}/results_openood_baselines" \
    --seeds "${SEED_LIST[@]}" --num-workers 8 --continue-on-error
}

run_extended_baseline_smoke() {
  run_once "baselines_extended_smoke" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 \
    --methods ash dice she rmds rankfeat \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${JOURNAL_ROOT}/smoke_baselines_extended" \
    --seeds 0 --num-workers 8 --max-eval-samples 128 \
    --continue-on-error
}

run_extended_feature_baselines() {
  run_once "baselines_extended_feature" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
    --methods ash dice she \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${PROJECT}/results_openood_baselines" \
    --seeds "${SEED_LIST[@]}" --num-workers 8 --continue-on-error
}

run_rankfeat_baselines() {
  run_once "baselines_rankfeat" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
    --methods rankfeat \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${PROJECT}/results_openood_baselines" \
    --seeds "${SEED_LIST[@]}" --num-workers 8 --continue-on-error
}

run_rmds_baselines() {
  run_once "baselines_rmds" \
    "${PYTHON}" -u run_openood_baselines.py \
    --benchmarks cifar10 cifar100 imagenet200 imagenet1k \
    --methods rmds \
    --data-root "${DATA_ROOT}" \
    --openood-results-root "${OPENOOD_CKPT_ROOT}" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-root "${PROJECT}/results_openood_baselines" \
    --seeds "${SEED_LIST[@]}" --num-workers 8 --continue-on-error
}

run_extended_baseline_summary() {
  run_once "baselines_extended_summary" \
    "${PYTHON}" -u summarize_extended_baselines.py \
    --baseline-root "${PROJECT}/results_openood_baselines" \
    --output-dir "${JOURNAL_ROOT}/summary_extended_baselines"
}

run_extended_baselines() {
  run_extended_feature_baselines
  run_rankfeat_baselines
  run_rmds_baselines
  run_extended_baseline_summary
}

run_efficiency() {
  for benchmark in cifar10 cifar100 imagenet200 imagenet1k; do
    local batch_size=64 search_root="${OUTPUT_ROOT}"
    if [[ "${benchmark}" == cifar10 || "${benchmark}" == cifar100 ]]; then
      batch_size=256
      search_root="${JOURNAL_ROOT}/cifar_trajectory/checkpoints"
    elif [[ "${benchmark}" == imagenet200 ]]; then
      batch_size=128
    fi
    run_once "efficiency_${benchmark}" \
      "${PYTHON}" -u benchmark_inference_efficiency.py \
      --benchmark "${benchmark}" --seed 0 \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${OPENOOD_CKPT_ROOT}" \
      --openood-root "${PROJECT}/OpenOOD" \
      --detector-search-root "${search_root}" \
      --locked-results-root "${JOURNAL_ROOT}" \
      --ctm-root "${JOURNAL_ROOT}/ctm" \
      --output "${JOURNAL_ROOT}/efficiency/efficiency.csv" \
      --steps 0 1 3 10 --batch-size "${batch_size}" \
      --warmup 10 --repeats 50 --num-workers 8
  done
}

run_efficiency_batch1() {
  for benchmark in cifar10 cifar100 imagenet200 imagenet1k; do
    run_once "efficiency_ctm_b1_v3_${benchmark}" \
      "${PYTHON}" -u benchmark_ctm_paired_efficiency.py \
      --benchmark "${benchmark}" --seed 0 \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${OPENOOD_CKPT_ROOT}" \
      --openood-root "${PROJECT}/OpenOOD" \
      --ctm-root "${JOURNAL_ROOT}/ctm" \
      --output "${JOURNAL_ROOT}/efficiency/efficiency.csv" \
      --batch-size 1 --warmup 10 --repeats 50 --num-workers 8 \
      --geometry-weight 0.8
  done
}

run_extended_efficiency() {
  for benchmark in cifar10 cifar100 imagenet200 imagenet1k; do
    local setup_batch=64
    if [[ "${benchmark}" == cifar10 || "${benchmark}" == cifar100 ]]; then
      setup_batch=256
    elif [[ "${benchmark}" == imagenet200 ]]; then
      setup_batch=128
    fi
    run_once "efficiency_extended_${benchmark}" \
      "${PYTHON}" -u benchmark_extended_baseline_efficiency.py \
      --benchmark "${benchmark}" --seed 0 \
      --methods ash dice she rmds rankfeat \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${OPENOOD_CKPT_ROOT}" \
      --openood-root "${PROJECT}/OpenOOD" \
      --baseline-root "${PROJECT}/results_openood_baselines" \
      --setup-cache-root "${JOURNAL_ROOT}/efficiency/setup_cache" \
      --output "${JOURNAL_ROOT}/efficiency/extended_posthoc_efficiency.csv" \
      --batch-size 1 --setup-batch-size "${setup_batch}" \
      --warmup 10 --repeats 50 --num-workers 8
  done
}

run_scale_efficiency() {
  for benchmark in cifar10 cifar100 imagenet200 imagenet1k; do
    local setup_batch=64
    if [[ "${benchmark}" == cifar10 || "${benchmark}" == cifar100 ]]; then
      setup_batch=256
    elif [[ "${benchmark}" == imagenet200 ]]; then
      setup_batch=128
    fi
    run_once "efficiency_scale_${benchmark}" \
      "${PYTHON}" -u benchmark_extended_baseline_efficiency.py \
      --benchmark "${benchmark}" --seed 0 --methods scale \
      --data-root "${DATA_ROOT}" \
      --openood-results-root "${OPENOOD_CKPT_ROOT}" \
      --openood-root "${PROJECT}/OpenOOD" \
      --baseline-root "${PROJECT}/results_openood_baselines" \
      --setup-cache-root "${JOURNAL_ROOT}/efficiency/setup_cache" \
      --output "${JOURNAL_ROOT}/efficiency/extended_posthoc_efficiency.csv" \
      --batch-size 1 --setup-batch-size "${setup_batch}" \
      --warmup 10 --repeats 50 --num-workers 8
  done
}

run_required_efficiency_summary() {
  run_once "efficiency_required_summary_v2" \
    "${PYTHON}" -u summarize_efficiency_comparison.py \
    --current "${JOURNAL_ROOT}/efficiency/efficiency.csv" \
    --extended "${JOURNAL_ROOT}/efficiency/extended_posthoc_efficiency.csv" \
    --output-dir "${JOURNAL_ROOT}/efficiency/summary_required" \
    --batch-size 1
}

run_ctm_efficiency_summary() {
  run_once "efficiency_ctm_summary_v2" \
    "${PYTHON}" -u summarize_ctm_efficiency.py \
    --input "${JOURNAL_ROOT}/efficiency/efficiency.csv" \
    --output-dir "${JOURNAL_ROOT}/efficiency/summary_ctm" \
    --batch-size 1 --seed 0
}

run_all_method_summary() {
  run_once "all_method_summary" \
    "${PYTHON}" -u summarize_all_method_comparison.py \
    --baseline-root "${PROJECT}/results_openood_baselines" \
    --component-summary \
      "${JOURNAL_ROOT}/summary_component_ablation/component_mean_std.csv" \
    --output-dir "${JOURNAL_ROOT}/summary_all_methods"
}

run_extended_baseline_audit() {
  run_once "extended_baseline_audit" \
    "${PYTHON}" -u audit_extended_baseline_reproduction.py \
    --baseline-root "${PROJECT}/results_openood_baselines" \
    --openood-root "${PROJECT}/OpenOOD" \
    --output-dir "${JOURNAL_ROOT}/audit_extended_baselines"
}

run_required_summaries() {
  run_extended_baseline_summary
  run_all_method_summary
  run_extended_baseline_audit
  run_required_efficiency_summary
}

usage() {
  cat <<'EOF'
Usage: bash run_journal_experiments.sh <stage>

Stages (recommended order):
  smoke                  128-sample compatibility check; never report metrics
  trajectory_cifar       Gaussian/IMQ, T=0/1/3/10, three seeds
  trajectory_imagenet200 Gaussian/IMQ, T=0/1/3; existing T=10 is retained
  trajectory_imagenet1k  Gaussian/IMQ, T=0/3, three seeds
  trajectory_imagenet1k_t10
                         Complete Gaussian/IMQ T=10 with three seeds
  ctm_smoke              Non-reportable CTM compatibility matrix (11 model runs)
  ctm_full               Formal same-protocol CTM matrix; full ID-train means
  ctm_summary            Compare CTM with setup-bank centroid and locked fusion
  static_reduction_bridge_smoke
                         Non-reportable validation-only six-stage bridge check
  static_reduction_bridge
                         Formal validation-only six-stage static-reduction bridge
  required_additions_audit
                         Strictly verify CTM, bridge, and ImageNet-1K T=10 outputs
  mass_cifar             effective-rank normalization ablation, T=0, 3 seeds
  mass_imagenet200       effective-rank normalization ablation, T=0, 3 seeds
  baselines_cheap        EBO/MLS/GEN/ReAct/Scale under local OpenOOD
  baselines_heavy        KNN/ViM (run separately because setup is expensive)
  baselines_extended_smoke
                         ASH/DICE/SHE/RMDS/RankFeat CIFAR-10 smoke test
  baselines_extended_feature
                         ASH/DICE/SHE on all four benchmarks
  baselines_rankfeat     RankFeat full-SVD reproduction (small SVD batches)
  baselines_rmds         RMDS streaming/exact-statistics reproduction
  baselines_extended_summary
                         Validate matrix and write CSV/LaTeX paper tables
  baselines_extended     Run the preceding three formal extended stages
  efficiency             Paired MSP/CTM/current-method and T=0/1/3/10 efficiency
  efficiency_b1          Paired batch-1 MSP/CTM/current-method latency and VRAM
  efficiency_extended    Batch-1 ASH/DICE/SHE/RMDS/RankFeat efficiency
  efficiency_scale       Batch-1 Scale efficiency on all four benchmarks
  efficiency_ctm_summary Build the paired batch-1 CTM/current-method table
  efficiency_required_summary
                         Merge all batch-1 efficiency results for the paper
  all_method_summary     Unified ranking and locked-method win/loss audit
  baselines_extended_audit
                         Check 50 formal runs and RMDS algebraic equivalence
  required_summaries     Build all final summaries after efficiency completes
EOF
}

case "${1:-help}" in
  smoke) run_smoke ;;
  trajectory_cifar) run_cifar_trajectory ;;
  trajectory_imagenet200) run_imagenet200_trajectory ;;
  trajectory_imagenet1k) run_imagenet1k_trajectory ;;
  trajectory_imagenet1k_t10) run_imagenet1k_t10 ;;
  ctm_smoke) run_ctm_smoke ;;
  ctm_full) run_ctm_full ;;
  ctm_summary) run_ctm_summary ;;
  static_reduction_bridge_smoke) run_static_reduction_bridge_smoke ;;
  static_reduction_bridge) run_static_reduction_bridge_full ;;
  required_additions_audit) run_required_additions_audit ;;
  mass_cifar) run_cifar_mass ;;
  mass_imagenet200) run_imagenet200_mass ;;
  baselines_cheap) run_cheap_baselines ;;
  baselines_heavy) run_heavy_baselines ;;
  baselines_extended_smoke) run_extended_baseline_smoke ;;
  baselines_extended_feature) run_extended_feature_baselines ;;
  baselines_rankfeat) run_rankfeat_baselines ;;
  baselines_rmds) run_rmds_baselines ;;
  baselines_extended_summary) run_extended_baseline_summary ;;
  baselines_extended) run_extended_baselines ;;
  efficiency) run_efficiency ;;
  efficiency_b1) run_efficiency_batch1 ;;
  efficiency_extended) run_extended_efficiency ;;
  efficiency_scale) run_scale_efficiency ;;
  efficiency_ctm_summary) run_ctm_efficiency_summary ;;
  efficiency_required_summary) run_required_efficiency_summary ;;
  all_method_summary) run_all_method_summary ;;
  baselines_extended_audit) run_extended_baseline_audit ;;
  required_summaries) run_required_summaries ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
