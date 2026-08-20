# Hamiltonian OOD 完整实验手册（AutoDL）

本文档对应当前 `Round6` 代码。实验顺序已经调整为：先完成 CIFAR-10/CIFAR-100 的复现、方法消融和势能比较，再做 ImageNet-200，最后做 ImageNet-1K。CIFAR 的逐步操作见 `CIFAR_AUTODL_GUIDE.md`。

目标是完成三组可写进论文的实验：

1. 五种势能函数消融：Gaussian、Laplacian、Cauchy、IMQ、Matérn-3/2。
2. 论文实现一致性消融：`uniform/effective_rank` 质量与 `static/trajectory` 带宽目标。
3. 标准 OpenOOD v1.5：ImageNet-200 和 ImageNet-1K 的 Near-OOD、Far-OOD、FPR@95、AUROC、AUPR-IN、AUPR-OUT。

默认基线是 `uniform mass + static loss + backbone prediction`，用于和旧程序公平比较。论文公式对应设置是 `effective_rank mass + trajectory loss`。两种设置不会共用检测器检查点，也不会覆盖彼此的结果。

## 1. AutoDL 目录

登录服务器后执行：

```bash
export PROJECT=/root/autodl-tmp/hamood/Round6
export DATA_ROOT=$PROJECT/data
export OPENOOD_CKPT_ROOT=$PROJECT/openood_pretrained
export OUTPUT_ROOT=$PROJECT/outputs
export LOG_ROOT=$PROJECT/logs
export TORCH_HOME=$PROJECT/cache/torch
export ARCHIVE_ROOT=$PROJECT/archives

cd "$PROJECT"
mkdir -p "$DATA_ROOT" "$OPENOOD_CKPT_ROOT" "$OUTPUT_ROOT" \
  "$LOG_ROOT" "$TORCH_HOME" "$ARCHIVE_ROOT"
export PYTHONPATH="$PROJECT/OpenOOD:${PYTHONPATH:-}"
```

`/root/autodl-tmp` 是数据盘；不要把数据集放在只有 30 GB 的系统盘 `/root` 其他位置。

## 2. 安装依赖

当前 AutoDL 镜像已有 PyTorch 时，不要重新安装 PyTorch。执行：

```bash
cd "$PROJECT"
python -m pip install "numpy>=1.24,<2" "Cython>=0.29.30,<3"
python -m pip install -r requirements_server.txt
python -m pip install --no-build-isolation "libmr>=0.1.9"
python -m pip check
```

`libmr` 必须使用 `--no-build-isolation`，否则它的构建环境找不到 NumPy，这正是之前报错的原因。

验证：

```bash
python - <<'PY'
import torch, libmr, timm, foolbox
from openood.evaluation_api import Evaluator
from hamiltonian_detector import POTENTIAL_NAMES, MASS_MODES, BANDWIDTH_LOSSES

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Potentials:", POTENTIAL_NAMES)
print("Mass modes:", MASS_MODES)
print("Bandwidth losses:", BANDWIDTH_LOSSES)
PY

python -m pytest -q tests
bash -n run_server_experiments.sh
```

## 3. Google Drive 无法连接时如何上传 ZIP

程序会先查找 `$ARCHIVE_ROOT/<名称>.zip`，存在就直接校验和解压；只有找不到时才访问 Google Drive。

例如 ImageNet-200 权重必须命名为：

```text
imagenet200_checkpoint.zip
```

在 AutoDL 网页文件管理器中进入：

```text
/root/autodl-tmp/hamood/Round6/archives
```

点击“上传”，选择本地 ZIP。上传完成后执行：

```bash
ls -lh "$ARCHIVE_ROOT"
python -m zipfile -t "$ARCHIVE_ROOT/imagenet200_checkpoint.zip"
```

然后只准备 ImageNet-200 权重：

```bash
python -u prepare_openood.py \
  --benchmarks imagenet200 \
  --data-root "$DATA_ROOT" \
  --results-root "$OPENOOD_CKPT_ROOT" \
  --archive-dir "$ARCHIVE_ROOT" \
  --no-datasets
```

看到 `[local archive]`、`[extract]` 和 `OpenOOD preparation complete.` 才算成功。

其他可上传压缩包的固定名称如下：

```text
benchmark_imglist.zip
imagenet_1k.zip
ssb_hard.zip
ninco.zip
inaturalist.zip
texture.zip
openimage_o.zip
imagenet_v2.zip
imagenet_c.zip
imagenet_r.zip
imagenet_es.zip
```

上传未完成的文件不是合法 ZIP，程序会停止，不会把网页错误信息当数据集。若你知道 SHA256，还可增加：

```bash
--sha256 imagenet200_checkpoint=<64位SHA256>
```

## 4. 准备标准 OpenOOD v1.5 数据

准备 ImageNet-200：

```bash
python -u prepare_openood.py \
  --benchmarks imagenet200 \
  --data-root "$DATA_ROOT" \
  --results-root "$OPENOOD_CKPT_ROOT" \
  --archive-dir "$ARCHIVE_ROOT"
```

准备 ImageNet-1K：

```bash
python -u prepare_openood.py \
  --benchmarks imagenet1k \
  --data-root "$DATA_ROOT" \
  --results-root "$OPENOOD_CKPT_ROOT" \
  --archive-dir "$ARCHIVE_ROOT" \
  --no-checkpoints
```

ImageNet-1K 默认使用 torchvision ResNet-50 V1，不需要 OpenOOD 的 ImageNet-1K 权重 ZIP。标准目录应包含：

```text
data/
├── benchmark_imglist/
├── images_classic/
│   └── texture/
└── images_largescale/
    ├── imagenet_1k/
    ├── ssb_hard/
    ├── ninco/
    ├── inaturalist/
    ├── openimage_o/
    ├── imagenet_v2/
    ├── imagenet_c/
    ├── imagenet_r/
    └── imagenet_es/       # 仅 ImageNet-1K 需要
```

检查磁盘：

```bash
du -sh "$DATA_ROOT"/*
df -h /root/autodl-tmp
```

## 5. 先做最小代码测试

不运行完整数据集，先确认新参数能建立检测器：

```bash
python -m pytest -q tests

python Imagenet_ood_experiment.py --help | grep -E \
  "mass-mode|bandwidth-loss|trajectory-train-steps|archive-dir"
```

如果测试通过，再进入正式实验。不要把 `FakeData`、STL-10 代理数据产生的结果写进论文；`ood_experiment.py` 现在默认禁止代理数据，只有显式传入 `--allow_proxy_data` 才会启用旧行为。

## 6. 实验 A：五种势能函数

控制变量：`uniform mass + static loss + backbone prediction`。除势能函数外参数完全相同。

ImageNet-200，先跑 seed 0：

```bash
nohup python -u Imagenet_ood_experiment.py \
  --id-data imagenet200 \
  --data-root "$DATA_ROOT" \
  --openood-results-root "$OPENOOD_CKPT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed 0 --skip-download \
  --potentials gaussian laplacian cauchy imq matern32 \
  --mass-mode uniform \
  --bandwidth-loss static \
  --prediction-source backbone \
  > "$LOG_ROOT/imagenet200_seed0_potentials.log" 2>&1 &
```

ImageNet-1K，先跑 seed 0：

```bash
nohup python -u Imagenet_ood_experiment.py \
  --id-data imagenet1k \
  --data-root "$DATA_ROOT" \
  --openood-results-root "$OPENOOD_CKPT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed 0 --skip-download \
  --potentials gaussian laplacian cauchy imq matern32 \
  --mass-mode uniform \
  --bandwidth-loss static \
  --prediction-source backbone \
  > "$LOG_ROOT/imagenet1k_seed0_potentials.log" 2>&1 &
```

不要在一张 4090 上同时启动这两条命令。先完成 ImageNet-200，再执行 ImageNet-1K。

## 7. 实验 B：质量与轨迹目标的 2×2 消融

先固定 Gaussian，只改变以下两项：

| 组别 | `--mass-mode` | `--bandwidth-loss` | 含义 |
|---|---|---|---|
| B1 | `uniform` | `static` | 旧代码兼容基线 |
| B2 | `effective_rank` | `static` | 只加入论文质量 |
| B3 | `uniform` | `trajectory` | 只加入 Eq. 13-14 |
| B4 | `effective_rank` | `trajectory` | 最接近正文公式 |

B1 已包含在实验 A。以 ImageNet-200 seed 0 为例，依次执行 B2-B4：

```bash
for setting in "effective_rank static" "uniform trajectory" "effective_rank trajectory"; do
  read MASS LOSS <<< "$setting"
  python -u Imagenet_ood_experiment.py \
    --id-data imagenet200 \
    --data-root "$DATA_ROOT" \
    --openood-results-root "$OPENOOD_CKPT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --seed 0 --skip-download \
    --potential gaussian \
    --mass-mode "$MASS" \
    --mass-normalization none \
    --bandwidth-loss "$LOSS" \
    --trajectory-train-steps 0 \
    --ham-batch-size 16
done
```

`--trajectory-train-steps 0` 表示使用完整的 `--n-steps`。如果轨迹反传显存不足，只减小 `--ham-batch-size`（例如 16→8→4），不要擅自减少轨迹步数后仍声称是同一设置。

`--mass-normalization none` 对应正文直接使用有效秩。`class_mean` 是额外稳定化消融，不应和原公式混写。

ImageNet 默认 `--mass-resolution 0`，即在 OpenOOD 的 224×224 输入图像矩阵上计算有效秩。若仅做调试，可临时设为 64 加速，但该结果必须标记为低分辨率近似，不能与默认设置混写。

正文只给出了单个图像矩阵的公式，没有说明 RGB 如何变成矩阵。程序采用固定且可复现的扩展：先撤销 ImageNet 标准化，再按 `0.2989R + 0.5870G + 0.1140B` 转灰度，整幅图像减均值，最后由归一化奇异值熵计算有效秩；常量图像质量定义为 1。论文实验设置中必须写明这条 RGB 约定。

ImageNet 默认 `--candidate-k 20`，训练和推理都只计算固定的 20 个候选类；训练时程序强制把真类放进候选集。这是显式的大规模近似。若要验证全类别 Eq. 13-14，可在较小的 ImageNet-200 子实验上使用 `--candidate-k 0`；ImageNet-1K 全类别轨迹计算会非常慢。

## 8. 三个随机种子与一键运行

先用 seed 0 完成全流程；确认 CSV 正常后，再跑 0、1、2：

```bash
cd "$PROJECT"
SEEDS="0 1 2" \
RUN_IMAGENET200=1 \
RUN_IMAGENET1K=1 \
RUN_METHOD_ABLATIONS=1 \
DATA_ROOT="$DATA_ROOT" \
RESULTS_ROOT="$OPENOOD_CKPT_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
ARCHIVE_ROOT="$ARCHIVE_ROOT" \
nohup bash run_server_experiments.sh \
  > "$LOG_ROOT/server_experiments.log" 2>&1 &

echo $! > "$LOG_ROOT/server_experiments.pid"
```

只跑 ImageNet-200：

```bash
SEEDS="0" RUN_IMAGENET200=1 RUN_IMAGENET1K=0 \
RUN_METHOD_ABLATIONS=0 \
nohup bash run_server_experiments.sh \
  > "$LOG_ROOT/imagenet200_only.log" 2>&1 &
```

脚本可通过环境变量调节：`N_ANCHORS`、`SETUP_SAMPLES_PER_CLASS`、`HAM_EPOCHS`、`N_STEPS`、`DT`、`CANDIDATE_K`、`SIM_BATCH`、`HAM_BATCH_SIZE`。

## 9. 监控、停止和续跑

查看日志：

```bash
tail -f "$LOG_ROOT/server_experiments.log"
```

另开一个终端查看 GPU：

```bash
watch -n 2 nvidia-smi
```

查看进程：

```bash
ps -ef | grep -E "Imagenet_ood_experiment|run_server_experiments" | grep -v grep
```

正常停止一键脚本：

```bash
kill "$(cat "$LOG_ROOT/server_experiments.pid")"
```

检测器、共享特征和有效秩都有独立缓存。重新运行相同命令会自动加载；改变势能、质量模式、质量归一化、轨迹目标、随机种子或主要动力学参数时会建立新的检查点。

## 10. 输出文件

结果目录层次为：

```text
outputs/<benchmark>/<backbone>/seed<seed>/
  mass-<mode>-<normalization>_loss-<loss>-ts<steps>_pred-<source>/
```

每个配置中包含：

- `openood_metrics_<potential>.csv`：OpenOOD 原生指标。
- `openood_metrics_all_potentials.csv`：论文整理用长表。
- `run_config.json`：完整参数、模型来源和随机种子。
- `setup_features_*.pt`：所有势能共享的 class-balanced 特征。
- `anchor_image_effective_rank_*.pt`：有效秩质量缓存。
- `detector_*.pt`：不会跨实验设置误用的检测器。
- 可选 `openood_scores_*.pkl`：传入 `--save-scores` 后保存原始分数。

论文中至少报告三个种子的均值和标准差，并明确写出：候选类数 `candidate-k`、锚点数、轨迹步数、步长、质量模式、质量归一化、带宽目标、预测来源、GPU 和 OpenOOD 版本。

全部 seed 完成后自动汇总：

```bash
python summarize_openood_results.py \
  --output-root "$OUTPUT_ROOT" \
  --expected-seeds 0 1 2 \
  --strict-seeds
```

生成 `openood_all_runs_combined.csv` 和 `openood_summary_mean_std.csv`。若某个配置少了 seed，`--strict-seeds` 会停止并指出缺失项。

## 11. CIFAR 实验

CIFAR 不再使用 torchvision 的完整测试集、DTD 自带 split 或随机 10k 抽样。`ood_experiment.py` 默认读取 OpenOOD v1.5 的精确列表，并自动按种子寻找官方 ResNet-18 checkpoint。

从数据准备到 mean/std 汇总的完整步骤见 `CIFAR_AUTODL_GUIDE.md`。阶段入口为：

```bash
bash run_cifar_experiments.sh prepare
bash run_cifar_experiments.sh verify
bash run_cifar_experiments.sh smoke
SEEDS="0" bash run_cifar_experiments.sh reproduce
SEEDS="0" bash run_cifar_experiments.sh components
SEEDS="0" bash run_cifar_experiments.sh potentials
```

seed 0 验证成功后才运行 seed 1、2。正式结果文件是 `results/cifar/ood_results_cifar_v3.csv`；其中同时记录 OpenOOD 原生 `FPR95` 和另一种常见定义 `FPR95_IDTPR`，论文的 OpenOOD 表格使用前者。
