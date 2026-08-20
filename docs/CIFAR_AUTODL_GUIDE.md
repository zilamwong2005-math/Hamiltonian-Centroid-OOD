# CIFAR-10/CIFAR-100 实验：AutoDL 新手操作手册

这部分必须先于 ImageNet 实验完成。ImageNet 数据保留在原位置即可，不需要删除或重新下载。

正式实验使用以下固定协议：

- OpenOOD v1.5 官方 `benchmark_imglist`；
- OpenOOD 官方 CIFAR ResNet-18，随机种子 0、1、2；
- CIFAR-10/CIFAR-100 各自的 Near-OOD 与 Far-OOD 全量数据；
- 禁止 STL-10、FakeData 或随机抽取 10,000 张等代理方式；
- 正式结果 `MaxEvalSamples=0`；
- 论文主表的 FPR@95 按 OpenOOD v1.5 原生实现计算。

## 1. 登录服务器并设置目录

每次重新开机或新开终端，都重新执行：

```bash
export PROJECT=/root/autodl-tmp/hamood/Round6
export DATA_ROOT=$PROJECT/data
export OPENOOD_CKPT_ROOT=$PROJECT/openood_pretrained
export ARCHIVE_ROOT=$PROJECT/archives
export RESULT_ROOT=$PROJECT/results/cifar
export LOG_ROOT=$PROJECT/logs/cifar
export TORCH_HOME=$PROJECT/cache/torch
export PYTHONPATH="$PROJECT/OpenOOD:${PYTHONPATH:-}"

cd "$PROJECT"

mkdir -p "$DATA_ROOT" "$OPENOOD_CKPT_ROOT" "$ARCHIVE_ROOT" \
  "$RESULT_ROOT" "$LOG_ROOT" "$TORCH_HOME"
```

确认位置：

```bash
pwd
echo "$DATA_ROOT"
echo "$OPENOOD_CKPT_ROOT"
df -h /root/autodl-tmp
```

## 2. 更新代码以后先做检查

```bash
cd "$PROJECT"

python -m py_compile \
  hamiltonian_detector.py \
  openood_cifar.py \
  ood_experiment.py \
  prepare_openood.py \
  verify_cifar_openood.py \
  summarize_cifar_results.py

python -m pytest -q tests
bash -n run_cifar_experiments.sh
```

三条命令必须全部成功。不要在测试报错的情况下开始租用 GPU 跑正式实验。

## 3. 自动准备 CIFAR OpenOOD 数据

执行：

```bash
cd "$PROJECT"

bash run_cifar_experiments.sh prepare \
  2>&1 | tee "$PROJECT/logs/cifar_prepare.log"
```

程序优先从 AutoDL 加速的 Hugging Face 镜像下载以下 OpenOOD 目录版 ZIP：

```text
cifar10.zip
cifar100.zip
tin.zip
mnist.zip
svhn.zip
texture.zip
places365.zip
```

每个 Hugging Face 文件都固定了大小和 SHA256。下载中断后，保留 `.part` 文件并重新执行相同命令，程序会续传。

数据会解压到：

```text
data/images_classic/
├── cifar10/
├── cifar100/
├── tin/
├── mnist/
├── svhn/
├── texture/
└── places365/
```

官方分类器会解压到：

```text
openood_pretrained/
├── cifar10_resnet18_32x32_base_.../s0|s1|s2/best.ckpt
└── cifar100_resnet18_32x32_base_.../s0|s1|s2/best.ckpt
```

### Google Drive 权重下载失败怎么办

CIFAR 数据有 Hugging Face 镜像，但 OpenOOD 官方 CIFAR 分类器仍来自 Google Drive。若准备命令停在权重下载：

1. 在本地浏览器下载 OpenOOD 官方 CIFAR-10 和 CIFAR-100 checkpoint ZIP；
2. 打开 AutoDL 网页文件管理器；
3. 进入 `/root/autodl-tmp/hamood/Round6/archives`；
4. 上传并严格改名为：

```text
cifar10_checkpoint.zip
cifar100_checkpoint.zip
```

5. 在服务器验证：

```bash
python -m zipfile -t "$ARCHIVE_ROOT/cifar10_checkpoint.zip"
python -m zipfile -t "$ARCHIVE_ROOT/cifar100_checkpoint.zip"
```

6. 只解压权重，不重新下载数据：

```bash
python -u prepare_openood.py \
  --benchmarks cifar10 cifar100 \
  --data-root "$DATA_ROOT" \
  --results-root "$OPENOOD_CKPT_ROOT" \
  --archive-dir "$ARCHIVE_ROOT" \
  --no-datasets \
  --keep-archives
```

## 4. 严格验证全部数据和权重

```bash
cd "$PROJECT"
bash run_cifar_experiments.sh verify \
  2>&1 | tee "$PROJECT/logs/cifar_verify.log"
```

该命令会逐行检查 OpenOOD 列表，并验证每一张图片确实存在。最后必须看到：

```text
CIFAR-10/CIFAR-100 OpenOOD v1.5 数据全部正常
```

随后应打印 6 个 checkpoint：两个数据集乘以三个种子。

## 5. 先运行冒烟测试

冒烟测试只取每个列表前 128 张图片，检测器只使用 5 个锚点、3 个轨迹步和 1 个训练 epoch。它只能检查程序，不能写入论文。

```bash
cd "$PROJECT"

nohup bash run_cifar_experiments.sh smoke \
  > "$PROJECT/logs/cifar_smoke_master.log" 2>&1 &

echo $! > "$PROJECT/logs/cifar_smoke.pid"
```

查看进度：

```bash
tail -f "$PROJECT/logs/cifar_smoke_master.log"
```

另开一个终端：

```bash
watch -n 2 nvidia-smi
```

冒烟结果单独保存在 `results/cifar_smoke`，不会混入正式 CSV。

## 6. 第一阶段：Gaussian 基线复现

先只跑 seed 0：

```bash
cd "$PROJECT"

SEEDS="0" nohup bash run_cifar_experiments.sh reproduce \
  > "$PROJECT/logs/cifar_reproduce_seed0_master.log" 2>&1 &

echo $! > "$PROJECT/logs/cifar_reproduce_seed0.pid"
```

完成后检查：

```bash
tail -100 "$PROJECT/logs/cifar_reproduce_seed0_master.log"
ls -lh "$RESULT_ROOT/ood_results_cifar_v3.csv"
```

确认 CIFAR-10 ID accuracy 通常高于 90%、CIFAR-100 高于 65%，六个 OOD 数据集都产生指标后，再补 seed 1、2：

```bash
SEEDS="1 2" nohup bash run_cifar_experiments.sh reproduce \
  > "$PROJECT/logs/cifar_reproduce_seed12_master.log" 2>&1 &

echo $! > "$PROJECT/logs/cifar_reproduce_seed12.pid"
```

这一阶段固定：

```text
Gaussian + uniform mass + static bandwidth loss
```

它对应旧程序的实现思路，但数据和模型已经严格对齐 OpenOOD v1.5。

## 7. 第二阶段：方法组件消融

仍然先跑 seed 0：

```bash
SEEDS="0" nohup bash run_cifar_experiments.sh components \
  > "$PROJECT/logs/cifar_components_seed0_master.log" 2>&1 &
```

脚本会补齐三组，B1 已由 reproduction 产生：

| 组别 | mass | bandwidth loss | 含义 |
|---|---|---|---|
| B1 | uniform | static | 旧实现基线 |
| B2 | effective_rank | static | 只加入有效秩质量 |
| B3 | uniform | trajectory | 只加入历史轨迹目标 |
| B4 | effective_rank | trajectory | 正文完整方法 |

seed 0 正常后：

```bash
SEEDS="1 2" nohup bash run_cifar_experiments.sh components \
  > "$PROJECT/logs/cifar_components_seed12_master.log" 2>&1 &
```

## 8. 第三阶段：五种势能函数

固定 B4 完整方法，比较 Gaussian、Laplacian、Cauchy、IMQ 和 Matérn-3/2：

```bash
SEEDS="0" nohup bash run_cifar_experiments.sh potentials \
  > "$PROJECT/logs/cifar_potentials_seed0_master.log" 2>&1 &
```

检查 seed 0 后再运行：

```bash
SEEDS="1 2" nohup bash run_cifar_experiments.sh potentials \
  > "$PROJECT/logs/cifar_potentials_seed12_master.log" 2>&1 &
```

同一数据集和种子使用同一个官方编码器；检测器缓存包含协议、模型来源、势能、质量、损失及动力学参数，不会跨设置误用。

## 9. 可选：两种骨干网络稳健性实验

这不是 OpenOOD 官方模型横向对比，而是补充实验。ResNet-18 与 DenseNet-BC-100 都从头训练：

```bash
SEEDS="0" nohup bash run_cifar_experiments.sh backbones \
  > "$PROJECT/logs/cifar_backbones_seed0_master.log" 2>&1 &
```

seed 0 正常后再补 1、2。不要把自训练 DenseNet 的结果和 OpenOOD 官方 ResNet-18 基线混成同一行。

## 10. 汇总 mean ± std

三个种子全部结束以后执行：

```bash
bash run_cifar_experiments.sh summarize
```

生成：

```text
results/cifar/summary/cifar_per_dataset_mean_std.csv
results/cifar/summary/cifar_group_mean_std.csv
```

第二个文件先在每个种子内部对数据集做宏平均，再跨种子计算 mean 和 std，不会因为 MNIST 图片较多而给它更大的权重。

## 11. 断点续跑、停止和异常处理

脚本只有在一个配置完整成功后才写 `.done` 文件。服务器中断后重新执行完全相同的阶段即可：

```bash
SEEDS="0" bash run_cifar_experiments.sh potentials
```

检测器 checkpoint 会自动恢复；CSV 按实验指纹和 OOD 数据集去重。

查看任务：

```bash
ps -ef | grep -E "ood_experiment|run_cifar_experiments" | grep -v grep
```

正常停止主脚本：

```bash
kill "$(cat "$PROJECT/logs/cifar_reproduce_seed0.pid")"
```

停止主脚本以后，再检查是否还有 Python 子进程。不要使用 `kill -9`，除非普通 `kill` 等待一段时间后仍无效。

## 12. 计算量提示

`candidate_k=0` 表示对所有类别精确计算；CIFAR-100 会明显慢于 CIFAR-10。第一次正式运行必须先完成 seed 0，不要直接同时开多个阶段。

如果只是判断显存能否承受，可以临时运行：

```bash
SEEDS="0" \
N_ANCHORS=20 \
N_STEPS=20 \
HAM_EPOCHS=2 \
TRAJECTORY_TRAIN_STEPS=5 \
HAM_TRAIN_SAMPLES_PER_CLASS=20 \
bash run_cifar_experiments.sh components
```

这属于性能试跑，参数已经改变，不能写进正式表格。正式实验恢复默认值，并保证 CSV 中：

```text
MaxEvalSamples = 0
Anchors = 80
Steps = 120
CandidateK = 0
HamTrainSamplesPerClass = 0
```

