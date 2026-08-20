# ASH / DICE / SHE / RMDS / RankFeat：AutoDL 复现实验

本批实验沿用此前的 OpenOOD v1.5 数据、官方分类器权重、Near-OOD / Far-OOD
划分和指标。CIFAR-10、CIFAR-100、ImageNet-200 各运行三个官方分类器种子；
ImageNet-1K 使用同一个固定的 torchvision ResNet-50 V1，因此只运行 seed 0，
不能把同一个分类器重复三次伪装成三个独立种子。

## 1. 更新后先检查

```bash
export PROJECT=/root/autodl-tmp/hamood/Round6
export DATA_ROOT=$PROJECT/data
export OPENOOD_CKPT_ROOT=$PROJECT/openood_pretrained
export TORCH_HOME=$PROJECT/cache/torch
export PYTHONPATH="$PROJECT/OpenOOD:${PYTHONPATH:-}"

cd "$PROJECT"

python -m py_compile \
  run_openood_baselines.py \
  summarize_extended_baselines.py \
  summarize_journal_experiments.py \
  ood_experiment.py \
  OpenOOD/openood/postprocessors/dice_postprocessor.py \
  OpenOOD/openood/postprocessors/she_postprocessor.py \
  OpenOOD/openood/postprocessors/rmds_postprocessor.py

bash -n run_journal_experiments.sh
python -m pytest -q tests
```

## 2. 先做 CIFAR-10 冒烟测试

冒烟结果每个测试数据集最多 128 张图片，只检查接口、显存和输出格式，不能写入论文。
DICE、SHE 和 RMDS 的统计量仍由完整 ID 训练集建立。

```bash
mkdir -p "$PROJECT/logs"

bash run_journal_experiments.sh baselines_extended_smoke \
  2>&1 | tee "$PROJECT/logs/baselines_extended_smoke.log"
```

应当得到 5 个 CSV，且没有失败文件：

```bash
find "$PROJECT/results/journal/smoke_baselines_extended" \
  -type f -name "*.csv" | wc -l

find "$PROJECT/results/journal/smoke_baselines_extended" \
  -type f -name "*.failed.txt" -print
```

## 3. 正式实验一：ASH、DICE、SHE

```bash
FEATURE_LOG=$PROJECT/logs/baselines_extended_feature.log
FEATURE_PID=$PROJECT/logs/baselines_extended_feature.pid

nohup bash run_journal_experiments.sh baselines_extended_feature \
  > "$FEATURE_LOG" 2>&1 &
echo $! | tee "$FEATURE_PID"
```

实时查看：

```bash
tail -f "$FEATURE_LOG"
```

按 `Ctrl+C` 只会退出查看，不会终止后台实验。重新检查进程和最新状态：

```bash
PID=$(cat "$FEATURE_PID")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd

tail -c 30000 "$FEATURE_LOG" | tr '\r' '\n' | tail -80
```

## 4. 正式实验二：RankFeat

RankFeat 对两层特征做逐样本 SVD。程序已自动把批大小设为 CIFAR=64、
ImageNet-200=32、ImageNet-1K=8。正式阶段采用完整 SVD；不要添加
`--rankfeat-accelerate`，否则就变成近似幂迭代版本。

```bash
RANK_LOG=$PROJECT/logs/baselines_rankfeat.log
RANK_PID=$PROJECT/logs/baselines_rankfeat.pid

nohup bash run_journal_experiments.sh baselines_rankfeat \
  > "$RANK_LOG" 2>&1 &
echo $! | tee "$RANK_PID"
```

查看状态：

```bash
PID=$(cat "$RANK_PID")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader
tail -c 30000 "$RANK_LOG" | tr '\r' '\n' | tail -80
```

## 5. 正式实验三：RMDS

RMDS 最耗时，放在最后。当前实现与 OpenOOD 的均值、协方差和相对马氏距离公式
等价，但使用流式协方差和 GPU 向量化评分，不再保存完整训练特征，也不再逐类别
重复计算二次型。

```bash
RMDS_LOG=$PROJECT/logs/baselines_rmds.log
RMDS_PID=$PROJECT/logs/baselines_rmds.pid

nohup bash run_journal_experiments.sh baselines_rmds \
  > "$RMDS_LOG" 2>&1 &
echo $! | tee "$RMDS_PID"
```

查看状态：

```bash
PID=$(cat "$RMDS_PID")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
free -h
nvidia-smi
tail -c 30000 "$RMDS_LOG" | tr '\r' '\n' | tail -80
```

## 6. 完整性检查与论文表格

五种方法应生成 50 个正式 CSV：三个小/中型基准各 5 方法 × 3 种子，
ImageNet-1K 为 5 方法 × 1 固定分类器。

```bash
BASELINE_ROOT=$PROJECT/results_openood_baselines

echo "正式 CSV 数量："
find "$BASELINE_ROOT" -type f \
  \( -name "ash.csv" -o -name "dice.csv" -o -name "she.csv" \
     -o -name "rmds.csv" -o -name "rankfeat.csv" \) | wc -l

echo "残留失败："
find "$BASELINE_ROOT" -type f -name "*.failed.txt" -print

bash run_journal_experiments.sh baselines_extended_summary \
  2>&1 | tee "$PROJECT/logs/baselines_extended_summary.log"
```

汇总文件位于：

```text
/root/autodl-tmp/hamood/Round6/results/journal/summary_extended_baselines/
```

重点文件是 `extended_baselines_paper_table.csv`、
`extended_baselines_paper_table.tex` 和 `extended_baselines_completeness.csv`。

## 7. 失败后如何续跑

每个方法/基准/种子的 CSV 都是独立断点。成功项会自动跳过，失败项会保留
`.failed.txt`。修复后重新执行同一个阶段即可，不要加 `--force`，否则会重跑全部结果。

ASH 是唯一启用 OpenOOD APS 的本批方法，它只使用标准 ID validation 和 OOD
validation 选 percentile；Near-OOD / Far-OOD 测试集不参与选择。这个协议必须在论文中披露。
