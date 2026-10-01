# CADRef 与 LogitGap：AutoDL 小白运行指南

这套程序只更换 OOD 分数，其他实验条件沿用此前版本：OpenOOD v1.5
数据划分、官方分类器权重、图像预处理、Near/Far-OOD 定义和评估指标均不变。

- CADRef：复现作者默认的 **CADRef-Energy**。正式实验用完整 ID 训练集计算
  类别特征均值与全局平均 Energy，不使用任何 OOD 样本调参。
- LogitGap：复现固定超参数版本。它不需要训练或统计 ID 训练集，也不使用 OOD
  验证集；比较 logit 数量按论文规则由类别数固定决定。
- CIFAR-10、CIFAR-100、ImageNet-200：各使用 3 个真实 checkpoint seed。
- ImageNet-1K ResNet-50、DenseNet-121：模型权重是固定的，所以各运行 1 次，
  不把同一个模型重复标成 3 个 seed。

## 1. 获取代码

在 AutoDL 数据盘上克隆本仓库（若仓库已经存在则执行 `git pull`）：

```bash
mkdir -p /root/autodl-tmp/hamood
cd /root/autodl-tmp/hamood
git clone https://github.com/zilamwong2005-math/Hamiltonian-Centroid-OOD.git Round6

export PROJECT=/root/autodl-tmp/hamood/Round6
cd "$PROJECT"
```

按照主 README 安装依赖和固定版本的 OpenOOD。数据集与 checkpoint 不随仓库
分发；已有 Round6 数据盘的用户可直接继续使用原路径。

## 2. 恢复与此前完全相同的环境变量

每次新开终端都重新执行这一段：

```bash
export PROJECT=/root/autodl-tmp/hamood/Round6
export DATA_ROOT="$PROJECT/data"
export OPENOOD_CKPT_ROOT="$PROJECT/openood_pretrained"
export OUTPUT_ROOT="$PROJECT/results_openood"
export JOURNAL_ROOT="$PROJECT/results/journal"
export LOG_ROOT="$PROJECT/logs/journal"
export TORCH_HOME="$PROJECT/cache/torch"
export PIP_CACHE_DIR="$PROJECT/cache/pip"
export TMPDIR="$PROJECT/cache/tmp"
export PYTHONPATH="$PROJECT/OpenOOD:${PYTHONPATH:-}"

mkdir -p "$JOURNAL_ROOT" "$LOG_ROOT" "$TORCH_HOME" \
  "$PIP_CACHE_DIR" "$TMPDIR" "$PROJECT/cache/pretrained"
cd "$PROJECT"
```

## 3. 先做代码验收

```bash
python -m py_compile \
  run_cadref_logitgap.py \
  summarize_cadref_logitgap.py

bash -n run_journal_experiments.sh

python -m pytest -q \
  tests/test_cadref_logitgap.py \
  tests/test_summarize_cadref_logitgap.py
```

只有 pytest 全部通过，才继续运行实验。再确认 GPU、数据和 checkpoint 仍在：

```bash
nvidia-smi

find "$OPENOOD_CKPT_ROOT" -name best.ckpt | sort

test -d "$DATA_ROOT/benchmark_imglist" \
  && echo "数据列表存在" \
  || echo "数据列表缺失，请先停止"
```

## 4. 先跑冒烟测试（不能写入论文）

冒烟测试每个 OOD 数据集只取 128 张图，CADRef 每类只取 2 张 ID 训练图，
目的是发现接口、显存和路径错误。

```bash
SMOKE_LOG="$LOG_ROOT/cadref_logitgap_smoke_master.log"
SMOKE_PID="$LOG_ROOT/cadref_logitgap_smoke_master.pid"

nohup bash run_journal_experiments.sh cadref_logitgap_smoke \
  > "$SMOKE_LOG" 2>&1 &

echo $! | tee "$SMOKE_PID"
echo "PID：$(cat "$SMOKE_PID")"
echo "日志：$SMOKE_LOG"
```

实时查看：

```bash
tail -f "$LOG_ROOT/cadref_logitgap_smoke_master.log"
```

退出实时显示只按 `Ctrl+C`，不会终止后台实验。

检查是否结束：

```bash
PID=$(cat "$LOG_ROOT/cadref_logitgap_smoke_master.pid")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd

cat "$JOURNAL_ROOT/cadref_logitgap/cadref_logitgap_smoke_completed.json"

find "$JOURNAL_ROOT/cadref_logitgap/smoke" \
  -type f \( -name 'cadref.csv' -o -name 'logitgap.csv' \) | wc -l

grep -E "Traceback|Error|failed|RuntimeError" \
  "$LOG_ROOT/cadref_logitgap_smoke_master.log" | tail -30
```

正确状态应为：

- 主进程已经不存在；
- completion JSON 中 `completed` 为 `true`；
- CSV 数量为 **22**；
- 错误检查无输出。

## 5. 运行正式全量实验

冒烟测试通过后再执行：

```bash
FULL_LOG="$LOG_ROOT/cadref_logitgap_full_master.log"
FULL_PID="$LOG_ROOT/cadref_logitgap_full_master.pid"

nohup bash run_journal_experiments.sh cadref_logitgap_full \
  > "$FULL_LOG" 2>&1 &

echo $! | tee "$FULL_PID"
echo "PID：$(cat "$FULL_PID")"
echo "日志：$FULL_LOG"
```

正式实验中，CADRef 会完整扫描 ID 训练集。ImageNet-1K 的两次统计最慢，
日志长时间停留在 `CADRef setup` 并不代表卡死，只要计数仍在增长即可。

实时监控：

```bash
watch -n 10 '
echo "===== 进程 ====="
PID=$(cat /root/autodl-tmp/hamood/Round6/logs/journal/cadref_logitgap_full_master.pid)
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
echo "===== GPU ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader
echo "===== 已完成 CSV ====="
find /root/autodl-tmp/hamood/Round6/results/journal/cadref_logitgap/full \
  -type f \( -name cadref.csv -o -name logitgap.csv \) | wc -l
echo "===== 最新日志 ====="
tail -c 20000 /root/autodl-tmp/hamood/Round6/logs/journal/cadref_logitgap_full_master.log \
  | tr "\r" "\n" | tail -30
'
```

若系统没有 `watch`，使用：

```bash
tail -f "$LOG_ROOT/cadref_logitgap_full_master.log"
```

## 6. 验收正式结果

```bash
PID=$(cat "$LOG_ROOT/cadref_logitgap_full_master.pid")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd

cat "$JOURNAL_ROOT/cadref_logitgap/cadref_logitgap_full_completed.json"

echo "正式 CSV 数量："
find "$JOURNAL_ROOT/cadref_logitgap/full" \
  -type f \( -name 'cadref.csv' -o -name 'logitgap.csv' \) | wc -l

echo "合并文件行数："
wc -l "$JOURNAL_ROOT/cadref_logitgap/cadref_logitgap_full_all_runs.csv"

echo "错误检查："
grep -E "Traceback|Error|failed|RuntimeError|incomplete" \
  "$LOG_ROOT/cadref_logitgap_full_master.log" | tail -50
```

正确状态应为：

- 正式 CSV 数量为 **22**；
- 合并 CSV 为 **167 行**（1 行表头 + 166 行结果）；
- completion JSON 中 `completed: true`；
- `fixed_imagenet_models_repeated: false`；
- 错误检查无输出。

## 7. 生成 Near/Far 汇总表

```bash
bash run_journal_experiments.sh cadref_logitgap_summary \
  2>&1 | tee "$LOG_ROOT/cadref_logitgap_summary.log"

find "$JOURNAL_ROOT/summary_cadref_logitgap" -type f -maxdepth 1 -print | sort
```

会生成原始 Near/Far 行、均值标准差、论文 CSV 与 LaTeX 表。归档结果时应同时
保存终端日志和 `cadref_logitgap_full_completed.json`，以便核对实验矩阵与协议。

## 8. 中断与续跑

所有正式结果是逐方法、逐模型保存的。实例意外中断时，重新执行第 2 节环境变量，
然后再次运行 `cadref_logitgap_full`；程序会显示 `[resume]` 并跳过已经完整保存的结果。
不要加 `--force`，否则会重复计算。
