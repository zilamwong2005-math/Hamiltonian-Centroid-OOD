# 论文重写前最后一批必要实验（AutoDL 新手版）

这批工作不再重复已经完成的 OOD 准确率实验，只补齐审稿人很可能追问的三项：

1. 所有方法在同一协议下的统一排名、逐任务胜负和完整性检查；
2. ASH、DICE、SHE、RMDS、RankFeat、Scale 与本文方法在同一 RTX 4090 D、
   batch size=1 下的延迟、显存、附加状态和一次性准备开销；
3. 50 个扩展基线结果的复现审计，以及 RMDS 流式/向量化实现的代数等价检查。

以下命令都在 AutoDL 终端执行。以 `$` 开头的名字是环境变量，复制整段即可，
不要把命令提示符 `root@...#` 一起复制。

## 1. 解压更新包

先把 `hamood_final_required_experiments_20260818.zip` 上传到：

```text
/root/autodl-tmp/hamood/Round6/
```

然后执行：

```bash
export PROJECT=/root/autodl-tmp/hamood/Round6
cd "$PROJECT"

sha256sum hamood_final_required_experiments_20260818.zip
python -m zipfile -t hamood_final_required_experiments_20260818.zip
python -m zipfile -e hamood_final_required_experiments_20260818.zip "$PROJECT"
```

## 2. 更新后的静态检查和单元测试

```bash
export DATA_ROOT=$PROJECT/data
export OPENOOD_CKPT_ROOT=$PROJECT/openood_pretrained
export JOURNAL_ROOT=$PROJECT/results/journal
export LOG_ROOT=$PROJECT/logs
export TORCH_HOME=$PROJECT/cache/torch
export PYTHONPATH="$PROJECT/OpenOOD:${PYTHONPATH:-}"

mkdir -p "$LOG_ROOT" "$JOURNAL_ROOT"

python -m py_compile \
  benchmark_extended_baseline_efficiency.py \
  summarize_efficiency_comparison.py \
  summarize_all_method_comparison.py \
  audit_extended_baseline_reproduction.py

bash -n run_journal_experiments.sh
python -m pytest -q tests
```

## 3. 先做复现审计（不占 GPU）

```bash
bash run_journal_experiments.sh baselines_extended_audit \
  2>&1 | tee "$LOG_ROOT/extended_baseline_audit.log"
```

成功时日志末尾必须包含：

```text
"expected_formal_runs": 50
"audited_formal_runs": 50
"expected_matrix_complete": true
"no_smoke_results": true
"rankfeat_full_svd_all_runs": true
"passed": true
```

两项 RMDS 数值等价检查的 `max_abs_difference` 应不大于 `1e-10`。
这证明我们的流式协方差和 GPU 向量化改写与直接公式一致；它不等于“完全复现原论文
已发表数字”，论文里必须使用“local OpenOOD v1.5 reproduction”这一准确表述。

## 4. 统一生成全方法比较表（不占 GPU）

```bash
bash run_journal_experiments.sh all_method_summary \
  2>&1 | tee "$LOG_ROOT/all_method_summary.log"
```

结果目录：

```text
/root/autodl-tmp/hamood/Round6/results/journal/summary_all_methods/
```

其中：

- `all_methods_task_metrics.csv`：每种方法在 8 个 Near/Far 任务上的均值、方差和排名；
- `all_methods_ranking.csv`：仅在任务覆盖数可比时解释平均排名；
- `locked_pairwise_deltas.csv`：本文锁定方法相对每个基线的逐任务差值；
- `locked_win_summary.csv`：AUROC、FPR95 和两者同时获胜的次数；
- `main_comparison_paper_table.tex`：论文主表候选。

## 5. 本文方法和 MSP 的 batch-1 效率

```bash
EFF_B1_LOG=$LOG_ROOT/efficiency_batch1.log
EFF_B1_PID=$LOG_ROOT/efficiency_batch1.pid

nohup bash run_journal_experiments.sh efficiency_b1 \
  > "$EFF_B1_LOG" 2>&1 &
echo $! | tee "$EFF_B1_PID"
```

实时查看：

```bash
PID=$(cat "$EFF_B1_PID")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader
tail -c 30000 "$EFF_B1_LOG" | tr '\r' '\n' | tail -80
```

若 `ps` 只显示表头，说明进程已经结束，再检查日志即可。

## 6. 五个扩展基线的同硬件效率

DICE、SHE、RMDS 第一次运行会各自遍历完整 ID 训练集建立统计量，因此这一阶段
明显比普通推理计时慢。这正是要测量的一次性离线成本。完成后统计量会缓存，意外中断时
重新运行同一命令即可续跑；不要删除：

```text
/root/autodl-tmp/hamood/Round6/results/journal/efficiency/setup_cache/
```

启动命令：

```bash
EFF_EXT_LOG=$LOG_ROOT/efficiency_extended.log
EFF_EXT_PID=$LOG_ROOT/efficiency_extended.pid

nohup bash run_journal_experiments.sh efficiency_extended \
  > "$EFF_EXT_LOG" 2>&1 &
echo $! | tee "$EFF_EXT_PID"
```

实时查看：

```bash
PID=$(cat "$EFF_EXT_PID")
ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader
tail -c 30000 "$EFF_EXT_LOG" | tr '\r' '\n' | tail -80
```

ASH 使用正式实验中已经锁定的 APS percentile。我们不重新调参，也不把数据加载时间
冒充校准时间，所以其 `SetupSeconds` 为缺失值，并在 `SetupProtocol` 中明确说明。
RankFeat 不需要离线训练集统计，因此离线 setup 为 0。

## 7. 补测最强直接对手 Scale

Scale 不需要遍历训练集，四个计时任务通常很快：

```bash
bash run_journal_experiments.sh efficiency_scale \
  2>&1 | tee "$LOG_ROOT/efficiency_scale.log"
```

## 8. 生成最终效率表

前两个效率阶段都结束后执行：

```bash
bash run_journal_experiments.sh efficiency_required_summary \
  2>&1 | tee "$LOG_ROOT/efficiency_required_summary.log"
```

成功后应生成 4 个基准 × 9 个端点，共 36 行：

```bash
wc -l \
  "$JOURNAL_ROOT/efficiency/summary_required/efficiency_comparison_batch1.csv"
```

由于 CSV 有一行表头，预期输出是 `37`。论文表位于：

```text
/root/autodl-tmp/hamood/Round6/results/journal/efficiency/summary_required/
```

## 9. 最终总检查

```bash
cat \
  "$JOURNAL_ROOT/audit_extended_baselines/extended_baseline_reproduction_audit.json"

wc -l \
  "$JOURNAL_ROOT/summary_all_methods/all_methods_task_metrics.csv" \
  "$JOURNAL_ROOT/efficiency/summary_required/efficiency_comparison_batch1.csv"

find "$JOURNAL_ROOT" -type f -name "*.failed.txt" -print
```

如果审计 `passed=true`、效率表 37 行、没有 `.failed.txt`，这批实验就完成了。
之后不再根据测试结果反复改规则或追加无计划试验，直接进入论文重写，以避免形成
test-set-driven 的实验叙事。
