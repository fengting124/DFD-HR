# DFD-HR 标准化实验执行指南

本指南把实验从“临时运行”变成可定位、可恢复、可审查、可归档的工程流程。当前仓库大部分自动化尚未实现，实施状态以 `TASK_INDEX.md` 为准。

## 1. 设计目标

- 更换机器或 Codex 会话后，可以只依赖仓库文档恢复上下文。
- 任一结果都能追溯到代码提交、冻结配置、数据协议、环境和初始权重。
- Jupyter 负责交互验证与分析，长期训练独立运行。
- 节点本地空间有限时，只保存必要产物并安全迁移。
- 失败实验也有编号、日志、原因和结论。
- 公共 Git 仓库不暴露实验室拓扑和用户信息。

## 2. 文件和存储职责

### 2.1 Git 仓库

保存：代码、配置模板、源 Notebook、测试、小型文档、实验摘要和注册表。

不保存：数据、环境、缓存、权重、执行后 Notebook、完整日志、特征和服务器快照。

### 2.2 数据根

`${DFDHR_DATA_ROOT}` 视为只读。训练、评估和 Notebook 不得向其中写入缓存、索引、预测或 checkpoint。

### 2.3 运行根

每个实验使用：

```text
${DFDHR_RUNTIME_ROOT}/runs/${RUN_ID}/
```

最小结构：

```text
manifest.yaml
config.resolved.yaml
command.sh
environment.txt
git.txt
metrics.jsonl
training.log
summary.md
checksums.sha256
checkpoints/best.pth
checkpoints/last.pth
notebooks_executed/
```

### 2.4 归档根

`${DFDHR_ARCHIVE_ROOT}` 只保存已经完成、校验且需要长期保留的产物。同步后比较 SHA-256，校验成功前不得删除本地副本。

## 3. RUN_ID

格式：

```text
<project>_<train-set>_<core-variable>_<YYYYMMDD>_<sequence>
```

示例：

```text
dfdhr_ffppc23_official-eval_20260720_001
dfdhr_ffppc23_amp-b1-acc8_20260720_002
```

RUN_ID 一经使用不得复用，包括失败和中止实验。

## 4. Git 工作流

### 4.1 分支类型

- `fix/*`：正确性缺陷。
- `feat/*`：可复用功能。
- `infra/*`：Jupyter、运行目录、归档和脱敏审计工具。
- `exp/*`：单一实验变量。
- `docs/*`：文档。
- `release/*`：复现版本冻结。

### 4.2 标准流程

```bash
git status --short --branch
git fetch origin
git switch -c <type>/<topic> origin/main
```

提交前：

```bash
git status --short
git diff --check
git diff -- <明确文件>
```

使用 `git add <明确文件>`，一个提交只处理一个逻辑变化。服务器审计、训练代码和实验配置不得混入同一分支。

### 4.3 推荐提交顺序

正式复现前建议独立完成：

1. `fix/validation-protocol`
2. `fix/final-test-metrics`
3. `feat/amp-grad-accum`
4. `feat/checkpoint-resume`
5. `fix/ddp-validation-sync`
6. `infra/jupyter-standard`
7. `feat/experiment-registry`

## 5. 实验生命周期

### 5.1 问题定义

启动前写清：研究问题、基线、唯一变化、成功标准和失败可排除的假设。

创建：实验分支、RUN_ID、manifest 和冻结配置。禁止正式启动时手工修改未提交配置。

### 5.2 Preflight

必须检查：

- Git 工作树干净，提交已记录。
- `DFDHR_PYTHON` 指向预期环境。
- 核心依赖、CUDA 和 GPU 可用。
- GPU 型号、数量和 VRAM 符合实验配置。
- 本地磁盘满足预算并保留安全余量。
- 数据根、JSON、train/validation/test 可解析。
- 初始权重严格加载，SHA-256 已记录。
- 输出目录不在代码目录和数据目录。

任一失败都不能启动正式训练。

### 5.3 两批次 Smoke Test

顺序：

1. 模型实例化。
2. 读取一个真实 batch。
3. forward、loss、backward、optimizer step。
4. 重复第二个 batch。
5. 检查 loss 有限、无 OOM。
6. 检查 backbone 冻结模块无梯度，Router/Adapter/Head 有梯度。
7. 保存 checkpoint 并严格重新加载。
8. 记录峰值显存、step time 和数据等待时间。

Smoke Test 只验证链路，不得作为研究结果。

### 5.4 Mini Run

固定小子集、固定随机种子、与正式训练一致的有效 batch，运行 1–3 epoch。验证：

- loss 与 validation 指标更新。
- best/last 语义正确。
- last 恢复后 epoch、optimizer、scheduler、scaler 和 RNG 连续。
- 日志、metrics.jsonl、退出状态完整。

### 5.5 正式训练

- 使用 `tmux` 或调度器启动。
- 多卡使用 `torchrun`。
- 记录每卡 micro-batch、world size、累积步数、有效 batch。
- 默认只保留 best 和 last。
- checkpoint 先写临时文件，再原子替换。
- 定期检查磁盘，但不高频遍历共享目录。
- 长期训练不依赖 Jupyter kernel 或浏览器存活。

### 5.6 验证和测试

- 训练集只训练。
- 明确 validation 只用于选 best。
- validation 缺失必须失败，禁止回退到 test。
- 跨域测试只在训练与选模完成后执行。
- 结果记录帧级、视频级指标、聚合方法、样本数和划分。
- 每个结果关联代码、配置、数据 JSON 和 checkpoint 哈希。

### 5.7 归档

长期保留：manifest、冻结配置、命令、环境、Git 信息、结构化指标、日志、best、last、总结和校验文件。

通常不保留：缓存、重复 epoch 权重、可再生特征和临时文件。

## 6. Jupyter 标准

### 6.1 使用场景

适合：环境检查、路径审计、checkpoint strict load、数据协议审计、单卡显存测试、小规模推理、结果绘图和只读训练监控。

不适合：长期训练、多卡长期 DDP、需要浏览器持续在线的任务。

### 6.2 Kernel

Kernel 必须绑定现有 DFD-HR Conda Python，而不是 Jupyter Server 自身环境。Notebook 第一单元记录：

```python
import os, sys, subprocess, torch
print(sys.executable)
print(os.getcwd())
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
```

### 6.3 Notebook 清单

当前实现状态以 `TASK_INDEX.md` 为准：

```text
[DONE] 00_environment_and_paths.ipynb
[DONE] 01_checkpoint_strict_load.ipynb
[DONE] 02_dataset_protocol_audit.ipynb
[DONE] 03_single_gpu_memory_smoke.ipynb
[DONE] 04_official_weight_eval.ipynb
[DONE] 05_training_monitor.ipynb
```

每个 Notebook 只解决一个问题，必须在 Restart Kernel 后 Run All 成功。源文件进入 Git，执行后副本进入运行目录。

## 7. 实验生命周期工具

运行前必须设置 `AGENTS.md` 规定的六个 `DFDHR_*` 路径变量，并使用 `${DFDHR_PYTHON}` 调用工具。初始化要求 Git 工作树干净、RUN_ID 未出现在 runtime/archive/registry，且输出、缓存和归档根均位于仓库与数据根之外。

```bash
scripts/init_experiment.sh \
  --run-id "$RUN_ID" \
  --config training/config/detector/dfd_hr.yaml \
  --base-config training/config/test_config.yaml \
  --dataset-json "$DATASET_JSON" \
  --objective "$OBJECTIVE" \
  --hypothesis "$HYPOTHESIS" \
  --success-criterion "$SUCCESS_CRITERION" \
  --single-changed-variable "$CHANGED_VARIABLE" \
  --node-role "$NODE_ROLE" \
  --command "$COMMAND" \
  --epochs "$EPOCHS" \
  --precision "$PRECISION" \
  --gpu-count "$GPU_COUNT" \
  --gpu-indices "$GPU_INDICES" \
  --per-gpu-batch "$PER_GPU_BATCH" \
  --gradient-accumulation-steps "$ACCUMULATION" \
  --workers "$WORKERS" \
  --maximum-local-size-gib "$LOCAL_BUDGET_GIB"
```

初始化会创建 T1.4 的最小目录、合并并冻结配置、把训练输出重定向到该 RUN_ID、关闭特征保存、采集白名单环境/Git 信息并生成 SHA-256。随后使用：

```bash
scripts/capture_experiment_metadata.sh --run-id "$RUN_ID"
scripts/freeze_experiment_config.sh \
  --run-id "$RUN_ID" \
  --config training/config/detector/dfd_hr.yaml \
  --base-config training/config/test_config.yaml
scripts/checksum_experiment.sh --run-id "$RUN_ID"
scripts/verify_experiment.sh --run-id "$RUN_ID"
```

实验完成后再执行 `experiment_lifecycle.py register` 写入脱敏 registry。registry 禁止绝对内部路径、IP、SSH、token 或凭据，只保存角色和 `${DFDHR_*}` 相对位置。

归档默认只预演：

```bash
scripts/archive_experiment.sh --run-id "$RUN_ID"
```

只有明确批准后才增加 `--execute`。执行模式复制到 archive、比较全部 SHA-256，且无论成功或失败都保留 runtime 源目录；工具不提供自动删除能力。

## 8. 节点迁移和硬件审计

真实节点别名和路径保存在 `.local/infrastructure.yaml`，公共文档仅使用角色名。

新控制节点先完成自身环境、Git、测试和 SSH 探测能力检查。两台新增节点分别检查：CPU、RAM、GPU、VRAM、驱动、CUDA compatibility、当前利用率、scratch、数据目录、JSON 路径、环境、官方权重和单元测试。

同名 `/scratch` 路径不代表相同数据。跨节点训练前必须比较 JSON 哈希、数据有限计数、抽样路径和环境清单。

审计阶段不安装软件、不复制数据、不修改系统设置、不读取其他用户的私有内容。

## 9. 存储受限节点

存储受限节点用于显存 Smoke、推理或边界明确的小实验。要求：

- manifest 写明最大本地空间。
- `save_feat: false`。
- 仅保留 best/last。
- 缓存与日志有明确目录。
- 先本地完整原子写入，再同步到归档节点。
- 同步成功并校验哈希后才清理。

不建议直接把远端网络目录作为 `torch.save()` 目标。

## 10. 结构化指标

`metrics.jsonl` 每行至少包含：时间戳、RUN_ID、epoch、global_step、loss、learning rate、step/data time、显存、validation 指标和磁盘余量。训练监控 Notebook 只读取该文件和日志，不访问训练进程内存。

生命周期冻结配置将 `metrics_jsonl` 指向 RUN_ID 根目录。Trainer 在首批、`metrics_interval`（默认沿用 `rec_iter`）和末批追加 train 事件，并在 validation/test 完成后追加聚合事件；DDP 仅 rank 0 写入。每个事件使用单次追加写入，完整行立即同步，非有限观测值记录为 `null`。读取器只容忍最后一行因中断而不完整，中间损坏仍立即报错。

`05_training_monitor.ipynb` 仅依赖标准库和当前仓库模块，读取 `${DFDHR_METRICS_JSONL}` 与可选 `${DFDHR_TRAINING_LOG}`，最多显示最近 20 个训练事件和 40 行日志。执行后副本必须写入 `${DFDHR_RUNTIME_ROOT}`，不得提交到 Git。

## 11. 完成定义

一个实验只有同时具备 RUN_ID、问题定义、干净提交、冻结配置、数据协议、环境记录、命令、日志、指标、best/last、归档校验和明确结论，才算完成。

只有 checkpoint、只有 Notebook 输出或只有口头结果，均不构成可复现实验。
