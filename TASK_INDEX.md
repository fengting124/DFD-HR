# DFD-HR 当前任务索引

本文件是新机器、新 Codex 会话和实验中断后的任务入口。通用规范见 `AGENTS.md`，详细实验流程见 `docs/EXPERIMENT_WORKFLOW.md`。

> 当前状态：此前提出的标准化实验目录、模板、Jupyter Notebook、训练代码修复和正式复现大部分尚未完成。除有明确证据的项目外，全部按 `TODO` 处理，不得把计划描述为已实现。

## 1. 启动顺序

每次更换机器或聊天记录丢失后：

```bash
git status --short --branch
git fetch --all --prune
git switch main
git pull --ff-only
git switch <当前任务分支>
git log --oneline --decorate -12
```

随后按顺序阅读：

1. `AGENTS.md`
2. `TASK_INDEX.md`
3. `docs/EXPERIMENT_WORKFLOW.md`
4. `docs/controller_migration_and_node_audit_plan.md`
5. 与当前任务直接相关的代码、配置和测试

真实节点别名、内部路径和连接方式只写入 Git 外文件：

```text
.local/infrastructure.yaml
```

模板为 `templates/infrastructure.local.example.yaml`。该文件应位于共享项目目录中，使更换控制节点后仍可读取，但必须保持未跟踪状态。

## 2. 状态定义

- `TODO`：尚未开始或无完成证据。
- `ACTIVE`：当前正在执行。
- `BLOCKED`：存在已记录阻塞。
- `DONE`：已完成并有提交、日志或报告证据。
- `SUPERSEDED`：已由新方案替代。

Current task branch: `feat/amp-grad-accum`

Completed scope: T2.3 AMP and gradient accumulation

Next task branch: `feat/checkpoint-resume`

## 3. 当前里程碑

建立新的控制节点，审计两台新增计算节点的硬件、数据、环境和存储状态；然后完成 DFD-HR 官方权重校准、单卡 Smoke Test、两卡 DDP Smoke Test，最后才进入 Mini Run 和完整训练。

## P0：控制节点迁移与新增节点审计

### T0.1 建立新控制节点

**状态：DONE**

- [x] SSH 连接稳定。
- [x] 共享项目目录可见。
- [x] `docs/experiment-workflow` 可检出，工作树干净。
- [x] `.local/infrastructure.yaml` 已根据模板创建且未被 Git 跟踪。
- [x] 预期 Conda/Python 环境存在并可导入 PyTorch、Transformers、OpenCV 等核心依赖。
- [x] CUDA 和 GPU 可见。
- [x] 当前测试集通过。
- [x] 能对目标节点执行单次、有限超时、非交互 SSH 检查。
- [x] 审计输出只写入当前用户的 Git 外运行目录。

证据：时间戳、hostname、Git 分支/提交、Python 路径、PyTorch/CUDA、测试摘要、scratch 空间。

完成证据（2026-07-20 16:36 +08:00，原始节点名和内部路径仅保存在 Git 外本地审计记录）：

- 控制节点候选可见共享仓库，分支 `docs/experiment-workflow`，提交 `366b84eb157aac9f78edf30d80ebbafa4aeb0f0c`。
- `.local/infrastructure.yaml` 已创建，`git check-ignore -v` 确认为 `.gitignore` 的 `.local/` 规则忽略。
- 约定 DFD-HR Python 可执行，导入通过：PyTorch `2.3.1+cu121`、Transformers `4.44.2`、OpenCV `4.5.4`、lmdb `1.4.1`、sklearn `1.3.0`；CUDA runtime `12.1`，CUDA available，GPU count `2`。
- 本机 GPU：2 x NVIDIA GeForce RTX 2080 Ti，单卡 11264 MiB，采样时 GPU 利用率 0%。
- 本机 CPU/RAM：Intel Core i7-9700K，1 socket，8 physical cores，8 threads；RAM 62 GiB。
- 本机 scratch：约 1.8 TiB total，1005 GiB available，inode 使用约 7%。
- 单元测试：`${DFDHR_PYTHON} -m unittest discover -s tests -v`，9 tests OK。
- 远端有限 SSH 探测能力确认：新增节点 A/B 均可一次性非交互返回 hostname。

提交：未提交，当前仅更新任务索引和 Git 外 `.local/infrastructure.yaml`。

下一步：修复新增节点 A/B 的本地环境、JSON 注册表和官方权重缺口后，才能进入 Smoke Test 前置检查。

### T0.2 原控制节点有限重检

**状态：BLOCKED**

当前观察为连接超时。只允许从新控制节点执行一次有限重检：

- [x] 记录 DNS/别名解析。
- [x] 记录 TCP/SSH 结果。
- [x] 记录时间戳。

不得高频重试，不得修改 SSH、网络、路由或防火墙。只有新鲜成功连接证据才能恢复其控制、训练或归档角色。

阻塞证据（2026-07-20 16:37 +08:00）：从新控制节点执行一次有限超时 SSH 重检，原控制节点返回 `No route to host`。未重试，未修改 SSH、网络、路由或防火墙。

提交：未提交。

下一步：等待外部网络/节点恢复证据；恢复前不得作为控制、训练或归档节点。

### T0.3 扩展审计目标

**状态：DONE**

从 `.local/infrastructure.yaml` 读取全部现有节点和两台新增节点。每台节点仅收集：

- [x] 可达性和规范 hostname。
- [x] CPU 型号、插槽、物理核心、线程。
- [x] 总 RAM、可用 RAM。
- [x] GPU 型号、数量、VRAM、驱动、CUDA compatibility。
- [x] 当前 GPU 利用率和是否适合新的独占任务。
- [x] scratch 文件系统、总量、普通用户可用空间、使用率、inode。
- [x] 共享代码目录可见性。
- [x] 当前用户本地运行目录可用性。
- [ ] 是否存在调度器和监控能力；不安装任何组件。

禁止读取其他用户的命令行、环境变量、shell 历史、项目内容、数据内容、私钥或凭据。

完成证据（2026-07-20 16:37-16:38 +08:00）：已从 `.local/infrastructure.yaml` 读取控制节点候选、原控制节点、新增节点 A、新增节点 B、高显存 Smoke 候选和长期占用节点角色；本轮只对控制节点候选、原控制节点、新增节点 A/B 执行有限审计。新增节点 A/B 均可达并可见共享仓库；原控制节点不可达。未安装组件，未复制数据，未修改系统配置。

提交：未提交。

下一步：若需要选择归档节点或 Smoke 候选，单独审计高显存 Smoke 候选和长期占用节点的当前状态。

### T0.4 新增节点 A 审计

**状态：DONE**

硬件：

- [x] CPU/RAM。
- [x] GPU 型号、数量、单卡 VRAM。
- [x] NVIDIA 驱动和 CUDA compatibility。
- [x] 当前 GPU 状态。
- [x] scratch 和根分区容量。
- [x] 从控制节点可达。

DFD-HR 就绪性：

- [x] 共享仓库可见。
- [x] 数据根存在。
- [ ] 必需数据集目录存在。
- [ ] 数据集 JSON 注册表存在。
- [ ] train/validation/test 抽样路径可解析。
- [x] DFD-HR 环境是否存在；若不存在只记录，不安装或复制。
- [ ] 官方 checkpoint 是否存在，SHA-256 是否匹配。
- [ ] 环境存在时运行单元测试。
- [x] 输出空间是否适合 Smoke Test 或训练。

完成证据（2026-07-20 16:37-16:38 +08:00）：

- 可达；共享仓库分支 `docs/experiment-workflow`，提交 `366b84eb157aac9f78edf30d80ebbafa4aeb0f0c`。
- CPU/RAM：Intel Core i9-12900K，1 socket，16 physical cores，24 threads；RAM 62 GiB，采样 available 55 GiB。
- GPU：2 x NVIDIA GeForce RTX 3090，单卡 24576 MiB；driver `580.159.03`，CUDA compatibility `13.0`。
- GPU 采样状态：两卡正在使用，utilization 约 99%/49%，free VRAM 约 16.1/16.6 GiB；不建议立即调度独占训练。
- scratch：约 3.6 TiB total，2.5 TiB available，inode 使用约 4%；根分区约 457 GiB total，356 GiB available。
- 数据根存在；有限目录检查存在 FaceForensics++、Celeb-DF-v2、DF40、DFDC、DFDCP、WildDeepfake，缺 DeeperForensics、FFIW。
- JSON 注册表 symlink 存在但目标缺失，无法进行 JSON SHA-256 和 train/validation/test 抽样路径解析。
- 约定 DFD-HR Python 环境缺失；因此未运行单元测试。
- 官方 checkpoint 未在约定搜索深度内发现，未获得 SHA-256。
- 推荐角色：暂不作为训练节点；可在补齐环境、JSON、权重并重新检查 GPU 空闲后再考虑 Smoke/训练。

提交：未提交。

下一步：只在取得确认后准备本地环境、JSON 注册表和官方权重；补齐前不得启动 Smoke Test 或训练。

工作树：当前因本次任务索引更新而非干净；`.local/infrastructure.yaml` 被 Git 忽略。

### T0.5 新增节点 B 审计

**状态：DONE**

使用与 T0.4 相同的清单，单独记录。不得根据节点 A 推断节点 B 的硬件、数据或环境一致性。

完成证据（2026-07-20 16:37-16:38 +08:00）：

- 可达；共享仓库分支 `docs/experiment-workflow`，提交 `366b84eb157aac9f78edf30d80ebbafa4aeb0f0c`。
- CPU/RAM：Intel Core i9-10900K，1 socket，10 physical cores，20 threads；RAM 62 GiB，采样 available 22 GiB。
- GPU：2 x NVIDIA GeForce RTX 3090，单卡 24576 MiB；driver `580.159.03`，CUDA compatibility `13.0`。
- GPU 采样状态：两卡满载，free VRAM 约 1.6/1.9 GiB；不适合新的独占任务。
- scratch：约 3.6 TiB total，376 GiB available，使用率 90%，inode 使用约 17%；根分区约 457 GiB total，356 GiB available。
- 数据根存在；有限目录检查存在 FaceForensics++、Celeb-DF-v2、DF40、DFDC、DFDCP、WildDeepfake，缺 DeeperForensics、FFIW。
- JSON 注册表 symlink 存在但目标缺失，无法进行 JSON SHA-256 和 train/validation/test 抽样路径解析。
- 约定 DFD-HR Python 环境缺失；因此未运行单元测试。
- 官方 checkpoint 未在约定搜索深度内发现，未获得 SHA-256。
- 推荐角色：暂不作为 Smoke 或训练节点；即使补齐环境，也需先释放 GPU 并缓解 scratch 空间压力。

提交：未提交。

下一步：优先确认是否需要清理/扩容当前用户 scratch 预算和补齐环境；补齐前不得启动 Smoke Test 或训练。

工作树：当前因本次任务索引更新而非干净；`.local/infrastructure.yaml` 被 Git 忽略。

### T0.6 数据集与环境一致性

**状态：BLOCKED**

第一阶段至少检查：

- [ ] FaceForensics++ c23：train/validation/test。
- [x] Celeb-DF-v2。
- [ ] 当前测试配置引用的 DFD、DFDC、DFDCP、DeeperForensics、WildDeepfake、FFIW。
- [x] 当前复现计划需要的 DF40 方法子集。

方法：目录存在性和有限计数、JSON SHA-256、各 split 抽样路径、环境包清单、官方权重哈希。审计阶段不得复制数据或环境。

阻塞证据（2026-07-20 16:38 +08:00）：

- 控制节点候选数据根存在，且本地官方 checkpoint `dfd_hr_ffpp.pth` 存在，size 1.6 GiB，SHA-256 `bbf2b1c805fe545a1ac1ead36e3c4341a78b5a6334766dd875dc6d1a940944ec`。
- 新增节点 A/B 的数据根均存在，但 JSON 注册表 symlink 目标缺失，无法计算 JSON SHA-256 或解析 split 抽样路径。
- 新增节点 A/B 约定 DFD-HR Python 环境缺失，无法采集环境包清单或运行单元测试。
- 新增节点 A/B 未找到官方 checkpoint，无法比较 SHA-256。
- 新增节点 A/B 有 FaceForensics++、Celeb-DF-v2、DF40、DFDC、DFDCP、WildDeepfake；缺 DeeperForensics、FFIW。DFD/DeepFakeDetection 需进一步确认是否映射到现有 DFDCP/FaceForensics++ 目录或另有目录名。

提交：未提交。

下一步：补齐或重新指向新增节点 A/B 的 JSON 注册表、环境和官方权重后，重新执行 JSON SHA-256、split 抽样路径和单元测试审计。

### T0.7 生成脱敏结果报告

**状态：DONE**

目标：`docs/controller_migration_and_node_audit_results.md`

允许：时间戳、匿名节点角色、通用硬件、当前用户可见容量、数据/环境就绪状态、调度建议。

禁止：真实服务器名、IP/端口、其他用户身份/PID/目录大小、SSH 配置、凭据、内部绝对路径。

完成证据：

- 报告：`docs/controller_migration_and_node_audit_results.md`
- 任务提交：`75ea940`
- 报告提交：`e236846`
- 合并 PR：`#1`
- `main` 合并提交：`fc37a2f`

下一步：基础设施审计已收口；新增节点就绪性作为独立任务处理，不阻塞控制节点上的 Jupyter 标准化和代码正确性修复。

## P1：标准化实验基础设施

### T1.1 本地路径与环境变量

**状态：TODO**

- [ ] 建立并验证 `DFDHR_REPO_ROOT`、`DFDHR_PYTHON`、`DFDHR_DATA_ROOT`、`DFDHR_RUNTIME_ROOT`、`DFDHR_ARCHIVE_ROOT`、`DFDHR_CACHE_ROOT`。
- [ ] 数据目录只读。
- [ ] 训练输出不写入代码目录或共享数据目录。
- [ ] 缓存不写入仓库。

已知前置问题：`00_environment_and_paths.ipynb` 的只读审计确认当前 detector YAML 使用仓库相对 `log_dir`/`logdir`，按仓库根解析时会把日志和 checkpoint 写入仓库。正式运行前必须由运行配置显式覆盖到 `${DFDHR_RUNTIME_ROOT}`；本次 Jupyter 任务未修改训练配置。

### T1.2 实验注册表

**状态：TODO**

- [ ] 建立 `registry/experiments.csv`。
- [ ] 每个运行使用唯一 `RUN_ID`。
- [ ] 记录状态、目标、分支、提交、配置、节点角色、GPU、结果和归档位置。

### T1.3 实验模板

**状态：TODO**

以下模板已经规划，但只有文件存在不代表流程已实现：

- [ ] `templates/experiment_manifest.yaml`
- [ ] `templates/experiment_summary.md`
- [ ] `templates/infrastructure.local.example.yaml`
- [ ] `templates/pull_request_template.md`
- [ ] 运行目录初始化脚本
- [ ] 配置冻结与哈希脚本
- [ ] 环境与 Git 元数据采集脚本
- [ ] 产物同步与哈希校验脚本

### T1.4 运行目录与存储预算

**状态：TODO**

每个运行目录至少包含：

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
```

默认只保留 `best` 和 `last` checkpoint，关闭特征转储；本地空间预算必须写入 manifest。

## P2：训练与评估正确性阻塞项

### T2.1 显式验证集协议

**状态：DONE**

- [x] FaceForensics++ validation 显式配置。
- [x] validation 缺失时立即失败，禁止回退到 test。
- [x] 增加有效和无效划分测试。

完成证据（2026-07-20）：

- 两个 DFD-HR detector YAML 均显式设置 `validation_dataset: [FaceForensics++]` 和 `frame_num.val: 32`；训练 CLI 支持 `--validation_dataset` 覆盖。
- `resolve_eval_loader_names` 对缺失或空 validation 配置立即抛出 `ValueError`，不再允许无 validation 继续训练。
- 数据 split 解析只接受请求的 mode；缺少 `val` 时立即抛出 `ValueError`，不再以 `test` 替代。
- 当前 FaceForensics++ JSON 的 5 类标签均已有限验证存在 `train`/`val`/`test` 和 `val/c23`，未遍历数据集文件。
- 回归测试覆盖空 validation、缺失 `val`、显式 `val` 和 CLI 覆盖；完整测试为 12 tests OK。
- YAML 解析、Python 编译、隐式回退模式扫描、`git diff --check` 和敏感信息扫描通过。
- 本任务未启动训练，未修改数据、权重、系统或 CUDA 配置。

提交：`ad51336`（失败回归测试）、`a1293e2`（协议强制）、`8fb2ddf`（显式配置、CLI 和文档）。

下一步：合并本分支后从更新的 `main` 创建 `fix/final-test-metrics`，完成 T2.2；不在本分支混入指标保存修改。

### T2.2 最终测试指标

**状态：DONE**

- [x] `save_best=False` 时返回本次测试指标。
- [x] 保存数据集级和平均指标为稳定 JSON。
- [x] 增加回归测试。

完成证据（2026-07-20）：

- `Trainer.test_epoch(..., save_best=False)` 返回本次测试的数据集级指标和 `avg`，不再返回历史 `best_metrics_all_time`。
- 最终测试报告原子写入 `${log_dir}/test/metrics.json`，schema version 为 `1`，包含 `phase`、`metric_scoring`、`datasets` 和 `average`。
- JSON 报告排除逐样本 `pred`/`label` 数组，并将 NumPy scalar 转为稳定的原生 JSON 数值。
- validation 的 `save_best=True` checkpoint 选择和既有 best metric 保存行为未改变。
- 回归测试使用临时运行目录和 mock 数据验证当前指标返回、数据集/平均 JSON 内容及历史 best 隔离；未加载模型或数据集。
- 完整测试为 13 tests OK；Python 编译、`git diff --check` 和敏感信息扫描通过。
- 本任务未启动训练，未修改数据、权重、系统或 CUDA 配置。

提交：`2311869`（失败回归测试）、`4e523e4`（当前指标返回和 JSON 持久化）。

下一步：合并本分支后从更新的 `main` 创建 `feat/amp-grad-accum`，完成 T2.3；结构化逐 step `metrics.jsonl` 仍属于后续运行基础设施，不以本次最终指标 JSON 代替。

### T2.3 AMP 与梯度累积

**状态：DONE**

- [x] 配置驱动 AMP。
- [x] 梯度累积。
- [x] 记录每卡 micro-batch、world size、累积步数、有效 batch。
- [x] 检查有限 loss 和梯度。

完成证据（2026-07-20）：

- 两个 detector YAML 显式声明保守默认值 `amp: false` 和 `gradient_accumulation_steps: 1`；AMP 仅在配置开启且运行设备为 CUDA 时生效。
- 非 SAM 优化器按累积窗口归一化 loss，只在窗口边界执行 optimizer step；最后一个不足窗口使用实际 micro-batch 数作为除数。
- DDP 非边界 micro-batch 使用 `no_sync`；完整 DDP 同步正确性仍由 T2.5 和两进程 Smoke 单独验证。
- 运行配置和日志记录每卡 micro-batch、world size、累积步数、有效 batch、AMP 请求值和实际启用值。
- FP32/非 AMP 路径发现非有限 loss 或梯度立即失败；AMP 梯度溢出记录警告并交由 GradScaler 跳过 step、降低 scale。
- SAM 与 AMP 或累积步数大于 1 的组合明确拒绝，避免不正确的两阶段梯度累积。
- 回归测试覆盖 optimizer step 边界、末尾残余窗口、CPU AMP 禁用、配置默认值、非有限 loss/梯度、SAM 边界和真实 CUDA GradScaler 溢出恢复。
- 完整测试为 21 tests OK；真实 CUDA 验证仅使用 1 x 1 合成张量，未加载项目模型或数据，未启动实验训练。
- Python 编译、`git diff --check` 和敏感信息扫描通过。

提交：`cb64a62`（失败合同测试）、`ffbf9f3`（AMP 与累积实现）、`fe980fb`（配置默认值）、`822d292`（GradScaler 溢出恢复）。

下一步：合并本分支后从更新的 `main` 创建 `feat/checkpoint-resume`，完成 T2.4，包括 GradScaler 和 RNG 的完整恢复与 round-trip 测试。

### T2.4 可恢复 checkpoint

**状态：TODO**

- [ ] 保存/恢复 model、optimizer、scheduler、epoch、best metrics、GradScaler、RNG。
- [ ] 原子写入。
- [ ] 默认仅保留 best/last。
- [ ] checkpoint round-trip 测试。

### T2.5 DDP 验证同步

**状态：TODO**

- [ ] 明确分布式评估或 rank-0 + barrier 方案。
- [ ] 防止 rank 失步。
- [ ] 两进程有限 Smoke Test。

### T2.6 官方权重评估校准

**状态：TODO**

- [ ] 官方 checkpoint `strict=True` 加载。
- [ ] 固定小子集评估。
- [ ] 至少一个完整外部数据集评估。
- [ ] 记录 checkpoint、配置、JSON 哈希和指标。

## P3：Jupyter 标准化

### T3.1 注册现有环境为 Kernel

**状态：DONE**

- [x] 确认 `ipykernel`。
- [x] 注册稳定显示名。
- [x] Kernel 必须指向现有 DFD-HR Conda Python。
- [x] Jupyter runtime/config 放入当前用户运行目录。
- [x] Server 仅监听 localhost，经 SSH 隧道访问。

完成证据（2026-07-20）：

- 经用户明确批准，在既有 DFD-HR 环境中安装 ipykernel `6.31.0`、JupyterLab `4.6.1` 和 nbconvert `7.17.1`；未使用或修改系统 Python。
- 用户级 `dfd-hr` kernelspec 注册成功，显示名为 `Python (DFD-HR)`，`kernel.json` 的 `argv[0]` 与 `${DFDHR_PYTHON}` 完全一致。
- 真实 Kernel 启动和执行通过：解释器匹配，PyTorch `2.3.1+cu121`，CUDA available。
- `scripts/register_jupyter_kernel.sh` 连续执行两次成功，注册结果一致。
- `scripts/start_jupyter_local.sh` 实际启动验证仅监听 `127.0.0.1`；runtime/config 位于 `${DFDHR_RUNTIME_ROOT}/jupyter/` 且权限为 `700`，验证后 Server 已停止。
- 未修改 CUDA、系统配置、训练代码、数据或权重，未启动训练。
- 最终验证：`pip check` 无依赖破损；`${DFDHR_PYTHON} -m unittest discover -s tests -v` 为 9 tests OK；Shell 语法和源 Notebook nbformat/空输出检查通过。

提交：`eab2030`（Kernel 注册、localhost 启动脚本和使用说明）。

下一步：Kernel 基础设施已完成；后续节点需使用同一脚本在各自现有环境中注册，不提交节点本地 kernelspec。

### T3.2 标准 Notebook

**状态：TODO**

- [x] `00_environment_and_paths.ipynb`
- [ ] `01_checkpoint_strict_load.ipynb`
- [ ] `02_dataset_protocol_audit.ipynb`
- [ ] `03_single_gpu_memory_smoke.ipynb`
- [ ] `04_official_weight_eval.ipynb`
- [ ] `05_training_monitor.ipynb`

源 Notebook 进入 Git；执行后的副本和大型输出进入 `${DFDHR_RUNTIME_ROOT}`。Notebook 必须 Restart Kernel 后 Run All 成功。

`00_environment_and_paths.ipynb` 完成证据（2026-07-20）：

- 源文件：`notebooks/00_environment_and_paths.ipynb`，提交前已清空输出，nbformat 校验通过，kernelspec 为 `dfd-hr`。
- 执行副本：`${DFDHR_RUNTIME_ROOT}/jupyter-validation/20260720_172902/00_environment_and_paths.executed.ipynb`，位于 Git 外。
- Restart/Run All 等价执行通过：7 个代码单元全部执行，无错误，解释器匹配，CUDA available，runtime 不在仓库或数据根内。
- 只读配置审计发现当前 detector YAML 的相对日志目录会解析到仓库内；已记录到 T1.1，未在本任务修改训练配置。
- 未加载模型，未遍历或修改数据，未启动训练；源 Notebook 不含真实节点信息或内部绝对路径。
- 提交：`df9736a`。

下一步：其余 5 个 Notebook 保持 TODO，并随各自依赖任务处理；本分支完成后优先进入 `fix/validation-protocol`，不继续扩展 Notebook。

## P4：Smoke Test 与正式复现

### T4.1 单卡两批次 Smoke Test

**状态：BLOCKED by P0/P2**

从 micro-batch 1 开始，分别记录 FP32 和 AMP 峰值显存、step time、有限 loss、可训练模块梯度、冻结 backbone 无梯度、checkpoint round trip。

### T4.2 两卡 DDP Smoke Test

**状态：BLOCKED by T4.1/T2.5**

有限步数验证同步、日志、有效 batch、checkpoint、正常退出和恢复。

### T4.3 Mini Run

**状态：BLOCKED by T4.2**

固定小子集，1–3 epoch。只验证训练闭环，不作为论文结果。

### T4.4 完整训练

**状态：BLOCKED by T4.3**

从 CLIP 初始化开始，不加载发布的 DFD-HR 权重。使用验证集选择 best，并维护可恢复 last。

### T4.5 跨数据集最终评估

**状态：BLOCKED by T4.4**

冻结 best checkpoint 后一次性执行，保存帧级/视频级指标、聚合方法、样本数、代码/配置/数据/权重哈希。

## 4. 每次任务结束时必须更新

- 当前状态和完成证据。
- 分支与提交 SHA。
- 测试、Notebook 或报告路径。
- 尚未知信息。
- 下一项有限动作。
- 工作树是否干净。
