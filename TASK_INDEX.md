# DFD-HR 当前任务索引

本文件是新机器、新 Codex 会话和实验中断后的任务入口。通用规范见 `AGENTS.md`，详细实验流程见 `docs/EXPERIMENT_WORKFLOW.md`。

> 当前状态：控制节点审计、实验生命周期、Jupyter 00-05、训练正确性修复、单/双卡 Smoke 及随机架构 Mini Run 已有证据；完整训练、归档实跑和新增节点准备仍未完成。除有明确证据的项目外，全部按 `TODO` 处理。

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

Current task branch: `infra/full-reproduction-preflight`

Completed scope: T4.4 full reproduction resource preflight and node selection

Next task branch: awaiting approval 1 for pinned CLIP asset transfer

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

**状态：DONE**

- [x] 建立并验证 `DFDHR_REPO_ROOT`、`DFDHR_PYTHON`、`DFDHR_DATA_ROOT`、`DFDHR_RUNTIME_ROOT`、`DFDHR_ARCHIVE_ROOT`、`DFDHR_CACHE_ROOT`。
- [x] 数据目录只读。
- [x] 训练输出不写入代码目录或共享数据目录。
- [x] 缓存不写入仓库。

已知前置问题：`00_environment_and_paths.ipynb` 的只读审计确认当前 detector YAML 使用仓库相对 `log_dir`/`logdir`，按仓库根解析时会把日志和 checkpoint 写入仓库。正式运行前必须由运行配置显式覆盖到 `${DFDHR_RUNTIME_ROOT}`；本次 Jupyter 任务未修改训练配置。

完成证据（2026-07-20）：生命周期工具要求六个变量全部存在、解释器与 `${DFDHR_PYTHON}` 一致、repo 与 active Git root 一致；runtime/archive/cache 自动建立在 repo 和数据根之外。冻结配置把 `log_dir`/`logdir` 重定向到独立 RUN_ID，强制 `save_feat=false`、`save_ckpt=true`。集成验证未向数据根写入文件。

### T1.2 实验注册表

**状态：DONE**

- [x] 建立 `registry/experiments.csv`。
- [x] 每个运行使用唯一 `RUN_ID`。
- [x] 记录状态、目标、分支、提交、配置、节点角色、GPU、结果和归档位置。

完成证据（2026-07-20）：RUN_ID 受格式、runtime、archive 和 registry 四重唯一性检查；registry schema 由测试锁定，写入前拒绝内部绝对路径、IP、SSH、token、password 和 private key 文本。真实 lifecycle-only 验证行绑定提交 `8cc2599`，公共路径仅使用 `${DFDHR_*}` 角色变量。

### T1.3 实验模板

**状态：DONE**

以下模板已经规划，但只有文件存在不代表流程已实现：

- [x] `templates/experiment_manifest.yaml`
- [x] `templates/experiment_summary.md`
- [x] `templates/infrastructure.local.example.yaml`
- [x] `templates/pull_request_template.md`
- [x] 运行目录初始化脚本
- [x] 配置冻结与哈希脚本
- [x] 环境与 Git 元数据采集脚本
- [x] 产物同步与哈希校验脚本

完成证据（2026-07-20）：`scripts/experiment_lifecycle.py` 提供 init/freeze/capture/checksums/verify/register/archive 子命令，另有 6 个单一职责 shell 入口。manifest 递归拒绝 `REPLACE_ME`，环境采集使用白名单，配置和数据/权重记录 SHA-256。archive 默认 dry-run，`--execute` 才复制并逐文件比对哈希，且无删除源目录能力；本轮仅验证 dry-run，未复制归档。

### T1.4 运行目录与存储预算

**状态：DONE**

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

完成证据（2026-07-20）：Git 外 lifecycle 验证 RUN_ID `_002` 成功执行 init、两次 verify、capture、freeze、checksums 和 archive dry-run；九个最小文件、`checkpoints/`、`notebooks_executed/`、command 可执行位、无 symlink、输出边界、预算和全文件 SHA-256 均通过。RUN_ID `_001` 因 PyTorch `TorchVersion` YAML 序列化失败而保留为失败编号，修复提交 `8cc2599` 后未复用。

提交：`b3ab5d6`（生命周期工具）、`8cc2599`（runtime 版本序列化修复）。完整测试为 46 tests OK。

### T1.5 结构化训练指标流

**状态：DONE**

- [x] Trainer 将 train/validation 的 epoch、global step、loss、学习率、step/data time、显存和磁盘余量追加到 `${RUN_DIR}/metrics.jsonl`。
- [x] 多卡仅 rank 0 写入，记录 world size、micro-batch、累积步数和有效 batch。
- [x] 每行独立 JSON、原子追加边界明确，异常中断后已有行仍可解析。
- [x] `05_training_monitor.ipynb` 只读消费该文件和日志。

完成证据（2026-07-20）：`JSONLMetricsWriter` 以单次 `O_APPEND` 写入和可选 `fsync` 保存标准 JSON 行，非有限观测值转为 `null`；读取器保留中断尾行之前的全部完整事件。Trainer 记录首批、固定间隔、末批和 validation/test 事件，包含 batch 语义、学习率、耗时、CUDA 内存和磁盘余量；非零 DDP rank 不创建指标文件。生命周期冻结配置显式写入 RUN_ID 和运行根下的 `metrics_jsonl`，verify 同时检查二者。

`05_training_monitor.ipynb` 源文件无执行输出；Git 外 `_001` 因 kernel 未安装 matplotlib 失败后保持原样，未安装软件；零额外依赖版本在 `_002` Restart/Run All 成功，读取 3 个 train 事件和 1 个 validation 事件，并输出趋势、资源字段和有限日志尾部。完整测试为 52 tests OK，本任务未加载模型、读取数据集或启动训练。

提交：`1b188e4`。下一步从更新后的 `main` 创建 `test/mini-run`，使用唯一 RUN_ID 和固定小子集执行 T4.3，不得扩大为正式训练。

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

**状态：DONE**

- [x] 保存/恢复 model、optimizer、scheduler、epoch、best metrics、GradScaler、RNG。
- [x] 原子写入。
- [x] 默认仅保留 best/last。
- [x] checkpoint round-trip 测试。

完成证据（2026-07-20）：

- 完整训练 checkpoint 保存并严格恢复 model、optimizer、scheduler、已完成 epoch、best metrics、GradScaler、配置及 Python/NumPy/Torch/CUDA RNG。
- `--resume` 只接受完整训练 checkpoint，拒绝把仅含模型权重的官方 checkpoint 误作训练恢复点；恢复后从 `epoch + 1` 继续。
- checkpoint 先写同目录临时文件，再通过 `os.replace` 原子提交；测试确认完成后无残留 `.tmp`。
- validation 只保留 `avg` best，训练每个 epoch 覆盖单一 `checkpoints/ckpt_last.pth`；`--no-save_ckpt` 同时禁止 best 和 last。
- DDP 下 last 仅由 rank 0 写入；跨 rank barrier 与验证同步仍由 T2.5 完成。
- round-trip 测试验证模型参数、optimizer momentum、scheduler、Scaler、best metrics、下一 epoch、四类 RNG 及 best/last 文件集合。
- 完整测试为 24 tests OK；Python 编译、`git diff --check` 和敏感信息扫描通过。
- 测试只使用临时目录、微型线性层和合成张量，未加载项目模型或数据，未启动实验训练。

提交：`4812ef2`（失败 round-trip 合同测试）、`4525284`（原子完整状态保存与恢复）。

下一步：合并本分支后从更新的 `main` 创建 `fix/ddp-validation-sync`，完成 T2.5 的 rank 对齐和两进程有限 Smoke。

### T2.5 DDP 验证同步

**状态：DONE**

- [x] 明确分布式评估或 rank-0 + barrier 方案。
- [x] 防止 rank 失步。
- [x] 两进程有限 Smoke Test。

完成证据（2026-07-20）：

- validation、last checkpoint 和最终测试统一使用 rank-0-only 操作，边界包含前 barrier、结果或错误广播、后 barrier。
- rank 0 的 validation inference 显式使用 DDP wrapper 下的 `model.module`，避免其他 rank 位于 barrier 时触发 DDP forward collective。
- validation best metrics 广播给所有 rank；rank-0 操作失败时错误也广播并在所有 rank 同步抛出，避免一侧继续训练。
- 两进程 gloo 测试实际构造 DDP-wrapped 微型模型，仅 rank 0 执行 validation forward；两个 rank 获得同一指标并在之后成功完成 all-reduce。
- 完整测试为 25 tests OK；Python 编译、`git diff --check` 和敏感信息扫描通过。
- 本测试验证 CPU/gloo 控制流和 collective 对齐；真实两卡 CUDA/NCCL 训练链路仍属于 T4.2，不以本证据替代。
- 未加载项目模型或数据，未启动实验训练。

提交：`2a9e290`（失败两进程同步测试）、`c541c0a`（rank-0 validation 同步与 DDP 推理解包）。

下一步：合并本分支后进入 T2.6 官方权重校准；先做 strict load 和有限评估前检查，不跳过数据协议与输出路径边界。

### T2.6 官方权重评估校准

**状态：DONE**

- [x] 官方 checkpoint `strict=True` 加载。
- [x] 固定小子集评估。
- [x] 至少一个完整外部数据集评估。
- [x] 记录 checkpoint、配置、JSON 哈希和指标。

完成证据（2026-07-20）：

- 官方 checkpoint 在强制 Hugging Face 离线模式下按 CLIP-L/14 公开架构构造并 `strict=True` 加载成功；1335 个 checkpoint 张量与 409,037,250 个模型参数完全匹配，无 missing/unexpected key。
- 官方 checkpoint SHA-256 为 `bbf2b1c805fe545a1ac1ead36e3c4341a78b5a6334766dd875dc6d1a940944ec`；上游默认 detector 配置 SHA-256 为 `168c1abd68d8719a27537b92d3ae9de88dcc74a26226f018e7a4356e1fc995e`，测试配置 SHA-256 为 `e29340d58ae09c15422b3f72c0cd48df3a8aa697268e04cc72e12f04bc2c8656`。
- 固定小子集校准使用外部数据集角色 `Celeb-DF-v2`，8 帧、2 视频，batch 1；帧级 AUC `1.0`，视频级 AUC `1.0`。该结果只证明链路可复现，不作为研究指标。
- 完整外部数据集校准使用 `e4s_ff` 的全部测试协议样本：2040 帧、255 视频，batch 1；帧级 AUC `0.9728590838509317`、ACC `0.8759803921568627`、AP `0.9719757913021212`、EER `0.07142857142857142`，视频级 AUC `0.986583850931677`、ACC `0.9176470588235294`、EER `0.05`。
- 完整评估关联代码提交 `166528b`；`e4s_ff` JSON SHA-256 为 `a2aa669d2403554eb41e2ba5e067ad50dc387a6b086f8a308cdd1ffda5f4fa61`。结构化报告和日志保存在 Git 外运行目录，不含逐样本预测或标签数组。
- 新增离线架构构造、checkpoint 键归一化冲突检查、确定性类别均衡子集、哈希和原子 JSON 报告工具；完整测试为 31 tests OK。
- 本任务只执行只读 checkpoint 加载和推理；未下载 backbone、未复制资产、未修改数据/环境/系统配置，未启动训练。

提交：`166528b`（离线严格加载与评估工具）、`c4fd678`（`01`/`04` 标准 Notebook）。

下一步：先完成 `02_dataset_protocol_audit.ipynb`，把已修复的 validation/test 协议和 JSON 有效性固化为可重复审计；随后进入 T4.1 单卡两批次 Smoke Test。

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
- [x] `01_checkpoint_strict_load.ipynb`
- [x] `02_dataset_protocol_audit.ipynb`
- [x] `03_single_gpu_memory_smoke.ipynb`
- [x] `04_official_weight_eval.ipynb`
- [ ] `05_training_monitor.ipynb`

源 Notebook 进入 Git；执行后的副本和大型输出进入 `${DFDHR_RUNTIME_ROOT}`。Notebook 必须 Restart Kernel 后 Run All 成功。

`00_environment_and_paths.ipynb` 完成证据（2026-07-20）：

- 源文件：`notebooks/00_environment_and_paths.ipynb`，提交前已清空输出，nbformat 校验通过，kernelspec 为 `dfd-hr`。
- 执行副本：`${DFDHR_RUNTIME_ROOT}/jupyter-validation/20260720_172902/00_environment_and_paths.executed.ipynb`，位于 Git 外。
- Restart/Run All 等价执行通过：7 个代码单元全部执行，无错误，解释器匹配，CUDA available，runtime 不在仓库或数据根内。
- 只读配置审计发现当前 detector YAML 的相对日志目录会解析到仓库内；已记录到 T1.1，未在本任务修改训练配置。
- 未加载模型，未遍历或修改数据，未启动训练；源 Notebook 不含真实节点信息或内部绝对路径。
- 提交：`df9736a`。

后续进展：`01` 和 `04` 已随 T2.6 完成；`02`、`03`、`05` 继续按各自依赖任务处理。

`01_checkpoint_strict_load.ipynb` 与 `04_official_weight_eval.ipynb` 完成证据（2026-07-20）：

- 两个源 Notebook 均为无输出、无执行计数、nbformat 4.5 有效文件，kernelspec 为 `dfd-hr`，不包含真实节点或内部绝对路径。
- 在代码提交 `c4fd678` 上使用真实 `dfd-hr` Kernel 执行通过：`01` 的 3 个代码单元确认 1335 个张量 strict load；`04` 的 3 个代码单元完成 8 样本外部集校准并读取结构化报告，均为 0 error。
- 执行副本和 `04` 生成的指标报告仅保存在 `${DFDHR_RUNTIME_ROOT}/jupyter-validation/`，未提交到 Git。
- `04` 默认限制 8 个确定性样本；只有显式设置本地 `DFDHR_EVAL_MAX_SAMPLES=0` 才执行完整数据集，避免误启动长评估。
- 提交：`c4fd678`。

后续进展：`02` 已完成；`03` 随 T4.1 完成，`05` 等待结构化训练指标流。

`02_dataset_protocol_audit.ipynb` 完成证据（2026-07-20）：

- 源 Notebook 为无输出、无执行计数、nbformat 4.5 有效文件，kernelspec 为 `dfd-hr`，不包含真实节点或内部绝对路径。
- 在干净提交 `ca3ced1` 上使用真实 `dfd-hr` Kernel 执行通过：5 个代码单元全部完成，0 error；执行副本和结构化 audit JSON 仅保存在 `${DFDHR_RUNTIME_ROOT}/jupyter-validation/`。
- FaceForensics++ c23 的 5 个标签均具有显式且非空的 train/val/test 元数据，各标签三组 video ID 两两互斥；有限检查 30 个 JSON 引用文件均存在。
- 当前 Dataset loader 分别解析 train `17248`、validation `13432`、test `3360` 个协议采样帧；validation 与 test 的已选路径集合完全互斥，确认没有 test fallback。
- 外部测试角色 `Celeb-DF-v2` 解析 `4139` 个协议采样帧，有限检查 8 个引用文件存在；JSON SHA-256 为 `754a72ede6a124602e2c1d8da1a8fa83fb539a71c837076fa4550d09eeca5672`。
- FaceForensics++ JSON SHA-256 为 `0f05209e5d9dfbb86038887d7a1bb5a1977d1fa70312b921bacd7f43604b7c3f`；Notebook 未遍历数据目录树、未解码图像、未修改数据或启动训练。
- 提交：`ca3ced1`。

后续进展：T4.1 和 `03_single_gpu_memory_smoke.ipynb` 已完成。

`03_single_gpu_memory_smoke.ipynb` 完成证据（2026-07-20）：

- 源 Notebook 为无输出、无执行计数、nbformat 4.5 有效文件，kernelspec 为 `dfd-hr`，不包含真实节点或内部绝对路径。
- 在干净提交 `c5595ab` 上使用真实 `dfd-hr` Kernel 执行通过：3 个代码单元全部完成，0 error；FP32 与 AMP 分别在独立进程中执行严格两批次 Smoke。
- 两种精度均使用 FaceForensics++ 两个类别平衡的真实样本、micro-batch 1、448 分辨率、关闭数据增强；未进入 epoch 训练循环。
- FP32 loss 为 `0.2698855996`、`0.7747207284`，step time 为 `2.0834s`、`1.5513s`；峰值 allocated/reserved 为 `6792459264`/`7321157632` bytes。
- AMP loss 为 `0.2712574303`、`0.7789944410`，step time 为 `1.7743s`、`1.4726s`；峰值 allocated/reserved 为 `6059573760`/`6295650304` bytes，GradScaler scale 在两批后保持 `1024`。
- 两种精度的第二批未缩放 Adapter、Router、Head、Query 梯度均存在且有限，optimizer 参数确实更新；392 个冻结 backbone/visual-projection 参数保持无梯度。
- FP32/AMP 分别原子保存 `2465258500`/`2465258692` bytes 的完整 checkpoint，临时文件清理、模型扰动后恢复、optimizer/Scaler/RNG 恢复和 next epoch `1` 均通过；报告和 checkpoint 仅保存在 `${DFDHR_RUNTIME_ROOT}/jupyter-validation/`。
- 真实 Smoke 发现并修复两项此前微型测试未覆盖的问题：CUDA resume 将 CPU RNG state 错映射到 GPU，以及 MoE autocast gate/expert dtype 不一致；新增 CPU staging、AMP initial scale、未缩放梯度 observer 与真实 CUDA MoE backward 测试。
- 完整测试为 37 tests OK；提交：`c5595ab`。

下一步：合并后从更新的 `main` 创建 `test/two-gpu-ddp-smoke`，完成 T4.2 的有限双卡 NCCL Smoke；不进入 Mini Run。

## P4：Smoke Test 与正式复现

### T4.1 单卡两批次 Smoke Test

**状态：DONE**

从 micro-batch 1 开始，分别记录 FP32 和 AMP 峰值显存、step time、有限 loss、可训练模块梯度、冻结 backbone 无梯度、checkpoint round trip。

完成证据：见 T3.2 的 `03_single_gpu_memory_smoke.ipynb` 记录；代码提交 `c5595ab`，Git 外结构化报告绑定同一干净提交。

### T4.2 两卡 DDP Smoke Test

**状态：DONE**

有限步数验证同步、日志、有效 batch、checkpoint、正常退出和恢复。

完成证据（2026-07-20）：

- 在干净提交 `f930ab4` 上使用 `torchrun --standalone --nproc_per_node=2` 完成真实双卡 NCCL Smoke；每 rank 固定 20 optimizer steps，micro-batch `1`、world size `2`、accumulation `1`、有效 batch `2`，未进入 epoch 训练循环。
- 固定 FaceForensics++ 40 样本由 DistributedSampler 分配；每 rank 均为 10 真/10 假，两个 rank 均完成 20 步并正常退出。
- rank 0/1 首末 loss 分别为 `0.2712574303 -> 0.2241023779` 和 `0.3696714044 -> 0.1313642859`；平均 step time 为 `1.8392s`/`1.8385s`。
- rank 0/1 峰值 allocated 为 `6557471744`/`6644567552` bytes，reserved 为 `6784286720`/`6838812672` bytes；两 rank GradScaler final scale 均为 `1024`。
- 两 rank 最后一步的未缩放 Adapter、Router、Head、Query 梯度均存在、有限且经 DDP 同步一致；训练参数 probe 完全一致，冻结 backbone 保持无梯度。
- 真实 rank-zero-only 操作结果成功广播到两 rank；未发生 collective 失步。
- last checkpoint 原子写入，size `2465285464` bytes，SHA-256 `a203bf5cc76f7d1fafcb7d72e0dcafc1c983cff878c4f872166bd4268b84de1a`；包含 2 份 rank-local RNG，两个 rank 均完成模型、optimizer、Scaler、RNG 和 next epoch 恢复，且恢复后的随机流保持 rank 间不同。
- 结构化报告确认 `process_group_destroyed=true`，运行结束后两张 GPU 均释放；报告、日志和 checkpoint 只保存在 Git 外运行目录。
- 完整测试为 40 tests OK；新增 DDP last checkpoint 多 rank RNG 收集/恢复，同时保持单 rank旧 checkpoint 兼容。
- 本任务只执行有限 20-step Smoke，未运行 validation/test epoch、Mini Run 或正式训练，未复制资产或修改系统配置。

提交：`f930ab4`。

下一步：T4.2 与 T1.1-T1.5 均已完成；按完整生命周期进入 T4.3 Mini Run。

### T4.3 Mini Run

**状态：DONE**

固定小子集，1–3 epoch。只验证训练闭环，不作为论文结果。

完成证据（2026-07-20）：

- 提交 `95034a3` 修正 base/experiment 配置优先级，增加 train/val/test 确定性二分类内存子集和 Mini 配置生成器；未复制数据或 JSON。提交 `240e8ac` 将单样本 batch 不可计算的 AUC/EER 记录为 JSON `null`，并增加 lifecycle status 原子收口命令。完整测试为 56 tests OK。
- Git 外失败编号 `_001` 在首批后因不可计算指标的 `None` 适配缺失而退出，已标记 `failed`、写明原因、刷新 checksums 并通过 verify；编号未复用。
- RUN_ID `dfdhr_ffppc23_mini-random_20260720_002` 绑定干净提交 `240e8ac`，固定 FaceForensics++ c23 train/val/test 为 16/8/8 个平衡样本，所有 32 个路径存在。单卡 AMP，micro-batch `1`、累积 `16`、有效 batch `16`、1 epoch、seed `1024`。
- 本机和共享 scratch 未定位到可验证的 CLIP ViT-L/14 预训练资产；本次 manifest 明确记录 `clip_pretrained=false`、无 DFD-HR checkpoint，使用 architecture-only random initialization。结果只验证工程闭环。
- 正常产生 16 个 train、1 个 validation 和 1 个 final-test 结构化事件；validation/test frame AUC 与 video AUC 均为 `0.4375`，仅作链路证据。峰值 CUDA allocated `5169022464` bytes，运行目录 `4930651542` bytes。
- best/last 均为 `2465259076` bytes，无 `.tmp`；last SHA-256 `c98e5f60f2631049418104379482cd1df82f70314f99e1ca9086a4519aa9544a`，best SHA-256 `437c0939b81b81f1b05577b8324b926b69c59d331cbe927824ba8882d778e388`。
- 独立恢复进程从 last 恢复到 epoch `1`，未新增训练 epoch 并完成有限 final test；进程退出后两张 GPU 均释放。completed manifest、35 项全目录 checksums、8 GiB 预算和输出边界 verify 通过；archive 仅 dry-run，目标不存在。
- 脱敏 registry 已记录完成行；详细路径、日志、执行结果和资产缺口只保存在 Git 外运行目录与 `.local/asset_sources.yaml`。

下一步：T4.3 已完成；按 T4.4 的独立批准边界推进完整复现资源准备，不得直接启动完整训练。

### T4.4 完整训练

**状态：BLOCKED by CLIP asset and formal-training approval**

从 CLIP 初始化开始，不加载发布的 DFD-HR 权重。使用验证集选择 best，并维护可恢复 last。

资产来源审计：**DONE**。官方仓库 `openai/clip-vit-large-patch14` 的只读元数据审计固定到 revision `32bd64288804d66eefd0ccbe215aa642df71cc41`；首选 `model.safetensors` 大小 `1710540580` bytes，LFS SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`，配套 `config.json` 大小 `4519` bytes。精确来源记录在 Git 外 `.local/asset_sources.yaml`，本轮未下载文件。

阻塞证据（2026-07-20）：默认 Hugging Face cache、当前用户共享 scratch 模型文件和现有本地资产记录中均未找到可离线加载并校验的该预训练资产。官方 DFD-HR checkpoint 只允许校准与评估，不能替代独立训练初始化。

完整复现资源 Preflight：**DONE**。完成证据（2026-07-20）：

- 对 `controller node`、`3090 candidate A`、`3090 candidate B` 和 `evaluation node` 分别完成新鲜有限检查；远程节点使用有限超时、非交互 SSH，每个角色独立记录。未读取进程明细或其他用户信息。
- 四个角色均可达、共享仓库可见、提交一致且干净；四处均未发现 pinned CLIP。相同的 30 个 FF++ c23 train/validation/test 样本路径在四处全部存在，说明 3090 节点不需要复制 FF++ 数据本体。
- `controller node`：2 × RTX 2080 Ti，采样时空闲；FF++ JSON SHA-256、`2156/420/420` 视频计数、split 两两互斥和有限路径均通过；环境完整，56 tests OK；约 983 GiB 可用。结论 `READY_AFTER_SMALL_ASSET_FIX`，仅缺 pinned CLIP，是当前首选训练与评估角色。
- `3090 candidate A`：2 × RTX 3090，存储约 2.48 TiB 可用，但采样时两卡有负载；环境和节点本地 JSON 缺失。结论 `BLOCKED_BY_GPU / ENVIRONMENT / CLIP`；在 GPU 可预约、环境重建、JSON 与 CLIP 就绪并通过获批吞吐 Smoke 后，可切换为速度优先节点。
- `3090 candidate B`：2 × RTX 3090 显存被占用，scratch 使用率 90%，环境和 JSON 缺失。结论 `NOT_RECOMMENDED`。
- `evaluation node`：2 × RTX 3090 采样时空闲，FF++ JSON/路径与 56 tests 通过，但缺 `ipykernel`、CLIP，scratch 使用率 98%。结论 `BLOCKED_BY_STORAGE`，不承担新训练或归档任务。
- 实测完整 checkpoint `2465259076` bytes；`3 × checkpoint + 10 GiB` 为 `18133195468` bytes（约 16.89 GiB）。计入日志后 manifest 硬下限取 20 GiB，建议维持 30 GiB；启动前仍要求至少 50 GiB 普通用户可用空间。
- 首选方案只移动 pinned CLIP snapshot；不移动数据、JSON、环境、外部评估数据或官方 DFD-HR checkpoint。若改选 3090 A，额外只准备约 11.5 MB FF++ JSON，并从 pinned requirements 重新创建约 5.9 GiB 安装规模的环境；禁止把 Conda 前缀直接 rsync 后视为可靠。
- 训练后保留 producing node 的 best/last 和完整元数据；另行获批后才把冻结 best 与 checksums 复制到 `controller node` 评估。长期 `archive candidate` 尚未专项审计，校验前不得删除源副本。
- Git 外证据：`.local/full_reproduction_preflight.yaml`、`.local/node_inventory_latest.json`、`.local/asset_transfer_plan.yaml` 及受控 raw audit 目录；均保持未跟踪。公开脱敏报告为 `docs/full_reproduction_preflight_public.md`，提交 `0497dab`。
- 完整测试为 56 tests OK；本轮未安装软件、未下载或复制资产、未修改系统配置、未运行 GPU benchmark、未启动 Mini Run 或完整训练。

下一步：等待批准 1（在首选节点下载或复制 pinned CLIP snapshot）。批准 1 只允许资产落地与 revision/size/SHA-256/必要文件校验，不包含节点环境准备、pretrained Smoke 或完整训练；后续批准 2、3、4 仍需分别取得。

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
