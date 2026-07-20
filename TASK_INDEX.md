# DFD-HR 当前任务索引

本文件是新机器、新 Codex 会话和实验中断后的任务入口。通用规范见 `AGENTS.md`，详细实验流程见 `docs/EXPERIMENT_WORKFLOW.md`。

> 当前状态：控制节点审计、实验生命周期、Jupyter 00-05、训练正确性修复、单/双卡 Smoke、pinned CLIP Mini Run 及正式训练协议修复已有证据；完整训练、归档实跑和部分节点准备仍未完成。除有明确证据的项目外，全部按 `TODO` 处理。

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

Current task branch: `fix/ddp-validation-timeout`

Completed scope: paper-spec MoE alignment, 3090 gates, and full-validation DDP timeout diagnosis

Next task branch: merge `fix/ddp-validation-timeout`, then restart the formal run with a new RUN_ID

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

**状态：ACTIVE**

从 CLIP 初始化开始，不加载发布的 DFD-HR 权重。使用验证集选择 best，并维护可恢复 last。

资产来源审计：**DONE**。官方仓库 `openai/clip-vit-large-patch14` 的只读元数据审计固定到 revision `32bd64288804d66eefd0ccbe215aa642df71cc41`；首选 `model.safetensors` 大小 `1710540580` bytes，LFS SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`，配套 `config.json` 大小 `4519` bytes。精确来源记录在 Git 外 `.local/asset_sources.yaml`；来源审计阶段未下载，后续获批落地结果见下文。

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

补充候选比较：**DONE**。完成证据（2026-07-20）：

- 对五个指定候选分别执行一次有限超时、非交互、只读 SSH 检查；只采集聚合 GPU 状态、当前用户可见存储、环境、FF++ 数据/JSON、pinned CLIP 和单元测试，不读取进程明细。
- `additional 3090 candidate C` 的 FF++ JSON 哈希、split 计数与互斥性、30/30 路径和 56 tests 均通过，约 2.71 TiB 可用；训练环境可用但缺 Jupyter 依赖，pinned CLIP 缺失，采样时一卡轻载、另一卡空闲。结论 `READY_AFTER_SMALL_ASSET_FIX / COORDINATION_REQUIRED`，成为明确预约后的速度优先候选。
- `additional 3090 candidate D` 的 FF++ 数据 30/30 路径存在且约 1.56 TiB 可用，但采样时两卡满载，环境、JSON 和 pinned CLIP 缺失。结论 `BLOCKED_BY_GPU / ENVIRONMENT / CLIP`。
- 原 `3090 candidate A/B` 与 `evaluation node` 的新鲜状态和既有结论一致；所有五个指定候选均缺 pinned CLIP，因此没有节点达到可立即启动正式训练的状态。
- Git 外证据：`.local/candidate_comparison_latest.yaml` 和受控 raw audit 目录；公开报告只增加匿名角色，不记录真实节点名、内部路径、端口或进程信息。
- 提交：`3c10c94`（候选比较与脱敏报告更新）。
- 本轮未安装软件、未复制数据或权重、未修改系统配置、未运行 benchmark、Smoke、Mini Run 或完整训练。

Pinned CLIP 资产落地：**DONE**。完成证据（2026-07-20）：

- 经用户单独批准，固定 revision 的八个必要文件先下载到 `controller node`；默认 Xet 传输停滞且大文件保持 0 bytes 后，停止本次自有会话并保留失败证据，改用标准 HTTP/LFS 在同一节点完成，未并发或切换下载源。
- `model.safetensors` 大小 `1710540580` bytes，SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`；processor 与 427,616,513 参数的完整 CLIP 模型在强制离线模式加载成功。
- 资产通过 `tmux` 中的可恢复传输复制到 `additional 3090 candidate C`；目标端重新执行大小、SHA-256、必要文件和完全离线加载校验，结果一致。
- 补充检查确认一个 2 × RTX 2080 Ti 候选采样时空闲、环境存在且空间充足；一个 4 × RTX 2080 Ti 候选环境和空间存在但四卡均满载。当前不向二者追加复制：controller node 本身已是可用的 2 卡 2080 Ti 回退，4 卡候选未满足 GPU 可用门槛。
- Git 外证据：`.local/asset_sources.yaml`、`.local/asset_transfer_plan.yaml`、下载/复制日志和逐节点原始审计；公开记录保持匿名。
- 提交：`926f9d5`（pinned CLIP 落地、复制和节点选择结果）。
- 未复制数据集、Conda 环境或官方 DFD-HR checkpoint；未安装软件、修改系统配置、运行 GPU benchmark、Smoke、Mini Run 或正式训练。

下一步：优先协调 `additional 3090 candidate C` 的连续双卡预约；获得单独批准后，使用 pinned CLIP 和正式候选配置执行有限 pretrained initialization Smoke。若无法预约 3090，则在 controller node 的 2 × RTX 2080 Ti 上执行同一 Smoke；4 × RTX 2080 Ti 候选仅在四卡明确释放后重新评估。

Pretrained initialization Smoke：**DONE on fallback node**。完成证据（2026-07-20）：

- 提交 `7fd60a6` 为单卡两批次和双卡 20-step Smoke 增加显式本地 pinned CLIP 初始化；该路径与 DFD-HR checkpoint 加载互斥，强制 Transformers 离线读取 safetensors，并在报告中记录初始化类型、大小、SHA-256 和 `dfd_checkpoint_loaded=false`。
- 旧的 DFD-HR checkpoint Smoke 参数和来源报告字段保持兼容；mocked loader 测试确认显式本地路径、`local_files_only=True` 和 `use_safetensors=True` 均传入 Transformers。
- `controller node` 与 `additional 3090 candidate C` 的现有环境分别运行 59 tests，全部通过；代码提交 `7fd60a6`，草稿 PR 为 `#18`。
- 首选 3090 候选按 60 秒间隔执行五次聚合采样，一张卡持续 98-99% 利用率，另一张在 3-41% 波动；等待会话停止并记录 `smoke_started=false`，未实例化模型、读取 batch 或写 checkpoint。
- 新鲜全候选扫描确认 `controller node` 两张 RTX 2080 Ti 连续空闲，环境、数据、JSON、pinned CLIP 和约 981 GiB 空间齐全，因此按回退方案执行。另有一个 2 卡 2080 Ti 角色仅缺 CLIP；4 卡 2080 Ti 角色四卡均为 97-99% 利用率；空闲 3090 角色要么缺环境/JSON/CLIP，要么 scratch 使用率 98%。
- 单卡 AMP 严格两批次 Smoke 使用 pinned CLIP、本地离线加载且 `dfd_checkpoint_loaded=false`；loss `0.5838524103/1.3313060999` 均有限，Adapter/Router/Head/Query 梯度有限，392 个冻结 backbone/projection 参数无梯度，optimizer 更新成功。
- 单卡峰值 allocated/reserved 为 `6015711232/6211764224` bytes，step time `2.1014s/1.6530s`；完整 checkpoint `2465258820` bytes，原子写入、SHA-256 和 round-trip 均通过。
- 双卡 NCCL Smoke 每 rank 固定 20 optimizer steps，micro-batch `1`、有效 batch `2`；两 rank 各 10 真/10 假，训练参数与必需梯度同步一致，rank-zero 广播、两份 rank-local RNG、checkpoint round-trip 和正常进程组销毁全部通过。
- 双卡 rank 0/1 平均 step time 为 `2.0265s/2.0081s`，峰值 reserved 为 `6817841152/6798966784` bytes；last checkpoint `2465285656` bytes，SHA-256 `c5c49301443dc721dc0d74d5978f92a5c2219eba057defde67466a8285ecd2d2`，无临时文件残留。
- 两项报告均绑定干净提交 `8d66d47`、相同 FF++ JSON SHA-256 和 pinned CLIP SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`；未加载官方 DFD-HR checkpoint。
- Git 外证据：pretrained Smoke 单卡/DDP 报告、日志、checkpoint 和 `.local/idle_candidate_scan_latest.yaml`；公开记录保持匿名。
- 未复制新资产、安装软件、修改系统配置、运行 Mini Run 或正式训练。

下一步：审查并合并 PR `#18`，从更新后的 `main` 创建 `exp/pretrained-mini-run`；冻结 pinned CLIP 初始化、micro-batch `1`、有效 batch 和小型确定性 FF++ train/validation/test 子集，先完成可恢复 pretrained Mini Run。首选 3090 候选仍需连续双卡预约；预约未落实时可在已验证的 2 卡 2080 Ti 回退角色执行 Mini Run，但不得直接启动完整训练。

Pretrained Mini Run：**DONE on fallback node**。完成证据（2026-07-20）：

- 提交 `bafe451` 为 Mini 配置生成器增加显式 pinned CLIP 本地离线初始化，并修复 lifecycle manifest 对 `clip_pretrained` 与 `dfd_checkpoint` 初始权重类型的区分；完整测试为 60 tests OK。
- RUN_ID `dfdhr_ffppc23_mini-pretrained_20260720_001` 绑定干净提交 `bafe451`；manifest 为 `clip_pretrained=true`、`dfd_hr_checkpoint=null`、`independent_reproduction=true`，初始权重 SHA-256 为 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`。
- 使用固定平衡 FF++ c23 train/validation/test `16/8/8` 样本，1 epoch、单卡 AMP、micro-batch `1`、累积 `16`、有效 batch `16`、seed `1024`；未加载官方 DFD-HR checkpoint。
- 正常产生 16 个 train、1 个 validation 和 1 个首次 final-test 结构化事件；validation frame/video AUC 为 `0.875/0.875`，有限 final-test frame/video AUC 为 `0.625/0.625`。这些小样本指标只验证链路，不作为研究结果。
- 峰值 CUDA allocated `5858192384` bytes，平均 train step time 约 `1.5382s`；运行目录约 4.6 GiB，满足 8 GiB manifest 预算。
- best/last 均为 `2465259268` bytes，SHA-256 分别为 `2511c360cc5cb97fdb23d8b5be5d382fdcc0599b108b37563d7b0f8446665f2c` 和 `7739cfb5293d28005d602e1cfe6c707dfad7d5fb420e41bd6c286c77dd43ade5`，无临时 checkpoint 残留。
- 独立恢复进程从 last 恢复到 next epoch `1`；恢复前后 train/validation 事件保持 `16/1`，未新增训练 epoch，只新增一次结果一致的 final-test 事件。完成后 GPU 释放，checksums、路径保护和本地预算再次通过。
- lifecycle status 为 `completed`，summary 已补全，archive 仅 dry-run，runtime 源目录保留；脱敏 registry 已登记完成行。
- Git 外证据：完整 manifest、冻结配置、命令、环境、日志、结构化指标、summary、best/last、恢复日志和 checksums；不提交内部绝对路径或执行后产物。
- 未复制新资产、安装软件、修改系统配置、执行完整数据集评估或启动正式训练。

下一步：确定正式训练节点和连续 GPU 时段。首选 3090 角色仍具备完整资产和空间但需预约；已验证的 2 卡 2080 Ti 回退角色可以运行 micro-batch `1`，但预计更慢。获得正式训练批准后再建立冻结 RUN_ID/config/manifest，执行最后一次 GPU、存储、数据 JSON、pinned CLIP 和输出预算 Preflight；未批准前不得启动完整 20 epoch 训练。

正式训练协议修复：**DONE**。完成证据（2026-07-21）：

- `validation_checks_per_epoch` 取代训练循环中的隐式 step 取模；正式配置显式固定首 epoch 末尾验证一次、后续 epoch 中点和末尾各验证一次，奇数长度 DataLoader 也得到确定的 epoch 内触发点。生命周期 manifest 记录该选模频率。
- 正式训练配置将 `test_dataset` 固定为空并关闭训练后的自动 final test；跨数据集评估继续由冻结 best checkpoint 后的独立任务执行，外部测试资产缺失不会阻断训练生命周期。
- Python、NumPy、Torch、全局 rank 和 DataLoader worker seed 均有显式策略；DataLoader 使用有 seed 的 generator，视频级增强在固定 seed 后恢复外层 RNG 状态。deterministic 模式强制关闭 cuDNN benchmark、开启 cuDNN deterministic 和 deterministic algorithms；不支持时必须显式改记为 `seeded_best_effort`。
- `scripts/build_formal_training_config.py` 生成 Git 外正式配置，固定 paper-aligned FF++ c23、8/32/32 帧、离线 pinned CLIP、20 epoch、AMP、双卡有效 batch `16`、Adam `1e-4`、每 rank workers `4`、关闭特征保存并保留 best/last 所需 checkpoint。
- 提交：`c48a6ff`。`${DFDHR_PYTHON} -m unittest discover -s tests -v` 为 72 tests OK；`git diff --check` 通过。本任务未加载模型、读取正式数据、占用 GPU、执行吞吐检查或启动正式训练。

下一步：审查并合并 `fix/formal-training-protocol`；从更新后的 `main` 创建 `exp/full-reproduction`，生成并冻结 Git 外 config、RUN_ID 和 manifest。合并时执行最终节点决策：优先使用已明确预约的 3090 角色；预约仍未落实时准备空闲 3090 备选，准备失败则回退已验证的 2 卡 2080 Ti 角色。只在最终节点执行一次受限吞吐检查，正式 20 epoch 训练仍需独立批准。

正式配置候选冻结：**DONE; initial RUN_ID/manifest gate was BLOCKED by final node reservation**。完成证据（2026-07-21）：

- PR `#20` 已通过隐私检查并 squash 合并到 `main`，合并提交 `e97a6c3`；`exp/full-reproduction` 从该提交创建。
- Git 外候选配置由正式生成器产生并通过字段断言，SHA-256 为 `8244984878cee51f07766c6005e0daffebc773ac7bbc6b319ea8ba7a07ca2a69`。有效 batch、验证频率、离线 pinned CLIP、空 test 列表和 deterministic 设置均与已合并协议一致。
- 最终节点门执行新鲜有限检查：首选 3090 角色的两卡仍有活动负载，虽环境、JSON、CLIP 和空间就绪，但没有连续双卡预约证据；3090 备选一张卡满载且环境、JSON、CLIP 缺失。未读取进程明细。
- 因最终节点未确定，未创建会记录错误 GPU/runtime 的 RUN_ID 和 manifest；未复制资产、安装软件、加载模型、读取正式 batch、运行吞吐检查或启动训练。真实节点和候选配置路径只保存在 Git 外 `.local/full_reproduction_preflight.yaml`。

提交：协议实现 `c48a6ff`，任务记录 `d926053`，PR `#20` 合并提交 `e97a6c3`；本次候选配置与节点门记录待当前分支提交。下一步：取得首选 3090 角色的连续双卡预约；若仍无法预约，则等待 3090 备选双卡空闲后再准备小资产。两者均未满足时，按计划绑定已完整验证的 2 卡 2080 Ti 回退角色，再创建 RUN_ID/manifest 并执行受限吞吐检查；20 epoch 正式训练继续等待独立批准。

回退节点冻结与受限吞吐检查：**DONE**。完成证据（2026-07-21）：

- 首选 3090 角色没有连续双卡预约，备选 3090 角色一张卡满载且缺环境、JSON、CLIP；目标端没有 Conda/Mamba/Micromamba，源端没有现成 `conda-pack`。未直接复制 Conda 前缀或临时安装包管理器，按既定决策切换到已完整验证的 2 卡 2080 Ti 回退角色。
- 回退角色新鲜 preflight 确认两卡空闲、约 971 GiB 可用、工作树干净，FF++ JSON 与 pinned CLIP SHA-256 均匹配。正式配置固定 workers `4`、validation batch `1`、metrics interval `16`，Git 外配置 SHA-256 为 `0713b78135109bbc39df0fdbfef851039f213ee56aec2b0bb73572a94202d157`。
- 第一次受限检查在读取 batch 前发现现代 `torchrun` 的 `LOCAL_RANK` 未被训练入口读取；提交 `fe58162` 修复环境变量/CLI rank 解析。第二次在首个 forward 发现严格 deterministic 需要 cuBLAS workspace；提交 `2b80d19` 将 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 纳入配置、启动校验和 manifest。
- 第三次在首个 backward 确认当前运行时的 `adaptive_avg_pool2d_backward_cuda` 没有确定性实现。按协议将实际复现模式显式记录为 `seeded_best_effort`，仍保留 Python/NumPy/Torch/rank/worker seed、`cudnn_benchmark=false`、`cudnn_deterministic=true` 和固定 cuBLAS workspace；提交 `8c11418` 增加受控回退生成选项。
- 第四次受限检查完成每 rank 128 个训练 micro-batch 和 rank 0 的 64 个 validation 样本；有效 batch `16`，训练墙钟估算约 `1.27 s/micro-batch`，validation 约 `2.28 samples/s`，峰值 CUDA reserved `6633291776` bytes。记录点 data time 均值低于 `0.02 s`，无需调整 workers 或 batch。
- 检查未写 `.pth` 或 `.tmp`，未进入完整 FF++ epoch，GPU 已释放。有限 validation 指标只证明链路，不作研究解释；详细日志、metrics 和报告仅在 Git 外 throughput 目录。完整测试为 78 tests OK。
- 先前 formal RUN_ID 因代码、配置或合并提交变化被取代，均按不可复用规则标记 `aborted`；每个编号都未执行保存的 20 epoch 命令。最终 planned RUN_ID 在本次纯任务索引合并后生成并只记录在 Git 外，避免公开索引提交再次改变其绑定的 `main` SHA。

PR `#21` 已通过隐私检查并 squash 合并到 `main`，合并提交 `d4486ae`。下一步：合并本次纯任务索引收口后，重新生成唯一 planned RUN_ID/config/manifest 并执行最终只读 verify。只有用户明确批准 20 epoch 正式训练后才执行保存的命令；训练完成并冻结 best SHA-256 后，T4.5 才能开始。

Paper-spec 协议校准：**DONE**。完成证据（2026-07-21）：

- 论文公式的 MoA gate 在训练阶段加入噪声，并对全部 `N` 个专家输出加权求和；现有 planned 配置的 `top_k=2, noise=true` 既不等于论文公式，也不等于官方发布实现的 `top_k=4, noise=false`。
- paper-spec 定义为论文明确内容优先、论文未说明项采用官方仓库默认。正式生成器和 paper-aligned detector 配置现固定 4 个专家、`top_k=4`、`noise=true`，并在 manifest 记录 MoE、epoch 与 AMP/batch 适配来源。
- 提交 `71f1421`；初始完整测试 78 tests OK。旧 planned RUN_ID 已在训练前标为 `aborted`，没有正式训练 step 或 checkpoint。
- 两个空间健康的 3090 候选均已有 FF++ 数据本体；FF++ JSON、pinned CLIP 与 `conda-pack` 环境包已完成可恢复复制和 SHA-256 校验，未复制数据集本体。环境解包后均完成 `conda-unpack`、核心依赖/CUDA 检查和全量测试。
- 优先候选的一张空闲 3090 完成 paper-spec pinned-CLIP 单卡两批次 Smoke：loss 和必需梯度有限，optimizer 更新、冻结 backbone、checkpoint 原子写入与 round-trip 均通过；峰值 CUDA reserved 约 5.72 GiB。
- 同一张卡的受限扫描确认 physical batch `8` 可完成两次训练 micro-batch，峰值 CUDA reserved 约 21.46 GiB；validation batch `32` 可完成 64 个样本，峰值 reserved 约 8.99 GiB。指标只作链路证据，不作研究解释。
- 正式配置生成器现显式接收 GPU 数、每卡 batch、梯度累积和 validation batch，并拒绝有效 batch 不等于 `16` 的组合；默认 `1 x 2 x 8` 保持兼容，新增 `8 x 2 x 1` 与非法组合回归测试。完整测试为 80 tests OK，`git diff --check` 通过。
- 3090 双卡队列只读取聚合 GPU 利用率/显存和当前用户可见存储；候选仍有活动负载时不抢占。详细节点名、内部路径、日志、资产哈希和报告只保存在 Git 外运行目录。
- 一个双卡空闲但存储余量较低的 3090 角色仅用于临时门禁：补齐 pinned CLIP 后哈希和完全离线加载通过，目标环境 80 tests OK；未复制数据集或环境。该角色门禁前后仍有约 80 GiB 可用，不承担长期归档。
- Paper-spec 双卡 Smoke 每 rank 固定 20 steps，参数同步、rank-local RNG、必需梯度、rank-zero 广播、checkpoint round-trip 和进程组销毁全部通过；每卡峰值 CUDA reserved 约 6.23 GiB，提交绑定 `d7dc523`。
- 双卡 physical batch `8`、累积 `1` 的受限门禁完成每 rank 4 个训练 micro-batch，并使用 validation batch `32` 完成 64 个验证样本；有效 batch 精确为 `16`，稳定训练 step 约 3.09 秒，峰值 CUDA reserved 约 21.84 GiB。workers `4` 在预热后 data time 接近零，无需增加。
- 最终 3090 配置冻结为每卡 batch `8`、累积 `1`、validation batch `32`、每 rank workers `4`；这比梯度累积回退更接近论文 physical batch 16。受限指标只证明链路，不作研究解释，门禁未写正式 checkpoint。
- PR `#23` 已 squash 合并到 `main`，合并提交 `6ce9bfb`。正式 RUN `_007` 绑定该提交并通过 config/manifest/资产哈希、双卡、70 GiB 空间门与 lifecycle verify 后启动；epoch 0 的 1078 个训练 step 全部完成，末步 loss 有限，显存与磁盘稳定。
- `_007` 在首次完整 validation 的第 `160/420` batch 失败：rank 0 验证运行超过 30 分钟，rank 1 等待后续 broadcast 时触发 PyTorch 默认 NCCL watchdog timeout。无 validation 事件、best 或 last checkpoint 产生，RUN 已如实标记 `failed` 并通过路径、checksum 和预算 verify，禁止复用或伪装恢复。
- 正式配置现显式固定 `ddp_timeout_minutes: 180`；训练入口验证其为正整数并将其传给 process-group 初始化，生命周期 manifest 同步记录。未设置该字段的旧配置仍保持 30 分钟默认；默认、180 分钟和非法值均有回归测试，本地和目标 3090 环境均为 83 tests OK。
- 提交 `035c8b9` 的目标端受限双卡门禁在两 rank 日志中均记录 180 分钟 timeout，完成每 rank 4 个 physical batch `8` 训练 step、rank-zero validation 和后续同步后正常退出；未写 checkpoint，GPU 已释放。

下一步：在真实双卡上执行包含 rank-zero validation 的 timeout 配置门禁；审查并合并 `fix/ddp-validation-timeout` 后，从新 `main` 创建 RUN `_008` 并重新开始 20 epoch。不得复用 `_007`、旧 RUN_ID 或旧 `top_k=2` checkpoint。

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
