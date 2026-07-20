# DFD-HR 当前任务索引

本文件是新机器、新 Codex 会话和实验中断后的任务入口。通用规范见 `AGENTS.md`，详细实验流程见 `docs/EXPERIMENT_WORKFLOW.md`。

> 当前状态：此前提出的标准化实验目录、模板、Jupyter Notebook、训练代码修复和正式复现大部分尚未完成。除有明确证据的项目外，全部按 `TODO` 处理，不得把计划描述为已实现。

## 1. 启动顺序

每次更换机器或聊天记录丢失后：

```bash
git status --short --branch
git fetch --all --prune
git switch docs/experiment-workflow
git pull --ff-only
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

## 3. 当前里程碑

建立新的控制节点，审计两台新增计算节点的硬件、数据、环境和存储状态；然后完成 DFD-HR 官方权重校准、单卡 Smoke Test、两卡 DDP Smoke Test，最后才进入 Mini Run 和完整训练。

## P0：控制节点迁移与新增节点审计

### T0.1 建立新控制节点

**状态：TODO**

- [ ] SSH 连接稳定。
- [ ] 共享项目目录可见。
- [ ] `docs/experiment-workflow` 可检出，工作树干净。
- [ ] `.local/infrastructure.yaml` 已根据模板创建且未被 Git 跟踪。
- [ ] 预期 Conda/Python 环境存在并可导入 PyTorch、Transformers、OpenCV 等核心依赖。
- [ ] CUDA 和 GPU 可见。
- [ ] 当前测试集通过。
- [ ] 能对目标节点执行单次、有限超时、非交互 SSH 检查。
- [ ] 审计输出只写入当前用户的 Git 外运行目录。

证据：时间戳、hostname、Git 分支/提交、Python 路径、PyTorch/CUDA、测试摘要、scratch 空间。

### T0.2 原控制节点有限重检

**状态：BLOCKED**

当前观察为连接超时。只允许从新控制节点执行一次有限重检：

- [ ] 记录 DNS/别名解析。
- [ ] 记录 TCP/SSH 结果。
- [ ] 记录时间戳。

不得高频重试，不得修改 SSH、网络、路由或防火墙。只有新鲜成功连接证据才能恢复其控制、训练或归档角色。

### T0.3 扩展审计目标

**状态：TODO**

从 `.local/infrastructure.yaml` 读取全部现有节点和两台新增节点。每台节点仅收集：

- [ ] 可达性和规范 hostname。
- [ ] CPU 型号、插槽、物理核心、线程。
- [ ] 总 RAM、可用 RAM。
- [ ] GPU 型号、数量、VRAM、驱动、CUDA compatibility。
- [ ] 当前 GPU 利用率和是否适合新的独占任务。
- [ ] scratch 文件系统、总量、普通用户可用空间、使用率、inode。
- [ ] 共享代码目录可见性。
- [ ] 当前用户本地运行目录可用性。
- [ ] 是否存在调度器和监控能力；不安装任何组件。

禁止读取其他用户的命令行、环境变量、shell 历史、项目内容、数据内容、私钥或凭据。

### T0.4 新增节点 A 审计

**状态：TODO**

硬件：

- [ ] CPU/RAM。
- [ ] GPU 型号、数量、单卡 VRAM。
- [ ] NVIDIA 驱动和 CUDA compatibility。
- [ ] 当前 GPU 状态。
- [ ] scratch 和根分区容量。
- [ ] 从控制节点可达。

DFD-HR 就绪性：

- [ ] 共享仓库可见。
- [ ] 数据根存在。
- [ ] 必需数据集目录存在。
- [ ] 数据集 JSON 注册表存在。
- [ ] train/validation/test 抽样路径可解析。
- [ ] DFD-HR 环境是否存在；若不存在只记录，不安装或复制。
- [ ] 官方 checkpoint 是否存在，SHA-256 是否匹配。
- [ ] 环境存在时运行单元测试。
- [ ] 输出空间是否适合 Smoke Test 或训练。

### T0.5 新增节点 B 审计

**状态：TODO**

使用与 T0.4 相同的清单，单独记录。不得根据节点 A 推断节点 B 的硬件、数据或环境一致性。

### T0.6 数据集与环境一致性

**状态：TODO**

第一阶段至少检查：

- [ ] FaceForensics++ c23：train/validation/test。
- [ ] Celeb-DF-v2。
- [ ] 当前测试配置引用的 DFD、DFDC、DFDCP、DeeperForensics、WildDeepfake、FFIW。
- [ ] 当前复现计划需要的 DF40 方法子集。

方法：目录存在性和有限计数、JSON SHA-256、各 split 抽样路径、环境包清单、官方权重哈希。审计阶段不得复制数据或环境。

### T0.7 生成脱敏结果报告

**状态：TODO**

目标：`docs/controller_migration_and_node_audit_results.md`

允许：时间戳、匿名节点角色、通用硬件、当前用户可见容量、数据/环境就绪状态、调度建议。

禁止：真实服务器名、IP/端口、其他用户身份/PID/目录大小、SSH 配置、凭据、内部绝对路径。

## P1：标准化实验基础设施

### T1.1 本地路径与环境变量

**状态：TODO**

- [ ] 建立并验证 `DFDHR_REPO_ROOT`、`DFDHR_PYTHON`、`DFDHR_DATA_ROOT`、`DFDHR_RUNTIME_ROOT`、`DFDHR_ARCHIVE_ROOT`、`DFDHR_CACHE_ROOT`。
- [ ] 数据目录只读。
- [ ] 训练输出不写入代码目录或共享数据目录。
- [ ] 缓存不写入仓库。

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

**状态：TODO**

- [ ] FaceForensics++ validation 显式配置。
- [ ] validation 缺失时立即失败，禁止回退到 test。
- [ ] 增加有效和无效划分测试。

### T2.2 最终测试指标

**状态：TODO**

- [ ] `save_best=False` 时返回本次测试指标。
- [ ] 保存数据集级和平均指标为稳定 JSON。
- [ ] 增加回归测试。

### T2.3 AMP 与梯度累积

**状态：TODO**

- [ ] 配置驱动 AMP。
- [ ] 梯度累积。
- [ ] 记录每卡 micro-batch、world size、累积步数、有效 batch。
- [ ] 检查有限 loss 和梯度。

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

**状态：TODO**

- [ ] 确认 `ipykernel`。
- [ ] 注册稳定显示名。
- [ ] Kernel 必须指向现有 DFD-HR Conda Python。
- [ ] Jupyter runtime/config 放入当前用户运行目录。
- [ ] Server 仅监听 localhost，经 SSH 隧道访问。

### T3.2 标准 Notebook

**状态：TODO**

- [ ] `00_environment_and_paths.ipynb`
- [ ] `01_checkpoint_strict_load.ipynb`
- [ ] `02_dataset_protocol_audit.ipynb`
- [ ] `03_single_gpu_memory_smoke.ipynb`
- [ ] `04_official_weight_eval.ipynb`
- [ ] `05_training_monitor.ipynb`

源 Notebook 进入 Git；执行后的副本和大型输出进入 `${DFDHR_RUNTIME_ROOT}`。Notebook 必须 Restart Kernel 后 Run All 成功。

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
