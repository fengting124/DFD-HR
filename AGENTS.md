# DFD-HR 实验与工程规范

本文件是 Codex 和项目成员进入仓库后的第一入口。详细执行流程见 `docs/EXPERIMENT_WORKFLOW.md`，当前任务和完成状态见 `TASK_INDEX.md`。

> 当前事实：实验模板、Jupyter 00-04、训练正确性修复及单/双卡 Smoke 已有证据；Mini Run、完整训练、训练监控和归档实跑仍未完成。只有存在明确提交、测试或运行证据的项目才能标记为完成。

## 1. 新机器与新会话启动顺序

每次更换机器、Codex 聊天记录丢失或长期中断后：

```bash
git status --short --branch
git fetch --all --prune
git switch main
git pull --ff-only origin main
git switch <当前任务分支>
git log --oneline --decorate -12
```

依次阅读：

1. `AGENTS.md`
2. `TASK_INDEX.md`
3. `docs/EXPERIMENT_WORKFLOW.md`
4. `docs/controller_migration_and_node_audit_plan.md`
5. 当前任务涉及的代码、配置和测试

之后检查：

- Git 工作树是否干净。
- 当前 Python/Conda 环境。
- PyTorch、CUDA 和 GPU。
- 数据根、JSON 注册表和官方权重。
- 运行目录与空间预算。
- 当前单元测试。

不得在完成 Preflight 前启动正式训练。

## 2. 本地基础设施信息

本仓库公开，不提交真实服务器名、用户名、IP、端口、SSH 配置、内部绝对路径、其他用户进程或目录信息。

真实节点映射只保存在 Git 外：

```text
.local/infrastructure.yaml
```

使用模板：

```text
templates/infrastructure.local.example.yaml
```

该本地文件可位于共享项目目录，使控制节点切换后仍能读取，但必须保持未跟踪状态。

当前基础设施任务按角色记录：

- 新控制节点候选。
- 当前不可用的原控制节点。
- 两台硬件、数据和环境未知的新增节点。
- 高显存但存储受限的 Smoke/推理节点。
- 长期被占用、未经协调不得调度的新任务节点。

实时状态必须重新检查。历史审计只能作为线索，不能作为当前调度依据。

## 3. 优先级与冲突处理

优先级：

1. 用户当前明确要求。
2. 本文件。
3. `TASK_INDEX.md` 和专项文档。
4. 既有实现惯例。

发现协议含糊、可能数据泄漏、可能覆盖结果或涉及系统级改动时，停止对应写操作，记录冲突并取得确认。

## 4. 核心原则

- 代码、配置、测试、小型源 Notebook、模板和文档进入 Git。
- 数据、环境、缓存、权重、日志、执行后 Notebook 和训练产物留在 Git 外。
- Notebook 用于检查、诊断、分析和监控，长期训练由 `tmux`/调度器和 `torchrun` 执行。
- 每个实验绑定唯一 RUN_ID、冻结配置、Git 提交、数据协议和独立运行目录。
- 每个实验只验证一个主要变量。
- 失败实验也记录状态、原因和结论。
- 训练集用于训练，验证集用于选模，跨域测试只在训练完成后执行。
- validation 缺失必须失败，禁止回退到 test。
- 公共仓库只保存最小、可公开、匿名化和可复用的信息。

## 5. 工程质量

### 5.1 最小实现

- 只实现当前需求及其必要验证。
- 优先复用现有接口、配置和工具函数。
- 不提前增加与当前任务无关的抽象层、兼容层或依赖。
- 不顺手重构无关代码。
- 行为变化同步更新测试、配置示例和文档。
- 数据协议、指标、checkpoint 和恢复逻辑修改必须有回归验证。

### 5.2 文件归属

| 内容 | 位置 |
| --- | --- |
| 模型、训练、评估 | `training/` |
| 运行配置 | `training/config/` |
| 自动化测试 | `tests/` |
| 可复用文档 | `docs/` |
| 源 Notebook | `notebooks/` |
| 实验模板 | `templates/` |
| 实验注册表 | `registry/` |
| 本地节点映射 | `.local/`，Git 忽略 |
| 日志、指标、权重 | `${DFDHR_RUNTIME_ROOT}` |

不要为了理想目录结构一次性移动现有文件。目录迁移必须单独提交并验证导入和链接。

## 6. 路径与存储边界

运行时使用：

```text
DFDHR_REPO_ROOT       Git 仓库根目录
DFDHR_PYTHON          DFD-HR 环境 Python
DFDHR_DATA_ROOT       只读数据集根目录
DFDHR_RUNTIME_ROOT    当前节点运行根目录
DFDHR_ARCHIVE_ROOT    已校验长期归档根
DFDHR_CACHE_ROOT      模型与依赖缓存根
```

要求：

- 代码目录只保存 Git 跟踪内容和小文件。
- 数据根只读，不写缓存、索引、预测、日志或 checkpoint。
- 每个 RUN_ID 使用独立运行目录。
- 缓存、环境和大型产物不写入共享代码目录。
- 同名本地路径不代表不同节点拥有相同内容。
- 存储受限节点必须在 manifest 中设置最大本地预算。
- 默认关闭特征转储，只保留 best 和 last checkpoint。
- 归档前本地完整写入，传输后比较 SHA-256，校验成功前不删除本地副本。

## 7. Git 工作流

### 7.1 分支职责

- `main`：已验证代码、稳定配置、测试和公开文档。
- `fix/*`：单一缺陷修复。
- `feat/*`：可复用能力。
- `infra/*`：Jupyter、运行管理、匿名审计和归档工具。
- `exp/*`：单一研究变量。
- `docs/*`：文档和模板。
- `release/*`：复现版本冻结。

禁止直接在 `main` 开发。服务器审计、训练代码和实验参数不得混入同一分支。

### 7.2 修改和提交

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

规则：

- 使用 `git add <明确文件>`，避免未检查的 `git add .`。
- 一个提交只包含一个逻辑变化。
- 提交信息使用 `<type>(<scope>): <summary>` 或仓库既有简洁英文风格。
- 不覆盖用户未提交修改，不改写他人未合并历史。
- 推送前运行与修改风险相称的验证。

## 8. 实验定义与模板

RUN_ID：

```text
<project>_<train-set>_<core-variable>_<YYYYMMDD>_<sequence>
```

模板：

```text
templates/experiment_manifest.yaml
templates/experiment_summary.md
templates/pull_request_template.md
templates/infrastructure.local.example.yaml
registry/experiments.csv
```

模板文件已经建立不代表自动化流程已经完成。实施状态以 `TASK_INDEX.md` 为准。

每个正式实验记录：问题、假设、基线、唯一变化、成功标准、RUN_ID、分支、提交、冻结配置及哈希、数据协议/JSON 哈希、初始权重哈希、环境、GPU、命令、随机种子、有效 batch、日志、指标、best/last、失败原因和归档校验。

状态统一为：

```text
planned
preflight
smoke_passed
running
completed
failed
aborted
archived
```

## 9. 实验生命周期

### 9.1 Preflight

必须确认：

- Git 干净且提交已记录。
- Python 和依赖正确。
- GPU、VRAM、磁盘符合配置和预算。
- 数据、JSON 和 split 可解析。
- 初始权重 strict load，哈希已记录。
- 输出位于运行根，不在代码/数据目录。

任何一项失败都不能开始正式训练。

### 9.2 两批次 Smoke Test

验证：模型实例化、两个真实 batch 的 forward/loss/backward/step、有限 loss、无 OOM、冻结参数无梯度、Router/Adapter/Head 有梯度、checkpoint 保存和严格加载、峰值显存和时间记录。

Smoke 只验证链路，不产生研究结论。

### 9.3 Mini Run

固定小子集和 seed，使用与正式训练一致的有效 batch，运行 1–3 epoch。验证 validation、best/last、完整 resume、结构化日志和正常退出。不得作为论文主结果。

### 9.4 正式训练

- 通过 `tmux`、调度器或等价持久会话启动。
- 多卡使用 `torchrun`。
- 记录 micro-batch、world size、累积步数和有效 batch。
- checkpoint 原子写入。
- 默认只保留 best/last。
- 不依赖浏览器或 Notebook kernel 存活。

### 9.5 最终评估

validation 选择 best；训练完成后冻结 checkpoint，再进行跨数据集测试。帧级/视频级指标必须记录聚合方法、样本数、划分和所有来源哈希。

## 10. Jupyter 规范

适合：环境与路径检查、checkpoint strict load、数据协议审计、单卡显存 Smoke、小规模推理、绘图和只读训练监控。

不适合：长期训练、多卡长期 DDP、依赖浏览器持续在线的任务。

计划中的 Notebook，目前均按未完成处理：

```text
00_environment_and_paths.ipynb
01_checkpoint_strict_load.ipynb
02_dataset_protocol_audit.ipynb
03_single_gpu_memory_smoke.ipynb
04_official_weight_eval.ipynb
05_training_monitor.ipynb
```

要求：第一单元记录解释器、工作目录、Git 和 CUDA；固定随机种子；无隐藏状态；Restart Kernel 后 Run All 成功；源 Notebook 进入 Git，执行副本进入运行目录。

## 11. 基础设施审计边界

允许无需额外批准：有限 SSH 检查、公开系统硬件/文件系统状态、当前用户项目/环境/数据/权重存在性、项目单元测试和有限 Smoke。

需要明确批准：系统级安装、SSH/网络/驱动/CUDA/挂载/调度器/监控修改、大规模数据或环境复制、删除文件、清理他人目录、终止/抢占进程、启动长时间训练。

不得读取或提交其他用户命令行、环境、shell 历史、项目文件、数据内容、密钥或凭据。

## 12. 验证强度

- 文档：链接、命令、占位符、敏感信息、`git diff --check`。
- 配置：解析与关键字段测试。
- 通用代码：相关单测和完整 CPU 测试。
- 数据协议：fixture 验证划分、采样和泄漏防护。
- 训练逻辑：CPU 单测、单卡两批次 Smoke，必要时两进程 Smoke。
- Checkpoint：保存、严格加载、恢复连续性。
- 指标：手工可计算固定输入。

不能运行 GPU 验证时，明确记录未验证项和原因，不能用 CPU 单测代替 GPU 链路证据。

## 13. 任务交接与完成标准

每次任务结束：

1. 更新 `TASK_INDEX.md` 状态和证据。
2. 记录分支与提交。
3. 记录测试、Notebook、配置、报告和运行目录。
4. 写明未知项和下一项有限动作。
5. 保持工作树干净，或明确记录未提交内容。

一个实验只有同时具备 RUN_ID、问题定义、干净提交、冻结配置、数据协议、环境、命令、日志、指标、best/last、校验归档和明确结论才算完成。

只有 checkpoint、只能依靠隐藏状态运行的 Notebook、或缺少提交与配置关联的结果，均不构成可复现实验。
