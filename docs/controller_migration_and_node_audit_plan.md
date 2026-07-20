# 控制节点迁移与新增节点审计计划

本计划用于在原控制节点不可用时，将控制职责迁移到新的候选节点，并对两台新增节点进行只读资源和 DFD-HR 就绪性检查。

真实别名、端口和内部路径从 `.local/infrastructure.yaml` 读取。公共报告只使用节点角色名。

## 1. 范围

本阶段允许：

- 有限超时的 SSH 可达性检查。
- CPU、RAM、GPU、驱动、CUDA compatibility、文件系统和当前利用率检查。
- 当前用户项目、环境、数据根、JSON 和官方权重存在性检查。
- 项目单元测试和不占用长期资源的有限验证。

本阶段禁止：

- 安装系统软件或服务。
- 修改 SSH、网络、驱动、CUDA、挂载、调度器、crontab 或 systemd。
- 复制大数据集或 Conda 环境。
- 删除文件或清理其他用户目录。
- 终止或抢占其他用户进程。
- 启动长时间训练。

## 2. 新控制节点自检

在 `controller_candidate` 上执行：

```bash
hostname
git status --short --branch
git fetch --all --prune
git switch docs/experiment-workflow
git pull --ff-only
git log --oneline --decorate -12
```

读取 `AGENTS.md`、`TASK_INDEX.md` 和本文件。

验证环境变量：

```bash
printf '%s\n' "$DFDHR_REPO_ROOT" "$DFDHR_PYTHON" "$DFDHR_DATA_ROOT"
printf '%s\n' "$DFDHR_RUNTIME_ROOT" "$DFDHR_ARCHIVE_ROOT" "$DFDHR_CACHE_ROOT"
```

验证 Python：

```bash
"$DFDHR_PYTHON" - <<'PY'
import sys
import torch
import transformers
print(sys.executable)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

运行当前测试：

```bash
"$DFDHR_PYTHON" -m unittest discover -s tests -v
```

任何失败都记录在任务索引中，不在本阶段自行安装或修复系统依赖。

## 3. 有限 SSH 探测

从 `.local/infrastructure.yaml` 获取目标别名。每个节点只尝试一次：

```bash
ssh -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=yes \
    <host-alias> 'hostname'
```

记录：时间戳、成功/失败类别、规范 hostname。不要把真实别名或地址写入公共报告。

## 4. 硬件与存储清单

在可达节点上只执行公开系统状态命令：

```bash
hostname
lscpu
free -h
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,utilization.memory,driver_version --format=csv,noheader
nvidia-smi
findmnt /scratch 2>/dev/null || true
df -h /scratch / 2>/dev/null || true
df -ih /scratch / 2>/dev/null || true
```

报告应提取：

- CPU 插槽、物理核心、线程。
- RAM 总量和可用量。
- GPU 型号、数量、单卡 VRAM。
- 驱动和 CUDA compatibility。
- 当前 GPU 利用率。
- scratch 总量、普通用户可用量、使用率和 inode。

GPU 当前空闲只代表采样时刻，不构成长期预约。

## 5. DFD-HR 就绪性

### 5.1 共享代码

- 仓库可见。
- 目标分支可读取。
- 工作树状态不被审计修改。

### 5.2 数据

检查 `${DFDHR_DATA_ROOT}` 和项目 JSON 注册表。至少覆盖：

- FaceForensics++ c23 train/validation/test。
- Celeb-DF-v2。
- 当前测试配置引用的 DFD、DFDC、DFDCP、DeeperForensics、WildDeepfake、FFIW。
- 需要的 DF40 方法。

每个数据集执行：目录存在性、有限文件/目录计数、JSON SHA-256、各 split 抽样路径解析。

禁止在审计中遍历全量数据做昂贵哈希。数据一致性先使用 JSON 哈希、有限计数和抽样路径；正式跨节点训练前再设计受控 manifest。

### 5.3 环境

检查 `${DFDHR_PYTHON}` 是否存在并能导入依赖。若不存在，只记录 `environment_missing`。

### 5.4 官方权重

检查权重文件存在性、大小和 SHA-256。不得因文件名相同就判断内容一致。

### 5.5 测试

环境存在时运行单元测试。测试失败记录命令、退出码和简要错误；不在审计提交中混入训练代码修复。

## 6. 两台新增节点

节点 A、B 必须分别产生记录：

```text
reachability
hardware
storage
gpu_state
repo_visibility
data_readiness
json_path_sampling
environment_readiness
official_weight_hash
test_result
recommended_role
unknowns
```

禁止根据一台节点的结果推断另一台节点。

## 7. 角色选择

完成审计后按以下顺序选择：

1. 单卡 VRAM 与实际 DFD-HR Smoke 峰值。
2. 当前独占可用性。
3. 本地数据就绪性。
4. 本地输出空间和安全余量。
5. 环境就绪性。
6. 归档目标可达性。

角色可包括：控制节点、Jupyter 节点、显存 Smoke 节点、正式训练节点、归档节点、暂不可用节点。

历史表格只能作为线索，不能替代启动前实时检查。

## 8. 输出

本地完整结果写入 `${DFDHR_RUNTIME_ROOT}` 下的受控审计目录，保持 Git 外。

公共脱敏摘要写入：

```text
docs/controller_migration_and_node_audit_results.md
```

摘要只记录匿名角色和复现所需的通用结论。

## 9. 完成条件

- 新控制节点自检有证据。
- 原控制节点状态有一次时间戳重检。
- 两台新增节点分别完成硬件、数据、环境和存储检查。
- 已形成匿名调度建议。
- `.local/infrastructure.yaml` 未进入 Git。
- 未读取或提交其他用户敏感信息。
- `TASK_INDEX.md` 已更新下一步 Smoke Test 节点和阻塞项。
