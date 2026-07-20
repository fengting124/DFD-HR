# DFD-HR 完整复现资源 Preflight

## 目标

从固定 revision 的官方 CLIP ViT-L/14 预训练权重初始化 DFD-HR，不加载官方已训练的 DFD-HR checkpoint；在 FaceForensics++ c23 上训练，以 validation 选择 best，冻结后再执行预先声明的最终测试。

先完成只读资源 Preflight 和节点选择；取得单独批准后，在控制节点角色下载并校验 pinned CLIP，再复制到首选 3090 候选并重复校验。未启动训练或 GPU benchmark。

## 必需资产

- FaceForensics++ c23 train、validation 和 test。
- 三个 split 明确、非空、互斥且有限抽样路径可解析的 JSON。
- 可运行 DFD-HR 且全测试通过的 Python 环境。
- 固定 revision 和 SHA-256 的 CLIP ViT-L/14 snapshot。
- 可容纳 best、last、一次原子临时 checkpoint、日志、指标和安全余量的本地存储。
- 启动前可连续独占的 GPU。

官方已训练的 DFD-HR checkpoint 仅用于校准和对照，不属于独立训练初始化资产。

## 训练节点硬门槛

实测完整 checkpoint 为 `2465259076` bytes。三份 checkpoint 写入窗口需要 `7395777228` bytes，加 10 GiB 安全余量后的硬下限约为 `16.89 GiB`；计入日志和指标后，manifest 不应低于 `20 GiB`，当前建议保持 `30 GiB`。启动时普通用户可用空间不得低于 `50 GiB`。

FaceForensics++ JSON 审计得到 train/validation/test 视频计数 `2156/420/420`，三组 video ID 两两无交集。每个候选节点均检查了同一组 30 个有限样本路径。

## 候选节点结论

| Node role | GPU | FF++ train/val/test | Environment | CLIP | Scratch | Current status |
|---|---|---|---|---|---|---|
| controller node | 2 × RTX 2080 Ti | JSON 与 30/30 路径通过 | Ready，56 tests OK | Verified | 44% used，约 983 GiB free | `READY_FOR_PRETRAINED_SMOKE` |
| 3090 candidate A | 2 × RTX 3090，当前有负载 | 数据 30/30，JSON 缺失 | Missing | Missing | 29% used，约 2.48 TiB free | `BLOCKED_BY_GPU / ENVIRONMENT / CLIP` |
| 3090 candidate B | 2 × RTX 3090，显存被占用 | 数据 30/30，JSON 缺失 | Missing | Missing | 90% used，约 375 GiB free | `NOT_RECOMMENDED` |
| evaluation node | 2 × RTX 3090，采样时空闲 | JSON 与 30/30 路径通过 | Tests pass，Jupyter 依赖不完整 | Missing | 98% used，约 81 GiB free | `BLOCKED_BY_STORAGE` |
| additional 3090 candidate C | 2 × RTX 3090，一卡轻载、另一卡空闲 | JSON 与 30/30 路径通过 | Tests pass，Jupyter 依赖不完整 | Verified | 23% used，约 2.71 TiB free | `READY_FOR_PRETRAINED_SMOKE / COORDINATION_REQUIRED` |
| additional 3090 candidate D | 2 × RTX 3090，采样时满载 | 数据 30/30，JSON 缺失 | Missing | Missing | 55% used，约 1.56 TiB free | `BLOCKED_BY_GPU / ENVIRONMENT / CLIP` |
| archive candidate | 未完成专项审计 | 未要求 | 未要求 | 未要求 | 未验证 | `UNRESOLVED` |

GPU 状态只代表单次采样，不构成预约或连续可用承诺。

## 首选方案

**controller node** 已具备 FF++ 数据、JSON、环境、测试、输出空间、已验证的有限显存链路和 pinned CLIP，可以作为 2 × RTX 2080 Ti 回退角色。**additional 3090 candidate C** 同样已具备 pinned CLIP，并因训练资产完整、空间充足而成为速度优先候选，但仍需明确的连续双卡预约。

RTX 2080 Ti 的代价是预计训练更慢。现有证据只证明 micro-batch 1 配合梯度累积可运行，不能据此推断完整训练时长；CLIP 落地后仍需单独批准相同配置的有限吞吐 Smoke。

补充审计还确认一个 2 × RTX 2080 Ti 节点采样时空闲、环境存在且空间充足；一个 4 × RTX 2080 Ti 节点具备环境和空间，但采样时四卡均满载。前者可在需要时作为额外回退候选，后者在 GPU 明确释放前不准备资产。历史 Smoke 证明 11 GiB 单卡显存可用 micro-batch 1 跑通工程链路，但正式吞吐和四卡扩展性仍需 pinned CLIP 下的独立有限 Smoke 证据。

## 备选方案

**3090 candidate A** 是条件满足后的速度优先备选，不是当前可启动节点。切换条件为：

1. 两张 GPU 已协调并可连续独占。
2. 重新创建 DFD-HR 环境并通过全测试。
3. 只准备 FF++ JSON 并重跑 split、哈希和路径审计。
4. pinned CLIP snapshot 已落地并校验。
5. 获批的同配置有限吞吐 Smoke 表明迁移收益成立。

该节点已经通过 30/30 FF++ 样本路径检查，因此不计划复制 FaceForensics++ 数据本体。

## 训练资产与评估资产的区分

完整训练只要求 FF++ c23 train/validation/test、环境、JSON、CLIP、GPU 和运行空间。Celeb-DF-v2、DFD、DFDC、DFDCP、DeeperForensics、WildDeepfake、FFIW 和 DF40 不应为了训练提前复制到候选节点。

controller node 同时是当前首选评估角色。训练结束后，可以在独立批准下把冻结 best checkpoint 和最小元数据传到评估角色，再对完整外部测试协议做一次新鲜 Preflight。当前 evaluation node 因 scratch 使用率过高，不建议承担新的训练或归档任务。

## 最小资产移动结果

固定 revision 的 CLIP snapshot 已先在控制节点角色下载，再通过可恢复传输复制到 additional 3090 candidate C。两端 `model.safetensors` 大小均为 `1710540580` bytes，SHA-256 均为 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`；同 revision 的配置、processor、tokenizer 文件齐全，processor 和 427,616,513 参数的 CLIP 模型均可完全离线加载。

若改选 3090 candidate A，额外只需：

- 约 11.5 MB 的 FF++ JSON，复制后比较 SHA-256 并重跑有限路径审计。
- 重新创建约 5.9 GiB 安装规模的 Python 环境。不得把 Conda 目录直接 rsync 后默认视为可靠。

不需要复制 FF++ 数据本体、外部评估数据或官方 DFD-HR checkpoint。

## 尚需批准的动作

以下批准彼此独立，前一项不自动包含后一项：

1. 下载或复制 pinned CLIP snapshot。**DONE**
2. 若改选资产不完整的 3090 节点，准备其环境和 FF++ JSON；当前首选 3090 角色不需要此项。
3. 执行 pretrained initialization 的有限 Smoke 和同配置吞吐检查。
4. 启动完整训练。

本报告不构成以上任何动作的批准。

## 正式训练前剩余门槛

- pinned CLIP snapshot 在选定节点存在，revision、size、SHA-256 和必要配套文件全部通过。
- 选定节点启动前重新检查 GPU 独占可用性和至少 50 GiB 普通用户可用空间。
- pretrained initialization 的有限 Smoke 获批并通过。
- 冻结正式配置、有效 batch、RUN_ID、manifest、数据 JSON 哈希和初始权重哈希。
- 训练命令通过 `tmux` 或调度器执行，best/last、结构化指标和恢复方案已确认。
- 归档角色完成专项审计；在此之前不得删除训练节点本地副本。
- 获得独立的完整训练启动批准。
