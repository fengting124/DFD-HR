# DFD-HR 独立复现结果

## 实验范围

正式 RUN `_009` 从 pinned CLIP ViT-L/14 初始化，不加载发布的 DFD-HR
checkpoint，在 FaceForensics++ c23 上完成 20 epoch 训练。训练使用 2 张
GPU、每卡 batch 8、有效 batch 16、AMP、Adam `1e-4`，每个 epoch 对完整
validation split 做一次双卡无重复分片验证。

本轮已完成：

1. 环境、路径、JSON、权重哈希和 checkpoint round-trip 门禁。
2. 单卡 Smoke、双卡 DDP Smoke 和有限 Mini Run。
3. 20 epoch 正式训练及 validation-selected best checkpoint 冻结。
4. FaceForensics++ c23 held-out test。
5. 14 个跨数据集测试，共 `393658` 帧、`13232` 个视频。

`test_DFR` 和 `test_FFIW` 在当前声明协议名下没有可用资产，已明确标为
`not_evaluated`，未计入完成数量。

## 持久化 CSV

完整表格保存在 Git 外归档目录
`dfdhr_ffppc23_full-pretrained_20260722_009_evaluation_results/`。该目录不含
节点信息、日志、数据、环境或 checkpoint 文件。

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `01_experiment_overview.csv` | 1 | RUN、协议、配置和资产哈希 |
| `02_training_progress.csv` | 255 | 训练 loss、局部指标、耗时和显存采样 |
| `03_validation_history.csv` | 20 | 每个 epoch 的完整验证结果 |
| `04_in_domain_test_metrics.csv` | 1 | FF++ c23 held-out 指标 |
| `05_cross_dataset_test_metrics.csv` | 14 | 全部跨数据集指标 |
| `06_not_evaluated_datasets.csv` | 2 | 未评估协议名及原因 |
| `07_artifact_manifest.csv` | 6 | 前六个 CSV 的行数、大小和 SHA-256 |

关键归档哈希：

- `05_cross_dataset_test_metrics.csv`:
  `375ace544bc1a8432851233dac88bbc7cb580b4f5b0f0a77edce6e4af98f2a8a`
- 原始标准化 `summary.json`:
  `921f7750b6bb7f31d2efa51136cc57e644c048202e818cc598858ee7e26f97c2`

## 结果含义

同域 FF++ c23 的 frame AUC 为 `0.9891`，video AUC 为 `0.9953`，说明模型
能够很好地拟合训练协议内的伪造分布。跨域结果更能衡量泛化：

| 数据域 | Frame AUC | Video AUC | 观察 |
| --- | ---: | ---: | --- |
| DeepFakeDetection | 0.9287 | 0.9717 | 对该域迁移较强 |
| Celeb-DF-v2 | 0.8408 | 0.9071 | 有明显域差异 |
| DFDC | 0.8043 | 0.8285 | 泛化弱于同域结果 |
| DFDCP | 0.8217 | 0.8523 | 泛化仍有提升空间 |
| test_WDF | 0.8182 | 0.8625 | 复杂真实分布更困难 |
| e4s_ff | 0.9635 | 0.9802 | 对该 DF40 方法较强 |
| uniface_ff | 0.9543 | 0.9789 | 对该 DF40 方法较强 |

AUC 衡量阈值无关的排序能力，越高越好；AP 更关注正类排序质量；EER 是
假接受率和假拒绝率相等时的错误率，越低越好。video 指标先聚合同一视频
的帧预测，通常比单帧指标稳定。不同数据集的样本数、视频数和类别构成
不同，因此不能直接把 accuracy 当作唯一的跨数据集排名依据。

这些结果表明：模型在 FF++ 内部表现很强，也能迁移到多种 DF40 方法，
但在 DFDC、DFDCP 和 WDF 等分布上存在明显泛化落差。它证明了复现链路
有效，但不能单独证明优于论文、官方 checkpoint 或其他检测器。

## 必要对比

后续结论至少需要以下对比，且必须复用相同 JSON、采样、聚合和指标代码：

1. **官方 checkpoint**：在同一 14 数据集协议上评估，区分复现误差与方法
   本身的域泛化上限。
2. **DDF/CAFormer**：使用相同测试样本和视频聚合，与仓库中的另一后端做
   公平横向比较。
3. **论文表格**：仅在数据版本、split、帧采样和预处理完全一致时比较。
4. **核心消融**：移除 hierarchical routing、移除 MoE、关闭 gate noise、
   改变 routed layers 或 CLIP 初始化，分别判断各组件贡献。
5. **重复运行**：正式结论应使用至少 3 个 seed，报告均值、标准差，并在
   视频级 prediction 上给出 bootstrap 置信区间。

验证分片和 timeout 属于工程正确性改动，不应作为方法增益消融；它们只需
证明与单卡/旧协议在固定子集上指标一致。
