# 实验总结

## 基本信息

- RUN_ID：
- 状态：
- 日期：
- 分支：
- 提交：
- 配置及 SHA-256：
- 初始权重及 SHA-256：
- 数据 JSON SHA-256：
- 节点角色：
- GPU：

## 研究问题

## 假设

## 基线

## 唯一变化

## 数据协议

- 训练集：
- 验证集：
- 测试集：
- 压缩级别：
- 帧采样：
- 视频级聚合：

## 训练参数

- Epoch：
- Precision：
- 每卡 batch：
- GPU 数量：
- 梯度累积：
- 有效 batch：
- 优化器：
- 学习率：
- Seed：

## Preflight 与 Smoke Test

- [ ] 环境与路径通过
- [ ] 数据路径抽样通过
- [ ] 官方/初始权重 strict load 通过
- [ ] 单卡两批次 Smoke 通过
- [ ] 冻结/可训练参数梯度检查通过
- [ ] Checkpoint round trip 通过
- [ ] 两卡 DDP Smoke 通过或明确不适用

## 结果

| Dataset | Frame AUC | Video AUC | ACC | AP | EER | Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## 资源消耗

- 峰值显存：
- 平均 step time：
- 总运行时间：
- 本地最大占用：
- Checkpoint 大小：

## 异常与失败

## 结论

## 与假设的关系

## 可否合并到 main

- [ ] 是
- [ ] 否

理由：

## 归档

- 本地运行目录：
- 归档角色/位置：
- 校验状态：
- checksums 文件：

## 下一步
