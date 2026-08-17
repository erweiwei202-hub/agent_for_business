---
id: retail-grpo-006
title: 完成单卡在线 GRPO 和 Validation Checkpoint 选择
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 要构建什么

在约 32GB AutoDL GPU 上运行完整的单卡在线 GRPO。使用 4 个环境 worker、inference microbatch=2 起步，显存不足时降为 1，最多运行 100 个 optimizer steps，保存多个 checkpoint，并依据 14 个 validation task 选择最佳 checkpoint。

## 验收标准

- [ ] 训练可以使用 4 个环境 worker 运行，且环境数据库状态互不污染。
- [ ] inference microbatch 可以从 2 降到 1，显存不足时不会破坏轨迹契约。
- [ ] GRPO 最多运行 100 个 optimizer steps，并记录每步 Reward、advantage、loss 和有效轨迹数。
- [ ] 训练可以保存并恢复 checkpoint。
- [ ] 训练过程记录模型、数据、环境、随机种子和超参数版本。
- [ ] checkpoint 选择只读取 validation 结果，不读取 final test 结果。
- [ ] 训练失败或 Reward 无效比例过高时，能够输出诊断日志。

## 被以下事项阻塞

- retail-grpo-005

## 类型与覆盖范围

- 类型：AFK
- 覆盖用户故事：18、19、20、21、22、28

