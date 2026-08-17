---
id: retail-grpo-004
title: 完成 Action-only LoRA SFT、Validation Benchmark 和 GRPO Gate
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 要构建什么

把 accepted 教师轨迹渲染为 Qwen3.5-2B 的 Action-only SFT 数据，使用普通 LoRA 训练最多 2 个 epoch，并在 14 个 validation task 上运行与最终评估一致的 validation benchmark。SFT 的 2 个 epoch 是初始预算，只有在 Raw 与 SFT 的 validation benchmark 对比通过 SFT-to-GRPO Gate 后，才允许进入主线 GRPO。

Gate 需要确认 SFT 相对 Raw Baseline 的 benchmark 指标有明确改善或没有显著退化、工具格式已经稳定、严重策略违规已经接近零，并且剩余 Badcase 主要是规划、路径选择、效率、终止和错误恢复问题。

## 验收标准

- [ ] 用户消息、系统提示、工具定义和工具结果被保留为上下文。
- [ ] 只有 Assistant 工具调用和最终回复 token 参与 loss。
- [ ] action mask 经过独立测试，observation token 不产生 loss。
- [ ] Qwen3.5-2B 普通 LoRA 训练最多 2 个 epoch，并保存配置和 checkpoint。
- [ ] validation task 与训练 task、final test task 没有 task_id 重叠。
- [ ] 每个 epoch 后，Raw 和 SFT 都能够在相同 validation benchmark 协议下输出工具解析率、参数有效率、任务成功率和策略违规率。
- [ ] Validation benchmark 报告能够比较 Raw 与 SFT 的整体成功率、三类能力指标和 Badcase 分布。
- [ ] Gate 使用 validation benchmark 判断 SFT 是否优于或不显著劣于 Raw，而不是使用 final test。
- [ ] Gate 未通过时，系统不会自动启动主线 GRPO，并会生成诊断原因。
- [ ] Gate 通过时，能够选择明确的 SFT checkpoint 作为 GRPO 初始模型。

## 被以下事项阻塞

- retail-grpo-003

## 类型与覆盖范围

- 类型：AFK
- 覆盖用户故事：13、14、15、16、22、30、31
