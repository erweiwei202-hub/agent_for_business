---
id: retail-grpo-008
title: 实现 Action-only SFT 训练入口
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 父项

- retail-grpo-004

## 要构建什么

把已生成的 accepted-only SFT JSONL 接到 Qwen3.5-2B 的普通 LoRA 训练入口。入口能够读取 messages 和 assistant token mask，默认最多训练 2 个 epoch，并保存可复现配置；没有 transformers/peft 时不得在 import 阶段崩溃。

## 验收标准

- [ ] 训练配置默认使用 Qwen3.5-2B、LoRA、Action-only loss 和最多 2 个 epoch。
- [ ] 训练入口从 SFTDatasetStore 读取数据并生成 labels。
- [ ] 配置拒绝全参数微调、epoch 大于 2 或缺少 action-only mask。
- [ ] 通过单元测试验证配置和数据入口；真实训练在 AutoDL Python 3.12 执行。

## 被以下事项阻塞

- retail-grpo-003
