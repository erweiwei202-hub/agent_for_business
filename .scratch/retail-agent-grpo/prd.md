---
title: Retail Agent Online GRPO Training and Validation
status: needs-triage
labels:
  - needs-triage
prd: D:/agent_for_business/outputs/retail-agent-grpo-prd.md
---

# Retail Agent Online GRPO Training and Validation

当前 PRD 将已确认的 GRPO 方案发布到项目 issue tracker。

核心约定：

- 以通过 SFT-to-GRPO Gate 的 Qwen3.5-2B Action-only LoRA checkpoint 为起点；
- 从 60 个 train task 中采集 50 个候选 group，每组 4 条独立 rollout；
- 任意一条 `reward_valid=false` 时丢弃整个 group；有效 reward=0 或策略违规 reward=-1 的 group 保留；
- 对剩余 rollout batch 做两轮 minibatch 更新，不做跨 batch 长期 replay；
- loss 为 `-J_clip + 0.001 * KL(reference, current)`，只作用于 assistant action token；
- 使用固定的 14 个 validation task 做无梯度 checkpoint 选择；
- final test 不参与训练、调参或 checkpoint 选择。

完整 PRD：

D:/agent_for_business/outputs/retail-agent-grpo-prd.md

当前状态：needs-triage。
