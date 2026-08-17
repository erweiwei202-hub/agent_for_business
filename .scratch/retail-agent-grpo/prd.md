---
title: Retail Agent SFT + GRPO Training Pipeline
status: needs-triage
labels:
  - needs-triage
prd: D:/agent_for_business/outputs/retail-agent-grpo-prd.md
---

# Retail Agent SFT + GRPO Training Pipeline

该需求已根据已确认的项目范围整理完成。

目标是基于 τ³ Retail 构建完整的电商客服 Agent 后训练流水线，包含：

- DeepSeek Flash 教师轨迹采集；
- τ³ 官方 evaluator 与 Retail Policy Verifier；
- Action-only LoRA SFT；
- 基于 validation 行为的 SFT-to-GRPO Gate；
- 自定义单卡 GRPO；
- Raw、SFT、SFT+GRPO 三组实验；
- task-disjoint validation 和 final test；
- Badcase 分类与可审计实验报告。

完整 PRD 位于：

D:/agent_for_business/outputs/retail-agent-grpo-prd.md

当前状态：needs-triage。
