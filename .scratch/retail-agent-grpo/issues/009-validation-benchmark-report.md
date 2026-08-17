---
id: retail-grpo-009
title: 实现 Raw/SFT Validation Benchmark 报告
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 父项

- retail-grpo-004

## 要构建什么

在固定 validation task_id 上，用相同 runner、seed 和 verifier 分别运行 Raw 与 SFT，并输出可复现的 BenchmarkSummary、GateDecision 和 JSON 报告；无效 reward 必须阻断 Gate。

## 验收标准

- [ ] Raw 与 SFT 使用同一批 validation task_id 和相同评估协议。
- [ ] 报告包含成功率、策略违规率、tool error rate、reward_valid 和 Gate 原因。
- [ ] 报告只接受 validation 输入，不读取 final test。
- [ ] Gate 未通过时返回明确原因并阻止 GRPO 初始 checkpoint 选择。

## 被以下事项阻塞

- retail-grpo-001
- retail-grpo-002
