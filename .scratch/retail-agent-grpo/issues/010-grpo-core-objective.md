---
id: retail-grpo-010
title: 实现 GRPO Group Reward 与 Action-only Loss 核心
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 父项

- retail-grpo-005

## 要构建什么

实现不依赖 τ³ 运行时的 GRPO 数学核心：同一 prompt 的 group reward 标准化、全相同 reward 的稳定 advantage、action mask、clip objective 和 reference KL。该切片可独立用合成 rollout 验证，后续再接在线环境 rollout。

## 验收标准

- [ ] group reward 正确转换为 advantage。
- [ ] reward 全相同时 advantage 为零且不会产生 NaN。
- [ ] observation token 不产生 policy loss。
- [ ] 高 reward action 的 objective 方向正确，clip 和 KL 边界稳定。
- [ ] 只提供纯函数/小模块和测试，不修改 τ³ vendor。

## 被以下事项阻塞

- None - can start immediately
