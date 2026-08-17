---
id: retail-grpo-005
title: 实现 GRPO Rollout、Action Mask 和 Logprob 对齐
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 要构建什么

在通过 SFT-to-GRPO Gate 的 checkpoint 上实现单卡 GRPO 的最小真实闭环：同一个 prompt 生成 4 条完整 τ³ 轨迹，计算 Reward、group advantage、old/current/reference logprob，并只更新 Assistant action token。

## 验收标准

- [ ] 同一个 prompt 能生成 4 条完整环境交互轨迹。
- [ ] 4 条轨迹的 Reward 能被正确归一化为 group advantage。
- [ ] action token、user token 和 tool observation token 的边界可追踪。
- [ ] old logprob、current logprob 和 reference logprob 在 action token 上对齐。
- [ ] 高 Reward 轨迹的策略更新方向为提高概率。
- [ ] 严重策略违规轨迹的策略更新方向为降低概率。
- [ ] 至少完成一次真实 GRPO update，并产生可加载 checkpoint。
- [ ] 全部 Reward 相同、空 action、超长轨迹和无效轨迹等边界情况有测试。

## 被以下事项阻塞

- retail-grpo-001
- retail-grpo-002
- retail-grpo-004

## 类型与覆盖范围

- 类型：AFK
- 覆盖用户故事：18、20、21、22、30

