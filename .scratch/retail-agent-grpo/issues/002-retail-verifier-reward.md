---
id: retail-grpo-002
title: 实现 Retail Policy Verifier 和 Reward 合同
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 要构建什么

在规范化 trajectory 上实现可重放、可解释的 Retail Policy Verifier。它结合 τ³ 官方 evaluator、数据库终局、用户通信和 Retail 政策，判断任务成功、策略违规、基础设施有效性和 Badcase 类型，并输出 GRPO 使用的 scalar reward。

## 验收标准

- [ ] 能检查身份验证、订单归属和订单状态约束。
- [ ] 能检查退款、取消、修改和换货的确认机制。
- [ ] 能区分任务成功、部分完成、安全失败、策略违规和基础设施无效。
- [ ] 严重策略违规不会因为数据库终局看似正确而获得正向成功奖励。
- [ ] 能输出 task_success、db_match、communication_ok、policy_violation、reward_valid、first_error 和 reward。
- [ ] 人工构造的 goodcase 和 badcase 能通过自动化测试。
- [ ] Reward 的工具错误和重复调用惩罚有上限，不会压过终局成功奖励。

## 被以下事项阻塞

- retail-grpo-001

## 类型与覆盖范围

- 类型：AFK
- 覆盖用户故事：6、7、8、9、10、11、12、27、30

