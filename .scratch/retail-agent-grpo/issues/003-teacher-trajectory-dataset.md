---
id: retail-grpo-003
title: 完成教师轨迹采集和 SFT 数据构建
status: needs-triage
labels:
  - needs-triage
type: HITL
---

## 要构建什么

接入 DeepSeek Flash 教师 Agent 和 τ³ 官方 User Simulator，在 60 个训练任务上每个任务默认采集 5 条原始轨迹。所有动作必须在 τ³ Retail 环境中真实执行，再由 Retail Policy Verifier 过滤为 accepted SFT 数据、失败 Badcase 数据和基础设施无效数据。

## 验收标准

- [ ] DeepSeek Flash 可以通过 OpenAI-compatible API 作为教师 Agent 运行。
- [ ] 教师 Agent 使用与学生一致的工具定义、政策和环境。
- [ ] User Simulator 能够产生多轮用户回复、确认、拒绝和改变请求。
- [ ] 每条轨迹保存 task_id、seed、教师模型版本、完整交互和 Verifier 结果。
- [ ] 60 个训练任务按 task_id 生成数据，validation 和 final test 不参与教师 SFT 数据采集。
- [ ] accepted、failed 和 infrastructure_invalid 数据分开保存。
- [ ] 教师采集 acceptance rate 和每个 task 的成功情况能够汇总。
- [ ] 某个任务多次失败时可以单独重试，不会覆盖原始失败记录。

## 被以下事项阻塞

- retail-grpo-001
- retail-grpo-002

## 类型与覆盖范围

- 类型：HITL
- 覆盖用户故事：3、4、5、6、7、8、13、28

