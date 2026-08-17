---
id: retail-grpo-001
title: 跑通 τ³ Retail 可复现轨迹闭环
status: needs-triage
labels:
  - needs-triage
type: HITL
---

## 要构建什么

建立独立项目与 τ³-bench 的环境适配，使一个 Retail 任务可以经过 User Simulator、Agent 工具调用、环境执行、官方评估和规范化 trajectory 保存，形成后续 Verifier、数据采集和训练共同使用的轨迹契约。

需要固定 τ³-bench 版本或 commit，并在 AutoDL 上记录模型服务、环境和运行配置。

## 验收标准

- [ ] 能固定并记录 τ³-bench 版本、Python 环境和运行配置。
- [ ] 至少一个查询类任务能够完整经过 User Simulator 和 τ³ Retail 环境。
- [ ] Agent 工具调用和工具结果可以被规范化保存。
- [ ] 轨迹包含 task_id、seed、事件顺序、终局状态和官方 evaluator 结果。
- [ ] 同一 seed 和配置可以重新运行并得到可比较的轨迹结构。
- [ ] 失败、超时、工具解析错误和基础设施错误能够被区分。

## 被以下事项阻塞

None - can start immediately

## 类型与覆盖范围

- 类型：HITL
- 覆盖用户故事：1、2、3、4、6、28、30

