---
id: retail-grpo-007
title: 完成 Raw、SFT、GRPO Final Test 和 Badcase 报告
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 要构建什么

对 Raw Qwen3.5-2B、SFT checkpoint 和通过 Gate 后的 SFT+GRPO checkpoint，在固定的 30 个 final test task 上各运行 3 次，生成可比较的成功率、策略违规、工具错误、轨迹效率和 Badcase 报告。

## 验收标准

- [ ] 三个实验组使用完全相同的 30 个 final test task。
- [ ] 每个 task、每个模型运行 3 次，并保存原始 trajectory。
- [ ] final test 不参与 SFT 数据采集、Reward 调参或 checkpoint 选择。
- [ ] 报告整体成功率和查询、取消/退款、修改/换货三类能力指标。
- [ ] 报告策略违规率、错误工具率、错误参数率、平均调用次数和平均轨迹长度。
- [ ] 报告 wrong_tool、wrong_argument、missing_confirmation、authentication_failure、wrong_order、policy_violation、premature_stop 和 tool_loop 分布。
- [ ] 报告 Raw 到 SFT、SFT 到 GRPO 的差异，并保留失败案例。
- [ ] 如果 SFT-to-GRPO Gate 未通过，报告应明确阻止 GRPO 的原因，而不是伪造 SFT+GRPO 结果。
- [ ] 最终结果包含配置、数据版本、环境版本、task_id 和真实实验指标。

## 被以下事项阻塞

- retail-grpo-004
- retail-grpo-006

## 类型与覆盖范围

- 类型：AFK
- 覆盖用户故事：17、23、24、25、26、27、29、30

