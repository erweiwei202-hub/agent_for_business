---
id: retail-grpo-011
title: 增加 GRPO CLI 启动入口与训练配置落盘
status: needs-triage
labels:
  - needs-triage
type: AFK
---

## 父项

- retail-grpo-006

## 要构建什么

为 Retail Agent 增加统一的 `grpo` CLI 命令，把已通过 SFT-to-GRPO Gate 的 checkpoint 接入在线 GRPO trainer。命令负责解析并校验 GRPO 运行参数、加载项目环境配置、创建输出目录、把配置传递给公开 trainer 接口，并落盘可审计的训练配置和启动报告。

启动方式应覆盖当前已确认的训练协议：每个 rollout batch 采集 50 个候选 group、每组 4 条 rollout、无效 group 整组丢弃、有效 batch 做 2 轮 minibatch 更新、clip ratio=0.2、reference KL beta=0.001。CLI 不应把 rollout batch、batch epoch、minibatch 和 optimizer step 混成同一个参数。

## 验收标准

- [ ] `python -m agent_for_business.cli grpo --help` 能展示 GRPO 启动命令和关键参数。
- [ ] `grpo` 命令支持 checkpoint/model、output-dir、split-tasks、groups-per-batch、group-size、batch-epochs、max-workers、inference-microbatch、clip-ratio、kl-beta、seed 和训练预算参数。
- [ ] 默认配置为 50 groups、group size 4、batch epochs 2、4 个环境 worker、inference microbatch 2、clip ratio 0.2、KL beta 0.001。
- [ ] 命令拒绝非正的 group、epoch、worker、microbatch 和训练预算参数，并拒绝不合法的 clip/KL 值。
- [ ] 命令将解析后的配置传递给公开 GRPO trainer 接口，而不是在 CLI 内实现 rollout、reward 或 loss 逻辑。
- [ ] 命令加载 `.env` 中的项目配置，但不覆盖进程中已有的环境变量。
- [ ] 命令在输出目录落盘 JSON-safe 的 GRPO 配置、启动 manifest 和训练状态/结果报告。
- [ ] manifest 记录 model/checkpoint、task split、seed、采样协议、loss 超参数和环境/依赖版本字段。
- [ ] trainer 失败时 CLI 返回非零状态并保留诊断配置，不生成“训练成功”报告。
- [ ] parser、配置校验、trainer 参数传递和失败路径有行为测试；测试通过 fake trainer 验证公开 CLI 行为，不依赖真实 GPU 或 τ³ API。
- [ ] README 或 AutoDL runbook 包含一个可复制的 `grpo` 启动示例，并明确必须先通过 SFT-to-GRPO Gate。

## 被以下事项阻塞

- retail-grpo-005
- retail-grpo-006

