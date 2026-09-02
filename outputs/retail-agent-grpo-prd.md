---
title: Retail Agent Online GRPO Training and Validation
status: needs-triage
labels:
  - needs-triage
---

# Retail Agent Online GRPO Training and Validation

## 问题陈述

项目已经具备 τ³ Retail 轨迹契约、官方 reward 映射、Retail Policy Verifier、Action-only SFT、SFT-to-GRPO Gate，以及不依赖运行时的 GRPO 数学核心，但还不能把通过 Gate 的 SFT checkpoint 接入真实环境完成在线 GRPO。

当前缺少的闭环包括：

- 从同一个 Retail task 采集多个独立 rollout；
- 保存并对齐 assistant action token 的 action mask、old/current/reference logprob；
- 对无效 reward、全零 reward 和策略违规进行可审计处理；
- 对一个 rollout batch 做多轮 clipped GRPO 更新；
- 保存可恢复 checkpoint，并用固定 validation task 选择 checkpoint；
- 在训练过程中记录足够的 reward、advantage、loss、有效 group 数和异常原因。

如果直接把所有失败轨迹当成负样本，或者把基础设施评分失败当成模型失败，GRPO 会得到错误的训练信号。若只看训练 loss，也无法判断模型是否在 τ³ 环境中真的变好。

## 解决方案

在通过 SFT-to-GRPO Gate 的 Qwen3.5-2B Action-only LoRA checkpoint 上，实现自定义单卡在线 GRPO：

1. 从固定的 60 个 train task 池中采样 50 个候选 prompt group。
2. 每个 group 对同一个 task 生成 4 条完整 τ³ Retail rollout；每条 rollout 使用独立 seed 和隔离的环境状态。
3. 用官方终局 reward 和 Retail Policy Verifier 得到每条 rollout 的有效性与训练 reward。
4. 如果一个 group 中任意 rollout 的 `reward_valid` 为 false，则丢弃整个 group；有效但 reward 为 0 或 -1 的 group 保留。
5. 在每个保留 group 内计算 group-relative advantage，并保留采样时的 old logprob。
6. 对全部保留 group 组成的 rollout batch 做两轮 minibatch 更新；两轮之间不重新采样。
7. 使用 action-only clipped objective 和 SFT reference KL：

   `loss = -J_clip + 0.001 * KL(reference, current)`

8. 在训练过程保存多个 checkpoint，并在固定的 14 个 validation task 上做无梯度评估，只用 validation 指标选择 checkpoint。
9. 最终 30 个 final-test task 只用于最终报告，不参与 GRPO 更新或 checkpoint 选择。

第一版不实现跨多个 rollout batch 的长期 replay buffer。两轮复用仅限于同一批由同一个行为策略采集的数据；下一批数据必须由更新后的当前策略重新采样。

## 用户故事

1. 作为 GRPO 训练器，我希望从已通过 SFT-to-GRPO Gate 的 checkpoint 开始，以避免在错误的初始策略上解释 GRPO 结果。
2. 作为 GRPO 训练器，我希望只从 60 个 train task 采样，使 validation 和 final test 不被训练污染。
3. 作为 GRPO 训练器，我希望每个 rollout batch 包含 50 个候选 group，使一次更新覆盖足够多的任务行为。
4. 作为 GRPO 训练器，我希望每个 group 包含同一 task 的 4 条独立 rollout，使 reward 可以进行组内相对比较。
5. 作为环境执行器，我希望每条 rollout 使用独立 seed 和隔离数据库状态，使一条轨迹的 mutation 不影响同组其他轨迹。
6. 作为环境执行器，我希望 inference microbatch 默认从 2 开始，并能降到 1，而不改变 trajectory contract。
7. 作为数据管道，我希望保存完整 user message、assistant message、tool call、tool result、terminal state 和 evaluation，使训练样本可追溯。
8. 作为 reward 处理器，我希望沿用 τ³ 官方 reward 作为任务结果来源，而不是引入未经验证的 LLM judge。
9. 作为策略验证器，我希望策略违规获得 hard penalty，以便模型学习避免缺少认证、缺少确认和非法并发工具调用。
10. 作为策略验证器，我希望普通工具错误只产生有限扣分，而不是覆盖掉有效的任务结果。
11. 作为数据管道，我希望 `reward_valid=false` 被标识为基础设施或评分无效，而不是被当作模型的负奖励。
12. 作为数据管道，我希望 group 内只要有一条无效 reward 就丢弃整个 group，以保持 group size=4 的相对比较语义。
13. 作为训练器，我希望有效但失败的 rollout 仍被保留，因为 reward=0 或 reward=-1 可以与成功 rollout 形成相对偏好。
14. 作为训练器，我希望在每个 group 内计算 reward mean、标准差和 advantage，并在 reward 全相同时稳定地产生全零 advantage。
15. 作为训练器，我希望全零 reward 不被人为改成负分，避免模型学习错误的失败标签。
16. 作为训练器，我希望保存采样时的 old logprob，使同一个 rollout batch 可以进行有限的多轮 clipped 更新。
17. 作为训练器，我希望第二轮更新重新计算 current logprob，而不错误地复用第一轮的 current logprob。
18. 作为训练器，我希望 loss 只作用于 assistant action token，不作用于 user token、tool observation token 和 padding token。
19. 作为训练器，我希望使用 SFT checkpoint 作为冻结 reference policy，并通过小系数 KL 防止策略发生无约束漂移。
20. 作为训练器，我希望每个 rollout batch 只训练两轮，之后采集由新策略产生的新 batch，而不是无限复用旧轨迹。
21. 作为训练器，我希望 batch 内部可拆成 minibatch，以适应 32GB GPU 的显存限制。
22. 作为训练器，我希望记录候选 group 数、有效 group 数、丢弃 group 数、有效 rollout 数、零 advantage group 数和更新步数。
23. 作为训练器，我希望当整个 batch 没有非零 advantage 时得到明确诊断，而不是误以为模型获得了有效学习。
24. 作为训练器，我希望保存 optimizer、scheduler、随机数状态、当前 batch manifest 和训练配置，以便中断后恢复。
25. 作为实验开发者，我希望 checkpoint 的选择只依赖固定 validation task 的推理结果，而不依赖 final test。
26. 作为实验开发者，我希望 validation 记录成功率、平均 reward、策略违规率、工具错误率和 reward 有效率。
27. 作为实验开发者，我希望 validation 采用固定 task ID 和 seed，使不同 checkpoint 的差异可以归因于模型而不是评估采样变化。
28. 作为实验开发者，我希望策略违规和无效 reward 先作为硬门槛，再使用成功率和平均 reward 排序 checkpoint。
29. 作为项目维护者，我希望数学核心、token 对齐、rollout 过滤、训练循环和 validation 选择都能通过独立测试验证。
30. 作为项目审阅者，我希望最终报告能够区分“没有相对 reward 信号”“reward 无效”“模型任务失败”和“策略违规”。

## 实现决策

### 前置条件与范围

- GRPO 初始模型必须是通过 SFT-to-GRPO Gate 的 Action-only LoRA checkpoint。
- 主模型为 Qwen3.5-2B；训练方式为单卡 LoRA，不做全参数微调。
- 训练环境为 τ³ Retail；不修改 vendor 上游实现。
- 训练 task 只来自固定的 60 个 train task。
- validation 使用固定的 14 个 task；final test 使用固定的 30 个 task。
- 第一版不把“最多 100 optimizer steps”当成算法固定值。训练预算应按 rollout batch 数、minibatch 大小、训练轮数和实际 optimizer step 分开记录。

### Rollout batch 与 group

- 每个 rollout batch 采集 50 个候选 group。
- 每个 group 固定包含同一 task 的 4 条 rollout。
- 同组 rollout 的 task ID 相同，seed 不同，环境状态隔离。
- 50 个 group 从 train task 池按可复现的采样策略产生；task ID、seed 和 policy version 必须写入 batch manifest。
- 50 是候选 group 数；无效 group 被丢弃后，有效 group 数可以小于 50，不在第一版隐式补采样。
- 环境 worker 数默认 4；inference microbatch 默认 2，显存不足时允许降为 1。

### Reward 与有效性

- 以 trajectory evaluation 中的官方 reward 为任务结果来源。
- `reward_valid=false` 表示基础设施或评分无效，不得转成模型负奖励。
- group 内任意一条 rollout 的 `reward_valid=false`，整个 group 丢弃。
- 有效 group 中的 reward=0 失败轨迹保留。
- 策略违规 reward 为 -1.0。
- 无策略违规时，官方 reward 按工具错误数进行有限扣分：每次扣 0.1，最多扣 0.2。
- 重复调用等行为若已经由 Verifier 归类为策略违规，沿用 hard penalty；普通可恢复工具错误继续使用有限扣分。
- 如果有效 batch 没有任何非零 advantage，训练器必须记录 `zero_advantage_batch`；第一版不得通过修改 reward 人为制造信号。

### Advantage 与 loss

- 每个 group 独立计算 reward mean 和总体标准差。
- 标准差为零时，该 group 的 advantage 全部为 0，不产生 NaN。
- old policy 是采样 rollout 时的策略；reference policy 是冻结的 SFT checkpoint；current policy 是正在更新的 LoRA policy。
- 对 action token 计算 `ratio = exp(current_logprob - old_logprob)`。
- clipped objective 使用 `epsilon=0.2`：

  `J_clip = mean(min(ratio * A, clip(ratio, 0.8, 1.2) * A))`

- `mean` 只覆盖 action mask 选中的 token；user、tool observation 和 padding 不参与 loss。
- reference KL 使用 action-token sampled-action 近似。
- KL 系数固定为第一版初始值 `beta=0.001`。
- 训练 loss 为：

  `loss = -J_clip + beta * KL`

- 数学核心继续保持为可注入、可独立测试的纯函数；在线 trainer 只负责把真实 batch 组装并调用这些核心。

### Batch 复用与更新轮次

- 一个 rollout batch 采集完成后，固定其 reward、advantage、old logprob、reference logprob 和 action mask。
- 对有效 group 组成的 batch 做两轮 minibatch 遍历。
- 第二轮必须重新计算 current logprob，因为第一轮已经改变了当前模型参数。
- 两轮结束后丢弃该 batch，使用更新后的策略重新采样下一批。
- 第一版不实现跨 batch replay buffer、旧 policy age 过滤或长时间 stale trajectory 复用。
- 训练日志同时记录 `batch_index`、`epoch_in_batch`、`optimizer_step` 和 `valid_group_count`，避免把 rollout batch、训练轮次和 optimizer step 混为一个计数器。

### Checkpoint 与 validation

- 至少保存初始 SFT checkpoint 和每个 GRPO rollout batch 后的候选 checkpoint。
- checkpoint 必须可加载，并包含 LoRA 权重、训练配置、优化器状态、scheduler 状态和随机数状态。
- validation 不计算梯度，不生成 advantage，不更新模型。
- checkpoint screening 使用固定的 14 个 validation task 和固定 seed；第一版每个 task 跑一次以控制环境成本。
- 对接近的候选 checkpoint，可对候选模型增加重复 seed 评估；最终 final test 仍单独使用每题 3 次的既定协议。
- checkpoint 选择顺序：先排除 reward 无效或策略违规超过门槛的 checkpoint，再最大化 task success rate；成功率接近时依次比较平均 reward、策略违规率和 tool error rate。
- validation 和 final test 任务不得进入训练 rollout batch。

### 需要建设的深模块

- `RolloutGroupCollector`：按 task/seed 采集 4 条完整轨迹，并保证环境隔离和可恢复 manifest。
- `VerifiedRolloutBatch`：执行 reward_valid 过滤、group 保留/丢弃、reward/advantage 计算和批次统计。
- `ActionLogprobBatch`：把统一 trajectory 渲染成模型输入，生成 action-only mask，并保证 old/current/reference token 对齐。
- `GRPOBatchTrainer`：在固定 rollout batch 上执行两轮 minibatch loss、梯度更新、日志记录和 checkpoint 保存。
- `ValidationCheckpointSelector`：运行固定 validation benchmark，应用安全门槛并选择最佳 checkpoint。
- `TrainingManifestStore`：持久化模型、数据、环境、seed、policy version、超参数、批次和恢复状态。

## 测试决策

测试只验证公开行为和数据契约，不绑定内部类名、私有方法或具体深度学习框架实现。优先使用现有纯 Python fake runner、fake tokenizer、fake trainer 和合成 trajectory 测试模式；真实 τ³ 与 GPU 测试作为独立集成验证。

### Reward、group 和 batch

- 4 条 rollout 的 reward 能正确转换为 group advantage。
- `[0, 0, 0, 0]` 的 advantage 全为 0 且没有 NaN。
- `[1, 0, 0, 0]` 能产生正负相对信号。
- policy violation 的 reward 为 -1.0。
- 工具错误扣分最多为 0.2。
- 任意 `reward_valid=false` 都会丢弃整个 group。
- 有效 reward=0 的 group 不会被丢弃。
- 50 个候选 group 被过滤后，batch 统计准确记录有效和丢弃数量。
- 全 batch zero advantage 会输出明确诊断，不会伪造负 reward。

### Token mask、logprob 和 loss

- user token、tool observation token、padding token 不贡献 loss。
- assistant tool-call 和 final action token 被正确标记。
- old/current/reference logprob 与 action mask 按 token 对齐。
- 高 advantage 的 action 概率提高时 clipped objective 方向为正。
- 低 advantage 的 action 概率提高时 objective 方向为负。
- ratio 超过 clip 边界时使用 clipped 分支。
- reference KL 在 logprob 相同和明显偏离时都稳定且非负。
- 总 loss 正确组合 `-J_clip + 0.001 * KL`。
- 两轮 batch 更新第二轮读取更新后的 current logprob，而不是缓存的旧 current logprob。

### 在线训练与恢复

- 一个 group 能生成 4 条完整轨迹并保存 task/seed/policy version。
- 4 个 worker 不共享会污染彼此的环境状态。
- inference microbatch 从 2 降到 1 时轨迹契约和 group 结构不变。
- 训练可完成至少一次真实或 fake GRPO update 并产生可加载 checkpoint。
- checkpoint 恢复后 optimizer、scheduler、RNG 和 batch manifest 状态一致。
- 训练日志记录 reward、advantage、loss、KL、clip fraction、有效 group 数和异常原因。
- 空 action、超长轨迹、无效 trajectory 和全零 reward 不会静默破坏训练。

### Validation 与 checkpoint 选择

- 所有候选 checkpoint 使用相同的 14 个 task ID 和 seed。
- validation 不写入模型梯度或训练 optimizer 状态。
- invalid reward 会阻止 checkpoint 被选中。
- 策略违规超过门槛的 checkpoint 不会因为成功率高而胜出。
- success rate、average reward、policy violation rate 和 tool error rate 在报告中可追溯。
- final test task ID 不会被 validation selector 读取。

### 端到端验收

- 在 AutoDL Python 3.12 环境中，SFT Gate 通过后能够启动一次最小真实 GRPO batch。
- 真实 batch 能采集、过滤、训练两轮、保存 checkpoint 并重新加载。
- 至少一个后续 checkpoint 能完成 14-task validation，并输出选择原因。
- 所有核心单元测试继续通过；真实环境测试失败时能区分 API、环境、reward、显存和模型解析问题。

## 范围外

- 不做 Raw 直接 GRPO 主线。
- 不做全参数微调、多卡分布式训练或 DeepSpeed/FSDP 优化。
- 不修改 τ³-bench vendor 代码。
- 不实现跨多个 rollout batch 的长期 replay buffer。
- 不把所有失败 reward 重写成负分。
- 不引入 LLM judge、独立 reward model 或未经验证的 shaping reward。
- 不用 final test 选择超参数或 checkpoint。
- 不在本 PRD 中重新实现 SFT、教师采集或最终 30-task 报告，只复用它们作为 GRPO 前置和后续流程。
- 不把训练 loss 当作最终业务成功指标。

## 补充说明

- `50 groups × 4 rollouts` 是一个候选 rollout batch 的规模；无效 group 被丢弃后，实际有效 batch 可以更小。
- “两轮训练”是对整个有效 batch 做两轮 minibatch 遍历，不是对每个 group 单独启动两个独立训练过程。
- rollout batch、batch epoch、minibatch、optimizer step 必须在代码和日志中使用不同字段名。
- 如果 reward 全为 0，GRPO 的组内相对学习信号为 0；正确动作是记录诊断、检查 SFT/探索/reward pipeline，而不是伪造 reward。
- validation 的职责是监控泛化、安全性和 checkpoint 选择，不是参与训练 advantage。
- 第一阶段的成功标准是闭环正确、可审计、可恢复；超参数优化和更复杂的 off-policy replay 放在闭环稳定之后。
