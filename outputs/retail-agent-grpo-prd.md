---
title: Retail Agent SFT + GRPO Training Pipeline
status: needs-triage
labels:
  - needs-triage
---

# Retail Agent SFT + GRPO Training Pipeline

## 问题陈述

当前项目希望在真实可执行的电商客服环境中，训练一个能够稳定调用工具、遵守业务政策并完成多轮订单处理任务的 Qwen3.5-2B Agent。

直接对原始模型运行 GRPO 存在明显风险：

- 模型可能不会 τ³ Retail 的具体工具格式和业务流程；
- 只使用最终任务成功作为奖励，反馈稀疏且难以定位错误；
- 只让大模型生成文本答案，不能证明工具调用真实执行过；
- 如果没有数据库、通信和策略验证，教师轨迹和 Reward 都可能错误；
- 如果按轨迹而不是按 task_id 划分数据，训练和测试可能发生泄漏；
- 只报告一个最终成功率，无法解释模型是否减少了错误工具、错误参数、缺少确认或越权操作。

因此需要构建一个独立、可审计、可复现的后训练项目，使用 τ³ Retail 作为环境，完成教师轨迹采集、程序化筛选、Action-only SFT、自定义单卡 GRPO、Badcase 分析和独立测试评估。

## 解决方案

项目构建以下完整流水线：

1. 固定 τ³-bench 版本，并复用 τ³ Retail 官方工具、数据库、政策、任务和 User Simulator。
2. 使用 DeepSeek Flash 作为教师 Agent，在环境中真实执行工具调用。
3. 每个训练任务默认采集 5 条原始轨迹，记录完整多轮交互和环境终局。
4. 使用 τ³ 官方 evaluator 加自定义 Retail Policy Verifier 筛选成功且合规的轨迹。
5. 只对 Assistant 工具调用和最终回复进行 Action-only LoRA SFT。
6. 从 SFT 后的 Qwen3.5-2B 开始进行在线 GRPO。
7. GRPO 使用终局成功奖励、策略违规硬惩罚、工具错误惩罚和有限效率惩罚。
8. 使用 task-disjoint validation 和 final test 对 Raw、SFT、SFT+GRPO 三组模型进行公平比较。
9. 输出成功率、策略违规率、工具错误率、训练曲线和 Badcase 报告。

项目目标是完成一个小规模但完整、可审计、可复现的 Agent 后训练实验，不宣称训练出通用电商模型，也不把教师输出直接当作真值。

## 用户故事

1. 作为项目开发者，我希望固定 τ³-bench 版本和任务划分，使所有实验使用同一环境。
2. 作为项目开发者，我希望复用 τ³ Retail 的官方数据库和工具，而不是重新实现电商环境。
3. 作为教师轨迹采集器，我希望使用官方 User Simulator 进行真实多轮对话。
4. 作为教师轨迹采集器，我希望记录用户消息、Assistant 动作、工具结果、最终回复和终局状态。
5. 作为教师轨迹采集器，我希望每个训练任务生成 5 条候选轨迹。
6. 作为数据审核者，我希望能够区分模型错误、工具解析错误和基础设施错误。
7. 作为数据审核者，我希望只有成功、数据库正确、通信正确且无策略违规的轨迹进入 SFT。
8. 作为数据审核者，我希望失败轨迹被保留，用于 Badcase 分析和 Reward 调试。
9. 作为策略验证器，我希望检查用户身份、订单归属和订单状态限制。
10. 作为策略验证器，我希望检查退款、取消、修改和换货是否在需要时获得明确确认。
11. 作为策略验证器，我希望严重策略违规覆盖普通任务成功结果。
12. 作为策略验证器，我希望返回任务成功、数据库匹配、通信正确、策略违规、错误工具次数和首个错误位置。
13. 作为 SFT 数据构建器，我希望完整保留用户消息和工具结果作为模型上下文。
14. 作为 SFT 数据构建器，我希望只在 Assistant 工具调用和最终回复 token 上计算 loss。
15. 作为 SFT 训练器，我希望使用普通 LoRA，不进行全参数微调。
16. 作为 SFT 训练器，我希望默认训练 2 个 epoch，并保存配置和 checkpoint。
17. 作为实验开发者，我希望比较 Raw、SFT 和 SFT+GRPO 三个实验组。
18. 作为 GRPO 训练器，我希望每个 prompt 生成 4 条轨迹进行组内比较。
19. 作为 GRPO 训练器，我希望环境 worker 并行，但模型 inference microbatch 可独立控制。
20. 作为 GRPO 训练器，我希望保存 action mask、logprob、Reward 和 advantage 以便排查训练问题。
21. 作为实验开发者，我希望最多运行 100 个 GRPO optimizer steps，并保存多个 checkpoint。
22. 作为实验开发者，我希望只使用 validation 选择 checkpoint。
23. 作为评估者，我希望三个模型在相同的 30 个 final test task 上运行。
24. 作为评估者，我希望每个测试任务对每个模型运行 3 次。
25. 作为评估者，我希望分别统计查询、取消/退款、修改/换货三类能力。
26. 作为评估者，我希望分别统计策略违规率、错误工具率、错误参数率和平均调用次数。
27. 作为 Badcase 分析者，我希望按 wrong_tool、wrong_argument、missing_confirmation、authentication_failure、wrong_order、policy_violation、premature_stop 和 tool_loop 分类失败。
28. 作为项目维护者，我希望每个实验保存模型、数据、环境、随机种子和训练配置。
29. 作为求职者，我希望简历指标来自真实运行结果，而不是估计值。
30. 作为项目审阅者，我希望通过测试确认 Reward、token mask、logprob 对齐和数据划分正确。

## 实现决策

### 项目边界

- 项目作为独立项目构建，不直接修改 τ³-bench 上游源码。
- τ³-bench 使用固定版本或固定 commit。
- 当前目标是电商客服，不是商品搜索、商品比较、购物车或支付购买。
- 目标能力为订单查询与状态判断、取消订单或退款、修改订单或换货。
- 当前不扩展到其他 domain、语音模式或知识检索模式。
- 教师模型使用可通过 OpenAI-compatible API 调用的 DeepSeek Flash 类模型。
- 学生模型使用 Qwen3.5-2B。

### 数据划分

- 官方 Retail train split 中划出 60 个训练任务和 14 个 validation 任务。
- 官方 test split 中固定选择 30 个 final test task。
- 其余 10 个官方 test task 保留，不参与训练、调参或 checkpoint 选择。
- 任务按能力和复杂度分层选择，不能根据模型结果事后挑选。
- 同一个 task_id 不得同时出现在训练、validation 和 final test。
- 教师轨迹只在训练任务上采集。
- 训练数据默认每个任务采集 5 条原始轨迹；如果某个任务全部失败，只对该任务进行诊断和有限重试。

### 数据记录

每条规范化轨迹至少包含：

- task_id；
- 随机种子；
- 教师或学生模型标识；
- 用户消息；
- Assistant 工具调用；
- 工具返回结果；
- 环境状态或终局状态；
- 官方 evaluator 结果；
- Policy Verifier 结果；
- reward_valid；
- Reward；
- 首个错误类型；
- 采集和训练配置。

失败轨迹不直接丢弃。能够判定的失败进入 Badcase 数据；基础设施无效轨迹单独标记，不伪装成模型失败。

### 深模块

- τ³ Environment Adapter：统一任务加载、环境重置、工具执行、User Simulator、数据库终局和官方评估接口。
- Trajectory Collector：驱动教师或学生完成多轮交互，支持重试、并发、超时和轨迹规范化。
- Retail Policy Verifier：检查身份验证、订单归属、订单状态、确认机制、数据库状态和用户回复。
- Trajectory Dataset Builder：依据 Verifier 生成 SFT 正样本、validation 数据、失败数据和 Badcase 索引。
- SFT Dataset Renderer：渲染 Qwen3.5-2B chat template，并构造 action-only label mask。
- SFT Trainer：使用普通 LoRA 训练 2 个 epoch，保存 checkpoint、配置和验证结果。
- Agent Rollout Engine：让当前模型在环境中产生多轮工具交互，并记录 action token logprob。
- Retail Reward Function：把结构化 Verifier 结果转换为 GRPO scalar reward。
- GRPO Trainer：实现组内 Reward 归一化、advantage、clipped policy objective、reference KL 和 action-only loss。
- Badcase Analyzer：统计首个错误、策略违规、工具错误、终局失败和任务难度。
- Evaluation Runner：在固定 task、trial、模型和环境配置下运行统一评估。
- Experiment Registry：保存实验 ID、数据版本、环境版本、模型版本、随机种子、参数和结果。

### Reward 合同

- 完整任务成功且无策略违规：主要正奖励，目标为 +1.0。
- 安全但部分完成：依据已经验证的数据库子目标和通信子目标给出 0.2 到 0.8。
- 严重策略违规：hard negative，目标为 -1.0。
- 错误工具或错误参数：小幅扣分并设置上限。
- 重复或无效调用：小幅扣分并设置上限。
- 基础设施错误或不可判定轨迹：reward_valid=false，单独统计。
- 不给普通查询步骤持续发正奖励，防止循环查询。
- 错误后成功恢复且没有安全违规时，只进行有限小幅扣分。
- τ³ 官方 evaluator 负责主要任务正确性，Policy Verifier 负责策略安全和诊断。

### SFT

- 只使用通过 Verifier 的完整成功且合规轨迹。
- 使用 Action-only SFT。
- 用户消息、系统提示、工具定义和工具结果保留为上下文，但不计算 loss。
- Assistant 工具调用和最终回复参与 loss。
- 使用普通 LoRA，不做全参数微调。
- 默认最多训练 2 个 epoch；每个 epoch 后都可以在 validation 上评估。
- validation 只用于训练参数和 checkpoint 选择，不用于最终结果报告。

### SFT Validation Benchmark 与 SFT-to-GRPO Gate

SFT 的 2 个 epoch 是初始训练预算，不是自动进入 GRPO 的条件。SFT 完成后，必须在固定的 validation benchmark 上用与最终评估一致的协议和指标，与 Raw Baseline 进行对比。只有在 validation benchmark 上满足以下条件，才进入主线 GRPO：

- SFT 的整体任务成功率和关键能力指标相对 Raw Baseline 有明确改善，或至少没有出现显著退化；
- 工具调用可以稳定解析，工具名称和参数错误不再是主要失败来源；
- 严重策略违规为零或已经降到可接受的极低水平；
- 连续 epoch 的 validation benchmark 提升已经趋于稳定；
- 剩余 Badcase 主要是长程决策、路径选择、效率、终止或错误恢复，而不是基础工具格式问题。

如果 Gate 未通过，优先补充或修正 SFT 数据、工具模板、Policy Verifier 或训练配置，不直接用 GRPO 放大无效动作。

Gate 使用 validation benchmark，不读取 final test 的结果。Gate 的判定、Raw 与 SFT 对比指标和原因必须写入实验记录。

### GRPO

- 主线从通过 SFT-to-GRPO Gate 的 SFT checkpoint 开始，不从 Raw 模型直接进行 GRPO。
- 每个 prompt 生成 4 条完整轨迹。
- 使用 4 个环境 worker 并行运行。
- inference microbatch 从 2 开始，显存不足时降为 1。
- 最多运行 100 个 optimizer steps，并保存多个 checkpoint。
- 使用 validation 任务选择 checkpoint。
- 只对 Assistant action token 计算策略 loss。
- 用户消息和工具 observation 不作为策略更新目标。
- 训练前验证 old logprob、current logprob、reference logprob 和 action mask 的对齐。
- 以单卡实现为主，不引入多机多卡依赖。

### 实验与评估

固定三组实验：

- Raw Qwen3.5-2B；
- Qwen3.5-2B + LoRA SFT；
- Qwen3.5-2B + LoRA SFT + GRPO。

每组在相同的 30 个 final test task 上运行 3 次。

主要指标：

- 整体任务成功率；
- 查询或状态判断成功率；
- 取消或退款成功率；
- 修改或换货成功率；
- 策略违规率；
- 错误工具率；
- 错误参数率；
- 平均工具调用次数；
- 平均轨迹长度；
- reward_valid 比例；
- Badcase 类型分布。

LLM Judge 不作为训练 Reward 的主要来源；后续若加入，仅用于定性解释。

## 测试决策

测试只验证外部行为和数据契约，不依赖内部实现细节。

### Policy Verifier

必须覆盖：

- 正确身份验证后查询；
- 未验证用户就操作订单；
- 正确确认后退款；
- 未确认就退款；
- 对错误用户订单进行修改；
- 订单状态不允许操作时强行操作；
- 工具调用错误但后来成功恢复；
- 数据库状态正确但存在策略违规；
- 环境错误或无法判定的轨迹。

### 轨迹和数据集

必须验证：

- 事件顺序正确；
- 工具调用和工具结果一一对应；
- User Simulator 消息不丢失；
- 终局状态关联正确的 task_id；
- 失败和基础设施错误能区分；
- 保存后重新加载结果一致；
- 用户和工具 observation 正确进入上下文；
- 只有 Assistant action/final response 的 mask 有效；
- 非法角色顺序和空轨迹被拒绝；
- 训练、validation、test task_id 没有重叠。
- SFT-to-GRPO Gate 的输入指标、判定结果和失败原因可复现。

### GRPO Loss

必须验证：

- 同一 prompt 的 group reward 正确归一化；
- reward 全相同时 advantage 行为稳定；
- 高 reward 轨迹概率上升；
- 策略违规轨迹概率下降；
- observation token 不产生策略 loss；
- action 数量变化不会把 observation 错误纳入 loss；
- old、current、reference logprob 对齐；
- clip 和 KL 在边界情况下稳定。

### 端到端

必须验证：

- 查询、退款/取消、修改/换货各至少有完整端到端案例；
- 并行环境 worker 之间数据库状态隔离；
- 超时、工具错误和 User Simulator 异常可恢复或可标记；
- Raw、SFT、SFT+GRPO 三组都能产生规范结果；
- 如果 SFT-to-GRPO Gate 未通过，系统能够阻止主线 GRPO 并生成诊断结果；
- checkpoint 选择只使用 validation；
- 最终报告包含配置、数据版本、task_id、指标和 Badcase。

## 范围外

- 不重新实现 τ³ Retail 数据库和基础工具。
- 不新增商品搜索、商品比较、购物车或支付购买环境。
- 不扩展到其他 τ³ domain、语音或知识检索。
- 不训练独立的 LLM Reward Model。
- 不把 LLM Judge 作为主训练 Reward。
- 不做全参数微调。
- 不做多机多卡分布式 GRPO。
- 不加入 DPO、PPO、RLOO 等额外训练算法。
- 不将 Raw 到 GRPO 作为主线实验。
- 不对官方 test task 生成 SFT 教师数据。
- 不把失败轨迹直接当作 SFT 正样本。
- 不根据最终结果事后筛选 30 个 test task。
- 不宣称模型具备通用电商泛化能力。
- 不在没有真实实验结果时填写简历提升数字。

## 测试和资源说明

目标环境为 AutoDL 约 32GB 显存 GPU：

- Qwen3.5-2B 使用普通 LoRA；
- SFT 使用 2 个 epoch；
- GRPO 使用并行环境、受控 inference microbatch 和 gradient checkpointing；
- 教师模型通过远程 API 调用，不占用学生训练显存；
- 最终训练时间取决于 GPU 型号、工具调用延迟、API 限流和轨迹长度。

主要风险：

1. 教师轨迹通过率低，导致部分任务缺少 SFT 正样本。
2. τ³ 工具格式和 Qwen3.5-2B chat template 或 serving parser 不兼容。
3. 多轮工具交互中的 action token mask 和 logprob 对齐错误。
4. 30 个 test task、每题 3 次支持简历级工程评估，但不等同于大规模 benchmark。
5. SFT 数据规模有限，主要作用是学习 τ³ Retail 工具协议和基本业务流程；GRPO 负责继续优化策略。

完成标准：

- 教师轨迹能够在环境中真实执行并保存；
- Verifier 能区分成功、失败、策略违规和基础设施无效；
- SFT 只使用验证通过的正样本；
- Raw、SFT、SFT+GRPO 在同一 final test 上运行；
- GRPO rollout、Reward、action mask 和 logprob 对齐通过测试；
- 输出训练、评估和 Badcase 报告；
- 简历指标全部来自实际运行结果。

## 补充说明

完成后可以基于真实结果使用以下简历表达方向：

Built a policy-compliant e-commerce customer-service Agent on τ³ Retail. Implemented environment-executed teacher trajectory collection, deterministic database and policy verification, action-only LoRA SFT, and a single-GPU online GRPO loop. Evaluated Raw, SFT, and SFT+GRPO models on task-disjoint multi-turn scenarios and analyzed tool-use and policy-violation Badcases.

其中成功率、违规率和调用次数必须使用真实实验结果替换。
