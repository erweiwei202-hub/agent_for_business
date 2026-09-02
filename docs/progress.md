# 项目进度记录

更新时间：2026-08-17

## 项目位置

项目根目录：D:/agent_for_business

需求文档：D:/agent_for_business/outputs/retail-agent-grpo-prd.md

本地 issue tracker：D:/agent_for_business/.scratch/retail-agent-grpo/issues/

τ³-bench vendor：D:/agent_for_business/vendor/tau2-bench

## 已确认的项目方案

- 场景：τ³ Retail 电商客服，不是商品搜索和购买。
- 目标能力：订单查询/状态判断、取消/退款、修改订单/换货。
- 教师模型：远程 DeepSeek Flash 类 API。
- 学生模型：Qwen3.5-2B。
- 数据：官方 Retail train/test 划分；60 train、14 validation、30 final test，剩余 10 个 test task 保留。
- 教师采集：每个训练任务默认 5 条原始轨迹，真实经过 τ³ User Simulator 和环境执行。
- SFT：只使用最终成功且合规的轨迹；普通 LoRA；最多 2 个 epoch；Action-only loss。
- SFT validation benchmark：必须比较 Raw 和 SFT 的独立指标，不能只看 training loss，也不能使用 final test 做 checkpoint 选择。
- 主线实验：Raw、SFT、SFT+GRPO，不做 Raw 直接 GRPO 主线。
- GRPO：自定义单卡 loop；group size 4；4 个环境 worker；inference microbatch 从 2 开始；最多 100 steps。
- Final evaluation：30 个 final test task，每个模型每题运行 3 次。
- Reward：终局成功为主；策略违规 hard penalty；工具错误和重复调用小幅扣分；基础设施无效标记 reward_valid=false。

## 已生成文档和 issue

- D:/agent_for_business/AGENTS.md
- D:/agent_for_business/docs/agents/
- D:/agent_for_business/outputs/retail-agent-grpo-prd.md
- D:/agent_for_business/.scratch/retail-agent-grpo/prd.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/001-retail-runtime-trajectory.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/002-retail-verifier-reward.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/003-teacher-trajectory-dataset.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/004-action-only-sft-gate.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/005-grpo-rollout-loss.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/006-online-grpo-training.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/007-final-evaluation-report.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/008-sft-training-entrypoint.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/009-validation-benchmark-report.md
- D:/agent_for_business/.scratch/retail-agent-grpo/issues/010-grpo-core-objective.md
- D:/agent_for_business/docs/autodl-runbook.md

## 已实现代码

- src/agent_for_business/trajectory.py
  - TrajectoryEvent、Trajectory、TrajectoryRecorder。
  - 记录 user message、assistant message、tool call、tool result、终局状态和评估结果。
- src/agent_for_business/trajectory_store.py
  - JSONL trajectory append 和 reload。
- src/agent_for_business/tau_adapter.py
  - 将 τ³ SimulationRun 的公开 get_messages() 结果转换为统一 trajectory。
- src/agent_for_business/retail_runner.py
  - 将 simulation provider、adapter 和 trajectory store 串起来的单任务 runner。
- src/agent_for_business/tau_provider.py
  - 懒加载 τ³ TextRunConfig、get_tasks 和 run_single_task 的 Retail provider。
- src/agent_for_business/policy_verifier.py
  - 当前已实现身份验证、mutation 前操作详情、显式确认、每次 mutation 消耗一次确认、策略违规 hard penalty 和结构化 db/communication/reward_valid 字段。
- src/agent_for_business/teacher_collection.py
  - raw、accepted、failed trajectory 分流和采集统计。
- src/agent_for_business/sft_dataset.py
  - 将 trajectory 转成保留 tool observation 的 Qwen messages。
  - 只把 assistant tool-call/final message 标记为 trainable。
  - 拒绝没有 assistant training target 的空 SFT 样本。
  - accepted-only builder、Qwen assistant token mask 和 SFT JSONL store。
- src/agent_for_business/validation_gate.py
  - 比较 Raw 与 SFT validation benchmark。
  - 阻断空 benchmark、成功率明显退化、策略违规过高或工具错误率恶化的 SFT。
- src/agent_for_business/badcase.py
  - 将 verifier 结果映射为 missing_confirmation、tool_loop、tool_error、
    authentication_failure 和 infrastructure_invalid 等 badcase 类别。
- src/agent_for_business/sft_training.py
  - Qwen3.5-2B Action-only LoRA 训练配置，最多 2 个 epoch。
  - 懒加载 transformers/peft/trl，支持 fake trainer 测试和配置落盘。
- src/agent_for_business/grpo_training.py
  - 识别完整模型与 SFT LoRA adapter。
  - 直接加载 LoRA policy，保持 adapter 可训练，不要求 merge。
  - 将已加载 policy 交给在线 GRPO trainer factory，并在 CLI manifest 记录模型来源。
- src/agent_for_business/grpo_agent.py、grpo_rollout.py
  - 本地 Qwen3.5 XML tool-call parser、token trace 和 tau2 half-duplex agent。
  - 通过现有 tau2 Retail orchestrator 执行本地 policy rollout，并保留 invalid 结果。
- src/agent_for_business/grpo_objective.py、grpo_online.py
  - action-only torch logprob replay、clipped GRPO、reference KL、并行 rollout、
    optimizer update、checkpoint/resume。
- src/agent_for_business/validation_benchmark.py
  - 使用相同 validation task_ids/seed 运行 Raw 与 SFT 并生成 JSON-safe 报告。
- src/agent_for_business/grpo_core.py
  - group advantage、action-only clipped objective 和 reference KL 纯 Python 核心。
- src/agent_for_business/pipeline.py、cli.py
  - 提供 smoke、并行教师采集、SFT JSONL 构建和 SFT 训练命令。
  - CLI 自动读取根目录 `.env`，使用 `ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL` 和 Anthropic provider-qualified model。
- src/agent_for_business/task_partition.py
  - task_id 级 train/validation/final_test/reserve 划分。
  - 加载官方 Retail split_tasks.json，并应用固定 60/14/30/10 manifest。
- pyproject.toml、README.md、.gitignore。

## 已通过测试

在当前本地 Python 3.9 环境中，最后一次完整测试结果为：

    69 passed

已经覆盖：

- trajectory 事件顺序和字段保存；
- JSONL 持久化往返；
- τ³ 消息适配；
- 单任务 runner；
- τ³ Retail authentication；
- mutation confirmation；
- action summary；
- 每次 mutation 重新确认；
- 结构化 terminal evaluation；
- reward_valid=false 基础设施结果；
- teacher raw/accepted/failed 数据分流；
- tau2 provider 配置和 task 选择。
- task-level split 划分、重叠检测和官方 Retail manifest 加载。
- 官方 reward 的 task_success/reward_valid 映射；
- 可恢复 tool error 的有限扣分；
- multiple pending tool calls 的策略违规检测；
- Action-only SFT message rendering 和空 target 拒绝；
- SFT validation benchmark 的 τ²、Verifier/GRPO、DB、通信和错误统计。
- accepted-only SFT 数据构建和 JSONL round-trip；
- Qwen assistant token mask；
- badcase 分类和 infrastructure_invalid 隔离。
- SFT 配置约束、数据入口、fake trainer 和配置落盘；
- Raw/SFT validation benchmark 报告；
- GRPO group advantage、mask、clip、KL 及输入校验。
- CLI parser、smoke 报告、SFT 构建命令和并行教师采集。
- `.env` 加载、Anthropic Messages credentials 和模型配置。

## 当前 TDD 状态

刚刚新增并完成了测试：

- tests/test_task_partition.py
- tests/test_sft_dataset.py
- tests/test_validation_gate.py
- tests/test_badcase.py
- tests/test_sft_training.py
- tests/test_validation_benchmark.py
- tests/test_eval_scripts.py
- tests/test_grpo_core.py
- tests/test_pipeline_entrypoints.py
- tests/test_cli.py

这些测试覆盖 task_id 级划分、Action-only SFT messages、accepted-only 数据构建、token mask、badcase 分类、SFT 入口、Raw/SFT validation benchmark、综合 final benchmark、GRPO 数学核心、Qwen action parser、tau2 rollout 契约、torch objective、在线 trainer、checkpoint/resume、CLI、`.env` 配置和并行采集。当前本地不依赖 tau2 的测试为 116 passed；真实 tau2 集成需在 AutoDL Python 3.12 验证。

## 环境注意事项

- 上游 tau2-bench 当前要求 Python >=3.12,<3.14。
- 当前开发机 Python 为 3.9，只能运行不依赖 tau2 的核心测试。
- 上游 tau2 直接导入还需要完整依赖，例如 loguru；不应在本地 Python 3.9 上强行运行完整 Retail。
- 真实 τ³ Retail smoke test 应在 AutoDL Python 3.12 环境执行，使用 uv sync --extra dev。

## 下一步顺序

1. 在 AutoDL Python 3.12 上按 `docs/autodl-runbook.md` 执行 smoke test。
2. smoke 通过后，用 4 workers 采集 60 个训练 task 的教师轨迹。
3. 构建 SFT JSONL，执行 2 epoch LoRA SFT。
4. 运行真实 Raw/SFT validation benchmark 并落盘综合报告。
5. 根据 benchmark 报告选择 checkpoint，启动在线 GRPO rollout/update，再进行最终 30-task 评估。

## TDD 规则

继续遵守单个行为的 RED → 最小 GREEN → 全量测试循环；不要一次性写完所有 SFT、GRPO 代码。测试验证公开行为，不绑定内部实现细节。
