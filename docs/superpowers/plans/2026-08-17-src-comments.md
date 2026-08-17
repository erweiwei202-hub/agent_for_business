# `src` 注释补充 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `src/agent_for_business` 补充足以帮助新开发者理解模块职责、数据流和关键业务约束的中文注释，同时保持所有运行行为不变。

**Architecture:** 按现有流水线边界补充分层注释：轨迹模型和持久化负责统一数据契约，τ³ 适配/Runner/Provider 负责运行时接入，Verifier 和采集器负责合规筛选，SFT/Validation/GRPO 负责训练数据与优化核心。只修改 docstring 和解释性行内注释，不重构代码、不改变签名和数据格式。

**Tech Stack:** Python 3.9 核心单元测试环境；项目目标 Python 3.12–3.13；pytest；`python -m compileall`；现有 `pyproject.toml` 中的 Ruff 规则。

## Global Constraints

- 覆盖 `src/agent_for_business` 下所有项目自有 `.py` 文件，不处理 `__pycache__`、`vendor` 和测试代码。
- 注释使用中文，代码中的现有英文 API 名称、协议名和数学术语保持原样。
- 保持运行逻辑、公开 API、JSONL 结构、CLI 参数和异常行为不变。
- 注释解释“模块职责、数据流、业务约束和设计原因”，不为简单赋值或直接转发添加重复说明。
- 当前目录不是 Git 仓库，不能执行提交步骤；不得用删除或重置命令处理已有文件。

---

### Task 1: 轨迹契约、持久化与 τ³ 消息边界

**Files:**
- Modify: `src/agent_for_business/trajectory.py`
- Modify: `src/agent_for_business/trajectory_store.py`
- Modify: `src/agent_for_business/tau_adapter.py`

**Interfaces:**
- `Trajectory` 是后续 Runner、Verifier、SFT builder 和 JSONL store 共享的统一数据契约。
- `SimulationTrajectoryAdapter.from_simulation()` 将 τ³ `SimulationRun.get_messages()` 的消息转换成 `Trajectory`，不改变外部 simulation 对象。
- `TrajectoryStore.append()` / `iter_trajectories()` 负责 JSONL 的追加写入和恢复。

- [ ] **Step 1: 为轨迹模型写模块和字段说明**

在 `trajectory.py` 顶部增加模块 docstring；为 `TrajectoryEvent` 说明四类事件和可选字段的关系；为 `Trajectory` 说明 `events`、`terminal_state`、`evaluation` 分别保存什么，以及为什么把轨迹和最终评估放在同一条记录中；为 `TrajectoryRecorder` 说明它按事件发生顺序构造不可变快照。

使用类似以下风格，避免描述字段类型本身：

```python
"""统一记录 τ³ Retail 一次任务执行过程的数据契约。"""

@dataclass
class TrajectoryEvent:
    """按真实消息顺序保存 user、assistant、tool call 和 tool result。"""
```

- [ ] **Step 2: 解释 JSONL store 的追加与恢复边界**

在 `trajectory_store.py` 增加模块、类和两个公共方法的 docstring；注释说明父目录按需创建、追加模式适合长时间采集、锁只保护当前进程内的并发写入，以及空文件/不存在文件的读取行为。

- [ ] **Step 3: 解释 τ³ 消息适配规则**

在 `tau_adapter.py` 增加模块、类和 `from_simulation()` 的 docstring；在 assistant 同时包含文本和多个 tool call、tool 消息通过 call id 关联的代码处补充注释，说明适配器只负责格式转换，终局状态和评估由调用方提供。

- [ ] **Step 4: 运行针对契约和 store 的现有测试**

运行：`python -m pytest -q tests/test_trajectory_contract.py tests/test_trajectory_store.py tests/test_tau_adapter.py`

预期：测试通过；注释改动不应改变 trajectory 事件顺序、JSONL 往返结果或消息映射。

---

### Task 2: Retail 运行时组装、Provider、Pipeline 与 CLI

**Files:**
- Modify: `src/agent_for_business/tau_provider.py`
- Modify: `src/agent_for_business/retail_runner.py`
- Modify: `src/agent_for_business/pipeline.py`
- Modify: `src/agent_for_business/cli.py`

**Interfaces:**
- `Tau2RetailProvider` 是 τ³ 第三方依赖的懒加载边界。
- `RetailTaskRunner` 把 simulation、adapter 和 trajectory store 串成单任务执行接口。
- `pipeline.py` 提供 CLI 使用的高层组装函数；`cli.py` 只负责参数、环境和报告落盘。

- [ ] **Step 1: 解释 Provider 的懒加载和单任务约束**

为 `tau_provider.py` 增加模块、类和 `run()` 的 docstring；在 `run()` 构建 `TextRunConfig`、按 task id 只加载一个任务、最后调用 `run_single_task` 的位置加注释，说明懒加载是为了让当前 Python 3.9 核心测试不必导入 τ³，以及单任务校验避免静默运行错误任务。

- [ ] **Step 2: 解释 Runner 的评估归一化**

为 `retail_runner.py` 增加模块、类和公共方法说明；注释 `info.evaluation` 优先、旧式 `reward_info` 回退的兼容逻辑，以及 `reward == 1.0` 被归一化为成功的原因。

- [ ] **Step 3: 解释 pipeline 和 CLI 的职责边界**

为 `pipeline.py` 的三个公共函数补充用途和输入输出说明；为 `cli.py` 的模块、命令解析、`.env` 加载、运行时参数合并和报告写入补充说明。注释明确 `smoke` 必须先验证单任务，`collect-teacher` 才进入训练任务采集，`build-sft` 和 `train-sft` 不直接启动 τ³ runtime。

- [ ] **Step 4: 运行入口相关现有测试**

运行：`python -m pytest -q tests/test_retail_runner.py tests/test_tau_provider.py tests/test_pipeline_entrypoints.py tests/test_cli.py`

预期：测试通过；CLI 参数默认值、环境变量优先级和报告结构保持不变。

---

### Task 3: 任务划分、策略校验、Badcase 与教师采集

**Files:**
- Modify: `src/agent_for_business/task_partition.py`
- Modify: `src/agent_for_business/policy_verifier.py`
- Modify: `src/agent_for_business/badcase.py`
- Modify: `src/agent_for_business/teacher_collection.py`

**Interfaces:**
- `TaskPartition` 规定 train/validation/final_test/reserve 的 task id 边界。
- `RetailPolicyVerifier.verify()` 输出结构化策略、成功、工具错误和 reward 信息。
- `BadcaseAnalyzer` 使用 Verifier 结果分类；`TeacherTrajectoryCollector` 按 raw/accepted/failed 分流。

- [ ] **Step 1: 说明固定 task split 的实验隔离目的**

为 `task_partition.py` 增加模块、常量、数据类和公共函数 docstring；注释解释 validation 不能与 final test 混用、固定 manifest 的可复现性，以及重复/遗漏 task id 检查的意义。

- [ ] **Step 2: 说明 Verifier 的状态机和 policy hard penalty**

为 `VerificationResult`、`RetailPolicyVerifier` 和 `verify()` 补充说明；在 authentication、pending tool、confirmation、action summary、mutation 消耗确认、multiple tool calls 和 tool error 分支处加中文注释。明确：策略违规是 hard penalty，工具错误只做有限扣分，`reward_valid=False` 表示基础设施/评分无效而不是模型失败。

- [ ] **Step 3: 说明 Badcase 分类优先级与采集分流**

为 `BadcaseRecord`、`BadcaseAnalyzer` 和教师采集器补充 docstring；在 `_category()` 说明 infrastructure invalid 优先于其他分类，在 collector 中说明所有 raw 轨迹先记录，只有 reward 有效、任务成功且无策略违规的轨迹才进入 accepted。

- [ ] **Step 4: 运行校验和采集现有测试**

运行：`python -m pytest -q tests/test_task_partition.py tests/test_policy_verifier.py tests/test_badcase.py tests/test_teacher_collection.py`

预期：测试通过；确认状态机和 accepted/failed 计数未改变。

---

### Task 4: Action-only SFT 数据和训练入口

**Files:**
- Modify: `src/agent_for_business/sft_dataset.py`
- Modify: `src/agent_for_business/sft_training.py`

**Interfaces:**
- `ActionOnlySFTRenderer` 保留 tool observation，但只把 assistant action/final message 作为训练目标。
- `ActionOnlySFTDatasetBuilder` 对 trajectory 做验证、渲染和 accepted-only 筛选。
- `QwenActionOnlyTokenFormatter` 生成 assistant token mask；`SFTDatasetStore` 持久化 JSONL。
- `SFTTrainingConfig` 和 `train_sft()` 是懒加载的 Qwen/LoRA/TRL 训练入口。

- [ ] **Step 1: 说明 SFT message rendering 和 accepted-only 边界**

为 `sft_dataset.py` 的数据类、renderer、builder、formatter 和 store 补充模块/类/公共方法说明；注释解释 observation 需要保留以提供决策上下文，但 user/tool observation 不应产生 loss；说明没有 assistant target 的样本必须跳过，避免训练空样本。

- [ ] **Step 2: 说明 token mask 的对齐和训练依赖懒加载**

在 token formatter 的 assistant span 对齐、`sft_training.py` 的配置校验、依赖导入和 trainer 构建处添加注释，说明 mask 必须与 input ids 等长、最多两个 epoch 的项目约束，以及懒加载让数据构建/单元测试不依赖大型训练栈。

- [ ] **Step 3: 运行 SFT 现有测试**

运行：`python -m pytest -q tests/test_sft_dataset.py tests/test_sft_training.py`

预期：测试通过；SFT JSONL 字段、空 target 拒绝、token mask 和 fake trainer 行为保持不变。

---

### Task 5: Validation Gate、benchmark 报告与 GRPO 数学核心

**Files:**
- Modify: `src/agent_for_business/validation_gate.py`
- Modify: `src/agent_for_business/validation_benchmark.py`
- Modify: `src/agent_for_business/grpo_core.py`

**Interfaces:**
- `run_validation_benchmark()` 用同一 validation task ids/seed 生成 JSON-safe 报告。
- `SFTValidationGate.decide()` 只允许不明显退化且策略/工具错误可接受的 SFT 进入后续流程。
- `grpo_core.py` 提供无框架依赖的 group advantage、action mask、clipped objective 和 reference KL 纯 Python 核心。

- [ ] **Step 1: 说明 Raw/SFT validation gate 的实验规则**

为 `BenchmarkSummary`、`GateDecision`、`SFTValidationGate` 和 benchmark 公共函数增加 docstring；注释说明 benchmark 必须有任务、Raw/SFT 使用相同 task ids 和 seed、final test 不能用于 Gate，以及成功率退化、策略违规和工具错误恶化分别如何影响决策。

- [ ] **Step 2: 说明 GRPO 的数学输入和 action-only 约束**

为 `grpo_core.py` 的四个公共函数补充中文说明，保留现有数学 docstring 中的公式语义；在 group 标准化、零方差、mask 过滤、ratio clipping、优势广播、sampled-action KL 和输入长度校验处增加必要注释，明确这些函数不负责模型 forward、rollout 或优化器更新。

- [ ] **Step 3: 运行 Gate、benchmark 和 GRPO 现有测试**

运行：`python -m pytest -q tests/test_validation_gate.py tests/test_validation_benchmark.py tests/test_grpo_core.py`

预期：测试通过；Gate 的放行/阻断条件和 GRPO 数值结果不变。

---

### Task 6: 包导出、全量审阅与验证

**Files:**
- Modify: `src/agent_for_business/__init__.py`
- Review: `docs/superpowers/specs/2026-08-17-src-comments-design.md`
- Review: all modified files under `src/agent_for_business`

- [ ] **Step 1: 为包入口补充导出说明**

在 `__init__.py` 增加模块 docstring，按“轨迹/运行时/验证/数据/训练”解释 re-export 的用途；不改变 `__all__` 内容和导入顺序的行为。

- [ ] **Step 2: 做注释差异审阅**

运行：`rg -n "^(class|def|async def)" src/agent_for_business -g '*.py'` 对照源码，确认公共类/函数都有用途说明；再检查关键状态机和算法处没有出现“描述代码表面语义”的重复注释，也没有把过时的实现细节写进注释。

- [ ] **Step 3: 运行语法检查和全量测试**

运行：`python -m compileall -q src`，然后运行 `python -m pytest -q`。

预期：compileall 成功；现有测试全部通过，或在依赖限制下明确记录无法执行的测试及实际可执行结果。

- [ ] **Step 4: 检查行为和文件范围**

运行：`git diff --check`（若 Git 不可用则改为读取修改文件并检查行尾/空白），确认只修改 `src/agent_for_business/*.py` 和计划/设计文档；不生成新的业务数据，不删除用户已有文件。
