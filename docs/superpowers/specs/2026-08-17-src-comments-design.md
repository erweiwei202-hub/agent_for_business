# `src` 注释补充设计

## 目标

为刚接触项目的开发者解释 `src/agent_for_business` 的模块职责、主要数据流和关键业务约束，让读者可以从源码理解“τ³ Retail 轨迹采集 → 策略校验/奖励 → SFT 数据 → Validation Gate → GRPO 核心”的关系。

## 范围

- 覆盖 `src/agent_for_business` 下所有项目自有 `.py` 文件。
- 不处理 `__pycache__`、第三方 `vendor` 代码和测试代码。
- 只增加模块 docstring、类/公共函数 docstring，以及关键状态转换和算法边界的行内注释。
- 保持运行逻辑、公开 API、数据格式和异常行为不变。

## 注释分层

1. **模块层**：说明模块在流水线中的位置，以及主要输入/输出。
2. **模型与接口层**：解释数据类字段含义、依赖注入点和适配器边界。
3. **流程层**：解释轨迹记录、τ³ 消息转换、Verifier 状态机、教师数据分流和 SFT 构建的关键状态变化。
4. **算法层**：解释 Action-only token mask、Raw/SFT Gate、GRPO group advantage、clipped objective 和 reference KL 的设计意图及边界。
5. **入口层**：说明 CLI 命令和 pipeline 如何组装组件，以及环境变量/懒加载的原因。

简单赋值、直接转发和名称已经足够清晰的代码不添加重复注释。

## 关键模块地图

```text
cli.py
  └─ pipeline.py
       ├─ tau_provider.py → τ³ Retail
       ├─ retail_runner.py → tau_adapter.py → trajectory.py
       │                                      └→ trajectory_store.py
       ├─ policy_verifier.py
       ├─ teacher_collection.py
       └─ sft_dataset.py → sft_training.py

validation_benchmark.py → validation_gate.py
grpo_core.py             （独立的纯 Python 数学核心）
task_partition.py        （官方任务划分边界）
badcase.py               （Verifier 结果分类）
```

## 验证方式

- 使用 `python -m compileall src` 检查语法和 docstring 修改没有破坏导入。
- 使用项目现有测试命令运行测试；若当前环境无法安装或导入 τ³ 依赖，记录实际限制并运行可执行的核心测试。
- 对比修改前后公开行为，不改变 JSONL 结构、CLI 参数和核心计算结果。
