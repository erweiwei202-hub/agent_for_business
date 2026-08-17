# Retail Agent SFT + GRPO

面向 τ³ Retail 的电商客服 Agent 后训练项目。

当前已实现的基础能力：

- 统一 trajectory contract；
- JSONL trajectory 持久化；
- τ³ SimulationRun 消息适配；
- 单任务 trajectory runner 接口；
- Retail policy 的认证、操作详情、一次性确认和单工具调用校验；
- 官方 reward 到 task_success/reward_valid 的映射；
- Action-only SFT messages 渲染和 assistant loss target 标记；
- accepted-only SFT dataset 构建、JSONL 持久化和 badcase 分类；
- SFT LoRA 训练配置、懒加载训练入口和 validation benchmark 报告；
- GRPO group advantage、action-only clipped objective 和 reference KL 核心；
- Raw/SFT validation benchmark gate，阻止退化的 SFT 直接进入 GRPO。

## 可执行入口

在 AutoDL 上使用：

```bash
uv run python -m agent_for_business.cli smoke --task-id 0
uv run python -m agent_for_business.cli collect-teacher --max-workers 4
uv run python -m agent_for_business.cli build-sft --input outputs/teacher/accepted.jsonl --output outputs/sft/accepted.jsonl
uv run python -m agent_for_business.cli train-sft --dataset outputs/sft/accepted.jsonl --output-dir outputs/sft/checkpoint
```

先复制 `.env.example` 为 `.env` 并填写 Anthropic Messages 配置。完整操作顺序见 `docs/autodl-runbook.md`。smoke 通过前不要采集教师数据或启动训练。

## 开发环境

上游 tau2-bench 当前要求 Python 3.12 到 3.13。推荐在 AutoDL Linux 环境执行：

    uv sync --extra dev --extra training

运行测试：

    uv run pytest -q

当前 Windows 开发机的 Python 3.9 只用于运行不依赖 tau2 的核心单元测试；真实 τ³ Retail runner 需要在 AutoDL Python 3.12 环境执行。
