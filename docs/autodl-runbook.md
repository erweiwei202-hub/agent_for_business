# AutoDL 实验操作手册

这份手册对应当前项目已经实现的 CLI。不要直接运行 `src/` 里的内部 Module。

## 0. 准备项目

把整个 `D:/agent_for_business` 上传到 AutoDL，例如放在：

```bash
/root/agent_for_business
```

进入项目并安装依赖：

```bash
cd /root/agent_for_business

# AutoDL 通常已经安装 Conda。如果当前 shell 中 conda 命令不可用，先执行：
# source /root/miniconda3/etc/profile.d/conda.sh
# 如果 Conda 安装在其他位置，请改成对应的 profile.d/conda.sh 路径。
conda --version

# 创建并进入 Python 3.12 环境；如果环境已经存在，只执行 activate。
# 你的终端之前显示的是 agent-for-business；如果实际名字是
# agent_for_business，请把下面两处名字替换成实际的 conda 环境名。
conda create -n agent-for-business python=3.12 -y
conda activate agent-for-business

# 安装项目内置的 tau2，以及项目的开发和训练依赖
python -m pip install --upgrade pip
python -m pip install -e vendor/tau2-bench
python -m pip install -e ".[dev,training]"

python -m pytest -q
```

本地测试必须先通过。当前不依赖 tau2 的测试应为 116 passed；AutoDL 再运行两组
tau2 集成测试（允许 tau2 的 audioop 弃用警告）。

## 1. 配置教师模型和 User Simulator

项目会自动读取根目录 `.env`。先创建它：

```bash
cp .env.example .env
```

然后编辑 `.env`。其中 `USER_LLM` 是 tau2 的 User Simulator 模型，不是 GRPO
正在训练的本地 policy；GRPO policy 由命令行的 `--model` 指定：

```dotenv
ANTHROPIC_API_KEY=你的密钥
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
AGENT_LLM=gpt-5.6-luna
USER_LLM=gpt-5.6-luna
```

当前项目根目录 `.env` 的 `USER_LLM` 就是 `gpt-5.6-luna`。这里使用的是
Anthropic Messages 协议，不是 OpenAI Chat Completions 协议。CLI 会把同一个
`ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL` 同时用于 Agent 和 User Simulator，除非
单独设置 `USER_API_KEY` 或 `USER_API_BASE`。GRPO 也会读取这套 User Simulator 配置；
命令行 `--user-llm`、`--user-api-base` 优先级更高。

如果你使用的是其他 Anthropic-compatible 服务，只需要修改：

```bash
ANTHROPIC_BASE_URL=https://你的服务/v1/messages
AGENT_LLM=anthropic/你的模型名
USER_LLM=anthropic/你的模型名
```

## 2. 先跑一个 smoke task

先不要采集 300 条轨迹，先确认环境和 API 真能工作：

```bash
python -m agent_for_business.cli smoke \
  --task-id 0 \
  --seed 100 \
  --output-dir outputs/smoke
```

检查：

```bash
python -m json.tool outputs/smoke/smoke_report.json
wc -l outputs/smoke/trajectory.jsonl
```

这一步的最低要求是命令正常结束、`trajectory.jsonl` 非空、`reward_valid` 不为 false。任务是否成功要看报告，但如果 reward 无效或工具调用解析失败，不能进入下一步。

## 3. 并行采集教师轨迹

smoke 通过后，采集官方 Retail train 中固定的 60 个 task，每题 5 条轨迹：

```bash
python -m agent_for_business.cli collect-teacher \
  --attempts-per-task 5 \
  --max-workers 4 \
  --output-dir outputs/teacher
```

输出文件：

```text
outputs/teacher/raw.jsonl
outputs/teacher/accepted.jsonl
outputs/teacher/failed.jsonl
outputs/teacher/collection_report.json
```

`accepted.jsonl` 才是 SFT 候选；`failed.jsonl` 用于 Badcase 分析；无效评分会被 Verifier 排除，不当成模型失败。

## 4. 构建 SFT JSONL

```bash
python -m agent_for_business.cli build-sft \
  --input outputs/teacher/accepted.jsonl \
  --output outputs/sft/accepted.jsonl
```

## 5. 运行 Qwen3.5-2B LoRA SFT

```bash
python -m agent_for_business.cli train-sft \
  --dataset outputs/sft/accepted.jsonl \
  --output-dir outputs/sft/checkpoint \
  --epochs 2
```

训练入口会保存：

```text
outputs/sft/checkpoint/sft_training_config.json
outputs/sft/checkpoint/       # LoRA checkpoint
```

## 6. SFT 后先做 validation，不要直接 GRPO

下一步需要用同一批 14 个 validation task 比较 Raw 和 SFT。先检查综合 benchmark 中的 τ² reward、Verifier reward、策略违规、工具错误、DB 状态和通信指标，再决定是否启动 GRPO。final test 的 30 个 task 不能用于这个判断。

当前 `validation_benchmark.py` 提供 Raw/SFT 的纯逻辑报告入口，不会自动生成
`passed` 或 `gate_decision`。validation 报告只用于人工/实验规则选择 checkpoint；
final test 不能反过来参与这个选择。

## 7. 启动 vLLM：一次暴露 base 和 SFT 两个模型

推荐让一个 vLLM 进程同时提供 base 和 LoRA SFT。这样两个 benchmark 使用同一个
端口，但通过不同的 model name 访问不同模型。先确认 SFT LoRA 路径存在：

```bash
test -f outputs/sft/checkpoint-qwen/checkpoint-294/adapter_config.json
```

Qwen3.5 使用 `qwen3_coder` parser。不要用 `hermes` 解析 Qwen3.5 的 XML 工具调用：

```bash
python -u eval-scripts/serve_qwen.py \
  --model Qwen/Qwen3.5-2B \
  --served-model-name qwen-base \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --lora-modules qwen-sft=outputs/sft/checkpoint-qwen/checkpoint-294
```

另开终端检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

输出中应该能找到 `qwen-base` 和 `qwen-sft`。两个 benchmark 要串行运行，不能
同时启动两个占用 8000 端口的 vLLM 服务。

## 8. Final test：必须跑固定的 30 个 test task

`eval-scripts/run_final_benchmark.py` 内置的 final-test task 是下面 30 个：

```text
5 9 12 17 18 26 27 32 33 36
38 39 40 42 45 49 51 53 55 56
60 61 62 64 65 68 70 71 74 77
```

每个模型用 3 trials 时，期望运行数是 `30 × 3 = 90`。脚本启动时必须打印：

```text
Running 30 final-test tasks × 3 trials
```

先在 `.env` 中配置 User Simulator。若 User Simulator 使用 Anthropic-compatible
接口，推荐明确写出：

```dotenv
USER_LLM=anthropic/你的用户模拟模型
USER_API_BASE=https://你的服务/v1/messages
USER_API_KEY=你的密钥
```

如果使用 OpenAI-compatible 接口，则把 `USER_API_BASE` 改成对应的 `/v1` 地址。
不要把 `gpt-5.6-luna` 和 Anthropic endpoint 的协议混用，除非该服务明确兼容。

### 8.1 评测 base

保持 vLLM 服务运行，执行：

```bash
python -u eval-scripts/run_final_benchmark.py \
  --env-file .env \
  --agent-llm openai/qwen-base \
  --vllm-api-base http://127.0.0.1:8000/v1 \
  --vllm-api-key EMPTY \
  --output outputs/benchmarks/qwen-base-final.json \
  --summary-output outputs/benchmarks/qwen-base-final.md \
  --num-trials 3 \
  --seed 300 \
  --max-concurrency 1
```

### 8.2 评测 SFT LoRA

同一个 vLLM 服务不需要重启，只把 agent model name 和输出文件改成 SFT：

```bash
python -u eval-scripts/run_final_benchmark.py \
  --env-file .env \
  --agent-llm openai/qwen-sft \
  --vllm-api-base http://127.0.0.1:8000/v1 \
  --vllm-api-key EMPTY \
  --output outputs/benchmarks/qwen-sft-final.json \
  --summary-output outputs/benchmarks/qwen-sft-final.md \
  --num-trials 3 \
  --seed 300 \
  --max-concurrency 1
```

每次运行会生成 JSON 和 `.md` 汇总。检查是否完整：

```bash
python - <<'PY'
import json
from pathlib import Path

for name in ("qwen-base-final", "qwen-sft-final"):
    path = Path("outputs/benchmarks") / f"{name}.json"
    data = json.loads(path.read_text())
    summary = data["benchmark"]["summary"]
    print(name, {
        "expected_runs": summary["expected_runs"],
        "completed_runs": summary["completed_runs"],
        "incomplete_runs": summary["incomplete_runs"],
        "tau_reward_mean": summary["tau_reward_mean"],
        "verifier_reward_mean": summary["verifier_reward_mean"],
        "policy_violation_rate": summary["policy_violation_rate"],
    })
PY
```

只有 `expected_runs=90` 且 `incomplete_runs=0` 才算 30-task × 3-trial
评测完整。`completed_runs` 少于 90 时，不要把 Markdown 中的平均分当成最终结果。

如果要从头重跑某个模型，不能只删除用户可见的 `.json`；tau2 还会从隐藏 checkpoint
恢复。使用新的输出文件名，或同时删除对应的 checkpoint 目录：

```bash
rm -rf outputs/benchmarks/.qwen-base-final.tau2
rm -f outputs/benchmarks/qwen-base-final.json outputs/benchmarks/qwen-base-final.md
```

## 9. GRPO 操作方式

### 9.1 当前仓库状态

当前 `grpo` 命令会解析模型来源，并默认启动真实的单卡在线 GRPO trainer：
本地 Transformers policy 负责生成，tau2 Retail 负责环境/User Simulator 交互，
Retail Verifier 负责 reward，训练只更新 LoRA 参数。真实运行需要 AutoDL Python 3.12
和 `uv sync --extra dev --extra training`；当前 Windows 环境只能运行不依赖 tau2 的测试。

如果一个 group 的 rollout 都完成了但 reward 完全相同（例如全部为 `0.0`），这组没有
GRPO 的相对偏好信号，trainer 会跳过 optimizer update，并在日志/结果中记录
`skipped_no_signal_groups` 与 `no_relative_reward_signal`。如果 rollout 的
`reward_valid=false` 或没有 action trace，则属于无效运行，当前 batch 会报错并写入
`grpo_failure.json`，不能把它当成普通的负样本继续训练。

SFT 的 `checkpoint-294` 是 PEFT LoRA adapter，不是完整模型。现在不需要 merge：
`grpo_training.load_grpo_model()` 会读取 adapter 的 `adapter_config.json`，先加载
`Qwen/Qwen3.5-2B`，再用 `PeftModel.from_pretrained(..., is_trainable=True)` 挂载
adapter。实际在线 trainer 通过 `train_grpo()` 的 factory 接收这个 policy model。
如果 `--model` 直接写 `Qwen/Qwen3.5-2B` 或其他纯基模路径，加载层会使用
`get_peft_model()` 自动挂载一份全新 LoRA，冻结基模并只训练 LoRA 参数；因此该路径不是
完整模型微调模式，也需要安装 PEFT。

使用下面的命令启动真实 GRPO：

```bash
python -m agent_for_business.cli grpo \
  --model outputs/sft/checkpoint-qwen/checkpoint-294 \
  --output-dir outputs/grpo \
  --groups-per-batch 50 \
  --group-size 4 \
  --batch-epochs 2 \
  --max-workers 4 \
  --inference-microbatch 2 \
  --clip-ratio 0.2 \
  --kl-beta 0.001 \
  --user-llm anthropic/deepseek-v4-flash \
  --learning-rate 1e-5 \
  --temperature 0.7 \
  --device cuda \
  --max-rollout-batches 2
```

它会先落盘 `grpo_training_config.json`、`grpo_run_manifest.json`，其中
`model_source.kind` 会标记为 `lora`，并记录基座和 adapter 路径；失败时落盘
`grpo_failure.json`。成功后每个 optimizer step 会保存
`outputs/grpo/checkpoint-<step>/`，其中包含 LoRA adapter、optimizer state 和
`grpo_manifest.json`。

从某个 GRPO checkpoint 继续训练：

```bash
python -m agent_for_business.cli grpo \
  --model outputs/sft/checkpoint-qwen/checkpoint-294 \
  --resume-from outputs/grpo/checkpoint-10 \
  --output-dir outputs/grpo \
  --max-rollout-batches 2
```

resume 时 policy 从 GRPO checkpoint 恢复，reference policy 仍从原始 SFT
checkpoint 加载；因此 reference KL 不会错误地变成当前 policy 的自比较。

### 9.2 GRPO checkpoint 生成后的评测顺序

在线 trainer 生成至少一个 GRPO checkpoint 后：

1. 以 validation 报告选定的 SFT checkpoint 作为 GRPO 初始模型。
2. 使用固定 train task 做 rollout；默认运行 2 个 batch，每批 50 个 group、每组 4
   条 rollout，即每批 200 条、总计 400 条；batch 内训练 2 轮，action token 使用
   clipped objective，并加 reference KL。实际数量会写入
   `grpo_training_config.json`、`grpo_result.json` 的 `rollout_plan`。
3. 保存 GRPO checkpoint 和训练 manifest；不要用 validation/final task 做训练。
4. 将 GRPO checkpoint 暴露成 `qwen-grpo`：

```bash
python -u eval-scripts/serve_qwen.py \
  --model Qwen/Qwen3.5-2B \
  --served-model-name qwen-grpo \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --lora-modules qwen-grpo=outputs/grpo/你的实际checkpoint目录
```

5. 复制 8.2 的 final benchmark 命令，将 `--agent-llm` 改为
   `openai/qwen-grpo`，输出改为 `qwen-grpo-final.json/.md`。GRPO 也必须跑完整
   的 30 个 final-test task，不能只跑 validation task。

## 你现在实际要做什么

当前建议按以下顺序执行：

1. 准备 AutoDL，并把项目放到 `/root/agent_for_business`；
2. 配好 API key 后运行第 2 步的 smoke 命令；
3. 完成 SFT 后启动 vLLM，同时暴露 `qwen-base`/`qwen-sft`；
4. 依次运行 base 和 SFT 的 30-task × 3-trial final benchmark；
5. 检查 `expected_runs=90`、`incomplete_runs=0` 后，再进入 GRPO trainer。

smoke 没通过之前，不要采集教师数据、不要训练 SFT、不要启动 GRPO。
