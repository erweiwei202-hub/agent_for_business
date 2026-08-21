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

# 创建并进入 Python 3.12 环境
conda create -n agent-for-business python=3.12 -y
conda activate agent-for-business

# 安装项目内置的 tau2，以及项目的开发和训练依赖
python -m pip install --upgrade pip
python -m pip install -e vendor/tau2-bench
python -m pip install -e ".[dev,training]"

python -m pytest -q
```

本地测试必须先通过。当前目标是 69 个测试通过。

## 1. 配置教师模型和 User Simulator

项目会自动读取根目录 `.env`。先创建它：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```dotenv
ANTHROPIC_API_KEY=你的密钥
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
AGENT_LLM=anthropic/deepseek-v4-flash
USER_LLM=anthropic/deepseek-v4-flash
```

这里使用的是 Anthropic Messages 协议，不是 OpenAI Chat Completions 协议。CLI 会把同一个 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL` 同时用于 Agent 和 User Simulator。

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
outputs/teacher/runtime.jsonl
outputs/teacher/raw.jsonl
outputs/teacher/accepted.jsonl
outputs/teacher/failed.jsonl
outputs/teacher/collection_report.json
```

`runtime.jsonl` 是 Runner 的原始轨迹和运行检查点存储，用于断点续跑，不是超时日志。
任务开始时会写入 `evaluation.runtime_status=running`，正常完成写入 `completed`，
异常写入 `error` 和 `evaluation.runtime_error`；每一行对应一次 `(task_id, seed)`
状态快照。超时或其他运行失败要看该轨迹的 `evaluation` 和
`collection_report.json` 分类。`accepted.jsonl` 才是 SFT 候选；`failed.jsonl`
用于 Badcase 分析；无效评分会被 Verifier 排除，不当成模型失败。

默认 LLM 请求超时为 60 秒，可通过以下参数调整：

```bash
--request-timeout 60
```


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

下一步需要用同一批 14 个 validation task 比较 Raw 和 SFT。只有 SFT-to-GRPO Gate 通过后，才启动 GRPO。final test 的 30 个 task 不能用于这个判断。

当前代码已经有 validation benchmark 和 Gate 的纯逻辑入口；真实模型 runner 和 checkpoint serving 的 CLI 仍是下一项实现，不要自己猜命令。

## 你现在实际要做什么

当前只做两件事：

1. 准备 AutoDL，并把项目放到 `/root/agent_for_business`；
2. 配好 API key 后运行第 2 步的 smoke 命令。

smoke 没通过之前，不要采集教师数据、不要训练 SFT、不要启动 GRPO。
