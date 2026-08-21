print("__name__ =", __name__)
print("__package__ =", __package__)

"""Retail Agent 实验命令行入口：环境配置、运行编排和报告落盘。"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Union

from .pipeline import (
    build_sft_dataset,
    create_tau2_retail_runner,
    run_smoke,
)
from .policy_verifier import RetailPolicyVerifier
from .sft_training import SFTTrainingConfig, train_sft
from .task_partition import load_retail_task_partition
from .teacher_collection import TeacherTrajectoryCollector
from .trajectory_store import TrajectoryStore


def llm_args_from_env(
    prefix: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """按角色读取 API 参数，角色专属变量优先于共享 Anthropic 变量。"""
    values = environ or os.environ
    args: Dict[str, str] = {}
    api_key = values.get(prefix + "_API_KEY") or values.get("ANTHROPIC_API_KEY")
    api_base = values.get(prefix + "_API_BASE") or values.get(
        "ANTHROPIC_BASE_URL"
    )
    if api_key:
        args["api_key"] = api_key
    if api_base:
        args["api_base"] = api_base
    return args


def load_project_env(
    path: Union[str, Path] = ".env",
    *,
    environ: Optional[MutableMapping[str, str]] = None,
) -> int:
    """Load simple KEY=value entries without overwriting process variables."""
    env_path = Path(path)
    if not env_path.exists():
        return 0

    values = environ if environ is not None else os.environ
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in values:
            # 不覆盖进程中已有变量，方便命令行/运行平台显式注入密钥。
            values[key] = value
            loaded += 1
    return loaded


def build_parser() -> argparse.ArgumentParser:
    """声明 smoke、教师采集、SFT 构建和 SFT 训练四类命令。"""
    parser = argparse.ArgumentParser(description="Retail Agent experiment pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run and verify one Retail task")
    _add_runtime_options(smoke)
    smoke.add_argument("--task-id", required=True)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--output-dir", default="outputs/smoke")

    collect = subparsers.add_parser(
        "collect-teacher", help="collect teacher trajectories on train tasks"
    )
    _add_runtime_options(collect)
    collect.add_argument(
        "--split-tasks",
        default="vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json",
    )
    collect.add_argument("--attempts-per-task", type=int, default=5)
    collect.add_argument("--base-seed", type=int, default=1000)
    collect.add_argument("--max-workers", type=int, default=4)
    collect.add_argument("--output-dir", default="outputs/teacher")

    build = subparsers.add_parser(
        "build-sft", help="build accepted-only SFT JSONL from trajectories"
    )
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)

    train = subparsers.add_parser("train-sft", help="run Qwen LoRA SFT")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", default="outputs/sft")
    train.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-2B"),
    )
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--max-length", type=int)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析命令并调用 pipeline；CLI 不承载模型或 τ³ 业务逻辑。"""
    load_project_env(os.environ.get("RETAIL_AGENT_ENV_FILE", ".env"))
    args = build_parser().parse_args(argv)
    if args.command == "build-sft":
        summary = build_sft_dataset(
            input_path=args.input,
            output_path=args.output,
            verifier=RetailPolicyVerifier(),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train-sft":
        config = SFTTrainingConfig(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            max_length=args.max_length,
        )
        result = train_sft(config)
        print(json.dumps({"status": "trained", "result": repr(result)}))
        return 0

    # build-sft/train-sft 在上面提前返回；只有 smoke 和 teacher collection
    # 才需要创建真实 τ³ runner。
    runner = create_tau2_retail_runner(
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        trajectory_path=_runtime_trajectory_path(args),
        agent_llm_args=_runtime_llm_args(args, "agent"),
        user_llm_args=_runtime_llm_args(args, "user"),
        max_steps=args.max_steps,
        max_errors=args.max_errors,
    )

    if args.command == "smoke":
        report = run_smoke(
            runner=runner,
            verifier=RetailPolicyVerifier(),
            task_id=args.task_id,
            seed=args.seed,
        )
        output_dir = Path(args.output_dir)
        _write_json(output_dir / "smoke_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # 教师采集只使用固定 partition 的 train task，validation/final_test 留给评估。
    partition = load_retail_task_partition(args.split_tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_store = TrajectoryStore(output_dir / "runtime.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=runner,
        verifier=RetailPolicyVerifier(),
        raw_store=TrajectoryStore(output_dir / "raw.jsonl"),
        accepted_store=TrajectoryStore(output_dir / "accepted.jsonl"),
        failed_store=TrajectoryStore(output_dir / "failed.jsonl"),
    )
    summary = collector.collect(
        task_ids=partition.train,
        attempts_per_task=args.attempts_per_task,
        base_seed=args.base_seed,
        max_workers=args.max_workers,
        runtime_store=runtime_store,
    )
    report = {
        "task_count": len(partition.train),
        "attempts_per_task": args.attempts_per_task,
        "summary": asdict(summary),
    }
    _write_json(output_dir / "collection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    """为需要真实 simulation 的命令添加 LLM 和运行上限参数。"""
    parser.add_argument(
        "--agent-llm",
        default=os.environ.get("AGENT_LLM", "anthropic/deepseek-v4-flash"),
    )
    parser.add_argument(
        "--user-llm",
        default=os.environ.get("USER_LLM", "anthropic/deepseek-v4-flash"),
    )
    parser.add_argument("--agent-api-base")
    parser.add_argument("--user-api-base")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=5)


def _runtime_llm_args(args: argparse.Namespace, role: str) -> Dict[str, str]:
    """合并环境变量配置与当前命令对指定角色的显式覆盖。"""
    prefix = role.upper()
    values = llm_args_from_env(prefix)
    api_base = getattr(args, role + "_api_base")
    if api_base:
        values["api_base"] = api_base
    return values


def _runtime_trajectory_path(args: argparse.Namespace) -> Path:
    """为 smoke 和批量采集分配不同的轨迹文件名，避免混写。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        return output_dir / "trajectory.jsonl"
    return output_dir / "runtime.jsonl"


def _write_json(path: Path, payload: object) -> None:
    """创建父目录并以 UTF-8、可读格式写出一份报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
