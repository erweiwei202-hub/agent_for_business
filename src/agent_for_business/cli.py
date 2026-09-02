print("__name__ =", __name__)
print("__package__ =", __package__)

"""Retail Agent 实验命令行入口：环境配置、运行编排和报告落盘。"""

import argparse
import json
import math
import os
import platform
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Union

from .grpo_training import (
    GRPOTrainingConfig,
    resolve_grpo_model_source,
    train_grpo,
)
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
    """声明 Retail smoke、数据构建、SFT 和 GRPO 训练命令。"""
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

    grpo = subparsers.add_parser(
        "grpo", help="run online GRPO from an SFT checkpoint"
    )
    grpo.add_argument("--model", required=True)
    grpo.add_argument("--output-dir", default="outputs/grpo")
    grpo.add_argument(
        "--split-tasks",
        default="vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json",
    )
    grpo.add_argument("--groups-per-batch", type=_positive_int, default=50)
    grpo.add_argument("--group-size", type=_positive_int, default=4)
    grpo.add_argument("--batch-epochs", type=_positive_int, default=2)
    grpo.add_argument("--max-workers", type=_positive_int, default=4)
    grpo.add_argument(
        "--inference-microbatch", type=_positive_int, default=2
    )
    grpo.add_argument(
        "--parallel-generation",
        action="store_true",
        help="allow rollout workers to generate concurrently on the policy model",
    )
    grpo.add_argument("--clip-ratio", type=_non_negative_float, default=0.2)
    grpo.add_argument("--kl-beta", type=_non_negative_float, default=0.001)
    grpo.add_argument("--seed", type=int, default=42)
    grpo.add_argument("--max-rollout-batches", type=_positive_int, default=2)
    grpo.add_argument(
        "--user-llm",
        default=os.environ.get("USER_LLM", "anthropic/deepseek-v4-flash"),
    )
    grpo.add_argument("--user-api-base")
    grpo.add_argument("--learning-rate", type=_positive_float, default=1e-5)
    grpo.add_argument("--weight-decay", type=_non_negative_float, default=0.0)
    grpo.add_argument("--temperature", type=_non_negative_float, default=0.7)
    grpo.add_argument("--top-p", type=_probability, default=0.95)
    grpo.add_argument("--max-new-tokens", type=_positive_int, default=512)
    grpo.add_argument("--device", default="auto")
    grpo.add_argument("--checkpoint-every", type=_positive_int, default=1)
    grpo.add_argument("--resume-from")

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
    if args.command == "grpo":
        config = GRPOTrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            split_tasks=args.split_tasks,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            batch_epochs=args.batch_epochs,
            max_workers=args.max_workers,
            inference_microbatch=args.inference_microbatch,
            parallel_generation=args.parallel_generation,
            clip_ratio=args.clip_ratio,
            kl_beta=args.kl_beta,
            seed=args.seed,
            max_rollout_batches=args.max_rollout_batches,
            user_llm=args.user_llm,
            user_llm_args=_runtime_llm_args(args, "user"),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume_from,
        )
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_payload = asdict(config)
        config_payload["rollout_plan"] = config.rollout_plan
        model_source = resolve_grpo_model_source(config.model_name)
        config_payload["model_source"] = {
            "kind": "lora" if config.use_lora or model_source.is_lora else "full",
            "base_model_name_or_path": model_source.base_model_name_or_path,
            "adapter_path": model_source.adapter_path,
        }
        _write_json(output_dir / "grpo_training_config.json", config_payload)
        _write_json(
            output_dir / "grpo_run_manifest.json",
            {
                "command": "grpo",
                "status": "started",
                "runtime": _runtime_manifest(),
                **config_payload,
            },
        )
        try:
            result = train_grpo(config)
        except Exception as error:
            _write_json(
                output_dir / "grpo_failure.json",
                {"command": "grpo", "status": "failed", "error": repr(error)},
            )
            raise
        _write_json(
            output_dir / "grpo_result.json",
            {"command": "grpo", "status": "trained", "result": result},
        )
        print(
            json.dumps(
                {"status": "trained", "result": result},
                ensure_ascii=False,
                default=str,
            )
        )
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


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for resource and batch settings."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a finite non-negative float for objective coefficients."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and non-negative"
        )
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive finite float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _probability(value: str) -> float:
    """Parse a probability in the open-closed unit interval."""
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


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
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _runtime_manifest() -> Dict[str, object]:
    """Return reproducibility metadata without requiring optional training deps."""
    dependencies: Dict[str, str] = {}
    for package in ("torch", "transformers", "trl", "tau2"):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


if __name__ == "__main__":
    raise SystemExit(main())
