#!/usr/bin/env python3
"""Run the official tau2 Retail final-test benchmark for one served model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_TASK_IDS = (
    "5",
    "9",
    "12",
    "17",
    "18",
    "26",
    "27",
    "32",
    "33",
    "36",
    "38",
    "39",
    "40",
    "42",
    "45",
    "49",
    "51",
    "53",
    "55",
    "56",
    "60",
    "61",
    "62",
    "64",
    "65",
    "68",
    "70",
    "71",
    "74",
    "77",
)


def load_dotenv(path: Union[str, Path]) -> int:
    """Load simple KEY=value entries without overwriting shell variables."""
    env_path = Path(path)
    if not env_path.exists():
        return 0

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def build_command(
    *,
    agent_model: str,
    vllm_api_base: str,
    vllm_api_key: str,
    user_model: str,
    user_api_base: str,
    user_api_key: str,
    output_path: Union[str, Path],
    num_trials: int,
    seed: int,
    max_concurrency: int,
) -> List[str]:
    """Build the official tau2 CLI command without starting a subprocess."""
    return [
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        "retail",
        "--task-split-name",
        "base",
        "--task-ids",
        *FINAL_TEST_TASK_IDS,
        "--agent",
        "llm_agent",
        "--agent-llm",
        agent_model,
        "--agent-llm-args",
        json.dumps(
            {
                "api_base": vllm_api_base,
                "api_key": vllm_api_key,
                "temperature": 0,
            },
            separators=(",", ":"),
        ),
        "--user",
        "user_simulator",
        "--user-llm",
        user_model,
        "--user-llm-args",
        json.dumps(
            {
                "api_base": user_api_base,
                "api_key": user_api_key,
                "temperature": 0,
            },
            separators=(",", ":"),
        ),
        "--num-trials",
        str(num_trials),
        "--seed",
        str(seed),
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        str(output_path),
        "--auto-resume",
    ]


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _default_summary_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".md")
    return output_path / "summary.md"


def _results_metadata_path(output_path: Path) -> Path:
    if output_path.is_dir():
        return output_path / "results.json"
    return output_path


def _tau2_checkpoint_dir(output_path: Path) -> Path:
    """Keep tau2's resumable checkpoint beside the requested output file."""
    output_path = output_path.resolve()
    if output_path.suffix:
        return output_path.parent / f".{output_path.stem}.tau2"
    return output_path / ".tau2"


def materialize_benchmark_output(
    *, checkpoint_results_path: Path, output_path: Path
) -> None:
    """Copy tau2's checkpoint JSON to the user-facing outputs file."""
    if not checkpoint_results_path.is_file():
        raise FileNotFoundError(checkpoint_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_results_path, output_path)


def _load_benchmark_payload(output_path: Path) -> dict[str, Any]:
    metadata_path = _results_metadata_path(output_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "simulations" not in payload:
        sims_dir = metadata_path.parent / "simulations"
        payload["simulations"] = []
        if sims_dir.exists():
            for sim_path in sorted(sims_dir.glob("*.json")):
                with sim_path.open("r", encoding="utf-8") as sim_file:
                    payload["simulations"].append(json.load(sim_file))
    return payload


def _reward(simulation: dict[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    return float(reward) if reward is not None else None


def build_comprehensive_benchmark(
    simulations: list[dict[str, Any]],
    *,
    expected_runs: int,
    deserializer: Callable[[dict[str, Any]], Any],
    adapter: Any,
    verifier: Any,
) -> dict[str, Any]:
    """Run the project Verifier over saved τ² simulations and aggregate them."""
    from agent_for_business.retail_runner import RetailTaskRunner
    from agent_for_business.validation_gate import BenchmarkRecord, BenchmarkSummary

    enriched_simulations: list[dict[str, Any]] = []
    records: list[BenchmarkRecord] = []

    for simulation_payload in simulations:
        payload = dict(simulation_payload)
        tau_reward = _reward(simulation_payload)
        task_id = str(simulation_payload.get("task_id", "unknown"))
        trial = simulation_payload.get("trial")
        termination_reason = simulation_payload.get("termination_reason")

        try:
            simulation = deserializer(simulation_payload)
            info = getattr(simulation, "info", {}) or {}
            evaluation = RetailTaskRunner.evaluation_from_simulation(simulation)
            evaluation.update(info.get("evaluation") or {})
            evaluation = RetailTaskRunner.normalise_evaluation(evaluation)
            trajectory = adapter.from_simulation(
                simulation,
                terminal_state=info.get("terminal_state", {}) or {},
                evaluation=evaluation,
            )
            verification = verifier.verify(trajectory)
            record = BenchmarkRecord.from_verification(
                task_id=task_id,
                trial=trial,
                tau_reward=tau_reward,
                termination_reason=termination_reason,
                result=verification,
            )
        except Exception as exc:  # noqa: BLE001 - preserve row-level failures.
            record = BenchmarkRecord(
                task_id=task_id,
                trial=trial,
                tau_reward=tau_reward,
                tau_reward_valid=tau_reward is not None,
                termination_reason=termination_reason,
                verifier_valid=False,
                verifier_error=f"{type(exc).__name__}: {exc}",
            )

        payload["verifier"] = record.to_dict()
        enriched_simulations.append(payload)
        records.append(record)

    summary = BenchmarkSummary.from_records(
        records,
        expected_runs=expected_runs,
    )
    return {
        "simulations": enriched_simulations,
        "summary": summary.to_dict(),
        "records": [record.to_dict() for record in records],
    }


def _deserialize_tau2_simulation(payload: dict[str, Any]) -> Any:
    """Lazily parse one saved τ² SimulationRun in the benchmark environment."""
    from tau2.data_model.simulation import SimulationRun

    return SimulationRun.model_validate(payload)


def enrich_benchmark_output(
    *,
    output_path: Path,
    expected_runs: int,
) -> dict[str, Any]:
    """Add project Verifier results to a materialized τ² results file."""
    from agent_for_business.policy_verifier import RetailPolicyVerifier
    from agent_for_business.tau_adapter import SimulationTrajectoryAdapter

    payload = _load_benchmark_payload(output_path)
    benchmark = build_comprehensive_benchmark(
        payload.get("simulations") or [],
        expected_runs=expected_runs,
        deserializer=_deserialize_tau2_simulation,
        adapter=SimulationTrajectoryAdapter(),
        verifier=RetailPolicyVerifier(),
    )
    payload["simulations"] = benchmark["simulations"]
    payload["benchmark"] = {
        "summary": benchmark["summary"],
        "records": benchmark["records"],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_rate(value: float | None) -> str:
    return "-" if value is None else _format_pct(value)


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_benchmark_summary(
    *,
    output_path: Path,
    summary_path: Path,
    agent_model: str,
    vllm_api_base: str,
    user_model: str,
    num_trials: int,
    seed: int,
    max_concurrency: int,
) -> None:
    payload = _load_benchmark_payload(output_path)
    simulations = payload.get("simulations") or []
    benchmark = payload.get("benchmark") or {}
    summary = benchmark.get("summary") or {}
    records = benchmark.get("records") or []
    expected_runs = int(summary.get("expected_runs", len(FINAL_TEST_TASK_IDS) * num_trials))
    completed_runs = int(summary.get("completed_runs", len(simulations)))

    def value(name: str, default: Any = None) -> Any:
        return summary.get(name, default)

    termination_counts = Counter(value("termination_counts", {}))
    task_rows: list[list[Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record.get("task_id", "unknown"))].append(record)

    for task_id in FINAL_TEST_TASK_IDS:
        task_records = by_task.get(task_id, [])
        task_tau_rewards = [
            record["tau_reward"]
            for record in task_records
            if record.get("tau_reward_valid") and record.get("tau_reward") is not None
        ]
        task_verifier_rewards = [
            record["verifier_reward"]
            for record in task_records
            if record.get("verifier_valid")
            and record.get("verifier_reward_valid")
            and record.get("verifier_reward") is not None
        ]
        task_policy_violations = sum(
            bool(record.get("policy_violation"))
            for record in task_records
            if record.get("verifier_valid")
        )
        task_tool_errors = sum(
            int(record.get("tool_error_count") or 0) > 0
            for record in task_records
            if record.get("verifier_valid")
        )
        task_db_rows = [
            record["db_match"]
            for record in task_records
            if record.get("db_match") is not None
        ]
        task_communication_rows = [
            record["communication_ok"]
            for record in task_records
            if record.get("communication_ok") is not None
        ]
        task_rows.append(
            [
                task_id,
                f"{len(task_records)}/{num_trials}",
                _format_float(
                    sum(task_tau_rewards) / len(task_tau_rewards)
                    if task_tau_rewards else None
                ),
                (
                    f"{sum(reward >= 1.0 for reward in task_tau_rewards)}"
                    f"/{len(task_tau_rewards)}"
                    if task_tau_rewards else "0/0"
                ),
                _format_float(
                    sum(task_verifier_rewards) / len(task_verifier_rewards)
                    if task_verifier_rewards else None
                ),
                task_policy_violations,
                task_tool_errors,
                _format_rate(
                    sum(bool(row) for row in task_db_rows) / len(task_db_rows)
                    if task_db_rows else None
                ),
                _format_rate(
                    sum(bool(row) for row in task_communication_rows)
                    / len(task_communication_rows)
                    if task_communication_rows else None
                ),
            ]
        )

    failed_rows: list[list[Any]] = []
    for record in records:
        tau_reward = record.get("tau_reward")
        clean = (
            record.get("tau_reward_valid")
            and tau_reward is not None
            and tau_reward >= 1.0
            and record.get("verifier_valid")
            and not record.get("policy_violation")
            and not record.get("first_error")
        )
        if clean:
            continue
        failed_rows.append(
            [
                record.get("task_id", "unknown"),
                record.get("trial", "-"),
                _format_float(tau_reward),
                _format_float(record.get("verifier_reward")),
                record.get("first_error") or record.get("verifier_error") or "-",
            ]
        )

    if not failed_rows:
        failed_section = "No failed, invalid, or policy-violating runs."
    else:
        failed_section = _markdown_table(
            ["Task", "Trial", "τ² reward", "Verifier reward", "Reason"],
            failed_rows,
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_path = _results_metadata_path(output_path)
    lines = [
        "# Final Benchmark Summary",
        "",
        f"Generated: {generated_at}",
        f"Source: `{source_path}`",
        "",
        "## Configuration",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ["Domain", "retail"],
                ["Task split", "base"],
                ["Final-test tasks", len(FINAL_TEST_TASK_IDS)],
                ["Trials per task", num_trials],
                ["Seed", seed],
                ["Max concurrency", max_concurrency],
                ["Agent model", agent_model],
                ["Agent endpoint", vllm_api_base],
                ["User model", user_model],
            ],
        ),
        "",
        "## Overall Results",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Expected runs", expected_runs],
                ["Completed runs", completed_runs],
                ["Incomplete runs", value("incomplete_runs", 0)],
                ["τ² valid reward runs", value("tau_reward_valid_count", 0)],
                ["τ² reward valid rate", _format_rate(value("tau_reward_valid_rate"))],
                ["τ² successes", value("tau_success_count", 0)],
                ["τ² success rate", _format_rate(value("tau_success_rate"))],
                ["τ² reward mean", _format_float(value("tau_reward_mean"))],
            ],
        ),
        "",
        "## τ² Metrics",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["db_match_true_count", value("db_match_true_count", 0)],
                ["db_match_present_count", value("db_match_present_count", 0)],
                ["db_match_missing_count", value("db_match_missing_count", 0)],
                ["db_match_rate", _format_rate(value("db_match_rate"))],
                ["communication_true_count", value("communication_true_count", 0)],
                ["communication_present_count", value("communication_present_count", 0)],
                ["communication_missing_count", value("communication_missing_count", 0)],
                ["communication_rate", _format_rate(value("communication_rate"))],
            ],
        ),
        "",
        "## Verifier / GRPO Metrics",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["verifier_evaluated_count", value("verifier_evaluated_count", 0)],
                ["verifier_invalid_count", value("verifier_invalid_count", 0)],
                ["verifier_reward_valid_count", value("verifier_reward_valid_count", 0)],
                ["verifier_reward_mean", _format_float(value("verifier_reward_mean"))],
                ["policy_violation_count", value("policy_violation_count", 0)],
                ["policy_violation_rate", _format_rate(value("policy_violation_rate"))],
                ["tool_error_run_count", value("tool_error_run_count", 0)],
                ["tool_error_rate", _format_rate(value("tool_error_rate"))],
                ["tool_error_total", value("tool_error_total", 0)],
            ],
        ),
        "",
        "## Verifier First Errors",
        "",
        _markdown_table(
            ["First error", "Count"],
            [
                [error, count]
                for error, count in sorted(
                    (value("first_error_counts", {}) or {}).items()
                )
            ]
            or [["-", 0]],
        ),
        "",
        "## Metric Definitions",
        "",
        "- `tau_reward`: official τ² `reward_info.reward`; it is the product of the enabled task reward components. For the Retail default basis, `tau_reward = db_reward * communicate_reward`.",
        "- The τ² DB reward (`db_reward`) is `1.0` when the predicted final agent and user database states match the gold end state, otherwise `0.0`; `db_match_rate` is true checks divided by present DB checks.",
        "- The τ² COMMUNICATE reward (`communicate_reward`) is `1.0` when every required communication item appears in assistant messages, otherwise `0.0`; `communication_rate` is true checks divided by present communication checks.",
        "- `tau_reward_valid_rate` is valid τ² reward runs divided by completed runs. Missing rewards are excluded from the τ² mean and success rate.",
        "- `verifier_reward`: project `VerificationResult.reward`, the GRPO scalar after replaying the trajectory. A policy violation gives `-1.0`; otherwise the evaluation reward is reduced by `min(0.2, 0.1 * tool_error_count)`.",
        "- `verifier_reward_mean` uses only Verifier-evaluated rows with `reward_valid=true`; `verifier_invalid_count` counts conversion or verification rows that could not be evaluated.",
        "- `policy_violation_rate` is policy-violating evaluated rows divided by Verifier-evaluated rows. `tool_error_rate` is rows with at least one tool error divided by Verifier-evaluated rows; `tool_error_total` counts all tool errors.",
        "- `termination_counts` counts τ² termination reasons across completed runs. `first_error_counts` groups the first Verifier error by its error name.",
        "- Per-task rates use the same denominators as the overall metrics; `-` means the corresponding check or reward was not present.",
        "",
        "## Termination Reasons",
        "",
        _markdown_table(
            ["Reason", "Count"],
            [[reason, count] for reason, count in termination_counts.most_common()]
            or [["-", 0]],
        ),
        "",
        "## Per-Task Results",
        "",
        _markdown_table(
            [
                "Task", "Runs", "τ² reward mean", "τ² successes",
                "Verifier reward mean", "Policy violations", "Tool-error runs",
                "DB match rate", "Communication rate",
            ],
            task_rows,
        ),
        "",
        "## Failed, Invalid, or Policy-Violating Runs",
        "",
        failed_section,
        "",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--agent-llm", default=None)
    parser.add_argument("--vllm-api-base", default=None)
    parser.add_argument("--vllm-api-key", default=None)
    parser.add_argument("--user-llm", default=None)
    parser.add_argument("--user-api-base", default=None)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/benchmarks/qwen-sft-final.json"),
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Markdown summary path. Defaults to the output path with a .md suffix.",
    )
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    agent_model = args.agent_llm or _first_env("AGENT_BENCHMARK_LLM") or "openai/qwen-sft"
    vllm_api_base = args.vllm_api_base or _first_env("VLLM_API_BASE") or "http://127.0.0.1:8000/v1"
    vllm_api_key = args.vllm_api_key or _first_env("VLLM_API_KEY") or "EMPTY"
    user_model = args.user_llm or _first_env("USER_LLM") or "gpt-5.6-luna"
    user_api_base = args.user_api_base or _first_env("USER_API_BASE", "ANTHROPIC_BASE_URL")
    user_api_key = args.user_api_key or _first_env("USER_API_KEY", "ANTHROPIC_API_KEY")
    if not user_api_base or not user_api_key:
        raise SystemExit(
            "Missing User Simulator credentials. Set USER_API_BASE and USER_API_KEY in .env."
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _tau2_checkpoint_dir(output_path)
    checkpoint_results_path = checkpoint_dir / "results.json"
    summary_path = Path(args.summary_output) if args.summary_output else _default_summary_path(output_path)
    command = build_command(
        agent_model=agent_model,
        vllm_api_base=vllm_api_base,
        vllm_api_key=vllm_api_key,
        user_model=user_model,
        user_api_base=user_api_base,
        user_api_key=user_api_key,
        output_path=checkpoint_dir,
        num_trials=args.num_trials,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
    )

    print(f"Running {len(FINAL_TEST_TASK_IDS)} final-test tasks × {args.num_trials} trials")
    print(f"Agent endpoint: {vllm_api_base}")
    print(f"User model: {user_model}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    return_code = subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    ).returncode

    if checkpoint_results_path.exists():
        try:
            materialize_benchmark_output(
                checkpoint_results_path=checkpoint_results_path,
                output_path=output_path,
            )
            enrich_benchmark_output(
                output_path=output_path,
                expected_runs=len(FINAL_TEST_TASK_IDS) * args.num_trials,
            )
            write_benchmark_summary(
                output_path=output_path,
                summary_path=summary_path,
                agent_model=agent_model,
                vllm_api_base=vllm_api_base,
                user_model=user_model,
                num_trials=args.num_trials,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
            )
            print(f"Wrote benchmark summary: {summary_path}")
        except Exception as exc:  # noqa: BLE001 - keep the benchmark exit code visible.
            print(f"Warning: failed to write benchmark summary: {exc}", file=sys.stderr)
    else:
        print(
            f"Warning: benchmark output not found, skipped summary: {output_path}",
            file=sys.stderr,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
