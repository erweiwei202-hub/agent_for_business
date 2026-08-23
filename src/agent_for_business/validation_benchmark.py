"""运行 Raw/SFT validation benchmark，并生成可序列化的指标报告。"""

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from .validation_gate import BenchmarkRecord, BenchmarkSummary


def run_validation_benchmark(
    *,
    task_ids: Iterable[str],
    runner: Any,
    verifier: Any,
    model_label: str,
    seed: int,
) -> dict[str, Any]:
    """用固定 task 集和 seed 运行一个模型，并构建 JSON-safe 报告。"""
    validation_task_ids = list(task_ids)
    if not validation_task_ids:
        raise ValueError("validation task_ids must not be empty")

    resolved_runner = _resolve_runner(runner)
    results = []
    records = []
    for task_id in validation_task_ids:
        # 同一个 seed/task 协议用于 Raw 和 SFT，差异才可归因于模型而非采样。
        trajectory = resolved_runner.run(task_id=task_id, seed=seed)
        result = verifier.verify(trajectory)
        results.append(result)
        records.append(
            BenchmarkRecord.from_verification(
                task_id=task_id,
                result=result,
            )
        )

    summary = BenchmarkSummary.from_records(
        records,
        expected_runs=len(validation_task_ids),
    )
    return {
        "model": model_label,
        "task_ids": validation_task_ids,
        "seed": seed,
        "results": [_as_json_dict(result) for result in results],
        "summary": summary.to_dict(),
    }


def compare_raw_sft_validation(
    *,
    task_ids: Iterable[str],
    raw_runner: Any,
    sft_runner: Any,
    verifier: Any,
    seed: int,
    raw_model_label: str = "raw",
    sft_model_label: str = "sft",
) -> dict[str, Any]:
    """用同一 validation 协议比较 Raw/SFT，并返回独立指标。"""
    validation_task_ids = list(task_ids)
    raw_report = run_validation_benchmark(
        task_ids=validation_task_ids,
        runner=raw_runner,
        verifier=verifier,
        model_label=raw_model_label,
        seed=seed,
    )
    sft_report = run_validation_benchmark(
        task_ids=validation_task_ids,
        runner=sft_runner,
        verifier=verifier,
        model_label=sft_model_label,
        seed=seed,
    )

    return {
        "model": {"raw": raw_model_label, "sft": sft_model_label},
        "task_ids": validation_task_ids,
        "seed": seed,
        "raw": raw_report,
        "sft": sft_report,
        "summary": {
            "raw": raw_report["summary"],
            "sft": sft_report["summary"],
        },
    }


def _resolve_runner(runner_or_factory: Any) -> Any:
    """接受现成 Runner 或返回 Runner 的工厂，统一为 ``run`` 对象。"""
    if callable(getattr(runner_or_factory, "run", None)):
        return runner_or_factory

    if callable(runner_or_factory):
        resolved_runner = runner_or_factory()
        if callable(getattr(resolved_runner, "run", None)):
            return resolved_runner

    raise TypeError("runner must provide run() or be a factory returning one")


def _as_json_dict(value: Any) -> dict[str, Any]:
    """将 dataclass 或普通结果对象转成报告可写入的字典。"""
    if is_dataclass(value):
        return asdict(value)
    return dict(vars(value))
