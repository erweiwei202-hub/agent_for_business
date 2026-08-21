"""并行采集教师轨迹，并按验证结果分流保存。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .policy_verifier import RetailPolicyVerifier
from .retail_runner import RetailTaskRunner
from .trajectory import TrajectoryRecorder
from .trajectory_store import TrajectoryStore


@dataclass
class CollectionSummary:
    """记录 raw、accepted、failed 三个输出流的数量。"""

    raw_count: int = 0
    accepted_count: int = 0
    failed_count: int = 0
    runtime_error_count: int = 0


class TeacherTrajectoryCollector:
    """为训练任务生成多次 rollout，并保留失败轨迹供分析。"""

    def __init__(
        self,
        *,
        runner: RetailTaskRunner,
        verifier: RetailPolicyVerifier,
        raw_store: TrajectoryStore,
        accepted_store: TrajectoryStore,
        failed_store: TrajectoryStore,
    ) -> None:
        self._runner = runner
        self._verifier = verifier
        self._raw_store = raw_store
        self._accepted_store = accepted_store
        self._failed_store = failed_store
        self._raw_store.ensure_file()
        self._accepted_store.ensure_file()
        self._failed_store.ensure_file()

    def collect(
        self,
        *,
        task_ids: Iterable[str],
        attempts_per_task: int,
        base_seed: int = 0,
        max_workers: int = 1,
        runtime_store: Optional[TrajectoryStore] = None,
        on_progress: Optional[Callable[[CollectionSummary], None]] = None,
    ) -> CollectionSummary:
        """按固定 seed 生成任务作业，并从已有轨迹断点续跑。"""
        if attempts_per_task < 1:
            raise ValueError("attempts_per_task must be at least 1")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        jobs: List[Tuple[str, int]] = []
        for task_index, task_id in enumerate(task_ids):
            for attempt in range(attempts_per_task):
                # seed 与 task/attempt 一一对应，使重跑时仍能定位同一采样配置。
                seed = base_seed + task_index * attempts_per_task + attempt
                jobs.append((task_id, seed))

        if runtime_store is not None:
            runtime_store.ensure_file()

        job_keys = {self._job_key(task_id, seed) for task_id, seed in jobs}
        raw_keys = self._load_keys(self._raw_store, job_keys)
        accepted_keys = self._load_keys(self._accepted_store, job_keys)
        failed_keys = self._load_keys(self._failed_store, job_keys)
        routed_keys = accepted_keys | failed_keys
        summary = CollectionSummary(
            raw_count=len(raw_keys),
            accepted_count=len(accepted_keys),
            failed_count=len(failed_keys),
        )

        runtime_trajectories: Dict[Tuple[str, Optional[int]], object] = {}
        if runtime_store is not None:
            for trajectory in runtime_store.iter_trajectories():
                key = self._trajectory_key(trajectory)
                if key in job_keys:
                    runtime_trajectories[key] = trajectory

        completed_runtime_trajectories = {
            key: trajectory
            for key, trajectory in runtime_trajectories.items()
            if self._runtime_status(trajectory) == "completed"
        }
        runtime_error_count = sum(
            self._runtime_status(trajectory) == "error"
            for trajectory in runtime_trajectories.values()
        )
        summary.runtime_error_count = runtime_error_count

        if on_progress is not None:
            on_progress(summary)

        for key, trajectory in completed_runtime_trajectories.items():
            if key in routed_keys:
                continue
            self._route_trajectory(
                trajectory,
                summary,
                raw_keys=raw_keys,
                accepted_keys=accepted_keys,
                failed_keys=failed_keys,
            )
            routed_keys.update(accepted_keys | failed_keys)
            if on_progress is not None:
                on_progress(summary)

        jobs = [
            (task_id, seed)
            for task_id, seed in jobs
            if self._job_key(task_id, seed) not in completed_runtime_trajectories
            and self._job_key(task_id, seed) not in routed_keys
        ]

        if jobs:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            future_jobs = {
                executor.submit(
                    self._runner.run,
                    task_id=task_id,
                    seed=seed,
                ): (task_id, seed)
                for task_id, seed in jobs
            }
            try:
                # 按完成顺序消费，先完成的任务立即落盘和分流，不被慢任务阻塞。
                for future in as_completed(future_jobs):
                    task_id, seed = future_jobs[future]
                    try:
                        trajectory = future.result()
                    except Exception as exc:
                        trajectory = self._runtime_error_trajectory(
                            task_id=task_id,
                            seed=seed,
                            error=exc,
                        )
                        if runtime_store is not None:
                            runtime_store.append(trajectory)

                    if self._runtime_status(trajectory) == "error":
                        summary.runtime_error_count += 1
                    else:
                        self._route_trajectory(
                            trajectory,
                            summary,
                            raw_keys=raw_keys,
                            accepted_keys=accepted_keys,
                            failed_keys=failed_keys,
                        )
                    if on_progress is not None:
                        on_progress(summary)
            except KeyboardInterrupt:
                for future in future_jobs:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

        return summary

    @staticmethod
    def _job_key(task_id: str, seed: Optional[int]) -> Tuple[str, Optional[int]]:
        return str(task_id), seed

    @classmethod
    def _trajectory_key(cls, trajectory) -> Tuple[str, Optional[int]]:
        return cls._job_key(trajectory.task_id, trajectory.seed)

    @staticmethod
    def _runtime_status(trajectory) -> str:
        return str(trajectory.evaluation.get("runtime_status", "completed"))

    @staticmethod
    def _runtime_error_trajectory(*, task_id: str, seed: int, error: Exception):
        message = " ".join(str(error).split())[:500]
        recorder = TrajectoryRecorder(task_id=task_id, seed=seed)
        return recorder.finish(
            terminal_state={},
            evaluation={
                "runtime_status": "error",
                "task_success": False,
                "reward": 0.0,
                "reward_valid": False,
                "communication_ok": None,
                "runtime_error": {
                    "type": type(error).__name__,
                    "message": message,
                },
            },
        )

    @classmethod
    def _load_keys(
        cls,
        store: TrajectoryStore,
        job_keys: Set[Tuple[str, Optional[int]]],
    ) -> Set[Tuple[str, Optional[int]]]:
        return {
            cls._trajectory_key(trajectory)
            for trajectory in store.iter_trajectories()
            if cls._trajectory_key(trajectory) in job_keys
        }

    def _route_trajectory(
        self,
        trajectory,
        summary: CollectionSummary,
        *,
        raw_keys: Set[Tuple[str, Optional[int]]],
        accepted_keys: Set[Tuple[str, Optional[int]]],
        failed_keys: Set[Tuple[str, Optional[int]]],
    ) -> None:
        """先保留所有 raw，再将有效且成功合规的轨迹送入 accepted。"""
        trajectory.evaluation = RetailTaskRunner.normalise_evaluation(
            trajectory.evaluation
        )
        key = self._trajectory_key(trajectory)
        if key not in raw_keys:
            self._raw_store.append(trajectory)
            raw_keys.add(key)
            summary.raw_count += 1

        result = self._verifier.verify(trajectory)
        accepted = (
            # reward 无效、任务失败或策略违规都不能进入 SFT 正例。
            result.reward_valid
            and result.task_success
            and not result.policy_violation
        )
        if accepted:
            if key not in accepted_keys:
                self._accepted_store.append(trajectory)
                accepted_keys.add(key)
                summary.accepted_count += 1
        elif key not in failed_keys:
            self._failed_store.append(trajectory)
            failed_keys.add(key)
            summary.failed_count += 1
