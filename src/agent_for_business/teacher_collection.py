"""并行采集教师轨迹，并按验证结果分流保存。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .policy_verifier import RetailPolicyVerifier
from .retail_runner import RetailTaskRunner
from .trajectory_store import TrajectoryStore


@dataclass
class CollectionSummary:
    """记录 raw、accepted、failed 三个输出流的数量。"""

    raw_count: int = 0
    accepted_count: int = 0
    failed_count: int = 0


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

    def collect(
        self,
        *,
        task_ids: Iterable[str],
        attempts_per_task: int,
        base_seed: int = 0,
        max_workers: int = 1,
        runtime_store: Optional[TrajectoryStore] = None,
    ) -> CollectionSummary:
        """按固定 seed 生成任务作业，并从 runtime.jsonl 断点续跑。"""
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

        job_keys = {self._job_key(task_id, seed) for task_id, seed in jobs}
        raw_keys = self._load_keys(self._raw_store, job_keys)
        accepted_keys = self._load_keys(self._accepted_store, job_keys)
        failed_keys = self._load_keys(self._failed_store, job_keys)
        routed_keys = raw_keys | accepted_keys | failed_keys
        summary = CollectionSummary(
            raw_count=len(raw_keys),
            accepted_count=len(accepted_keys),
            failed_count=len(failed_keys),
        )

        completed_runtime: Dict[Tuple[str, int], object] = {}
        if runtime_store is not None:
            for trajectory in runtime_store.iter_trajectories():
                key = self._trajectory_key(trajectory)
                if key in job_keys and self._runtime_status(trajectory) == "completed":
                    completed_runtime[key] = trajectory

        # 先把 runtime 中已经完成但尚未分流的轨迹补进 raw/accepted/failed。
        for key, trajectory in completed_runtime.items():
            self._route_trajectory(
                trajectory,
                summary,
                raw_keys=raw_keys,
                accepted_keys=accepted_keys,
                failed_keys=failed_keys,
            )
            routed_keys.update(accepted_keys | failed_keys)

        jobs = [
            (task_id, seed)
            for task_id, seed in jobs
            if self._job_key(task_id, seed) not in completed_runtime
            and self._job_key(task_id, seed) not in routed_keys
        ]

        if max_workers == 1:
            trajectories = (
                self._runner.run(task_id=task_id, seed=seed)
                for task_id, seed in jobs
            )
            for trajectory in trajectories:
                self._route_trajectory(
                    trajectory,
                    summary,
                    raw_keys=raw_keys,
                    accepted_keys=accepted_keys,
                    failed_keys=failed_keys,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._runner.run,
                        task_id=task_id,
                        seed=seed,
                    )
                    for task_id, seed in jobs
                ]
                for future in futures:
                    # 按提交顺序读取 future，保证统计/输出顺序稳定；执行本身仍并行。
                    self._route_trajectory(
                        future.result(),
                        summary,
                        raw_keys=raw_keys,
                        accepted_keys=accepted_keys,
                        failed_keys=failed_keys,
                    )

        return summary

    @staticmethod
    def _job_key(task_id: str, seed: int) -> Tuple[str, int]:
        return str(task_id), seed

    @classmethod
    def _trajectory_key(cls, trajectory) -> Tuple[str, int]:
        return cls._job_key(trajectory.task_id, trajectory.seed)

    @staticmethod
    def _runtime_status(trajectory) -> str:
        return str(trajectory.evaluation.get("runtime_status", "completed"))

    @classmethod
    def _load_keys(
        cls,
        store: TrajectoryStore,
        job_keys: Set[Tuple[str, int]],
    ) -> Set[Tuple[str, int]]:
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
        raw_keys: Set[Tuple[str, int]],
        accepted_keys: Set[Tuple[str, int]],
        failed_keys: Set[Tuple[str, int]],
    ) -> None:
        """先保留所有 raw，再将有效且成功合规的轨迹送入 accepted。"""
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
