"""并行采集教师轨迹，并按验证结果分流保存。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, List, Tuple

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
    ) -> CollectionSummary:
        """按固定 seed 生成任务作业，串行或并行执行后分流统计。"""
        if attempts_per_task < 1:
            raise ValueError("attempts_per_task must be at least 1")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        summary = CollectionSummary()
        jobs: List[Tuple[str, int]] = []
        for task_index, task_id in enumerate(task_ids):
            for attempt in range(attempts_per_task):
                # seed 与 task/attempt 一一对应，使重跑时仍能定位同一采样配置。
                seed = base_seed + task_index * attempts_per_task + attempt
                jobs.append((task_id, seed))

        if max_workers == 1:
            trajectories = (
                self._runner.run(task_id=task_id, seed=seed)
                for task_id, seed in jobs
            )
            for trajectory in trajectories:
                self._route_trajectory(trajectory, summary)
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
                    self._route_trajectory(future.result(), summary)

        return summary

    def _route_trajectory(self, trajectory, summary: CollectionSummary) -> None:
        """先保留所有 raw，再将有效且成功合规的轨迹送入 accepted。"""
        self._raw_store.append(trajectory)
        summary.raw_count += 1

        result = self._verifier.verify(trajectory)
        accepted = (
            # reward 无效、任务失败或策略违规都不能进入 SFT 正例。
            result.reward_valid
            and result.task_success
            and not result.policy_violation
        )
        if accepted:
            self._accepted_store.append(trajectory)
            summary.accepted_count += 1
        else:
            self._failed_store.append(trajectory)
            summary.failed_count += 1
