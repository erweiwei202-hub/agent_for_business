"""把单次 Retail simulation 统一成可校验、可持久化的轨迹。"""

from typing import Any, Callable, Dict, Optional

from .tau_adapter import SimulationTrajectoryAdapter
from .trajectory import Trajectory
from .trajectory_store import TrajectoryStore


class RetailTaskRunner:
    """组合 simulation provider、消息适配器和 trajectory store。"""

    def __init__(
        self,
        *,
        simulation_runner: Callable[[str, int], Any],
        trajectory_store: TrajectoryStore,
        adapter: Optional[SimulationTrajectoryAdapter] = None,
    ) -> None:
        self._simulation_runner = simulation_runner
        self._trajectory_store = trajectory_store
        self._adapter = adapter or SimulationTrajectoryAdapter()

    def run(self, *, task_id: str, seed: int) -> Trajectory:
        """执行一个 task，归一化评估信息，落盘并返回完整轨迹。"""
        simulation = self._simulation_runner(task_id, seed)
        info: Dict[str, Any] = getattr(simulation, "info", {}) or {}
        terminal_state = info.get("terminal_state", {})
        # 新版 simulation 把标准化 evaluation 放在 info 中；旧版则从 reward_info
        # 提取，兼容两种来源而不让后续 Verifier 感知 τ³ 版本差异。
        evaluation = info.get("evaluation") or self._evaluation_from_simulation(
            simulation
        )

        trajectory = self._adapter.from_simulation(
            simulation,
            terminal_state=terminal_state,
            evaluation=evaluation,
        )
        self._trajectory_store.append(trajectory)
        return trajectory

    @staticmethod
    def _evaluation_from_simulation(simulation: Any) -> Dict[str, Any]:
        """从旧式 reward_info 提取项目统一的 evaluation 字段。"""
        reward_info = getattr(simulation, "reward_info", None)
        reward = getattr(reward_info, "reward", None)
        evaluation: Dict[str, Any] = {"reward": reward}
        evaluation["task_success"] = RetailTaskRunner._is_successful_reward(reward)
        if reward is None:
            evaluation["reward_valid"] = False

        db_check = getattr(reward_info, "db_check", None)
        if db_check is not None:
            evaluation["db_match"] = getattr(db_check, "db_match", None)

        communicate_checks = getattr(reward_info, "communicate_checks", None)
        if communicate_checks is not None:
            evaluation["communication_ok"] = all(
                getattr(check, "met", False) for check in communicate_checks
            )

        return evaluation

    @staticmethod
    def _is_successful_reward(reward: Any) -> bool:
        """将 τ³ 的满分 reward 归一化为 task_success，容忍浮点误差。"""
        try:
            return abs(float(reward) - 1.0) <= 1e-6
        except (TypeError, ValueError):
            return False
