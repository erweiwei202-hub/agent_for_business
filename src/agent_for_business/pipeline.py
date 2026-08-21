"""面向 CLI 的高层流水线组装函数。"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union
from .policy_verifier import RetailPolicyVerifier
from .sft_dataset import ActionOnlySFTDatasetBuilder, SFTDatasetStore
from .retail_runner import RetailTaskRunner
from .tau_provider import Tau2RetailProvider
from .trajectory_store import TrajectoryStore


def build_sft_dataset(
    *,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    verifier: RetailPolicyVerifier,
) -> Dict[str, int]:
    """验证轨迹并把可训练样本写成 SFT JSONL，返回处理统计。"""
    trajectories = list(TrajectoryStore(input_path).iter_trajectories())
    result = ActionOnlySFTDatasetBuilder(verifier=verifier).build(trajectories)
    output_store = SFTDatasetStore(output_path)
    for example in result.examples:
        output_store.append(example)

    return {
        "input_count": len(trajectories),
        "written_count": len(result.examples),
        "skipped_count": len(result.skipped_task_ids),
    }


def run_smoke(
    *,
    runner: object,
    verifier: RetailPolicyVerifier,
    task_id: str,
    seed: int,
) -> Dict[str, object]:
    """运行一个 smoke task，返回可直接落盘的轨迹和 Verifier 报告。"""
    trajectory = runner.run(task_id=task_id, seed=seed)
    verification = verifier.verify(trajectory)
    return {
        "task_id": trajectory.task_id,
        "seed": trajectory.seed,
        "event_count": len(trajectory.events),
        "evaluation": trajectory.evaluation,
        "verification": asdict(verification),
    }


def create_tau2_retail_runner(
    *,
    agent_llm: str,
    user_llm: str,
    trajectory_path: Union[str, Path],
    agent_llm_args: Optional[Dict[str, object]] = None,
    user_llm_args: Optional[Dict[str, object]] = None,
    task_split_name: str = "base",
    max_steps: int = 100,
    max_errors: int = 5,
) -> RetailTaskRunner:
    """把真实 τ³ Provider 接到项目的单任务 Runner 和轨迹存储。"""
    provider = Tau2RetailProvider(
        agent_llm=agent_llm,
        user_llm=user_llm,
        task_split_name=task_split_name,
        agent_llm_args=agent_llm_args or {},
        user_llm_args=user_llm_args or {},
        max_steps=max_steps,
        max_errors=max_errors,
    )
    trajectory_store = TrajectoryStore(trajectory_path)
    return RetailTaskRunner(
        simulation_runner=provider.run,
        trajectory_store=trajectory_store,
        checkpoint_store=trajectory_store,
    )
