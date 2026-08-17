"""对外导出 Retail 后训练流水线的稳定核心接口。"""

# 按流水线层次 re-export，方便 notebook/测试从包入口访问核心对象；
# 具体实现仍保留在各自模块中，避免这里承担业务逻辑。
from .trajectory import Trajectory, TrajectoryEvent, TrajectoryRecorder
from .trajectory_store import TrajectoryStore
from .tau_adapter import SimulationTrajectoryAdapter
from .retail_runner import RetailTaskRunner
from .policy_verifier import RetailPolicyVerifier, VerificationResult
from .tau_provider import Tau2RetailProvider
from .teacher_collection import CollectionSummary, TeacherTrajectoryCollector
from .sft_dataset import (
    ActionOnlySFTDatasetBuilder,
    ActionOnlySFTRenderer,
    QwenActionOnlyTokenFormatter,
    SFTDatasetStore,
    SFTBuildResult,
    SFTExample,
)
from .validation_gate import BenchmarkSummary, GateDecision, SFTValidationGate
from .badcase import BadcaseAnalyzer, BadcaseRecord
from .sft_training import SFTTrainingConfig, load_sft_dataset, train_sft
from .validation_benchmark import (
    compare_raw_sft_validation,
    run_validation_benchmark,
)
from .grpo_core import (
    clipped_objective,
    compute_group_advantages,
    masked_mean,
    reference_kl,
)

__all__ = [
    "Trajectory",
    "TrajectoryEvent",
    "TrajectoryRecorder",
    "TrajectoryStore",
    "SimulationTrajectoryAdapter",
    "RetailTaskRunner",
    "RetailPolicyVerifier",
    "VerificationResult",
    "Tau2RetailProvider",
    "CollectionSummary",
    "TeacherTrajectoryCollector",
    "ActionOnlySFTRenderer",
    "ActionOnlySFTDatasetBuilder",
    "QwenActionOnlyTokenFormatter",
    "SFTDatasetStore",
    "SFTBuildResult",
    "SFTExample",
    "BenchmarkSummary",
    "GateDecision",
    "SFTValidationGate",
    "BadcaseAnalyzer",
    "BadcaseRecord",
    "SFTTrainingConfig",
    "load_sft_dataset",
    "train_sft",
    "run_validation_benchmark",
    "compare_raw_sft_validation",
    "compute_group_advantages",
    "masked_mean",
    "clipped_objective",
    "reference_kl",
]
