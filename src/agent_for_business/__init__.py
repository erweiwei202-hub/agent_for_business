"""对外导出 Retail 后训练流水线的稳定核心接口。"""

# 按流水线层次 re-export，方便 notebook/测试从包入口访问核心对象；
# 具体实现仍保留在各自模块中，避免这里承担业务逻辑。
from .badcase import BadcaseAnalyzer, BadcaseRecord
from .grpo_agent import (
    GenerationTrace,
    LocalQwenAgent,
    ParsedAssistant,
    ParsedToolCall,
    parse_qwen_response,
)
from .grpo_core import (
    clipped_objective,
    compute_group_advantages,
    masked_mean,
    reference_kl,
)
from .grpo_objective import ObjectiveResult, grpo_loss, sequence_logprobs
from .grpo_online import OnlineGRPOTrainer
from .grpo_rollout import RolloutResult, Tau2LocalRolloutRunner
from .grpo_training import (
    GRPOModelSource,
    GRPOTrainingConfig,
    load_grpo_model,
    resolve_grpo_model_source,
    train_grpo,
)
from .policy_verifier import RetailPolicyVerifier, VerificationResult
from .retail_runner import RetailTaskRunner
from .sft_dataset import (
    ActionOnlySFTDatasetBuilder,
    ActionOnlySFTRenderer,
    QwenActionOnlyTokenFormatter,
    SFTBuildResult,
    SFTDatasetStore,
    SFTExample,
)
from .sft_training import SFTTrainingConfig, load_sft_dataset, train_sft
from .tau_adapter import SimulationTrajectoryAdapter
from .tau_provider import Tau2RetailProvider
from .teacher_collection import CollectionSummary, TeacherTrajectoryCollector
from .trajectory import Trajectory, TrajectoryEvent, TrajectoryRecorder
from .trajectory_store import TrajectoryStore
from .validation_benchmark import (
    compare_raw_sft_validation,
    run_validation_benchmark,
)
from .validation_gate import BenchmarkRecord, BenchmarkSummary

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
    "BenchmarkRecord",
    "BenchmarkSummary",
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
    "GRPOTrainingConfig",
    "GRPOModelSource",
    "resolve_grpo_model_source",
    "load_grpo_model",
    "train_grpo",
    "ParsedToolCall",
    "ParsedAssistant",
    "GenerationTrace",
    "parse_qwen_response",
    "LocalQwenAgent",
    "ObjectiveResult",
    "grpo_loss",
    "sequence_logprobs",
    "RolloutResult",
    "Tau2LocalRolloutRunner",
    "OnlineGRPOTrainer",
]
