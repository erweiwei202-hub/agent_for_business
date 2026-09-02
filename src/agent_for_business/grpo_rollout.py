"""tau2-backed local-policy rollout boundary for online GRPO."""

from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

from .grpo_agent import GenerationTrace, LocalQwenAgent
from .policy_verifier import RetailPolicyVerifier, VerificationResult
from .retail_runner import RetailTaskRunner
from .tau_adapter import SimulationTrajectoryAdapter


@dataclass(frozen=True)
class RolloutResult:
    """One completed tau2 simulation plus the data needed for a GRPO update."""

    task_id: str
    seed: int
    simulation: Any
    trajectory: Any
    verification: VerificationResult
    traces: Tuple[GenerationTrace, ...]
    error: Optional[str] = None

    @property
    def valid_for_update(self) -> bool:
        """Whether this result supplies a valid reward and action-token trace."""

        return bool(
            self.verification.reward_valid
            and any(any(trace.action_mask) for trace in self.traces)
        )


class Tau2LocalRolloutRunner:
    """Run local Qwen policy rollouts through tau2's Retail orchestrator."""

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        user_llm: Optional[str] = None,
        user_llm_args: Optional[Dict[str, Any]] = None,
        task_split_name: str = "base",
        max_steps: int = 100,
        max_errors: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generation_lock: Optional[Any] = None,
        serialize_generation: bool = True,
        simulation_factory: Optional[Callable[..., Any]] = None,
        adapter: Optional[SimulationTrajectoryAdapter] = None,
        verifier: Optional[RetailPolicyVerifier] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {}
        self.task_split_name = task_split_name
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.serialize_generation = serialize_generation
        self.generation_lock = (
            generation_lock
            if generation_lock is not None
            else (RLock() if serialize_generation else nullcontext())
        )
        self._simulation_factory = simulation_factory
        self._adapter = adapter or SimulationTrajectoryAdapter()
        self._verifier = verifier or RetailPolicyVerifier()

    def run(self, *, task_id: str, seed: int) -> RolloutResult:
        """Run, normalize, adapt, and verify one task rollout."""

        try:
            factory = self._simulation_factory or self._run_tau2_simulation
            simulation, traces = factory(task_id, seed)
            info = getattr(simulation, "info", {}) or {}
            evaluation = RetailTaskRunner.evaluation_from_simulation(simulation)
            if isinstance(info, dict):
                evaluation.update(info.get("evaluation") or {})
                terminal_state = info.get("terminal_state", {}) or {}
            else:
                terminal_state = {}
            evaluation = RetailTaskRunner.normalise_evaluation(evaluation)
            trajectory = self._adapter.from_simulation(
                simulation,
                terminal_state=terminal_state,
                evaluation=evaluation,
            )
            verification = self._verifier.verify(trajectory)
            return RolloutResult(
                task_id=str(task_id),
                seed=seed,
                simulation=simulation,
                trajectory=trajectory,
                verification=verification,
                traces=tuple(traces),
            )
        except Exception as error:
            return self._invalid_result(task_id, seed, error)

    @staticmethod
    def _invalid_result(task_id: str, seed: int, error: Exception) -> RolloutResult:
        return RolloutResult(
            task_id=str(task_id),
            seed=seed,
            simulation=None,
            trajectory=None,
            verification=VerificationResult(
                task_success=False,
                policy_violation=False,
                first_error="infrastructure_invalid",
                reward=0.0,
                reward_valid=False,
            ),
            traces=(),
            error="{}: {}".format(type(error).__name__, error),
        )

    def _run_tau2_simulation(
        self,
        task_id: str,
        seed: int,
    ) -> tuple[Any, Tuple[GenerationTrace, ...]]:
        """Build and execute one real tau2 text orchestrator lazily."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Tau2LocalRolloutRunner requires model and tokenizer for live rollouts"
            )
        if not self.user_llm:
            raise RuntimeError(
                "Tau2LocalRolloutRunner requires user_llm for live rollouts"
            )
        try:
            from tau2.orchestrator.orchestrator import Orchestrator
            from tau2.run import get_tasks, run_simulation
            from tau2.runner.build import build_environment, build_user
        except ImportError as error:
            raise RuntimeError(
                "live tau2 rollouts require the tau2 training environment"
            ) from error

        tasks = get_tasks(
            "retail",
            task_split_name=self.task_split_name,
            task_ids=[str(task_id)],
        )
        if len(tasks) != 1:
            raise ValueError("expected exactly one Retail task for task_id=" + str(task_id))
        task = tasks[0]
        environment = build_environment("retail")
        agent = LocalQwenAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            generation_lock=self.generation_lock,
        )
        user = build_user(
            "user_simulator",
            environment,
            task,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
        )
        orchestrator = Orchestrator(
            domain="retail",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            seed=seed,
            validate_communication=True,
        )
        simulation = run_simulation(orchestrator)
        return simulation, agent.drain_generation_traces()
