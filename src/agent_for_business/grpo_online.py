"""Single-GPU online GRPO trainer for the local Retail policy."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .grpo_agent import GenerationTrace
from .grpo_core import compute_group_advantages
from .grpo_objective import ObjectiveResult, grpo_loss, sequence_logprobs
from .grpo_rollout import RolloutResult, Tau2LocalRolloutRunner
from .grpo_training import GRPOTrainingConfig
from .task_partition import load_retail_task_partition


class OnlineGRPOTrainer:
    """Collect rollout groups, update the policy, and save resumable state."""

    def __init__(
        self,
        *,
        config: GRPOTrainingConfig,
        policy_model: Any,
        tokenizer: Any,
        rollout_runner: Optional[Any] = None,
        task_ids: Optional[Sequence[str]] = None,
        reference_model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = self._resolve_device(config.device)
        if hasattr(self.policy_model, "to"):
            self.policy_model.to(self.device)
        self._task_ids = tuple(task_ids) if task_ids is not None else None
        self.rollout_runner = rollout_runner or Tau2LocalRolloutRunner(
            model=policy_model,
            tokenizer=tokenizer,
            user_llm=config.user_llm,
            user_llm_args=config.user_llm_args,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            max_steps=100,
            serialize_generation=not config.parallel_generation,
        )
        self.reference_model = reference_model or self._clone_reference(policy_model)
        if hasattr(self.reference_model, "to"):
            self.reference_model.to(self.device)
        self.optimizer = optimizer or self._build_optimizer(policy_model)
        self.optimizer_steps = 0
        self._next_batch_index = 0
        self.history: List[Dict[str, Any]] = []
        self._progress_path = Path(config.output_dir) / "grpo_progress.log"
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_lock = RLock()
        self._load_resume_state()

    def train(self) -> Dict[str, Any]:
        """Run configured rollout batches and return a JSON-safe summary."""

        task_ids = self._resolve_task_ids()
        if not task_ids:
            raise ValueError("GRPO has no training task ids")
        self._write_progress(
            "run_start",
            next_batch_index=self._next_batch_index,
            total_batches=self.config.max_rollout_batches,
        )
        valid_rollouts = 0
        invalid_rollouts = 0
        skipped_no_signal_groups = 0
        for batch_index in range(
            self._next_batch_index, self.config.max_rollout_batches
        ):
            if hasattr(self.policy_model, "eval"):
                self.policy_model.eval()
            batch_groups = []
            for group_index in range(self.config.groups_per_batch):
                task_id = task_ids[
                    (batch_index * self.config.groups_per_batch + group_index)
                    % len(task_ids)
                ]
                seeds = [
                    self.config.seed
                    + (
                        batch_index * self.config.groups_per_batch * self.config.group_size
                        + group_index * self.config.group_size
                        + member_index
                    )
                    for member_index in range(self.config.group_size)
                ]
                group = self._collect_group(
                    task_id,
                    seeds,
                    batch_index=batch_index,
                    group_index=group_index,
                )
                for result in group:
                    if result.valid_for_update:
                        valid_rollouts += 1
                    else:
                        invalid_rollouts += 1
                batch_groups.append(group)
                print(
                    "[GRPO] batch={}/{} group={}/{} rollouts={}".format(
                        batch_index + 1,
                        self.config.max_rollout_batches,
                        group_index + 1,
                        self.config.groups_per_batch,
                        len(group),
                    ),
                    flush=True,
                )

            objective, batch_stats, epoch_count = self._update_batch(batch_groups)
            skipped_no_signal_groups += int(
                batch_stats.get("skipped_no_signal_groups", 0)
            )
            if objective is None:
                self._write_progress(
                    "batch_skipped",
                    batch=batch_index + 1,
                    reason=batch_stats["skip_reason"],
                    reward_mean=batch_stats["reward_mean"],
                )
                print(
                    "[GRPO] batch={}/{} skipped: {}".format(
                        batch_index + 1,
                        self.config.max_rollout_batches,
                        batch_stats["skip_reason"],
                    ),
                    flush=True,
                )
                self._next_batch_index = batch_index + 1
                continue
            checkpoint_due = False
            for batch_epoch in range(epoch_count):
                self.optimizer_steps += 1
                metrics = {
                    "batch": batch_index,
                    "batch_epoch": batch_epoch,
                    "optimizer_step": self.optimizer_steps,
                    "loss": float(objective.loss.detach().cpu().item()),
                    "policy_objective": float(
                        objective.policy_objective.detach().cpu().item()
                    ),
                    "reference_kl": float(
                        objective.reference_kl.detach().cpu().item()
                    ),
                    **batch_stats,
                }
                self.history.append(metrics)
                self._write_progress(
                    "update",
                    batch=batch_index + 1,
                    epoch=batch_epoch + 1,
                    epochs=epoch_count,
                    optimizer_step=self.optimizer_steps,
                    loss=metrics["loss"],
                    reward_mean=batch_stats["reward_mean"],
                )
                if self.optimizer_steps % self.config.checkpoint_every == 0:
                    checkpoint_due = True

            self._next_batch_index = batch_index + 1
            if checkpoint_due:
                checkpoint = self.save_checkpoint(self.optimizer_steps)
                self._write_progress(
                    "checkpoint",
                    step=self.optimizer_steps,
                    path=checkpoint,
                )

        if (
            self.optimizer_steps
            and self.optimizer_steps % self.config.checkpoint_every != 0
        ):
            checkpoint = self.save_checkpoint(self.optimizer_steps)
            self._write_progress(
                "checkpoint",
                step=self.optimizer_steps,
                path=checkpoint,
            )

        self._write_progress(
            "run_done",
            optimizer_steps=self.optimizer_steps,
            next_batch_index=self._next_batch_index,
        )

        return {
            "optimizer_steps": self.optimizer_steps,
            "valid_rollouts": valid_rollouts,
            "invalid_rollouts": invalid_rollouts,
            "skipped_no_signal_groups": skipped_no_signal_groups,
            "next_batch_index": self._next_batch_index,
            "rollout_plan": self.config.rollout_plan,
            "history": self.history,
            "last_checkpoint": str(
                self._checkpoint_path(self.optimizer_steps)
                if self.optimizer_steps
                else ""
            ),
        }

    def save_checkpoint(
        self,
        step: int,
        *,
        next_batch_index: Optional[int] = None,
    ) -> Path:
        """Persist policy, tokenizer, optimizer, and a JSON manifest."""

        checkpoint = self._checkpoint_path(step)
        checkpoint.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy_model, "save_pretrained"):
            self.policy_model.save_pretrained(str(checkpoint))
        else:
            self._torch().save(self.policy_model.state_dict(), checkpoint / "model.pt")
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(checkpoint))
        self._torch().save(self.optimizer.state_dict(), checkpoint / "optimizer.pt")
        manifest = {
            "step": step,
            "next_batch_index": (
                self._next_batch_index
                if next_batch_index is None
                else int(next_batch_index)
            ),
            "model_name": self.config.model_name,
            "config": asdict(self.config),
            "rollout_plan": self.config.rollout_plan,
            "history": self.history,
        }
        (checkpoint / "grpo_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def _update_batch(
        self,
        groups: Iterable[Sequence[RolloutResult]],
    ) -> Tuple[Optional[ObjectiveResult], Dict[str, Any], int]:
        records = []
        rewards_for_stats = []
        valid_group_count = 0
        skipped_no_signal_groups = 0
        for group in groups:
            valid = [result for result in group if result.valid_for_update]
            if not valid:
                continue
            valid_group_count += 1
            rewards = [float(result.verification.reward) for result in valid]
            advantages = compute_group_advantages(rewards)
            rewards_for_stats.extend(rewards)
            if not any(advantage != 0.0 for advantage in advantages):
                skipped_no_signal_groups += 1
                continue
            for result, advantage in zip(valid, advantages):
                for trace in result.traces:
                    if any(trace.action_mask):
                        records.append((trace, float(advantage)))
        if not records:
            if valid_group_count and skipped_no_signal_groups == valid_group_count:
                return None, {
                    "action_token_count": 0,
                    "reward_mean": sum(rewards_for_stats)
                    / len(rewards_for_stats),
                    "rewards": rewards_for_stats,
                    "skipped_no_signal_groups": skipped_no_signal_groups,
                    "skip_reason": "no_relative_reward_signal",
                }, 0
            raise ValueError("GRPO batch has no valid action-token traces")

        total_action_tokens = sum(
            sum(1 for selected in trace.action_mask if selected)
            for trace, _ in records
        )
        if total_action_tokens == 0:
            raise ValueError("GRPO batch has no valid action-token traces")

        self.policy_model.train()
        last_objective = None
        for _ in range(self.config.batch_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            weighted_policy_objective = 0.0
            weighted_reference_kl = 0.0
            for start in range(0, len(records), self.config.inference_microbatch):
                microbatch = records[
                    start : start + self.config.inference_microbatch
                ]
                old_rows = []
                current_rows = []
                reference_rows = []
                masks = []
                advantages = []
                for trace, advantage in microbatch:
                    input_ids, response_start = self._trace_input(trace)
                    current_rows.append(
                        sequence_logprobs(
                            self.policy_model,
                            input_ids,
                            response_start,
                        ).squeeze(0)
                    )
                    with self._torch().no_grad():
                        reference_rows.append(
                            sequence_logprobs(
                                self.reference_model,
                                input_ids,
                                response_start,
                            ).squeeze(0)
                        )
                    old_rows.append(
                        self._torch().tensor(
                            trace.old_logprobs,
                            device=self.device,
                        )
                    )
                    masks.append(
                        self._torch().tensor(
                            trace.action_mask,
                            dtype=self._torch().bool,
                            device=self.device,
                        )
                    )
                    advantages.append(advantage)
                old = self._pad(old_rows)
                current = self._pad(current_rows)
                reference = self._pad(reference_rows)
                mask = self._pad(masks, padding_value=False).to(
                    dtype=self._torch().bool
                )
                objective = grpo_loss(
                    old_logprobs=old,
                    current_logprobs=current,
                    reference_logprobs=reference,
                    advantages=self._torch().tensor(
                        advantages,
                        device=self.device,
                    ),
                    action_mask=mask,
                    clip_ratio=self.config.clip_ratio,
                    kl_beta=self.config.kl_beta,
                )
                weight = objective.action_token_count / total_action_tokens
                (objective.loss * weight).backward()
                weighted_loss += float(objective.loss.detach().cpu().item()) * weight
                weighted_policy_objective += (
                    float(objective.policy_objective.detach().cpu().item()) * weight
                )
                weighted_reference_kl += (
                    float(objective.reference_kl.detach().cpu().item()) * weight
                )
            last_objective = ObjectiveResult(
                loss=self._torch().tensor(weighted_loss, device=self.device),
                policy_objective=self._torch().tensor(
                    weighted_policy_objective,
                    device=self.device,
                ),
                reference_kl=self._torch().tensor(
                    weighted_reference_kl,
                    device=self.device,
                ),
                action_token_count=total_action_tokens,
            )
            self.optimizer.step()
        return last_objective, {
            "action_token_count": last_objective.action_token_count,
            "reward_mean": sum(rewards_for_stats) / len(rewards_for_stats),
            "rewards": rewards_for_stats,
            "skipped_no_signal_groups": skipped_no_signal_groups,
        }, self.config.batch_epochs

    def _trace_input(self, trace: GenerationTrace) -> Tuple[Any, int]:
        values = trace.prompt_ids + trace.response_ids
        return self._torch().tensor([values], dtype=self._torch().long), len(
            trace.prompt_ids
        )

    def _run_rollout(self, task_id: str, seed: int) -> RolloutResult:
        if hasattr(self.rollout_runner, "run"):
            return self.rollout_runner.run(task_id=task_id, seed=seed)
        return self.rollout_runner(task_id, seed)

    def _collect_group(
        self,
        task_id: str,
        seeds: Sequence[int],
        *,
        batch_index: int,
        group_index: int,
    ) -> List[RolloutResult]:
        """Collect a live tau2 group concurrently, preserving seed order."""

        for rollout_index, seed in enumerate(seeds, start=1):
            self._write_progress(
                "rollout_start",
                batch=batch_index + 1,
                group=group_index + 1,
                rollout=rollout_index,
                total_rollouts=len(seeds),
                task_id=task_id,
                seed=seed,
            )

        if not isinstance(self.rollout_runner, Tau2LocalRolloutRunner):
            results = []
            for rollout_index, seed in enumerate(seeds, start=1):
                result = self._run_rollout(task_id, seed)
                results.append(result)
                self._write_progress(
                    "rollout_done",
                    batch=batch_index + 1,
                    group=group_index + 1,
                    rollout=rollout_index,
                    total_rollouts=len(seeds),
                    task_id=task_id,
                    seed=seed,
                    valid=result.valid_for_update,
                )
            return results

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._run_rollout, task_id, seed): (index, seed)
                for index, seed in enumerate(seeds)
            }
            results: Dict[int, RolloutResult] = {}
            for future in as_completed(futures):
                rollout_index, seed = futures[future]
                result = future.result()
                results[rollout_index] = result
                self._write_progress(
                    "rollout_done",
                    batch=batch_index + 1,
                    group=group_index + 1,
                    rollout=rollout_index + 1,
                    total_rollouts=len(seeds),
                    task_id=task_id,
                    seed=seed,
                    valid=result.valid_for_update,
                )
            return [results[index] for index in range(len(seeds))]

    def _write_progress(self, phase: str, **fields: Any) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        values = ["{}={}".format(key, str(value).replace(" ", "_")) for key, value in fields.items()]
        line = "{} phase={}{}\n".format(
            timestamp,
            phase,
            (" " + " ".join(values)) if values else "",
        )
        with self._progress_lock:
            with self._progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def _resolve_task_ids(self) -> Tuple[str, ...]:
        if self._task_ids is not None:
            return self._task_ids
        partition = load_retail_task_partition(self.config.split_tasks)
        self._task_ids = tuple(partition.train)
        return self._task_ids

    def _checkpoint_path(self, step: int) -> Path:
        return Path(self.config.output_dir) / "checkpoint-{}".format(step)

    @staticmethod
    def _clone_reference(policy_model: Any) -> Any:
        import copy

        reference = copy.deepcopy(policy_model)
        if hasattr(reference, "requires_grad_"):
            reference.requires_grad_(False)
        if hasattr(reference, "gradient_checkpointing_disable"):
            reference.gradient_checkpointing_disable()
        if hasattr(reference, "eval"):
            reference.eval()
        return reference

    def _build_optimizer(self, model: Any) -> Any:
        torch = self._torch()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("GRPO policy has no trainable parameters")
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def _resolve_device(requested: str) -> Any:
        torch = OnlineGRPOTrainer._torch()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _load_resume_state(self) -> None:
        if not self.config.resume_from:
            return
        checkpoint = Path(self.config.resume_from)
        model_path = checkpoint / "model.pt"
        if model_path.is_file() and hasattr(self.policy_model, "load_state_dict"):
            state_dict = self._torch().load(model_path, map_location=self.device)
            self.policy_model.load_state_dict(state_dict)
        optimizer_path = checkpoint / "optimizer.pt"
        manifest_path = checkpoint / "grpo_manifest.json"
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(self._torch().load(optimizer_path, map_location="cpu"))
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.optimizer_steps = int(manifest.get("step", 0))
            next_batch_index = manifest.get("next_batch_index")
            if next_batch_index is None:
                # Older checkpoints did not record a batch cursor and could
                # have been written mid-batch. Replaying that batch is safer
                # than skipping its remaining epochs.
                next_batch_index = self.optimizer_steps // self.config.batch_epochs
            self._next_batch_index = max(0, int(next_batch_index))
            self.history = list(manifest.get("history", []))

    @staticmethod
    def _pad(rows: Sequence[Any], padding_value: float = 0.0) -> Any:
        return OnlineGRPOTrainer._torch().nn.utils.rnn.pad_sequence(
            list(rows), batch_first=True, padding_value=padding_value
        )

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("online GRPO requires torch") from error
        return torch
