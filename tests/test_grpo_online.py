import json
from pathlib import Path

import torch

from agent_for_business.grpo_agent import GenerationTrace
from agent_for_business.grpo_online import OnlineGRPOTrainer
from agent_for_business.grpo_rollout import RolloutResult
from agent_for_business.grpo_training import GRPOTrainingConfig
from agent_for_business.policy_verifier import VerificationResult


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(10))

    def forward(self, *, input_ids, use_cache=False):
        assert use_cache is False
        return type(
            "Output",
            (),
            {"logits": self.logits.view(1, 1, -1).expand(1, input_ids.shape[1], -1)},
        )()


def make_rollout(task_id, seed, reward):
    return RolloutResult(
        task_id=task_id,
        seed=seed,
        simulation=object(),
        trajectory=object(),
        verification=VerificationResult(
            task_success=reward > 0,
            policy_violation=False,
            first_error=None,
            reward=reward,
            reward_valid=True,
        ),
        traces=(
            GenerationTrace(
                prompt_ids=(1, 2),
                response_ids=(3,),
                old_logprobs=(-2.302585,),
                action_mask=(True,),
            ),
        ),
    )


def test_online_trainer_updates_one_group_and_saves_checkpoint(tmp_path):
    calls = []

    def rollout_runner(task_id, seed):
        reward = 0.0 if len(calls) == 0 else 1.0
        calls.append((task_id, seed))
        return make_rollout(task_id, seed, reward)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
        learning_rate=0.1,
    )
    policy = TinyPolicy()
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=policy,
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    assert trainer.device == torch.device("cpu")

    result = trainer.train()

    assert result["optimizer_steps"] == 1
    assert result["valid_rollouts"] == 2
    assert result["rollout_plan"] == {
        "batches": 1,
        "groups_per_batch": 1,
        "rollouts_per_group": 2,
        "rollouts_per_batch": 2,
        "total_rollouts": 2,
    }
    assert calls == [("task-1", 42), ("task-1", 43)]
    assert list(Path(tmp_path).glob("checkpoint-1/*"))


def test_online_trainer_passes_parallel_generation_to_rollout_runner(
    tmp_path, monkeypatch
):
    import agent_for_business.grpo_online as grpo_online_module

    seen = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(grpo_online_module, "Tau2LocalRolloutRunner", FakeRunner)
    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        parallel_generation=True,
    )

    OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        reference_model=TinyPolicy(),
    )

    assert seen["serialize_generation"] is False


def test_online_trainer_reports_group_progress(tmp_path, capsys):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    trainer.train()

    assert "[GRPO] batch=1/1 group=1/1" in capsys.readouterr().out


def test_online_trainer_uses_configured_training_microbatches(
    tmp_path, monkeypatch
):
    import agent_for_business.grpo_online as grpo_online_module

    original_grpo_loss = grpo_online_module.grpo_loss
    action_batch_sizes = []

    def tracking_grpo_loss(**kwargs):
        action_batch_sizes.append(int(kwargs["action_mask"].shape[0]))
        return original_grpo_loss(**kwargs)

    monkeypatch.setattr(grpo_online_module, "grpo_loss", tracking_grpo_loss)

    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        batch_epochs=1,
        inference_microbatch=1,
        max_rollout_batches=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    trainer.train()

    assert action_batch_sizes == [1, 1]


def test_online_trainer_writes_compact_progress_log(tmp_path):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    trainer.train()

    progress = (Path(tmp_path) / "grpo_progress.log").read_text(
        encoding="utf-8"
    )
    assert "phase=rollout_start" in progress
    assert "phase=rollout_done" in progress
    assert "phase=update" in progress
    assert "phase=run_done" in progress


def test_online_trainer_saves_final_checkpoint_when_interval_is_larger(tmp_path):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
        checkpoint_every=10,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    result = trainer.train()

    checkpoint = Path(result["last_checkpoint"])
    assert checkpoint == Path(tmp_path) / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "grpo_manifest.json").is_file()


def test_online_trainer_excludes_invalid_rollouts_from_update(tmp_path):
    calls = []

    def rollout_runner(task_id, seed):
        calls.append(seed)
        result = make_rollout(task_id, seed, 1.0)
        if len(calls) <= 2:
            result = RolloutResult(
                task_id=result.task_id,
                seed=result.seed,
                simulation=result.simulation,
                trajectory=result.trajectory,
                verification=VerificationResult(
                    task_success=False,
                    policy_violation=False,
                    first_error="infrastructure_invalid",
                    reward=0.0,
                    reward_valid=False,
                ),
                traces=result.traces,
            )
        return result

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=3,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    result = trainer.train()

    assert result["valid_rollouts"] == 1
    assert result["invalid_rollouts"] == 2


def test_online_trainer_skips_group_without_relative_reward_signal(tmp_path):
    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=lambda task_id, seed: make_rollout(task_id, seed, 0.0),
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    result = trainer.train()

    assert result["optimizer_steps"] == 0
    assert result["valid_rollouts"] == 2
    assert result["invalid_rollouts"] == 0
    assert result["skipped_no_signal_groups"] == 1
    assert result["history"] == []
    assert not (Path(tmp_path) / "checkpoint-0").exists()


def test_online_trainer_resumes_optimizer_step_and_history(tmp_path):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    base_config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    first = OnlineGRPOTrainer(
        config=base_config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )
    first.policy_model.logits.data[0] = 3.0
    first_result = first.train()
    checkpoint = Path(first_result["last_checkpoint"])
    first_weights = first.policy_model.logits.detach().clone()

    resumed_config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=2,
        batch_epochs=1,
        resume_from=checkpoint,
    )
    resumed = OnlineGRPOTrainer(
        config=resumed_config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    assert resumed.optimizer_steps == 1
    assert torch.equal(resumed.policy_model.logits.detach(), first_weights)
    resumed_result = resumed.train()

    assert resumed_result["optimizer_steps"] == 2
    assert len(resumed_result["history"]) == 2


def test_online_trainer_resumes_from_next_batch_until_total_target(tmp_path):
    calls = []

    def rollout_runner(task_id, seed):
        calls.append(seed)
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    first_config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=1,
    )
    first = OnlineGRPOTrainer(
        config=first_config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1", "task-2"),
        reference_model=TinyPolicy(),
    )
    first.train()

    checkpoint = Path(tmp_path) / "checkpoint-1"
    resumed_config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=2,
        batch_epochs=1,
        resume_from=checkpoint,
    )
    resumed = OnlineGRPOTrainer(
        config=resumed_config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1", "task-2"),
        reference_model=TinyPolicy(),
    )

    result = resumed.train()

    assert result["optimizer_steps"] == 2
    assert calls == [42, 43, 44, 45]


def test_batch_epochs_performs_one_optimizer_step_per_epoch(tmp_path):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=2,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    result = trainer.train()

    assert result["optimizer_steps"] == 2


def test_checkpoint_cursor_is_saved_at_batch_boundary(tmp_path):
    def rollout_runner(task_id, seed):
        return make_rollout(task_id, seed, 1.0 if seed % 2 else 0.0)

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=2,
        max_rollout_batches=1,
        batch_epochs=2,
        checkpoint_every=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    result = trainer.train()

    manifest = json.loads(
        (Path(result["last_checkpoint"]) / "grpo_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["optimizer_steps"] == 2
    assert manifest["step"] == 2
    assert manifest["next_batch_index"] == 1
    assert not (Path(tmp_path) / "checkpoint-1").exists()


def test_online_trainer_rejects_batch_without_action_tokens(tmp_path):
    def rollout_runner(task_id, seed):
        result = make_rollout(task_id, seed, 1.0)
        return RolloutResult(
            task_id=result.task_id,
            seed=result.seed,
            simulation=result.simulation,
            trajectory=result.trajectory,
            verification=result.verification,
            traces=(
                GenerationTrace(
                    prompt_ids=(1,),
                    response_ids=(2,),
                    old_logprobs=(-1.0,),
                    action_mask=(False,),
                ),
            ),
        )

    config = GRPOTrainingConfig(
        model_name="tiny",
        output_dir=tmp_path,
        groups_per_batch=1,
        group_size=1,
        max_rollout_batches=1,
    )
    trainer = OnlineGRPOTrainer(
        config=config,
        policy_model=TinyPolicy(),
        tokenizer=None,
        rollout_runner=rollout_runner,
        task_ids=("task-1",),
        reference_model=TinyPolicy(),
    )

    import pytest

    with pytest.raises(ValueError, match="action-token traces"):
        trainer.train()
