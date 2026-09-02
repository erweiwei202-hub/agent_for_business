# Parallel GRPO Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in two-rollout parallel inference mode with progress logs and a 512-token launcher default.

**Architecture:** The existing rollout worker pool remains responsible for concurrent rollouts. A new config flag controls whether its shared generation lock is replaced with a no-op context; policy inference runs under `torch.inference_mode()` and training remains after rollout collection. Both launcher copies will expose two workers and 512 maximum new tokens.

**Tech Stack:** Python, PyTorch, Transformers generation, argparse, Bash, pytest.

## Global Constraints

- Keep serialized generation as the default when the new flag is absent.
- Never run optimizer updates concurrently with rollout inference.
- Use `torch.inference_mode()` for rollout generation.
- Do not implement padded interactive inference batching in this change.

---

### Task 1: Add failing coverage for the parallel mode and progress output

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_grpo_online.py`

- [ ] Add tests for the CLI flag/config field, runner wiring, and flushed group progress.
- [ ] Run the focused tests and verify they fail because the flag and output do not yet exist.

### Task 2: Implement safe opt-in parallel rollout inference

**Files:**
- Modify: `src/agent_for_business/grpo_training.py`
- Modify: `src/agent_for_business/cli.py`
- Modify: `src/agent_for_business/grpo_online.py`
- Modify: `src/agent_for_business/grpo_rollout.py`
- Modify: `src/agent_for_business/grpo_agent.py`

- [ ] Add `parallel_generation` config and `--parallel-generation` CLI wiring.
- [ ] Pass `serialize_generation=not config.parallel_generation` to the rollout runner.
- [ ] Use a real lock by default and a no-op context only in opt-in mode.
- [ ] Run rollout generation in eval/inference mode and print batch/group progress.
- [ ] Run focused tests and then the related GRPO test suite.

### Task 3: Update launchers and verify the handoff

**Files:**
- Modify: `GRPO_train.sh`
- Modify: `scripts/GRPO_train.sh`

- [ ] Set `MAX_WORKERS=2` and `MAX_NEW_TOKENS=512`.
- [ ] Add an editable `PARALLEL_GENERATION=1` switch and pass the CLI flag when enabled.
- [ ] Run static launcher checks and report that a running process must be restarted to use the changes.
