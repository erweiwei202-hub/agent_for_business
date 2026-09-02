# Parallel GRPO Inference Design

## Goal

Allow two independent GRPO rollouts to call the read-only policy model
concurrently, while preserving serialized optimizer updates and adding visible
rollout progress.

## Decisions

- Add an explicit `--parallel-generation` flag; the default remains serialized
  generation for safe fallback.
- The launcher enables the flag with `MAX_WORKERS=2` and sets
  `MAX_NEW_TOKENS=512`.
- Parallel generation uses `torch.inference_mode()` and model evaluation mode;
  no policy parameters or optimizer state are written during rollout.
- The optimizer update starts only after the current rollout batch is fully
  collected, so inference and training do not access the model concurrently.
- Print completed batch/group progress with `flush=True`.
- Do not implement true padded inference microbatching in this change; the
  interactive tau2 conversations can diverge after each tool turn.

## Testing

- Verify the CLI/configuration carries the parallel-generation flag.
- Verify the trainer passes the inverse serialization setting to the rollout
  runner.
- Verify a rollout group emits a progress line.
- Run the GRPO trainer, rollout, objective, and CLI tests.
