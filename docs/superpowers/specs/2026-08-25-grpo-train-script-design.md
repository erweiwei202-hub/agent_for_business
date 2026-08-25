# GRPO Training Shell Script Design

## Goal

Add a root-level `GRPO_train.sh` that launches the existing GRPO CLI with a
plain `python` command. The script should make the full GRPO configuration easy
to edit and support both fresh runs and checkpoint resume.

## Decisions

- Use `python -m agent_for_business.cli grpo`, assuming the current Python
  environment already has the project and training dependencies installed.
- Change into the repository root at script start so relative model, task split,
  and output paths work regardless of the caller's current directory.
- Keep every commonly changed GRPO setting in an editable variable block near
  the top of the script.
- Use shell arrays to assemble optional arguments safely. Empty
  `RESUME_FROM` and `USER_API_BASE` values will not be passed to the CLI.
- Preserve the trainer's existing checkpoint behavior through an editable
  `CHECKPOINT_EVERY` variable. Resume paths will be passed through
  `--resume-from` when configured.
- Use `set -euo pipefail` so a failed Python training process causes the shell
  script to fail.

## Parameters

The script will expose the existing GRPO CLI options for model and output
paths, task split, rollout group sizing, worker and microbatch settings,
clipping and KL coefficients, seed and rollout count, user simulator settings,
optimizer settings, generation settings, device, checkpoint frequency, and
resume path.

The default model will be the existing SFT checkpoint path
`outputs/sft/checkpoint-qwen/checkpoint-294`, and the default GRPO output path
will be `outputs/grpo`. These are script defaults only and remain editable.

## Data flow

```text
edit variables in GRPO_train.sh
        -> python -m agent_for_business.cli grpo
        -> existing GRPO config and trainer
        -> outputs/grpo/checkpoint-N
```

## Error handling

- The script will not hide Python errors or continue after a failed training
  command.
- Optional CLI arguments will only be emitted when their corresponding shell
  variables are non-empty, preventing malformed flags such as
  `--resume-from` without a value.
- Checkpoint creation and resume semantics remain owned by
  `OnlineGRPOTrainer`; the shell script only exposes their existing controls.

## Testing

- Run `bash -n GRPO_train.sh` for shell syntax validation.
- Inspect the generated command structure without starting GPU training.
- Run the project's existing CLI/parser tests if the environment supports the
  project's test dependencies.
