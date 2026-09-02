# GRPO Training Shell Script Design

## Goal

Add a root-level `GRPO_train.sh` that launches the existing GRPO CLI with a
plain `python` command. The script should make the full GRPO configuration easy
to edit and support both fresh runs and checkpoint resume.

## Decisions

- Use `python -m agent_for_business.cli grpo`, assuming the current Python
  environment already has the training dependencies installed. Add the local
  `src` directory to `PYTHONPATH` so the checkout is directly runnable.
- Change into the repository root at script start so relative model, task split,
  and output paths work regardless of the caller's current directory.
- Keep every commonly changed GRPO setting in an editable variable block near
  the top of the script.
- Do not override `USER_LLM` or `USER_API_BASE` in the script. Let the existing
  CLI load them from `.env` and the process environment.
- Set an editable checkpoint interval (default 10 optimizer steps), scan the
  output directory for the numerically latest `checkpoint-N`, and pass that
  checkpoint to `--resume-from` automatically when present.
- Save periodic checkpoints at completed batch boundaries so a resume never
  advances past an unfinished batch.
- Ensure the trainer saves a final checkpoint even when the final optimizer
  step is not an exact multiple of the periodic checkpoint interval.
- Use `set -euo pipefail` so a failed Python training process causes the shell
  script to fail.

## Parameters

The script will expose the existing GRPO CLI options for model and output
paths, task split, rollout group sizing, worker and microbatch settings,
clipping and KL coefficients, seed and rollout count, optimizer settings,
generation settings, device, and checkpoint frequency. User simulator
configuration remains in `.env`/process environment. Resume selection is
automatic rather than a manually edited path.

The default model will be the existing SFT checkpoint path
`outputs/sft/checkpoint-qwen/checkpoint-294`, and the default GRPO output path
will be `outputs/grpo`. These are script defaults only and remain editable.

## Data flow

```text
edit variables in GRPO_train.sh and/or `.env`
        -> detect latest checkpoint under the output directory
        -> python -m agent_for_business.cli grpo
        -> existing GRPO config and trainer
        -> outputs/grpo/checkpoint-N
```

## Error handling

- The script will not hide Python errors or continue after a failed training
  command.
- The script emits `--resume-from` only when a valid numeric checkpoint
  directory is found.
- Checkpoint creation remains owned by `OnlineGRPOTrainer`; the shell script
  selects the latest complete checkpoint and exposes the save interval.

## Testing

- Run `bash -n GRPO_train.sh` for shell syntax validation.
- Inspect the generated command structure without starting GPU training.
- Run the project's existing CLI/parser tests if the environment supports the
  project's test dependencies.
