# Serial GRPO Rollouts and User ID Prompt Rules

## Goal

Make the GRPO launcher run local-policy generation in parallel by default, and
give the local policy two explicit rules for handling unknown user IDs.

## Scope

- Set `MAX_WORKERS` to `2` and `PARALLEL_GENERATION` to `1` in both GRPO
  launcher scripts.
- Keep the existing CLI `--parallel-generation` option available for explicit
  experiments; this change only changes the launcher default.
- Add these exact rules to `LocalQwenAgent.system_prompt`:

  - `Never invent a user_id.`
  - `If the user_id is unknown, use find_user_id_by_email or find_user_id_by_name_zip.`

## Behavior

When the launcher is used, it will run two rollout workers and pass
`--parallel-generation`. The trainer will therefore allow two local model
generation calls at once. The prompt rules only guide the policy model; they do
not add parser-side validation or prevent legitimate retries with corrected
arguments.

## Verification

- Unit tests assert both prompt rules are present.
- Launcher checks assert parallel generation is disabled by default.
- Existing GRPO agent, rollout, online trainer, and CLI tests remain passing.
