# Serial GRPO and User ID Prompt Implementation Plan

> **For agentic workers:** Execute this plan inline with a test-first cycle for each behavior.

**Goal:** Run the checked-in GRPO launchers with two concurrent rollout workers by default and teach the local policy how to handle unknown user IDs.

**Architecture:** Keep the CLI escape hatch `--parallel-generation` unchanged. Change both launcher defaults so normal script runs use two rollout workers and pass that flag. Add the two user-ID rules to the existing `LocalQwenAgent.system_prompt`; no parser or retry behavior changes are included.

**Tech Stack:** Bash launchers, Python, pytest.

## Global Constraints

- `MAX_WORKERS` is `2` and `PARALLEL_GENERATION` is `1` in both launchers.
- The prompt contains the exact rules `Never invent a user_id.` and `If the user_id is unknown, use find_user_id_by_email or find_user_id_by_name_zip.`
- Explicit CLI `--parallel-generation` remains supported.
- Existing GRPO tests must remain passing.

### Task 1: Add user ID guidance to the policy prompt

**Files:**
- Modify: `src/agent_for_business/grpo_agent.py:242-263`
- Test: `tests/test_grpo_agent.py`

- [ ] Write a failing test asserting both exact user-ID rules appear in `LocalQwenAgent.system_prompt`.
- [ ] Run `pytest tests/test_grpo_agent.py -q` and confirm the new assertion fails because the rules are absent.
- [ ] Add the two prompt lines immediately after the existing ID rules.
- [ ] Run the focused test and confirm it passes.

### Task 2: Make both launchers serial by default

**Files:**
- Modify: `GRPO_train.sh:18`
- Modify: `scripts/GRPO_train.sh:18`
- Test: `tests/test_grpo_launchers.py`

- [ ] Write a failing test that reads both launchers and asserts `MAX_WORKERS="2"` and `PARALLEL_GENERATION="1"`.
- [ ] Run the focused launcher test and confirm it fails because both files currently use `"1"`.
- [ ] Change both launcher defaults to `"0"`; keep the conditional CLI flag logic and CLI option unchanged.
- [ ] Run the focused launcher test and confirm it passes.

### Task 3: Verify the complete change

- [ ] Run `pytest tests/test_grpo_agent.py tests/test_grpo_launchers.py tests/test_grpo_rollout.py tests/test_grpo_online.py tests/test_grpo_training.py tests/test_grpo_objective.py tests/test_grpo_core.py -q`.
- [ ] Run `git diff --check`.
- [ ] Confirm the final diff contains only the prompt, launcher defaults, tests, and this plan.
