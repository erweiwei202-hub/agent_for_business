# Comprehensive Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SFT admission gate with a reusable benchmark aggregation model that combines official τ² metrics, project Verifier/GRPO reward metrics, `db_match`, and communication checks in JSON and Markdown.

**Architecture:** `BenchmarkRecord` will be the normalized per-run contract. `BenchmarkSummary` in `validation_gate.py` will aggregate records without making a pass/fail decision. The final benchmark script will deserialize saved τ² simulations, adapt them to project trajectories, run `RetailPolicyVerifier`, create records, and render the shared summary into the output JSON and Markdown.

**Tech Stack:** Python 3.12, dataclasses, JSON, pytest, the vendored `tau2` Pydantic models at runtime, and the existing `SimulationTrajectoryAdapter`/`RetailPolicyVerifier`.

## Global Constraints

- Do not create a weighted composite score or automatic `passed`/`gate_decision` result.
- Official τ² reward remains the value from `simulation.reward_info.reward`; it is not replaced by Verifier reward.
- `verifier_reward` is `VerificationResult.reward`, and its mean uses only Verifier results with `reward_valid=True`.
- `db_match` and `communication_ok` rates use only non-null checks; missing checks have separate counts and are not failures.
- The final Markdown must explain every aggregate metric and the formulas for τ² reward and Verifier reward.
- Keep tau2 imports lazy in the final benchmark script so unit tests can run without installing tau2.
- Use public behavior-oriented tests and one red → green vertical slice at a time.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Replace the Gate Module with Per-Run Records and Aggregation

**Files:**
- Modify: `src/agent_for_business/validation_gate.py`
- Modify: `tests/test_validation_gate.py`

**Interfaces:**
- Produces `BenchmarkRecord` with fields `task_id`, `trial`, `tau_reward`, `tau_reward_valid`, `verifier_reward`, `verifier_reward_valid`, `verifier_valid`, `task_success`, `policy_violation`, `first_error`, `tool_error_count`, `db_match`, `communication_ok`, `termination_reason`, and `verifier_error`.
- Produces `BenchmarkRecord.from_verification(...)` for converting a `VerificationResult`-like object plus official τ² fields.
- Produces `BenchmarkSummary.from_records(records, expected_runs=None)` and `BenchmarkSummary.to_dict()`.
- Removes `GateDecision` and `SFTValidationGate` from the module.

- [ ] **Step 1: Write the failing aggregation test**

```python
def test_summary_keeps_tau_and_verifier_rewards_separate_and_counts_missing_checks():
    records = [
        BenchmarkRecord(
            task_id="5", trial=0, tau_reward=1.0, tau_reward_valid=True,
            verifier_reward=0.9, verifier_reward_valid=True,
            verifier_valid=True, task_success=True, policy_violation=False,
            first_error=None, tool_error_count=1, db_match=True,
            communication_ok=True, termination_reason="agent_end",
        ),
        BenchmarkRecord(
            task_id="9", trial=0, tau_reward=0.0, tau_reward_valid=True,
            verifier_reward=-1.0, verifier_reward_valid=True,
            verifier_valid=True, task_success=False, policy_violation=True,
            first_error="missing_confirmation", tool_error_count=0,
            db_match=False, communication_ok=None,
            termination_reason="agent_end",
        ),
    ]

    summary = BenchmarkSummary.from_records(records, expected_runs=3)

    assert summary.to_dict() == {
        "expected_runs": 3, "completed_runs": 2, "incomplete_runs": 1,
        "tau_reward_valid_count": 2, "tau_reward_invalid_count": 0,
        "tau_reward_valid_rate": 1.0, "tau_success_count": 1,
        "tau_success_rate": 0.5, "tau_reward_mean": 0.5,
        "db_match_true_count": 1, "db_match_present_count": 2,
        "db_match_missing_count": 0, "db_match_rate": 0.5,
        "communication_true_count": 1, "communication_present_count": 1,
        "communication_missing_count": 1, "communication_rate": 1.0,
        "termination_counts": {"agent_end": 2},
        "verifier_evaluated_count": 2, "verifier_invalid_count": 0,
        "verifier_reward_valid_count": 2, "verifier_reward_mean": -0.05,
        "policy_violation_count": 1, "policy_violation_rate": 0.5,
        "tool_error_run_count": 1, "tool_error_rate": 0.5,
        "tool_error_total": 1,
        "first_error_counts": {"missing_confirmation": 1},
    }
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest -q tests/test_validation_gate.py::test_summary_keeps_tau_and_verifier_rewards_separate_and_counts_missing_checks`

Expected: FAIL because `BenchmarkRecord` and the new summary interface do not exist.

- [ ] **Step 3: Implement the minimal aggregation model**

Implement frozen dataclasses. Use these denominator rules:

```python
tau_valid = [row for row in rows if row.tau_reward_valid and row.tau_reward is not None]
verifier_rows = [row for row in rows if row.verifier_valid]
verifier_rewards = [
    row.verifier_reward for row in verifier_rows
    if row.verifier_reward_valid and row.verifier_reward is not None
]
db_rows = [row.db_match for row in rows if row.db_match is not None]
communication_rows = [
    row.communication_ok for row in rows if row.communication_ok is not None
]
```

Compute `tau_success_rate` from valid τ² rewards `>= 1.0`, `verifier_reward_mean` from valid Verifier rewards, policy/tool rates from `verifier_rows`, and rates for DB/communication from their present rows. Return `None` for means/rates with no denominator. Keep termination and first-error counters as ordinary JSON dictionaries.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m pytest -q tests/test_validation_gate.py::test_summary_keeps_tau_and_verifier_rewards_separate_and_counts_missing_checks`

Expected: PASS.

- [ ] **Step 5: Add the invalid Verifier and conversion-error behavior test**

Add a test with one `verifier_valid=False` record containing `verifier_error="message_conversion_failed"` and one record with `tau_reward_valid=False`. Assert `verifier_invalid_count == 1`, `tau_reward_invalid_count == 1`, invalid rows are excluded from Verifier reward mean/rates, and `to_dict()` remains JSON serializable.

- [ ] **Step 6: Implement only the behavior needed by the new test and rerun both focused tests**

Run: `python -m pytest -q tests/test_validation_gate.py`

Expected: PASS.

- [ ] **Step 7: Commit the reporting model**

```powershell
git add tests/test_validation_gate.py src/agent_for_business/validation_gate.py
git commit -m "feat: replace validation gate with benchmark aggregation"
```

### Task 2: Remove Gate Decisions from Validation Benchmark Helpers

**Files:**
- Modify: `src/agent_for_business/validation_benchmark.py`
- Modify: `tests/test_validation_benchmark.py`
- Modify: `src/agent_for_business/__init__.py`

**Interfaces:**
- `run_validation_benchmark(...)` continues to return `model`, `task_ids`, `seed`, `results`, and a serialized `summary`, now produced from `BenchmarkRecord`/`BenchmarkSummary`.
- `compare_raw_sft_validation(...)` continues to compare two reports but returns no `gate_decision`; its `summary` contains the raw and SFT summaries only.

- [ ] **Step 1: Change one existing behavior test to assert no gate decision**

Change the raw/SFT test to assert:

```python
assert "gate_decision" not in report
assert report["summary"]["raw"]["verifier_reward_mean"] == 0.5
assert report["summary"]["sft"]["verifier_reward_mean"] == 1.0
```

- [ ] **Step 2: Run the changed test and verify it fails**

Run: `python -m pytest -q tests/test_validation_benchmark.py::test_compare_raw_sft_validation_uses_same_task_ids_seed_and_returns_gate`

Expected: FAIL because the current helper still returns `gate_decision` and the old summary shape.

- [ ] **Step 3: Implement the minimal helper migration**

For each Verifier result, create a `BenchmarkRecord` with `tau_reward=None`, `tau_reward_valid=False`, and the Verifier fields copied from the result. Build the report summary with `BenchmarkSummary.from_records(results, expected_runs=len(validation_task_ids))`. Remove the `gate` parameter and all `SFTValidationGate`/`GateDecision` imports and code. Remove those two names from the package exports.

- [ ] **Step 4: Run all validation helper tests**

Run: `python -m pytest -q tests/test_validation_benchmark.py tests/test_validation_gate.py`

Expected: PASS after updating assertions for the new summary contract.

- [ ] **Step 5: Commit the helper migration**

```powershell
git add tests/test_validation_benchmark.py src/agent_for_business/validation_benchmark.py src/agent_for_business/__init__.py
git commit -m "refactor: remove validation gate decisions"
```

### Task 3: Build Comprehensive Records from Saved τ² Simulations

**Files:**
- Modify: `src/agent_for_business/retail_runner.py`
- Modify: `eval-scripts/run_final_benchmark.py`
- Modify: `tests/test_retail_runner.py`
- Modify: `tests/test_eval_scripts.py`

**Interfaces:**
- `RetailTaskRunner.evaluation_from_simulation(simulation)` becomes the public normalization entry point for old `reward_info` and newer `info.evaluation` payloads.
- `build_comprehensive_benchmark(simulations, expected_runs, deserializer, adapter, verifier)` returns `{"simulations": enriched_simulations, "summary": summary_dict, "records": record_dicts}`.
- The final script provides a lazy `SimulationRun.model_validate` deserializer at runtime and injects `SimulationTrajectoryAdapter` and `RetailPolicyVerifier` into the public builder.

- [ ] **Step 1: Write the failing end-to-end conversion test**

Add fake τ²-shaped payloads and collaborators to `tests/test_eval_scripts.py`. Assert that the builder passes an adapted trajectory to the Verifier, stores `verifier_reward`, `policy_violation`, `db_match`, and `communication_ok` in the enriched simulation, and aggregates the official τ² reward separately from Verifier reward.

```python
def test_build_comprehensive_benchmark_verifies_serialized_simulations():
    class FakeSimulation:
        task_id = "5"
        trial = 0
        seed = 300
        info = {"terminal_state": {}, "evaluation": {"task_success": True}}

        def get_messages(self):
            return []

    seen = []

    class FakeAdapter:
        def from_simulation(self, simulation, *, terminal_state, evaluation):
            return SimpleNamespace(task_id=simulation.task_id, evaluation=evaluation)

    class FakeVerifier:
        def verify(self, trajectory):
            seen.append(trajectory)
            return VerificationResult(
                task_success=True, policy_violation=False,
                first_error=None, reward=0.75, reward_valid=True,
                db_match=True, communication_ok=False, tool_error_count=0,
            )

    report = module.build_comprehensive_benchmark(
        [{"task_id": "5", "trial": 0, "reward_info": {"reward": 1.0}}],
        expected_runs=1,
        deserializer=lambda payload: FakeSimulation(),
        adapter=FakeAdapter(),
        verifier=FakeVerifier(),
    )

    assert len(seen) == 1
    assert report["summary"]["tau_reward_mean"] == 1.0
    assert report["summary"]["verifier_reward_mean"] == 0.75
    assert report["simulations"][0]["verifier"]["db_match"] is True
    assert report["simulations"][0]["verifier"]["communication_ok"] is False
```

- [ ] **Step 2: Run the conversion test and verify it fails**

Run: `python -m pytest -q tests/test_eval_scripts.py::test_build_comprehensive_benchmark_verifies_serialized_simulations`

Expected: FAIL because the builder and public evaluation-normalization method do not exist.

- [ ] **Step 3: Implement the minimal conversion path**

Add the public `RetailTaskRunner.evaluation_from_simulation` method and have `run()` call it. In the final script, add the project `src` directory to `sys.path` only when needed, import the adapter/verifier/record types, and implement the builder loop:

```python
for simulation_payload in simulations:
    tau_reward = _reward(simulation_payload)
    try:
        simulation = deserializer(simulation_payload)
        info = getattr(simulation, "info", {}) or {}
        evaluation = RetailTaskRunner.evaluation_from_simulation(simulation)
        evaluation.update(info.get("evaluation") or {})
        evaluation = RetailTaskRunner.normalise_evaluation(evaluation)
        trajectory = adapter.from_simulation(
            simulation,
            terminal_state=info.get("terminal_state", {}) or {},
            evaluation=evaluation,
        )
        verification = verifier.verify(trajectory)
        record = BenchmarkRecord.from_verification(
            task_id=str(simulation_payload.get("task_id", "unknown")),
            trial=simulation_payload.get("trial"),
            tau_reward=tau_reward,
            termination_reason=simulation_payload.get("termination_reason"),
            result=verification,
        )
    except Exception as exc:
        record = BenchmarkRecord(
            task_id=str(simulation_payload.get("task_id", "unknown")),
            trial=simulation_payload.get("trial"),
            tau_reward=tau_reward,
            tau_reward_valid=tau_reward is not None,
            verifier_valid=False,
            verifier_error=f"{type(exc).__name__}: {exc}",
            termination_reason=simulation_payload.get("termination_reason"),
        )
```

Copy each payload before adding a JSON-safe `verifier` dictionary. Build the summary with `BenchmarkSummary.from_records(records, expected_runs=expected_runs)` and return enriched simulations, record dictionaries, and summary.

- [ ] **Step 4: Run the conversion test and the adapter/runner tests**

Run: `python -m pytest -q tests/test_eval_scripts.py::test_build_comprehensive_benchmark_verifies_serialized_simulations tests/test_tau_adapter.py tests/test_retail_runner.py`

Expected: PASS.

- [ ] **Step 5: Add the conversion-error test and implement it**

Use a deserializer that raises `ValueError("bad serialized simulation")`. Assert the returned summary has `verifier_invalid_count == 1`, the row contains the error text, and the official τ² reward remains counted independently.

- [ ] **Step 6: Commit the conversion path**

```powershell
git add tests/test_eval_scripts.py tests/test_retail_runner.py src/agent_for_business/retail_runner.py eval-scripts/run_final_benchmark.py
git commit -m "feat: verify saved tau simulations in benchmark"
```

### Task 4: Render Correct JSON and Markdown Summaries

**Files:**
- Modify: `eval-scripts/run_final_benchmark.py`
- Modify: `tests/test_eval_scripts.py`

**Interfaces:**
- `write_benchmark_summary(...)` reads the enriched `payload["benchmark"]["summary"]` and writes an explanation-rich Markdown report.
- The output JSON retains raw τ² fields and adds top-level `benchmark` plus per-simulation `verifier` details.

- [ ] **Step 1: Write the failing Markdown contract test**

Create a small enriched JSON fixture with one successful τ² run and one policy-violating run. Call `write_benchmark_summary(...)` and assert the Markdown contains `τ² Metrics`, `Verifier / GRPO Metrics`, `Metric Definitions`, `tau_reward`, `verifier_reward`, `db_match_rate`, `communication_rate`, `DB reward`, `COMMUNICATE`, and the `-1.0` policy-violation explanation. Assert it does not contain `gate_decision` or `SFTValidationGate`.

- [ ] **Step 2: Run the Markdown test and verify it fails**

Run: `python -m pytest -q tests/test_eval_scripts.py::test_markdown_summary_explains_tau_and_verifier_metrics`

Expected: FAIL because the current summary only has the old τ² table and no Verifier section or definitions.

- [ ] **Step 3: Implement the Markdown sections**

Render these sections in order:

1. Configuration.
2. Overall Results with expected/completed/incomplete runs, τ² valid reward count/rate, τ² success count/rate, and mean τ² reward.
3. τ² Metrics with DB match and communication true/present/missing/rate values.
4. Verifier / GRPO Metrics with evaluated/invalid counts, valid reward count/mean, policy violations/rate, tool-error runs/rate/total, and first-error counts.
5. `Metric Definitions` immediately after the aggregate metric tables. Explain the denominator and calculation for every displayed metric, including `tau_reward = product(enabled reward components)` and `verifier_reward = base evaluation reward - tool penalties`, with a hard `-1.0` policy-violation reward.
6. Termination Reasons, Per-Task Results, and Failed or Invalid Runs.

Use `-` for undefined means/rates and never coerce missing DB/communication checks to false. Remove the old `Average reward`-only table implementation and all gate wording.

- [ ] **Step 4: Run focused final benchmark tests**

Run: `python -m pytest -q tests/test_eval_scripts.py`

Expected: PASS.

- [ ] **Step 5: Commit JSON/Markdown output changes**

```powershell
git add tests/test_eval_scripts.py eval-scripts/run_final_benchmark.py
git commit -m "feat: render comprehensive benchmark report"
```

### Task 5: Remove Stale Gate References and Verify the Repository

**Files:**
- Modify: `docs/progress.md`
- Modify: `README.md` or other documentation files returned by the reference search
- Modify: tests whose names/assertions still describe the deleted gate

- [ ] **Step 1: Search for stale gate API references**

Run: `rg -n "SFTValidationGate|GateDecision|gate_decision|sft_ready_for_grpo|validation gate" src tests docs README.md`

Expected: no runtime/API references remain; historical design prose may mention the migration only when it clearly describes the old behavior.

- [ ] **Step 2: Update documentation and test names**

Describe the system as a comprehensive benchmark report. State that it reports facts and separate metrics and does not automatically approve GRPO.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m pytest -q`

Expected: PASS with no failures.

- [ ] **Step 4: Run static checks on changed Python files**

Run: `python -m compileall -q src eval-scripts tests; ruff check src eval-scripts tests`

Expected: no syntax errors and no Ruff violations. If Ruff is unavailable, report that fact and retain the successful compile/test results.

- [ ] **Step 5: Inspect the final diff and status**

Run: `git diff --check; git status --short`

Expected: no whitespace errors; only intended benchmark files are changed in addition to the user's pre-existing worktree changes.

