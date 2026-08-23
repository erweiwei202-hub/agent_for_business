# Comprehensive Benchmark Design

## Goal

Replace the Raw/SFT pass/fail gate with a benchmark report that combines the
official τ² Retail evaluation with the project's trajectory Verifier, and make
the final benchmark CLI emit that report in JSON and Markdown.

## Decisions

- Remove `SFTValidationGate` and `GateDecision`; the benchmark reports facts and
  does not make an automatic GRPO admission decision.
- Keep `src/agent_for_business/validation_gate.py` as the compatibility module
  for the benchmark aggregation types, but turn it into a reporting module
  rather than a gate.
- Do not invent a weighted composite score. τ² reward and the GRPO-compatible
  Verifier reward are reported as separate metrics; policy compliance and tool
  reliability remain separate diagnostic rates.
- Keep the official τ² run as the source of task execution and official reward.
  After the subprocess finishes, deserialize each saved `SimulationRun`, adapt
  it to the project's `Trajectory`, and run `RetailPolicyVerifier` over it.
- A verifier conversion or verification failure is reported explicitly as an
  invalid verifier result; it must not silently count as a clean run.

## Report model

The benchmark report will contain:

- run metadata: model label, task IDs, trial count, seed, and concurrency;
- official τ² metrics: run count, valid reward rate, success rate, average
  reward, and termination counts;
- project Verifier/GRPO metrics: the mean `VerificationResult.reward` used by
  GRPO, policy-violation rate, tool-error rate, verifier-invalid count, and
  first-error counts;
- per-task rows showing trial coverage, τ² rewards, Verifier outcomes, policy
  errors, tool errors, and termination reasons;
- explicit missing, infrastructure, and conversion-error counts.

Database-match and communication rates are not promoted to headline metrics.
They remain available in raw per-run Verifier details when supplied by the
trajectory, but they are not duplicated in the aggregate summary.

The Markdown summary will render these sections in the same order and will no
longer contain a `passed` or `gate_decision` section.

## Data flow

```text
τ² CLI subprocess
        │
        ▼
checkpoint/results.json
        │
        ├── official τ² metrics
        │
        └── SimulationRun deserialization
                    │
                    ▼
        SimulationTrajectoryAdapter
                    │
                    ▼
        RetailPolicyVerifier
                    │
                    ▼
        Comprehensive benchmark report
                    │
                    ├── user-facing JSON
                    └── user-facing Markdown
```

The existing `validation_benchmark.py` helpers remain available for direct
Raw/SFT validation experiments, but their output will use the new reporting
types and will not return a GateDecision.

## Error handling

- Missing or invalid τ² reward remains visible as `reward_valid=false` and is
  excluded from official reward averages.
- `VerificationResult.reward` is preserved as `verifier_reward` per run and
  aggregated as `verifier_reward_mean`; this is the reward signal used by the
  project's GRPO path, including policy penalties and tool-error penalties.
- Missing serialized messages or an inability to adapt a simulation produces a
  verifier-invalid row with an explicit error reason.
- Duplicate or missing task/trial keys are preserved in the report and counted
  as incomplete coverage; no result is silently discarded.
- The final benchmark subprocess exit code is preserved even when report
  materialization succeeds, so infrastructure failures remain observable.

## Testing

- Replace gate tests with aggregation tests for official τ² metrics, Verifier
  metrics, per-task grouping, invalid results, and error counts.
- Add an adapter test using a serialized τ²-shaped simulation and assert that
  its tool calls and tool results reach `RetailPolicyVerifier`.
- Add final benchmark tests proving the generated Markdown contains both τ²
  and Verifier sections and no Gate decision.
- Retain tests for checkpoint resume, output materialization, and command
  construction.
