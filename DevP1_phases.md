# Person 1: five-phase implementation plan

Person 1 owns **55% of the complete project**. This plan divides that share into
five phases and preserves the one-week delivery target, scientific protocol,
Person 2's ownership and all acceptance criteria. Percentages are shares of the
whole project, not estimates of elapsed time or completed work.

## Phase map

| Phase | Focus | Task mapping | Project share | Target |
| --- | --- | --- | ---: | --- |
| 1 | Analytical foundation and preregistration preparation | A1 | 8% | Day 1 |
| 2 | Primary GPU probe and trustworthy telemetry | A2 | 13% | Days 2–3 |
| 3 | Isolated coordinator, safety and boundary estimation | A3 | 13% | Days 3–4 |
| 4 | Empirical cache validation and discrepancy accounting | A4 | 12% | Days 4–5 |
| 5 | Transfer predictions, scientific conclusions and final integration | A5 + A6 | 9% | Days 6–7 |
| | **Person 1 total** | **A1–A6** | **55%** | **One week** |

Phases describe deliverable dependencies rather than a prohibition on parallel
preparation. Start published-source discovery with Person 2 on day 2; define
analysis columns before measurements. Implementing CPU code is permitted before
preregistration. Loading a model for memory checks or running any model forward
is gated by the completed, reviewed and pushed prediction plus its receipt.

## Phase 1 — Analytical foundation and preregistration preparation

**Outcome:** a tested, reproducible CPU calculation and a reviewable prediction
package with explicitly labelled uncertainty. A formal GPU-specific prediction
is frozen only when the actual environment and both reviewers are available.

### Work

1. Verify the pinned config bytes against provenance; extract layer count,
   hidden/intermediate sizes, query/KV heads, head dimension, vocabulary,
   embedding tying and position ceiling. Reject incompatible or malformed input.
2. Derive unique weight parameters by component, including Q/K/V biases, two
   RMSNorms per layer and final norm. Count the tied embedding once. Calculate
   the hypothetical duplicate-embedding error independently.
3. Expose closed-form KV bytes for arbitrary positive integer L and B, the naive
   query-head calculation, and separate tensor-peak and budget estimates. Preserve
   the primary B=1 experiment and FP16 policy.
4. Document the activation-envelope assumptions, fixed headroom, cold prefill,
   explicit DynamicCache, reduced logits and forced efficient SDPA. Show why
   shape-inspired scenarios are sensitivity assumptions, not proven bounds or
   measured confidence intervals.
5. Generate numerical predictions at L=512,1024,2048,4096,8192, plus an L(V)
   sensitivity grid, uncapped/config-capped limits and the quality-domain cap.
   Keep byte units canonical and label displayed MiB/GiB.
6. Prepare a deterministic JSON + Markdown draft package with hashes of all
   scientific inputs and generator code. Refuse to overwrite existing output.
   Keep the status explicitly draft while GPU inventory/review is missing.
7. Hand off to B1 for the context-only GPU/usable-budget inventory and B6 for
   independent derivation review. Then freeze a separately named formal
   prediction, push it before model measurements, and record its remote receipt
   in a subsequent commit. Never invent hardware, review or timestamp evidence.

### Deliverables

- `model_math.py`: validated config-derived math and B/L-aware formulas.
- CPU CLI for reproducible prediction-package preparation.
- `docs/DERIVATION.md`: equations, parameter breakdown, assumptions and examples.
- Focused tests for tied weights, GQA, B/L scaling, prediction feasibility,
  input integrity and reproducible artifacts.
- A generated, explicitly draft package under `prereg/drafts/`.
- Formal prediction/receipt only after the real B1/B6 handoffs are complete.

### Exit criteria

- [x] Config provenance matches the committed snapshot.
- [x] Unique parameters = 1543714304; FP16 weights = 3087428608 bytes.
- [x] KV slope = 28672 bytes/token at B=1; naive slope is 6x larger.
- [x] Required-length and VRAM-sensitivity predictions regenerate exactly.
- [x] Tests cover component accounting, B/L scaling and invalid inputs.
- [x] Draft/frozen status and omitted empirical evidence are unambiguous.
- [ ] Actual GPU identity/usable-budget observation and independent review exist.
- [ ] Formal prediction is pushed; its later receipt precedes model measurements.

The CPU deliverables can be completed locally. Missing B1/B6 evidence leaves
formal preregistration open; it does not prevent CPU implementation of later
phases, and it never authorizes GPU measurements.

## Phase 2 — Primary GPU probe and trustworthy telemetry

**Outcome:** one fresh-process single-prefill worker whose memory counters and
cache inspection can be trusted.

### Work

1. Implement the lifecycle in `docs/MEASUREMENT_PROTOCOL.md` with locally imported
   torch/NVML, exact version/model checks and logical-to-physical GPU UUID mapping.
2. Capture context-only and post-load baselines; verify actual unique parameter
   storage, FP16 dtype, embedding tie and buffer differences against Phase 1.
3. Enforce `attn_implementation="sdpa"` and
   `SDPBackend.EFFICIENT_ATTENTION`; reject fallback or unsupported kernels.
4. Reset peaks before input/position/cache allocation, run one cold prefill with
   `logits_to_keep=1`, synchronize and collect independent allocated/reserved
   peaks plus same-time snapshots and sampled NVML observations.
5. Inspect unique cache storage while retaining its logical shapes; release
   outputs/inputs and then cache in distinct snapshots. Keep only CPU metadata
   from inspection so a view cannot accidentally retain GPU tensors.
6. Classify CUDA OOM, unsupported backend, timeout and ordinary errors separately.
   Save partial evidence and exit after failure.
7. Land a stable primary-probe interface for Person 2's B4 chunked/static helpers.
   Review those helpers' measurement semantics without taking over their work.

### Deliverables and checks

- Primary `probe.py`, `telemetry.py`, structured request/result interface and
  version/backend/cache tests. Use mocks for CPU error/lifecycle tests.
- After preregistration: small real-GPU smoke record, efficient-kernel diagnostic,
  correct `(1,1,vocab)` logits and L-token cache evidence.
- At each synchronized snapshot reserved >= allocated; the tracked forward peak
  covers the live allocation; primary cache bytes agree with the formula within 1%.
- Diagnostic/profile results stay separate from primary measurements. No test
  consumes the project's deliberate OOM allowance casually.

**Handoff:** B2 validates the emitted schema; B4 adds diagnostic paths; Phase 3
uses the same worker entry point rather than duplicating model execution.

## Phase 3 — Isolated coordinator, safety and boundary estimation

**Outcome:** a CPU-only driver that safely schedules fresh workers and resumes
from durable results without changing the scientific experiment.

### Work

1. Validate the formal prediction receipt and protocol/model/config fingerprints.
   Reject a draft, mismatched environment or incomplete preregistration.
2. Launch a new interpreter for each point/repeat with allocator env set before
   import/CUDA initialization. Check default/expandable support and pair runs on
   the same GPU UUID with comparable context-adjusted available memory.
3. Integrate Person 2's B2 journal/lock/ledger APIs: durable start/result events,
   persistent mirroring, torn-line recovery, point-key resume and bounded logs.
4. Implement the conservative admission envelope, required grid, three repeats,
   safe escalation and 256-token safe-budget bisection. Save the original advance
   prediction separately from any calibrated scheduling estimate.
5. Track largest success, observed OOM upper point, estimated boundary interval
   and safe-budget crossing separately. Preserve model-card/config domain labels.
6. Consume the one global deliberate-attempt token before launch. On success,
   interruption or censoring, report the actual outcome without retrying. Stop
   escalation on an accidental OOM or material environment drift.

### Deliverables and checks

- `sweep.py`, CLI integration, scheduling manifest and worker timeout behavior.
- CPU synthetic-worker tests for restart, duplicate keys, malformed output,
  timeout, env isolation, interrupted launch and attempt-token reuse prevention.
- B2 integration tests prove persistence before starting bulk GPU work.
- Real paired safe sweeps use the frozen protocol; B4 owns run operation and
  coverage. Missing/failed/skipped points remain visible, never invented successes.

**Exit gate:** safety, persistence and isolation checks pass; bulk execution can
begin. An exact hardware threshold is not claimed from safe probes alone.

## Phase 4 — Empirical cache validation and discrepancy accounting

**Outcome:** quantitative evidence separating retained KV growth, transient
memory, allocator slack and the supported explanations of prediction error.

### Work

1. Validate and normalize immutable raw records, recording source hashes and code
   provenance. Separate protocol, mode, allocator, cache policy and repeats.
2. Regress directly inspected cache storage and corrected retained allocation
   against L. Report slope/intercept, R², repeat uncertainty and error from theory.
   Peak-vs-L is a separate total-memory fit, not a KV-cache measurement.
3. Calculate retained residual fractions, transient excess and direct measured
   weight/KV differences. Use lifetime/diagnostic evidence for activation claims.
4. Produce same-time allocator-slack tables and paired safe-limit deltas, with
   uncertainty/censoring. Do not call all cached capacity fragmentation or subtract
   unrelated maxima to create a peak-time accounting identity.
5. Compare advance tensor-peak and budget predictions against matching measured
   quantities at fixed L. Attribute supported deltas and retain an explicit
   unaccounted row; provide independent evidence instead of naming the remainder.
6. Supply stable processed tables to B3 and scientific methods/results to B5.
   Review the dynamic/chunked/static comparison without pooling incompatible fits.

### Deliverables and checks

- `analysis.py`, normalized/derived tables, cache regression and boundary summary.
- At least five primary lengths; cache formula error <=1%; slope error <=3%.
- Discrepancy closure within max(1 MiB, 0.1% of measured bytes); >=90% supported
  attribution for material gaps, with the existing small-gap rule preserved.
- If evidence fails a criterion, report that failure and the unexplained amount.
- All published figures trace back to saved data; B3 owns plotting implementation.

## Phase 5 — Transfer predictions, conclusions and final integration

**Outcome:** a coherent scientific contribution covering A5 and A6, integrated
into Person 2's independently reproduced final delivery.

### Work

1. Use B6's pinned config/source inventory for Qwen2.5-3B and Qwen2.5-7B; no
   checkpoint loads for either transfer model. Verify architecture-specific
   embedding/bias rules rather than applying a parameter formula blindly.
2. Predict four model/GPU combinations for A100-40GB and RTX 4090-24GB with
   parametric usable VRAM, activation assumptions, caps and transfer uncertainty.
3. Compare at least one compatible published quantity and report percentage error.
   A serving engine's KV capacity validates only the compatible KV submodel, not
   the primary HF single-prefill peak threshold.
4. Write scientific conclusions: what the advance prediction got right/wrong,
   what allocator intervention changed, what diagnostics show and what remains
   unaccounted. Keep planned, measured, calibrated and transferred results distinct.
5. Review B5's assembled report and B3's figures; resolve integration issues,
   verify that final claims map to evidence, and sign off the requirement checklist.
6. Preserve day 7 for necessary recovery/final fixes. Any new data triggers
   regeneration of affected tables, figures and conclusions.

### Deliverables and checks

- `generalize.py`, four transfer predictions, compatible cited comparison.
- Scientific report sections, limitations/deviation register and technical review.
- No claim that Person 1 owns or completed Person 2's report assembly,
  diagnostic implementation, plotting or independent reproduction.
- Final handoff includes input/code hashes, precise reproduction commands and
  all open criteria. Report unavailable verification honestly rather than passing it.

## Commit history and execution tracking

Use focused commits in this order; commit labels are examples, not pre-existing
history. Push the phase plan before starting implementation.

1. `docs(p1): define five-phase delivery plan` — this document and navigation.
2. `feat(p1-phase1): ...` — tested analytical/core package changes.
3. `docs(p1-phase1): ...` — derivation, generated draft and progress evidence.
4. `docs(prereg): freeze ...` — actual reviewed prediction only when ready.
5. `docs(prereg): record remote receipt ...` — subsequent commit before probes.
6. Later phases: separate probe, coordinator, analysis and transfer commits, each
   including relevant tests/evidence. Keep research-result commits distinct from
   fixes and preserve the original preregistration and raw observations.

Do not rewrite published history, amend a frozen prediction, commit credentials
or checkpoint weights, or combine unfinished phases into a misleading “done” commit.
Maintain `docs/P1_PROGRESS.md` with code checks, real evidence and outstanding
handoffs. A scaffold or passing CPU test does not establish GPU correctness.

**Current status (18 October 2026):** Phase 1 CPU implementation and draft are
complete; directly observed Colab preparation and checksum-verified raw evidence
are archived. Formal preregistration remains open pending runtime reconciliation,
independent review, prediction push and subsequent receipt. Phase 2's primary
engine and CPU validation are implemented; real GPU acceptance remains gated.
Later phases remain planned. See [progress evidence](docs/P1_PROGRESS.md) and
[the Phase 2 interface](docs/P2_PROBE_CONTRACT.md).
