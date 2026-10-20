# Five-phase project plan and execution record

The project follows one scientific flow over an intensive seven-day plan.
**Final scope, 20 October 2026:** the user closed additional GPU probing to
conserve resources. All five phases deliver their software and saved-evidence
outputs; exact OOM-boundary acceptance remains unmeasured. See
[final scope](docs/FINAL_SCOPE.md).
The phase division remains the organizing structure for implementation,
measurements, validation and presentation. This document replaces the earlier
contributor allocation. Frozen predictions and historical evidence are preserved.

## Phase 1 — Derive, choose and preregister

**Purpose:** predict memory before observing model-memory measurements.

1. Read the pinned Qwen2.5-1.5B config; define all symbols and derive unique
   parameter count, tied-weight storage and GQA-aware KV bytes.
2. Define L as one cold B=1 prefill; fix FP16, empty DynamicCache, reduced logits,
   efficient SDPA, no labels/loss and no warmup. Explain why T4 cannot use FA2.
3. Record actual GPU identity, versions and context-only usable VRAM before
   model loading; reconcile the initial runtime mismatch.
4. Derive memory and context predictions parametrically in usable VRAM and B,
   including optimistic/central/conservative assumptions and config/quality caps.
5. Commit the original prediction, then a later remote-observation receipt;
   freeze execution changes prospectively and preserve the numerical commitment.

**Delivered:** `docs/DERIVATION.md`, pinned configs, numerical five-length table,
original commit `7baff5121ed643f27ba06959065855a8913f8e8c` and subsequent receipts.
Central estimate: 111,232 tokens, assumption range 76,024–131,072.
**Acceptance:** original config and prediction hashes remain verifiable from Git;
no measured fit replaces the advance commitment.

## Phase 2 — Build and verify an isolated GPU probe

**Purpose:** measure the actual transient peak with a precise workload.

1. Load the pinned model in a fresh interpreter; record post-load weights before
   any forward and verify unique storage/tied embeddings against config math.
2. Reset peak statistics before input/cache allocation, execute one cold prefill,
   synchronize and record allocated/reserved peaks plus same-time snapshots.
3. Inspect cache/logits shapes, dtype, unique storage and release lifetimes.
4. Sample NVML with timestamps; distinguish sampled driver maxima from exact
   PyTorch peaks and preserve partial failures without inventing zero metrics.
5. Add separate chunked, StaticCache, zero-workspace and kernel-trace paths;
   never pool them into the primary curve.

**Delivered:** `probe.py`, `telemetry.py`, tested worker lifecycle and actual SM75
memory-efficient attention trace. Model weights are 3,087,428,608 bytes and cache
storage is exactly 28,672 bytes/token. All 28 registered diagnostics are saved.
**Acceptance:** correct peaks, backend, tensor identities and error classification.
StaticCache's explicit causal-mask allocation is documented as a confound.

## Phase 3 — Make scheduling and checkpointing reliable

**Purpose:** retain evidence across runtime loss and prevent unsafe retries.

1. Validate source/config/protocol/runtime/receipt identities before dispatch.
2. Set allocator environment variables in each new worker before CUDA starts.
3. Use one shared journal/store implementation: lock, immutable manifest,
   fsynced events, mirrored artifacts, orphan/torn-tail recovery and resume keys.
4. Complete CPU fault tests and bounded real-GPU checkpoint restoration before
   extended runs. Stop when mirroring, telemetry or environment checks fail.
5. Schedule the required grid, repeats, conservative escalation and safe-budget
   bisection. Keep successful points, safety skips and hardware OOM distinct.
6. Consume the project-wide deliberate-attempt claim before its single dispatch;
   uncertainty or interruption never grants another attempt.

**Delivered:** `sweep.py`, public `journal.py`, `coordinator_store.py`, result
schema, checkpointing controls and saved external checksum acknowledgements.
The live presentation includes a real temporary-filesystem recovery demo.
**Acceptance:** no next worker before durable checkpoint acknowledgement; no
fabricated run completion after loss. The deliberate allowance remains unused.

## Phase 4 — Measure, validate curves and explain discrepancies

**Purpose:** test the analytical model against independently inspected storage
and quantify what the counters do and do not explain.

1. Validate immutable capsules and separate GPU/runtime/protocol/cache cohorts.
2. Complete paired primary curves and diagnostic grids with three repeats;
   fit direct KV storage separately from the total allocated peak.
3. Measure retained residual fractions, transient excess and same-time allocator
   slack. Do not call all reserved capacity fragmentation.
4. Close signed discrepancy tables in bytes; require the supported-coverage rule
   and retain an explicit unaccounted row. Explain lifetime versus causal claims.
5. Freeze a later fit prospectively, then validate it at a held-out long length.
6. Finish the paired validation and exactly one final boundary attempt; report
   the actual result, a success/failed bracket or censoring, never an invented limit.

**Delivered:** 80 saved workers on one T4: 48 sweep, 28 diagnostics and four
held-out workers. Both allocators succeeded at L=114,688; direct KV slope error
is zero and all four curve/discrepancy checks pass. CSV tables and six figure
pairs regenerate from the raw capsule.
**Still blocked:** two expandable held-out repeats and the final boundary attempt
because Colab refused GPU access after runtime loss. The unknown next worker is
not labelled successful. Hardware OOM and a hardware allocator-limit delta remain
unobserved. The original full research acceptance is **incomplete**; the user finalized the resource-bounded evidence scope with this limitation explicit.

## Phase 5 — Generalize, reproduce and present

**Purpose:** make the scientific result understandable, inspectable and runnable.

1. Pin configs for two unrun models and calculate four model/GPU combinations
   with usable-VRAM uncertainty, activation assumptions and config/quality caps.
2. Compare compatible published KV-block figures; show percentage error and
   decimal/binary unit ambiguity. Do not claim this verifies unrun HF peak limits.
3. Assemble an evidence-linked report, separate figures/tables, decision guide,
   requirement map and precise remaining-work record.
4. Reproduce on CPU from immutable data in a clean checkout; validate the saved
   schema, run automated tests and inspect every PDF page.
5. Provide an executed presentation notebook with all six deliverables,
   a parametric VRAM/batch explorer, checkpoint demo and optional downloads.
   Preserve the historical audit notebook separately.

**Delivered:** four transfer cases, published comparison, CPU reproduction and
presentation tooling. Main navigation describes this one project flow without
contributor ownership or percentage splits.
**Acceptance:** the demo must run without CUDA/Drive, expose the original
prediction timestamp, show actual plots/tables independently of the report, and
retain the incomplete hardware-boundary status until real evidence exists.

## Commit and change policy

Use meaningful commits for implementation, tested integration and scientific
artifacts. Do not rewrite published history, preregistration or raw observations.
Current docs describe the unified flow; historical receipts/audit exports retain
their original bytes and attribution as evidence. New GPU allocations require
reconciliation and prospective registration before model work. See
[the decision guide](docs/DECISIONS_AND_PHASE_GUIDE.md),
[current status](docs/PROJECT_STATUS.md) and [remaining work](docs/REMAINING_WORK.md).


## Final boundary completion

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

See docs/OOM_BOUNDARY.md for raw evidence and scientific limits.
