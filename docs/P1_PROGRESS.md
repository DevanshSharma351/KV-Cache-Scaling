# Person 1 progress

Updated 19 October 2026. Person 1 owns 55%; work is organized by
[DevP1_phases.md](../DevP1_phases.md). Names and work-duration estimates are not
used in the allocation.

| Phase | Status | Evidence / remaining gate |
| --- | --- | --- |
| 1 — Analytical foundation | Accepted under user-authorized delegated review | Pinned T4 runtime, formal prediction 7baff51 and later receipt 976158c pushed before model work |
| 2 — Primary probe/telemetry | Bounded real-GPU acceptance passed | Six fresh L=512 primary probes; exact weights/cache/logits; separate SM75 efficient-SDPA CUDA trace |
| 3 — Coordinator/safety | Bounded external-checkpoint acceptance passed | Seven unique worker PIDs, checkpoint before each next step, loss of both Colab copies restored, no duplicate point or OOM token use; continuous bulk B2 integration remains a handoff |
| 4 — Empirical analysis | In progress; 42 bulk probes verified | Paired five-length grid and exact KV slope pass; expandable high-length points, controls and hardware boundary remain open |
| 5 — Transfer/integration | CPU pipeline implemented and reproduced | Four config-only transfer cases, published KV-only comparison, plots and report generate; final integrated GPU evidence remains open |

Latest bulk evidence: [42-probe interim checkpoint](../artifacts/experiments/2026-10-19-t4-interim/README.md).
104 CPU tests pass. Browser approval review capacity currently blocks retrieval of
primary-043; its cell must be inspected before any new dispatch. New local commits
remain unpushed while the active run requires remote HEAD `16e5005`.

Historical acceptance evidence: [acceptance report](../artifacts/acceptance/2026-10-18-t4/README.md),
[verified raw capsule summary](../artifacts/acceptance/2026-10-18-t4/verification.json).
83 CPU tests passed locally and in Colab. Later sections preserve the historical
implementation sequence; their earlier open gates are superseded by the final
acceptance record below where explicitly stated.

## Completed Phase 1 work

- Verified pinned config identity/revision and SHA-256 before draft preparation.
- Added disjoint unique-weight counts, tied-embedding accounting and direct
  `kv_bytes(L,B)` calculations.
- Added exact integer-byte predictions, theoretical batch scaling, domain caps,
  feasibility flags and optimistic/central/conservative sensitivity scenarios.
- Added strict draft input checks, deterministic JSON/Markdown preparation,
  scientific-input/generator hashes and overwrite protection.
- Extended CPU tests and CI with draft-package generation. Verified preparation
  does not import torch or NVML.
- Generated the review package under
  [`prereg/drafts/prefill-v1-phase1/`](../prereg/drafts/prefill-v1-phase1/prediction.md).

## Verification performed locally

Environment: macOS, CPU only, repository virtual environment. No allocated CUDA
GPU was available in this workspace. No checkpoint download, model load, forward,
allocator intervention or OOM attempt was performed.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts
.venv/bin/kvcache predict --usable-vram-gib 14.5 --batch-size 2
.venv/bin/kvcache prepare-prereg --output prereg/drafts/prefill-v1-phase1
```

Result: **27 tests passed**, syntax compilation passed and both CLI commands
completed. Tests cover hand-derived counts, KV/batch scaling, budget inversion,
malformed configs, tampered inputs, unsafe protocol changes, infeasible budgets,
repeated generation, manifest hashes and overwrite refusal. The committed package
is checked against current generator/input hashes and can be regenerated into a
different directory without changing its bytes. The preparation command above
will deliberately refuse a second run into the already committed directory.

Analytical checks: 1543714304 unique parameters, 3087428608 FP16 weight bytes,
28672 KV bytes/token at B=1, a 6x naive-head error and 466747392 bytes of possible
tied-embedding double-count error. These are config-derived quantities, not
post-load measurements.

## Open Phase 1 handoffs

Received a user-pasted T4 `nvidia-smi` snapshot on 18 October 2026. The
[partial inventory](../prereg/environment/t4-inventory-2026-10-18.partial.json)
records Tesla T4, driver 580.82.07, displayed total 15360 MiB and no running
processes reported at that snapshot. Its displayed CUDA 13.0 is driver
compatibility information, not the installed PyTorch runtime. The source's time
zone was not supplied, so its timestamp is preserved without conversion.

The displayed 0 MiB usage does not establish a CUDA-initialized usable budget.
At that point UUID, versions and the context-only budget were missing. The later
[context snapshot](../prereg/environment/t4-context-2026-10-18.user.json) supplies
those fields: usable free bytes are 15527116800 (14.46075439453125 GiB), and the
reported process list contains only the collector PID. The fresh-runtime
collector had no model calls. A later directly operated Colab collection is recorded below.

The installed Python 3.13 / torch 2.11+cu130 / Transformers 5.18 differs from the
proposal. The [runtime reconciliation record](COLAB_RUNTIME_RECONCILIATION.md)
documents the source-level GQA/backend concern and
[conditional original-scenario calculation](../prereg/environment/t4-context-2026-10-18.conditional-prediction.json).
No formal prediction or receipt was frozen from mismatched runtime evidence.

An [auditable Colab notebook](../notebooks/02_p1_phase1_colab.ipynb) is prepared
with explicit commands, logs and evidence export; saved cells remain unexecuted.
Local validation passes 36 tests, including environment rejection/mismatch
checks, exact-byte budget calculations and command-trail success/failure/timeout
checks. The supplied notebook was subsequently operated directly through Chrome:
all 36 tests also passed there. A fresh isolated context worker collected the
same 15527116800 free bytes, with zero live allocated/reserved tensors and no
other GPU PIDs. The [run record and checksum-verified mirrored snapshot](../prereg/environment/colab-20261018T123226Z-ad6f415d/README.md)
preserve provenance separately from the original user snapshot. Installed API
inspection and both draft calculations completed; no model execution occurred.

The evidence ZIP was recovered through a visible notebook text-transfer cell
after browser download failures. The committed original archive matches its
printed checksum; all 32 manifest entries were verified. It preserves raw
command logs, package inventory, API signatures, draft and conditional report.
The executed `.ipynb` remains saved at the Colab link in the run record and has
not been retrieved as a local file. Later export/transfer cells are visible there;
the ZIP covers preparation through archive creation. The repository template
itself remains unexecuted.

1. **B1 / Person 2:** the user-reported inventory is received. Resolve the
   observed/proposed runtime mismatch, preserve full package/API metadata in the
   notebook, confirm no prior model execution and refresh the context inventory
   after any stack change/restart. Do not double-count context or merge GPU runs.
2. **B6 / Person 2:** independently review the derivation, tensor-lifetime
   assumptions, numerical tables, quality/config domain distinction and frozen
   execution/acceptance policy. Actual GPU/backend/expandable support is untested.
3. **A1 / Person 1:** apply that real usable budget to the parametric prediction,
   resolve review findings and prepare a separately named formal prediction.
4. Push the reviewed formal prediction; then record its SHA, remote evidence and
   reviewers in a subsequent receipt commit before model-memory measurements.

Until these are complete, the package status is
`DRAFT_THEORY_ONLY_NOT_PREREGISTERED` and GPU measurement authorization is false.
No reviewer or remote receipt has been filled in speculatively. The original inventory remains labelled user-reported; the later directly
observed collection is kept separately and is not a model-memory measurement.

## Phase 2 implementation

The [primary probe contract](P2_PROBE_CONTRACT.md) documents `probe.py`,
`telemetry.py`, `ProbeRequest`, CPU-only `validate-probe`, result persistence,
measurement scopes and A3/B2/B4 integration. The original pinned-runtime policy
is enforced; the observed newer Colab stack is not silently adopted.

Implemented cold-forward ordering, separate load/forward peaks, UUID mapping,
weight/tied-storage inspection, cache formula/shape validation, separate
output/input/cache release snapshots, bounded sampled driver telemetry and
structured partial failures. Public execution was kept closed at the Phase 2
boundary; Phase 3 now wires the receipt/launch checks described below. CPU tests
use explicit doubles; no fake
measurement records are committed.

Verification performed locally on macOS: **55 CPU tests passed**; syntax
compilation passed. No Colab execution is claimed for this phase. A fresh CPU
interpreter imported the new probe/telemetry modules without importing torch or
NVML. `validate-probe --request configs/probe.example.json` returned the explicit
non-authorizing status and 14680064 theoretical KV bytes for L=512.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts
.venv/bin/python -m kv_cache_scaling validate-probe --request configs/probe.example.json
```

Real L=512 smoke, efficient-kernel trace, allocator capability checks and
persistent process-exit validation remain open after preregistration. Phase 2 is
not claimed fully accepted from CPU checks. No Colab tabs, access prompts,
package installs or model downloads were needed for this implementation.

## Phase 3 implementation

See [the coordinator contract](P3_COORDINATOR_CONTRACT.md). A3 now implements
CPU/Git receipt validation, immutable prediction/input/code checks, live pushed
head comparison, durable launch and duplicate-worker claims, per-point fresh
processes, child-only allocator configuration, bounded streamed logs, timeouts,
mirror-before-advance checkpoints, torn/orphan recovery, point-key resume,
three repeats, original-model admission and 256-token safe-budget bisection.
The global deliberate token is outside run/allocator folders and consumed before
launch. Success/interruption still consumes it; accidental OOM stops escalation.

The A3 reference adapter leaves Person 2's `journal.py` intact. B2 must review the
schema mapping and actual persistent filesystem behavior. No distributed locking
claim is made for independent Colab hosts or unverified cloud mounts. Actual
package inventories remain in per-worker records; the manifest references them.

Verification was local, with **77 CPU tests passed**, including synthetic Git
receipts, subprocess isolation, timeout/log bounds, mirror failure, torn recovery,
resume, global ledger reuse refusal and deliberate success/OOM distinctions.
No tests contacted GitHub or used CUDA. No model measurements, formal review or
preregistration evidence were invented. The current template remains incomplete;
the CLI rejects it. Real persistence/session-loss tests and GPU smoke/kernel/
allocator checks remain open after the real preregistration handoff.

## Commit boundaries

- `f8d1f53` — five-phase plan and navigation, pushed before implementation.
- `0bd78d0` — analytical validation, draft generator, focused tests and CI step.
- A subsequent documentation/artifact commit records this draft and progress.
  It is not a formal preregistration receipt.

Future commits follow the phase plan and preserve the original prediction and
raw data. No published history was rewritten.

## Completion work in progress: pinned runtime and bounded acceptance

The [fresh preparation](../prereg/environment/colab-20261018T141242Z-a337a6c3/README.md)
resolves the runtime mismatch: Python 3.11.13, torch 2.6.0+cu124, Transformers
4.51.3. All 77 tests at 072293c passed in that Colab runtime. No model was run.
The context inventory and a verified 22-member preparation archive persist
locally. Actual package and API evidence is preserved.

The user explicitly changed the review requirement to one contributor and
requested no Drive mount. New [acceptance controls](P3_ACCEPTANCE.md) restrict
execution to L=512, separate kernel-profile records, and support one-probe
external checkpoint barriers. The [concrete review candidate](../prereg/review-2026-10-18/README.md)
contains the actual-budget numerical prediction and full execution hashes.
A policy change is not recorded as an actual review. Formal push/receipt and
real GPU acceptance remain pending until approval of this candidate.

## Final Phases 1–3 acceptance, 18 October 2026

The user authorized one contributor review and delegated the technical review
to Codex on behalf of Person 1. The frozen review records that attribution; it
does not claim independent human or Person 2 approval. Review found and fixed
unavailable NVML process attribution before any model execution. All 83 tests
passed in the pinned Colab runtime before model work.

Prediction 7baff51 and later receipt 976158c were pushed and verified before
model attempts. An initial preflight failure was preserved: optional torchao
initialized CUDA during class imports. Removing unused torchao restored cold
imports without weakening the guard, changing scientific code or changing the
frozen context budget. The failed run was stopped; a new reviewed run followed.

Run 4353f74e-31b3-4574-93b0-7b0dc2a2beb2 passed seven real fresh workers:
three default and three expandable L=512 primary repeats, then one separately
profiled diagnostic. Unique parameter storage matched 3087428608 bytes; cache
storage and retained-minus-released bytes were exactly 14680064 throughout.
Logits were FP16 (1,1,151936). The trace includes the SM75 fmha_cutlass efficient
CUDA kernel. All workers exited to idle; no OOM occurred or token was consumed.

Drive remained disconnected. Verified/fsynced local capsules provided the
external persistence barrier after every worker. Both Colab checkpoint trees
were deleted and restored from the local copy; resume preserved the first point
and advanced without duplication. Recorded Linux CPU process-exit, torn-journal
and synthetic-token checks passed. A full cloud session restart was not tested.
Unattended bulk execution and Person 2's continuous B2 persistence adapter remain
separate integration work; this acceptance does not sign off Person 2's work.

The [complete acceptance record](../artifacts/acceptance/2026-10-18-t4/README.md)
links raw archives, transfer timestamps, exact numerical results, notebook cells
and the CPU-only verification command. Phase 4 research criteria (five lengths,
slope, context limit, allocator limit delta and discrepancy closure) are not
claimed from L=512 acceptance. Phases 4–5 remain the next work.
