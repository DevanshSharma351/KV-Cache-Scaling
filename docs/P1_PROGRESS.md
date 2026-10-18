# Person 1 progress

Updated 18 October 2026. Person 1 owns 55%; work is organized by
[DevP1_phases.md](../DevP1_phases.md). Names and work-duration estimates are not
used in the allocation.

| Phase | Status | Evidence / remaining gate |
| --- | --- | --- |
| 1 — Analytical foundation | CPU deliverables and remote preparation archived; runtime reconciliation and formal preregistration pending | 36 Colab tests passed; fresh T4 snapshot and raw ZIP verified; runtime decision, B6 review and receipt still open |
| 2 — Primary probe/telemetry | Engine and CPU checks implemented; GPU acceptance pending | Cold-prefill engine, telemetry, strict request validation and failure tests; no real model measurements |
| 3 — Coordinator/safety | Coordinator and CPU fault checks implemented; real integration pending | Receipt/launch gates, fresh processes, mirrored reference adapter, bounded logs, safe bisection and shared attempt ledger; B2 mount/schema review and GPU acceptance remain open |
| 4 — Empirical analysis | Planned | No measured dataset exists |
| 5 — Transfer/integration | Planned | No transfer validation or final research report claimed |

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
