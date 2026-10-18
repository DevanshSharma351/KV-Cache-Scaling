# Person 1 progress

Updated 18 October 2026. Person 1 owns 55%; work is organized by
[DevP1_phases.md](../DevP1_phases.md). Names and work-duration estimates are not
used in the allocation.

| Phase | Status | Evidence / remaining gate |
| --- | --- | --- |
| 1 — Analytical foundation | CPU deliverables complete; formal preregistration pending | Tested math/generator and committed draft; actual B1 inventory and B6 review still required |
| 2 — Primary probe/telemetry | Planned | No CUDA model measurements performed |
| 3 — Coordinator/safety | Planned | Journal integration and receipt enforcement remain implementation work |
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
UUID, installed versions, allocator env, context-only exact free/total bytes and
the no-prior-model confirmation remain missing. No numerical prediction or
preregistration receipt was altered using this partial evidence.

1. **B1 / Person 2:** supply the allocated GPU's identity/UUID, driver, Python,
   torch/CUDA/transformers versions, allocator configuration and a context-only
   free/total-memory snapshot before model load/forward. Record other users or
   unknown external usage. Derive usable VRAM without double-counting context.
2. **B6 / Person 2:** independently review the derivation, tensor-lifetime
   assumptions, numerical tables, quality/config domain distinction and frozen
   execution/acceptance policy. Actual GPU/backend/expandable support is untested.
3. **A1 / Person 1:** apply that real usable budget to the parametric prediction,
   resolve review findings and prepare a separately named formal prediction.
4. Push the reviewed formal prediction; then record its SHA, remote evidence and
   reviewers in a subsequent receipt commit before model-memory measurements.

Until these are complete, the package status is
`DRAFT_THEORY_ONLY_NOT_PREREGISTERED` and GPU measurement authorization is false.
No reviewer, remote receipt or hardware observation has been filled in speculatively.

## Commit boundaries

- `f8d1f53` — five-phase plan and navigation, pushed before implementation.
- `0bd78d0` — analytical validation, draft generator, focused tests and CI step.
- A subsequent documentation/artifact commit records this draft and progress.
  It is not a formal preregistration receipt.

Future commits follow the phase plan and preserve the original prediction and
raw data. No published history was rewritten.
