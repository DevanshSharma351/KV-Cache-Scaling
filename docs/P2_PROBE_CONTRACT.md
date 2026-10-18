# Phase 2: primary probe implementation and integration contract

Updated 18 October 2026. Person 1's A2 implementation is CPU-tested. It has **not**
been validated with a real GPU/model. No research measurement or deliberate OOM
was performed during development.

## Available code

- `probe.ProbeRequest`: strict CPU request validation, B=1 primary policy and
  protocol/config fingerprints; unknown JSON fields are rejected.
- `probe.validate_inputs`: checks exact config hash and supported experiment
  semantics before heavy imports. The pinned primary model and revision are
  unchanged.
- `probe._execute_probe`: internal cold single-prefill engine for A3 integration.
  It accepts an injected runtime only for CPU tests. Production must use its
  local imports and one call per fresh interpreter.
- `telemetry`: synchronized counter snapshots, physical UUID mapping, bounded
  NVML sampler and storage-deduplicated tensor metadata.
- `probe.write_probe_result`: exclusive JSON output, finite-value validation,
  flush/fsync and overwrite refusal. Torn/missing output must be handled by A3.

At the Phase 2 boundary, public execution was deliberately closed. Phase 3 now
wires the public entry to receipt verification, mirrored launch evidence and an
exclusive worker claim. No JSON boolean grants authorization. See the
[Phase 3 contract](P3_COORDINATOR_CONTRACT.md); the actual incomplete receipt still
blocks execution. The private engine is not a substitute for that launcher.

CPU request example:

```bash
kvcache validate-probe --request configs/probe.example.json
```

This prints `VALIDATED_REQUEST_NOT_AUTHORIZED` and expected cache bytes; it imports
no torch/NVML, loads no model and grants no authorization. The example uses the
observed T4 UUID and current scientific hashes solely to demonstrate validation.
A changed UUID/hash must be reconciled with the formal run, not blindly copied.

## Runtime policy

The engine currently targets the original proposal: Python 3.11, torch 2.6.0,
torch CUDA runtime 12.4 and Transformers 4.51.3. Metadata mismatches stop before
heavy GPU imports. The observed Python 3.13 / torch 2.11 / Transformers 5.18 Colab
stack is deliberately rejected. This implementation does not resolve or silently
change that proposal. A documented runtime decision remains necessary.

The worker compares CUDA's logical-device UUID to the requested physical UUID
and resolves NVML by UUID, honoring remapped device visibility without assuming
NVML index zero. CUDA must be uninitialized on entry. The native allocator is
required. Allocator variables are checked, never changed in the worker. For the
pinned 2.6 stack, expandable mode requires the older
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before interpreter startup;
the newer alias alone is insufficient. Conflicting aliases/options are rejected.
Unsupported-expansion warnings stop the variant before forward. Absence of a
warning is not proof that segment expansion occurred.

The exact model revision loads with safetensors, remote code disabled and SDPA,
then moves wholly to `cuda:0` in FP16. It verifies architecture, revision, unique
parameter count/storage and the embedding storage tie. Buffers are reported
separately and together with parameters using storage deduplication.

Implementation references: [PyTorch 2.6 CUDA UUID binding](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/cuda/Module.cpp),
[Transformers 4.51.3 Qwen2 forward](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py),
[DynamicCache](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/cache_utils.py),
and [SDPA adapter](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py).
These source checks do not establish T4 kernel correctness.

## Measured lifecycle implemented

1. Context-only baseline after tiny allocation/release and allocator cleanup.
2. Separate load-peak reset and exact pinned-model load; no warmup/forward.
3. Model/storage inspection, GC and allocator cleanup; post-load baseline.
4. Peak reset **before** inputs, positions and empty DynamicCache; start NVML
   sampling. Input token IDs use a seeded CPU generator then move to the GPU.
5. Exactly one inference-mode forward inside forced efficient-SDPA context,
   `logits_to_keep=1`, no mask/labels/hidden states/attention outputs.
6. Synchronize; read independent allocated/reserved peaks and after-forward
   snapshot. Validate `(1,1,vocab)` FP16 logits and cache identity, shape, populated
   length, dtype and unique backing bytes against theory (within 1%).
7. Delete output/input/position references; retained snapshot with cache live.
8. Delete cache; released snapshot. No `empty_cache()` inside this window.
9. Stop sampler, return CPU metadata and shut down NVML. The process must exit;
   it cannot be reused for another point.

## Result interface: `primary-probe-v1`

The result includes probe ID, L/repeat/allocator/attempt intent, scientific hashes,
worker code hashes, PID/timestamps, exact runtime/package versions and device
identity. `snapshots` contains only stages actually reached:
`context_only`, `post_load`, `after_forward`, `retained`, `released`.

Each snapshot records monotonic acquisition start/end, allocated/reserved/slack,
CUDA free/total, inactive split bytes and sequential NVML observations. Driver
usage minus reserved is explicitly a combined external/context/non-PyTorch term.
Unavailable process attribution is labelled unavailable, not an empty process list.
Observed other compute PIDs stop execution before load.

`peak_allocated_bytes` and `peak_reserved_bytes` are independent high-water marks
from the reset through forward completion. Never subtract them as a same-time
identity. Unreached peaks stay null. `load_peaks` uses a separate reset scope.
Weights, buffers, input/logits and cache metadata contain CPU numbers only.
Storage pointers are used transiently to deduplicate; no GPU tensor/view escapes
inside reports. Post-load allocation minus analytical weights is reported as a
mixed residual, not labelled activation memory.

The sampler starts before inputs and stops after release snapshots, a broader
scope than the exact forward peak. It retains at most 4096 observations, records
any eviction count and preserves the full-run sampled maximum. It reports its
requested interval and actual monotonic timestamps. Sampled NVML usage can miss
transients. Sampling failure invalidates a nominal success as `telemetry_error`.

Statuses include `success`, `cuda_oom`, `unsupported_backend`,
`unsupported_allocator`, `runtime_mismatch`, `timeout`, `telemetry_error` and
`error`. The exact exception type, stage, partial snapshots and available failure
counters are preserved. Error text alone never establishes CUDA OOM. A CUDA OOM
is labelled accidental for a safe request or `deliberate_request` for that intent;
**intent is not proof that the coordinator consumed the required ledger token**.
Only A3 may authorize that intent. No failure triggers a retry or recovery forward.

## Handoff and remaining acceptance

A3 must validate actual reviewed/pushed prediction and subsequent remote receipt,
validate code/input/runtime fingerprints, persist a start event before launch,
set env before a fresh interpreter, enforce timeout and process exit, validate
result schema/ID, mirror output and enforce the one global deliberate-attempt
ledger. A worker-raised timeout is distinct from a killed interpreter with no
JSON. The latter is a coordinator responsibility, never inferred as OOM.

B2 reviews schema/journal integration. B4 owns chunked/static and profiler helpers;
the primary implementation has no diagnostic mode or mathematical-backend
fallback. The coordinator should use the shared request, snapshots and result
writer, without copying model execution into a second implementation.

CPU tests exercise ordering, reference lifetimes, byte accounting, malformed
requests, hashes, aliases, identity/context drift, runtime mismatch, output
persistence and failure classification. Fake counters are test doubles and are
never published as experimental data. Before declaring Phase 2 fully accepted,
run the gated L=512 smoke, efficient-operator trace, both allocator support checks,
cache formula check and clean process/persistent-write checks on real hardware.

## Real acceptance outcome, 18 October 2026

The earlier pending-GPU status is superseded by the
[verified bounded acceptance report](../artifacts/acceptance/2026-10-18-t4/README.md).
All 83 CPU tests and seven real T4 workers passed, including both allocators,
exact cache/weight/logits checks, an actual SM75 efficient-attention kernel trace
and external checkpoint restoration. No OOM attempt was used. The user authorized
a delegated one-contributor technical review and kept Drive disconnected.
The operator-controlled checkpoint path is validated for the tested loss case;
full cloud session restart, unattended bulk persistence and full scientific
sweep results are not claimed. Person 2's continuous B2/B4 integration remains
a handoff. Frozen scientific and execution files were not edited after model
work; original predictions and raw failures remain intact.
