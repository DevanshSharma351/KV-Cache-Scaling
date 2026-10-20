# Targeted boundary continuation — prospective review

The user explicitly reopened the OOM objective after the resource-bounded delivery.
This plan supersedes the no-further-probes decision only for two fresh workers:

1. Default allocator, single prefill at 114,688 tokens, one repetition.
2. Only after success, idle verification, and an externally persisted complete checkpoint: default allocator at 131,072 tokens, the project's single deliberate attempt.

No diagnostics, repeated short sweep, or allocator rerun is scheduled. Earlier 80 core workers and two replication workers remain separate allocations; no pooling. The earlier validated peak fit schedules the first point by transfer, not by claiming a newly validated curve. The core allocation measured successful default 114,688 inputs; the second allocation only replicated 512. The new same-allocation success is mandatory before the deliberate attempt. A safe failure blocks all further probing, including the deliberate attempt.

The original 18 October prediction and its timestamp remain immutable. This is an informed continuation after observations, not a replacement blind prediction. The fresh context inventory and all worker/coordinator sources are frozen before any model load. Project-wide experiment identity and one-use allowance are preserved. Local fsynced evidence is the durable authority; Google Drive stays disconnected.

The coordinator retains receipt checks, fresh-process isolation, exact PyTorch forward peaks, per-worker allocator configuration, strict runtime/GPU identity checks, checkpoint journals, one-use launch claims, idle checks and typed CUDA OOM classification. Timeout, signal, frontend disconnection and generic errors never count as OOM. Uncertain dispatch consumes the allowance and cannot be retried.

Expected outcome: successful 114,688 and typed OOM at 131,072 establish an interval, not an exact token threshold. If 131,072 succeeds, the result is a lower bound at the model's configured stress ceiling; no additional attempt is authorized. No model probe follows the deliberate attempt. Allocator context-limit improvement remains unmeasured. A boundary interval does not identify a causal activation/fragmentation split.

Review performed by Codex under the user's standing authorization to review on their behalf; not independent human review. CPU checks cover ordering, single-use dispatch preconditions, failure and retry rejection; the full CPU suite passed 124 tests before registration.

## New allocation amendment

On 21 October (local time), the connected runtime supplied a new physical T4. The previous targeted anchor succeeded and its full checkpoint is preserved separately; no deliberate attempt was dispatched. This registration repeats only the mandatory same-allocation success before the one final attempt. Total prior successes: 83 across three distinct allocations. Original prediction and project-wide one-use allowance remain unchanged.
