# Resource-bounded replication on a separate T4

`closed-scope-checkpoint.zip` is the original exported final capsule, not a
repacked archive. SHA-256:
`080f8b7df8347753636c5a3f5bf0bf413cfef4850305511e4816832937f96fa5`.

It contains two successful fresh L=512 default-allocator workers, source/receipt
provenance, raw results, command logs, journals and `audit/scope-closure.json`.
GPU UUID: `GPU-440a7ac2-62c4-253c-ba99-111c2f1e8466`.
Execution source: `e2cacce` (full SHA in the raw manifest).
Pinned torch 2.6.0+cu124 / Transformers 4.51.3 / Python 3.11.13.

The user's resource-limit instruction superseded the planned full repeat sweep.
The control loop was interrupted at its input prompt after worker 2, outside any
GPU worker. The journal remains an honestly incomplete *original sweep*; it is
not relabelled `run_finished`. The closure record verifies idle state and declares
this bounded replication complete. No deliberate attempt was consumed. The two
observations must not be pooled with the 80-worker study on another GPU UUID.

See [final scope](../../../docs/FINAL_SCOPE.md). The published verification checks
these facts without claiming a completed OOM boundary or the old 83-worker gate.
