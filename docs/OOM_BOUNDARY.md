# Measured OOM boundary

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

## What was actually run

| Tokens | Result | Allocation |
| ---: | --- | --- |
| 114688 | Success | GPU-8fd0487b-72ce-c076-4966-c694a52185f8 |
| 131072 | Typed CUDA OOM during forward | Same allocation |

The original central prediction was 111232 tokens. Under a monotone threshold assumption, the largest fitting integer lies in [114688, 131071], so signed relative prediction error lies in [-15.136071289606397, -3.013392857142857] percent. The original prediction was conservative. No midpoint is presented as measured.

## How the check was kept small

The 80-worker core curve was reused for scheduling, without pooling GPU UUIDs. Two independent L=512 replications and a saved first boundary anchor were retained. Colab later reassigned the T4, requiring the same-allocation success to be checked again. Only one final deliberate attempt was dispatched, after its success checkpoint and external one-use intent were fsynced locally. Each worker used the pinned cold-prefill protocol, a fresh process and peak reset. Both workers exited and returned the GPU to idle. No model probe followed the OOM. Drive stayed disconnected.

## What the byte accounting establishes

The main discrepancy tables account for the measured completed-forward tensor-peak gap at fixed L, with explicit supported and unaccounted rows. The failed forward only supplies partial peak counters: {"allocated_bytes": 3493096960, "reserved_bytes": 3690987520, "partial_peak_allocated_bytes": 3493096960, "partial_peak_reserved_bytes": 3690987520, "scope": "prefill_inputs_through_forward"}. It cannot supply the completed-forward peak that would have occurred with more VRAM. The failure message records the failed allocation and allocator state; it does not causally isolate fragmentation from kernel workspace or activations. That causal threshold-error split remains unaccounted; it is not hidden in the fixed-length closure.

## Evidence

- [Boundary capsule](../artifacts/experiments/2026-10-21-boundary/checkpoint.zip)
- SHA-256: `aa5ac338423d0d2f0595fbac3b662ffc10dacb12bbc44f9b097d5eb69450c18b`
- [Prospective registration](../prereg/boundary-2026-10-21/prediction.json) and [later receipt](../prereg/boundary-2026-10-21/receipt.json)
- [Coordinator](../scripts/targeted_boundary.py)
- [Verifier](../src/kv_cache_scaling/boundary.py)

## Experimental limits

Exactly one deliberate OOM is compatible with a boundary interval, not a repeated crash-based binary search to an exact token. A paired default/expandable hardware-limit delta and the two missing original-allocation expandable repeats were not manufactured using later UUIDs. Transfer verification remains a published KV-block comparison, not an unrun HF threshold measurement.
