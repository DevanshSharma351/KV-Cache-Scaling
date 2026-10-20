# Final project status

All five implementation phases and six deliverable artifact types are integrated and reproducible. The single deliberate OOM check is now measured and checkpointed.

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

| Phase | Outcome |
| --- | --- |
| 1 — Derive and preregister | Original 18 October commitment and derivation preserved |
| 2 — Isolated measurement | Pinned runtime, cache, efficient SDPA, reduced logits, fresh workers, exact peak counters |
| 3 — Durable orchestration | Per-probe journals, external verified checkpoints, one-use OOM ledger |
| 4 — Explain memory | Core curve, direct KV slope, diagnostics, signed discrepancy table, separate observed OOM bracket |
| 5 — Generalize and present | Four transfer predictions, published KV comparison, report PDF, offline HTML and Colab board |

84 successful workers and one deliberate OOM are preserved across four separate allocations. See [final scope](FINAL_SCOPE.md) for experimental limits and [boundary evidence](OOM_BOUNDARY.md) for raw outcomes. No further GPU probes are planned.
