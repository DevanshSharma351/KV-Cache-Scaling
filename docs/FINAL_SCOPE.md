# Final scope and evidence

The boundary objective was reopened by the user after the initial resource-bounded delivery. It is now completed with one real typed forward CUDA OOM, preceded by a same-allocation successful long prefill. No further GPU model work is planned.

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

## Delivered work

All six required artifact types are available: symbol-defined derivation; original timestamped preregistration; checkpointing fresh-process harness; six analytical-overlay scientific figures; signed byte-level discrepancy tables; and four cross-model/GPU predictions with conditional published KV verification. The five phases remain documented. The final notebook presents the raw boundary outcome as well as the original core curves.

## Cohorts and counts

There are 84 preserved successful GPU workers plus one deliberate forward OOM: 80 core-study successes, two separate L=512 replication successes, one first boundary anchor on the subsequently lost T4, and one success plus the OOM on the final T4. Four physical GPU allocations remain separate. No old repeat counts or regression data were filled with another allocation.

## Honest acceptance limits

The exact per-token limit is bracketed, and the hardware-limit gain from expandable segments is unmeasured. Two original-allocation expandable held-out repetitions remain absent. Fixed-length byte-accounting closure is not a causal decomposition of the threshold error. Cross-system context estimates are conditional; published validation checks KV arithmetic. These limits remain visible in the report and presentation. The original strict 83-worker/cohort verifier has not been weakened; the final-scope verifier additionally verifies the new one-use boundary evidence.

See [measured boundary](OOM_BOUNDARY.md), [phase decisions](DECISIONS_AND_PHASE_GUIDE.md), and [live demo](LIVE_DEMO.md).
