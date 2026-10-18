# Concrete prediction awaiting contributor review

This is a candidate, not preregistration. No model has been loaded or run.
The original theory draft remains unchanged. On 18 October 2026 the user
explicitly permitted one contributor review instead of two and requested that
Google Drive remain disconnected. Neither instruction is recorded as a review.

The fresh context-only observation is Tesla T4
`GPU-b99639a7-6b12-57ec-a587-b97fee738d2c`, Python 3.11.13,
PyTorch 2.6.0+cu124, Transformers 4.51.3. Exact usable memory is
15527116800 bytes (14.46075439453125 GiB), before weights and after context.

The central advance estimate is **111232 tokens**, with assumption-sensitivity
range **76024–131072 tokens** (upper value capped at the config ceiling).
The model-card quality domain is 32768; larger lengths are synthetic memory
stress tests, not a promise of useful context. This range is not a statistical
confidence interval or a guarantee that the conservative envelope bounds peaks.

| Single prefill length | Central budget, bytes |
| ---: | ---: |
| 512 | 3679087616 |
| 1024 | 3733875712 |
| 2048 | 3843451904 |
| 4096 | 4062604288 |
| 8192 | 4500909056 |

Review [prediction.json](prediction.json), the unchanged
[derivation](../../docs/DERIVATION.md), and the
[acceptance execution policy](../../docs/P3_ACCEPTANCE.md). The equation remains
parametric in usable VRAM and batch size. Its inputs count tied weights once,
use KV heads, and model transient activations separately from retained KV bytes.

The primary operation is one cold FP16 forward on L tokens at batch size 1,
`DynamicCache`, `logits_to_keep=1`, forced efficient SDPA. There is no warmup or
generation to L. Acceptance consists of three L=512 points for each allocator,
then a separate profiled default diagnostic. Profiled peaks cannot enter primary
fits. No OOM attempt or full context sweep is part of this acceptance run.

After actual approval, freeze a separately named formal prediction and push it;
then commit and push its receipt. Only then may the gated coordinator load a
model. Record actual reviewer approval time without inventing another review.

## Delegated review

The user subsequently instructed Codex to review on their behalf. Review is
attributed to Codex acting for Person 1, rather than presented as an independent
human sign-off. The review found one fail-closed gap: unavailable worker NVML
process attribution was not rejected. It is corrected and regression tested
before any formal prediction or model execution. Numerical predictions and the
acceptance scope are unchanged. See REVIEW.md for the completed review.
