# Preregistered T4 prediction

Frozen before any model load or model forward. Review: Codex acting on behalf
of Person 1, explicitly delegated by the user, under the user-authorized
one-contributor review policy. This is not an independent human review.
The [completed review](../review-2026-10-18/REVIEW.md) is hashed with this
commitment. Original theory drafts and historical environment records are retained.

## Operational experiment

Qwen/Qwen2.5-1.5B at revision
`8faed761d45a263340a0528343f099c05c9a4323`, FP16, batch size 1.
L means one cold prefill forward on L tokens with an empty DynamicCache,
`logits_to_keep=1`, HF SDPA and forced `EFFICIENT_ATTENTION`. No warmup,
generation to L, FlashAttention-2, eager fallback or full-sequence logits.
Every point runs in a fresh process. Allocator configuration is set before
interpreter startup. Peak counters reset before input/cache construction;
post-forward resting allocation is a separate observation.

Python 3.11 / PyTorch 2.6.0+cu124 / Transformers 4.51.3.
Tesla T4 UUID `GPU-b99639a7-6b12-57ec-a587-b97fee738d2c`, driver 580.82.07.
The frozen context-only inventory records **15527116800 usable bytes**
(14.46075439453125 GiB), after context and before weights. Do not subtract
context again. Actual installed packages and API inspection are archived.

## Advance numerical commitment

Central config-capped memory estimate: **111232 tokens**.
Assumption-sensitivity range: **76024–131072 tokens**; optimistic raw estimate
146740 is capped at 131072. The useful/model-card domain remains 32768;
higher points are synthetic memory stress tests. The interval is not a measured
confidence interval or a guaranteed bound.

| Single prefill L | Central budget, bytes | Retained KV, bytes |
| ---: | ---: | ---: |
| 512 | 3679087616 | 14680064 |
| 1024 | 3733875712 | 29360128 |
| 2048 | 3843451904 | 58720256 |
| 4096 | 4062604288 | 117440512 |
| 8192 | 4500909056 | 234881024 |

## Derivation and usable-VRAM dependence

The pinned config gives N=28 layers, H=1536, I=8960, Q=12 query heads,
K=2 KV heads, D=128, vocabulary V=151936, tied input/output embeddings.
The [full derivation](../../docs/DERIVATION.md) defines each symbol and tensor
lifetime assumption. With q=QD, k=KD:

```text
P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H
P = 1543714304 unique parameters; W = 2P = 3087428608 bytes
KV(L,B) = 2 B L N K D * 2 = B L * 28672 bytes
M_budget(L,B) = W + B L (28672 + a) + H0
L_limit(U,B) = max(0, floor((U-W-H0)/(B*(28672+a))))
```

U is usable VRAM in bytes, already adjusted for context/other users. Weight
counting includes the tied matrix once; duplicating it adds 466747392 bytes.
Using Q in the KV formula overestimates it 6x. Fixed headroom is an assumption,
not measured fragmentation. Central a=78336 bytes/token, H0=536870912 bytes;
optimistic a=54272, H0=268435456; conservative a=120832, H0=1073741824.
The JSON preserves full component counts, scenario tables, B/L dependence,
VRAM sensitivity and all scientific/execution hashes.

## Acceptance, safety and discrepancy policy

The [frozen acceptance policy](../../docs/P3_ACCEPTANCE.md) initially allows only
three L=512 primary points per allocator plus a separate default kernel trace.
Profiled peaks cannot enter primary fits. No deliberate OOM or context-boundary
claim is part of acceptance. Primary bulk work retains the frozen 90% admission
policy, three repeats, 256-token safe-budget search and one global deliberate
attempt. Safe-budget crossings and actual OOM bounds are different quantities.

Cache storage must agree within 1%; later retained slope within 3%. Later
discrepancy accounting must close within max(1 MiB, 0.1% of measured bytes),
with >=90% supported attribution for gaps >=10 MiB. Smaller gaps require a
<=1 MiB residual and an ill-conditioned-ratio label. Unexplained bytes remain
explicitly unaccounted. Primary peak slope is not labelled KV slope.

Drive remains disconnected. Acceptance pauses after every new probe until the
operator verifies and fsyncs a complete checkpoint on the user's local disk.
Colab copies are ephemeral staging. This does not authorize unattended bulk
work without continuous, validated durable persistence.

A subsequent pushed receipt records this prediction's commit SHA and completed
review metadata. No model-memory measurement may precede that receipt.
