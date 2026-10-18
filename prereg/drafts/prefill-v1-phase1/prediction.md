# Phase 1 prediction draft

**Status: DRAFT_THEORY_ONLY_NOT_PREREGISTERED**

CPU calculations only. This package does not authorize GPU measurements.
Actual GPU inventory, independent review, formal freeze and remote receipt are pending.

## Operational definition

Model: `Qwen/Qwen2.5-1.5B` at `8faed761d45a263340a0528343f099c05c9a4323`.
One cold FP16 prefill of L tokens, B=1, empty DynamicCache, logits_to_keep=1.
HF SDPA with EFFICIENT_ATTENTION only; no fallback or generation.
Config ceiling 131072; lengths above the card's 32768 are synthetic memory stress.

## Derivation

```text
q = query_heads * head_dim; k = kv_heads * head_dim
P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H  (tied embeddings)
W = 2P; KV(L,B) = 2 * B * L * N * kv_heads * head_dim * 2
P_tensor(L,B) = W + B * L * [kv_bytes_per_token_at_B1 + a]
P_budget(L,B) = P_tensor(L,B) + H0
L(V,B) = max(0, floor((V-W-H0) / (B * [kv_bytes_per_token_at_B1+a])))
```

N=layers; H=hidden; I=intermediate; V=vocabulary in the parameter equation.
In L(V,B), V is usable VRAM in bytes after context/other users, before weights.
a is the proposed B=1 activation bytes/token; H0 is fixed byte headroom.
The shape envelope adds full final KV to a transient allowance; phase lifetimes may not coincide.
Neither the high scenario nor its L estimate is a guaranteed admission bound.

## Config and unique weights

| Architecture field | Value |
| --- | ---: |
| layers | 28 |
| hidden | 1536 |
| intermediate | 8960 |
| query_heads | 12 |
| kv_heads | 2 |
| head_dim | 128 |
| vocab | 151936 |
| tied_embeddings | True |
| max_positions | 131072 |

| Disjoint component | Parameters | FP16 bytes |
| --- | ---: | ---: |
| input_embedding | 233373696 | 466747392 |
| untied_output_embedding | 0 | 0 |
| attention_projection_weights | 154140672 | 308281344 |
| attention_qkv_biases | 57344 | 114688 |
| mlp_weights | 1156055040 | 2312110080 |
| layer_norm_weights | 86016 | 172032 |
| final_norm_weight | 1536 | 3072 |
| **TOTAL** | **1543714304** | **3087428608** |

Tied embedding double-count error: 466747392 bytes.
Correct/naive KV slopes: 28672 / 172032 bytes/token.
GQA overestimate factor: 6x.

## Proposed sensitivity assumptions

These scenarios are assumptions, not measured coefficients or confidence intervals.
H0 covers fixed logits/buffers/workspace/allocator uncertainty; context is already deducted.

| Scenario | a (bytes/token at B=1) | H0 (bytes) |
| --- | ---: | ---: |
| optimistic | 54272 | 268435456 |
| central | 78336 | 536870912 |
| conservative | 120832 | 1073741824 |

## Required-length predictions

Canonical values below are exact integer bytes; nothing in this table was measured.

| L | KV bytes | Naive KV bytes | Scenario | Tensor peak bytes | Budget bytes |
| ---: | ---: | ---: | --- | ---: | ---: |
| 512 | 14680064 | 88080384 | optimistic | 3129895936 | 3398331392 |
| 512 | 14680064 | 88080384 | central | 3142216704 | 3679087616 |
| 512 | 14680064 | 88080384 | conservative | 3163974656 | 4237716480 |
| 1024 | 29360128 | 176160768 | optimistic | 3172363264 | 3440798720 |
| 1024 | 29360128 | 176160768 | central | 3197004800 | 3733875712 |
| 1024 | 29360128 | 176160768 | conservative | 3240520704 | 4314262528 |
| 2048 | 58720256 | 352321536 | optimistic | 3257297920 | 3525733376 |
| 2048 | 58720256 | 352321536 | central | 3306580992 | 3843451904 |
| 2048 | 58720256 | 352321536 | conservative | 3393612800 | 4467354624 |
| 4096 | 117440512 | 704643072 | optimistic | 3427167232 | 3695602688 |
| 4096 | 117440512 | 704643072 | central | 3525733376 | 4062604288 |
| 4096 | 117440512 | 704643072 | conservative | 3699796992 | 4773538816 |
| 8192 | 234881024 | 1409286144 | optimistic | 3766905856 | 4035341312 |
| 8192 | 234881024 | 1409286144 | central | 3964038144 | 4500909056 |
| 8192 | 234881024 | 1409286144 | conservative | 4312165376 | 5385907200 |

## Usable-VRAM sensitivity

Budgets are hypothetical inputs, not observations of an allocated GPU.
A cap is a domain limit, not an experimentally observed failure. Zero means no token fits.

| Usable GiB | Scenario | Uncapped L | Config-capped L | Quality-domain L |
| ---: | --- | ---: | ---: | ---: |
| 12 | optimistic | 114885 | 114885 | 32768 |
| 12 | central | 86541 | 86541 | 32768 |
| 12 | conservative | 58351 | 58351 | 32768 |
| 13 | optimistic | 127830 | 127830 | 32768 |
| 13 | central | 96575 | 96575 | 32768 |
| 13 | conservative | 65533 | 65533 | 32768 |
| 14 | optimistic | 140775 | 131072 | 32768 |
| 14 | central | 106609 | 106609 | 32768 |
| 14 | conservative | 72715 | 72715 | 32768 |
| 14.5 | optimistic | 147248 | 131072 | 32768 |
| 14.5 | central | 111626 | 111626 | 32768 |
| 14.5 | conservative | 76306 | 76306 | 32768 |

## Frozen-policy proposal

```json
{
  "protocol_version": "prefill-v1",
  "model_id": "Qwen/Qwen2.5-1.5B",
  "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
  "config_path": "configs/models/qwen2.5-1.5b.json",
  "dtype": "float16",
  "batch_size": 1,
  "mode": "single_prefill",
  "attention_implementation": "sdpa",
  "sdpa_backend": "EFFICIENT_ATTENTION",
  "cache_type": "DynamicCache",
  "logits_to_keep": 1,
  "seed": 42,
  "warmup": false,
  "allocator_backend": "native",
  "allocator_variants": [
    "default",
    "expandable"
  ],
  "required_lengths": [
    512,
    1024,
    2048,
    4096,
    8192
  ],
  "candidate_lengths": [
    16384,
    32768,
    65536,
    98304,
    131072
  ],
  "quality_context_limit": 32768,
  "memory_stress_limit": 131072,
  "repeats": 3,
  "boundary_resolution_tokens": 256,
  "safe_budget_fraction": 0.9,
  "deliberate_oom_attempt_budget": 1,
  "per_probe_timeout_seconds": 1200,
  "diagnostics": {
    "chunked_prefill": {
      "chunk_size": 512,
      "lengths": [
        512,
        2048,
        8192,
        32768
      ]
    },
    "static_cache": {
      "lengths": [
        512,
        2048,
        8192
      ],
      "capacity_equals_length": true
    },
    "eager_attention": false,
    "full_logits": false
  }
}
```

## Review and experiment gate

Preserve three fresh-process repeats, safe-budget admission, and the global single deliberate-attempt ledger.
Validate cache bytes within 1% and retained slope within 3%; keep peak regression separate.
Require same-metric discrepancy closure within max(1 MiB, 0.1% measured bytes) and >=90% supported attribution
for gaps >=10 MiB; below that, residual <=1 MiB with the ratio labelled ill-conditioned.
Actual GPU/backend/allocator support is still untested; no CUDA/model load occurred during generation.
Obtain context-only inventory and Person 2 review, then freeze a separately named prediction and push it.
Record its SHA/remote evidence in a subsequent receipt before model-memory measurements.

## Input and generator provenance

```json
{
  "inputs": {
    "configs/experiment.json": "7198d509c816c58c36a1e16f62a325d3ec4a2a6cb3c8a249d8b1ac10add31f1b",
    "configs/models/provenance.json": "f8528701215fef3b0a94a6647d30bde8c449827504fd97622769f3f9119a0a27",
    "configs/models/qwen2.5-1.5b.json": "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77",
    "docs/DERIVATION.md": "54508b29832dcd4be09cfd9f0902f28921d8cde6731c70d31e053f590968bc5f"
  },
  "generator": {
    "preregistration.py": "003379da5d8618b3e7a6ce46c3a3d7d9984ed42b85755bc8a10e4cbadc4eaaf0",
    "model_math.py": "9ab3fa6d5b03128c956f62df0b77ce60598ede2c6dade4fde7c4f9d78cc23c6b"
  }
}
```
