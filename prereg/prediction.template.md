# Formal prediction template — NOT REGISTERED

Status: incomplete template. Replace this heading/status only when every field is
reviewed. Do not collect model memory measurements while this is incomplete.

## Identity and contract

- Protocol version and SHA-256:
- Model ID, exact revision, config SHA-256:
- Code commit containing prediction (referenced by later receipt):
- Contributors/reviewers:
- GPU identity and context-only budget observation (UTC):
- Dependency target and runtime lock reference:
- L definition: single prefill; batch 1; no warmup; no generation.
- FP16 weights/cache; explicit DynamicCache; logits_to_keep=1.
- SDPA EFFICIENT_ATTENTION only; unsupported kernel must fail.
- Primary domain <=131072; quality-domain summary <=32768.

## Derivation

Include config-derived unique parameter count, Q/K/V biases/norms, tied
embeddings, KV formula, GQA-vs-naive check, activation lifetime reasoning,
workspace/slack allowances, usable-VRAM definition and GB/GiB conversions.

## Required numerical predictions

Fill all cells. Predict tensor peak and total budget separately, in bytes and
MiB. Show optimistic/central/conservative values or ranges for each.

| L | KV bytes | Tensor-peak low / central / high | Budget low / central / high |
| ---: | ---: | --- | --- |
| 512 | | | |
| 1024 | | | |
| 2048 | | | |
| 4096 | | | |
| 8192 | | | |

Supply the explicit L(V_usable) equation and coefficients, with the meaning and
source of each uncertainty assumption. The current CPU scenarios are proposed
starting points, not automatically frozen facts.

| Usable VRAM (GiB) | Lower / central / upper uncapped L | Config-capped L | Quality-domain L |
| ---: | --- | --- | --- |
| 12.0 | | | |
| 13.0 | | | |
| 14.0 | | | |
| 14.5 | | | |
| Actual observed budget | | | |

## Frozen execution and acceptance

- Required grid, repetitions, diagnostics, pairing order and runtime timeout:
- Conservative safety envelope and 90% admission gate:
- Safe crossing resolution, prediction interval and censoring policy:
- Global single deliberate-attempt ledger; accidental OOM stop rule:
- Exact-one-OOM target and accepted/reportable exceptions:
- Peak window, input/cache allocation timing, snapshot lifecycle:
- Direct-cache slope tolerance 3%; storage/formula tolerance 1%:
- Gap-table closure and >=90% supported attribution rules:
- Transfer predictions and compatible external-validation target:
- Links/hashes for frozen plan, protocol and requirements:

## Review sign-off

Person 1:

Person 2:

Remote push evidence is recorded in the later receipt before any model-memory
measurement. Commit author dates alone are insufficient external evidence.
