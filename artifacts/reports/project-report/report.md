# KV-Cache Scaling: integrated evidence report

Status: **FINAL_WITH_MEASUREMENT_LIMITATIONS**. No missing experiment is presented as a successful result.

## Completed targeted OOM check

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

Raw evidence: artifacts/experiments/2026-10-21-boundary/checkpoint.zip; SHA-256 aa5ac338423d0d2f0595fbac3b662ffc10dacb12bbc44f9b097d5eb69450c18b.

## Final resource-bounded scope

The user reopened the boundary objective after the initial resource limit. The core curve, L=512 replication, and targeted boundary continuation are separate GPU allocations. The deliberate OOM check is now completed; the exact per-token threshold and allocator hardware-limit gain remain bounded or unmeasured.
See docs/FINAL_SCOPE.md and configs/delivery.json. No additional GPU run is planned.

### Separate replication checks

| Allocator | L | Repeat | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: | ---: |
| default | 512 | 1 | 3146681856 | 3330277376 |
| default | 512 | 2 | 3146681856 | 3330277376 |

## What we built

A config-derived memory predictor, preregistration gate, fresh-process CUDA probes, checkpointed coordinator, immutable-data analysis, reproducible scientific plots, and config-only transfer predictions. The measurement coverage below is computed from verified records.

## Scientific method

L means one cold FP16 prefill on L random tokens at B=1, with an empty DynamicCache, logits_to_keep=1 and forced efficient SDPA. Peak counters reset before inputs/cache allocation. Profiled diagnostics are separate.

P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H; W = 2P; KV(L,B) = 2BLNKD*2.

For Qwen2.5-1.5B: 1543714304 unique parameters, 3087428608 weight bytes and 28672 KV bytes/token at B=1. Query heads instead of KV heads cause a 6x error; duplicating tied embeddings adds 466747392 bytes.

The original advance central context estimate is 111232 tokens at 15527116800 usable bytes; assumption range 76024-131072 (config capped). These are not measured limits. The model-card quality domain is 32768.

## Saved primary measurements

52 primary probes across 9 measured lengths (512-114688 tokens).

| Allocator | L | Repeats | Median allocated peak, bytes | Median reserved peak, bytes | Median cache, bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 512 | 3 | 3146681856 | 3330277376 | 14680064 |
| default | 1024 | 3 | 3195448832 | 3403677696 | 29360128 |
| default | 2048 | 3 | 3295079936 | 3529506816 | 58720256 |
| default | 4096 | 3 | 3487002112 | 3693084672 | 117440512 |
| default | 8192 | 3 | 3877137920 | 4240441344 | 234881024 |
| default | 16384 | 3 | 4657409536 | 5171576832 | 469762048 |
| default | 32768 | 3 | 6217952768 | 6731857920 | 939524096 |
| default | 65536 | 3 | 9339039232 | 10171187200 | 1879048192 |
| default | 114688 | 3 | 14020668928 | 15319695360 | 3288334336 |
| expandable | 512 | 3 | 3144715776 | 3164602368 | 14680064 |
| expandable | 1024 | 3 | 3193482752 | 3202351104 | 29360128 |
| expandable | 2048 | 3 | 3291016704 | 3317694464 | 58720256 |
| expandable | 4096 | 3 | 3486084608 | 3529506816 | 117440512 |
| expandable | 8192 | 3 | 3876220416 | 3925868544 | 234881024 |
| expandable | 16384 | 3 | 4656492032 | 4764729344 | 469762048 |
| expandable | 32768 | 3 | 6217035264 | 6400507904 | 939524096 |
| expandable | 65536 | 3 | 9338121728 | 9695133696 | 1879048192 |
| expandable | 114688 | 1 | 14019751424 | 14644412416 | 3288334336 |

## Cache fits and residuals

| Allocator | KV slope bytes/token | KV slope error % | KV R-squared | Retained-minus-load intercept bytes | Total-peak slope bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 28672.000000000004 | 2.220446049250313e-14 | 1.0 | 8519679.99999988 | 95236.94686064174 |
| expandable | 28672.000000000004 | 2.220446049250313e-14 | 1.0 | 8519679.99999988 | 95248.00000000001 |

Retained residual fraction spans 0.133456% to 0.303312% across the observed primary points.

## Boundary evidence

- default: observed success at 114688 tokens; observed OOM upper point None. Original prediction error interval, (prediction-limit)/limit: None percent. This is an interval, not an exact threshold.
- expandable: observed success at 114688 tokens; observed OOM upper point None. Original prediction error interval, (prediction-limit)/limit: None percent. This is an interval, not an exact threshold.
- Expandable minus default conservative safe-limit delta: observed-success delta 0 tokens; bracket interval [-256, 256] tokens. A common analytical admission ceiling can censor both; this does not establish an unchanged hardware limit.
- Held-out default L=114688, n=3: prospective fitted peak 14019203874 bytes, observed median 14020668928 bytes; error -0.0104%. This fit was frozen after the safe sweep and before this held-out point; it does not replace the advance commitment.
- Held-out expandable L=114688, n=1: prospective fitted peak 14019751424 bytes, observed median 14019751424 bytes; error 0.0000%. This fit was frozen after the safe sweep and before this held-out point; it does not replace the advance commitment.

Only measured lengths are observations. Missing slope/R-squared values are null, not zero. Repeat ranges are descriptive, not confidence intervals. Separate peak maxima cannot be subtracted to infer same-time slack.

## Discrepancy accounting

| Allocator / L / repeat | Signed peak gap, bytes | Closure error, bytes | Absolute unaccounted, bytes | Coverage criterion |
| --- | ---: | ---: | ---: | --- |
| default / 114688 / 1 | -1339293184 | 0 | 0 | PASS |
| default / 114688 / 2 | -1339293184 | 0 | 0 | PASS |
| default / 114688 / 3 | -1339293184 | 0 | 0 | PASS |
| expandable / 114688 / 1 | -1340210688 | 0 | 0 | PASS |
| default / 512 / 1 | 4465152 | 0 | 0 | PASS |
| default / 512 / 2 | 4465152 | 0 | 0 | PASS |
| default / 512 / 3 | 4465152 | 0 | 0 | PASS |
| default / 1024 / 1 | -1555968 | 0 | 0 | PASS |
| default / 1024 / 2 | -1555968 | 0 | 0 | PASS |
| default / 1024 / 3 | -1555968 | 0 | 0 | PASS |
| default / 2048 / 1 | -11501056 | 0 | 0 | PASS |
| default / 2048 / 2 | -11501056 | 0 | 0 | PASS |
| default / 2048 / 3 | -11501056 | 0 | 0 | PASS |
| default / 4096 / 1 | -38731264 | 0 | 0 | PASS |
| default / 4096 / 2 | -38731264 | 0 | 0 | PASS |
| default / 4096 / 3 | -38731264 | 0 | 0 | PASS |
| default / 8192 / 1 | -86900224 | 0 | 0 | PASS |
| default / 8192 / 2 | -86900224 | 0 | 0 | PASS |
| default / 8192 / 3 | -86900224 | 0 | 0 | PASS |
| default / 16384 / 1 | -183238144 | 0 | 0 | PASS |
| default / 16384 / 2 | -183238144 | 0 | 0 | PASS |
| default / 16384 / 3 | -183238144 | 0 | 0 | PASS |
| default / 32768 / 1 | -375913984 | 0 | 0 | PASS |
| default / 32768 / 2 | -375913984 | 0 | 0 | PASS |
| default / 32768 / 3 | -375913984 | 0 | 0 | PASS |
| default / 65536 / 1 | -761265664 | 0 | 0 | PASS |
| default / 65536 / 2 | -761265664 | 0 | 0 | PASS |
| default / 65536 / 3 | -761265664 | 0 | 0 | PASS |
| expandable / 512 / 1 | 2499072 | 0 | 0 | PASS |
| expandable / 512 / 2 | 2499072 | 0 | 0 | PASS |
| expandable / 512 / 3 | 2499072 | 0 | 0 | PASS |
| expandable / 1024 / 1 | -3522048 | 0 | 0 | PASS |
| expandable / 1024 / 2 | -3522048 | 0 | 0 | PASS |
| expandable / 1024 / 3 | -3522048 | 0 | 0 | PASS |
| expandable / 2048 / 1 | -15564288 | 0 | 0 | PASS |
| expandable / 2048 / 2 | -15564288 | 0 | 0 | PASS |
| expandable / 2048 / 3 | -15564288 | 0 | 0 | PASS |
| expandable / 4096 / 1 | -39648768 | 0 | 0 | PASS |
| expandable / 4096 / 2 | -39648768 | 0 | 0 | PASS |
| expandable / 4096 / 3 | -39648768 | 0 | 0 | PASS |
| expandable / 8192 / 1 | -87817728 | 0 | 0 | PASS |
| expandable / 8192 / 2 | -87817728 | 0 | 0 | PASS |
| expandable / 8192 / 3 | -87817728 | 0 | 0 | PASS |
| expandable / 16384 / 1 | -184155648 | 0 | 0 | PASS |
| expandable / 16384 / 2 | -184155648 | 0 | 0 | PASS |
| expandable / 16384 / 3 | -184155648 | 0 | 0 | PASS |
| expandable / 32768 / 1 | -376831488 | 0 | 0 | PASS |
| expandable / 32768 / 2 | -376831488 | 0 | 0 | PASS |
| expandable / 32768 / 3 | -376831488 | 0 | 0 | PASS |
| expandable / 65536 / 1 | -762183168 | 0 | 0 | PASS |
| expandable / 65536 / 2 | -762183168 | 0 | 0 | PASS |
| expandable / 65536 / 3 | -762183168 | 0 | 0 | PASS |

The signed table closes using measured lifetime/counter deltas. Closure does not establish causal attribution. Any unsupported post-load storage or post-forward persistent growth remains explicitly unaccounted. For material gaps, both signed-gap and absolute-contribution coverage must reach 90%; for gaps below 10 MiB, the unaccounted amount must be <=1 MiB.

Matched L=512 zero-workspace controls support the attributed cuBLAS row where present. Other lengths use explicitly labelled fixed-size inference from the pinned PyTorch source, only for the same allocator/runtime and exact persistent delta.

## Transfer predictions

### Unrun target models and GPUs

No target checkpoint was loaded and no target GPU was measured.

| Model | GPU | Assumed usable GiB (low/central/high) | Raw central L | Config-capped L | Capped assumption range |
| --- | --- | --- | ---: | ---: | --- |
| Qwen/Qwen2.5-3B | A100-40GB | 37/38/39 | 251278 | 32768 | 32768–32768 |
| Qwen/Qwen2.5-3B | RTX-4090-24GB | 22/23/23.5 | 132571 | 32768 | 32768–32768 |
| Qwen/Qwen2.5-7B | A100-40GB | 37/38/39 | 109629 | 109629 | 72387–131072 |
| Qwen/Qwen2.5-7B | RTX-4090-24GB | 22/23/23.5 | 39097 | 39097 | 22613–55262 |

Activation/VRAM ranges are assumptions, not statistical confidence intervals. Config censoring is not an observed OOM.

## Published KV-only comparison

Source: [Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html).

The publication uses MB without defining decimal versus binary units. Both errors are reported; the 2-byte cache dtype is an explicit compatibility assumption.

| Base / matching Instruct KV architecture | Predicted 16-token block bytes | Published MB | Error if MiB | Error if decimal MB |
| --- | ---: | ---: | ---: | ---: |
| Qwen/Qwen2.5-3B | 589824 | 0.56 | 0.4464% | 5.3257% |
| Qwen/Qwen2.5-7B | 917504 | 0.87 | 0.5747% | 5.4602% |

This checks KV-byte scaling only. It does not verify transferred HF single-prefill peak context limits.


## Requirement status

| Requirement | Status | Evidence / remaining criterion |
| --- | --- | --- |
| FR1 | PASS | Pinned config, KV heads and tied-weight arithmetic |
| FR2 | PASS_FOR_SAVED_ACCEPTANCE | Exact weights/cache verified; residuals reported separately |
| FR3 | PASS_FOR_SAVED_ACCEPTANCE | Original prediction 7baff51 and later receipt 976158c precede model work |
| FR4 | CHECKPOINTED_GRID_PASS | Fresh workers and per-probe external checkpoint barriers; original acceptance restoration verified |
| FR5 | PASS | Five lengths, three repeats and retained-cache slope required |
| FR6 | OBSERVED_OOM_BRACKET_ALLOCATOR_DELTA_CENSORED | One-use OOM bracket and paired safe-budget delta are distinct; an exact hardware-limit delta is not established |
| FR7 | PASS | Closed signed table; unsupported bytes remain unaccounted |
| FR8 | CONDITIONAL_KV_COMPARISON | Four unrun transfer cells; published KV block comparison with dtype/unit caveats |

## Reproduction

```bash
python -m pip install -e . -r requirements-analysis.txt
kvcache analyze --capsule /Users/devanshsharma/Documents/KV-Cache/KV-Cache-Scaling/artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip --run-id 0cb5db86-c7be-41f1-99cf-c3fb73430756 --run-id 6e6aeb55-a540-43bc-a4e7-e881ec3d29f2 --run-id c6b37c71-23e8-4209-a278-2c9f65b6b763 \
  --output /tmp/kv-analysis --figures
kvcache generalize --output /tmp/kv-transfer
python scripts/build_report.py --analysis /tmp/kv-analysis \
  --transfer /tmp/kv-transfer --output /tmp/kv-report
```

Each output directory must be new. Source capsules, code, configs and output files carry hashes. No GPU is required to regenerate these analyses, figures, transfers or report. The executed Colab notebook retains the original command trail.



## Separate diagnostic comparisons

| Mode | Allocator | L | Primary peak bytes | Diagnostic peak bytes | Delta bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| chunked_dynamic | default | 512 | 3146681856 | 3147730432 | 1048576 |
| chunked_dynamic | default | 2048 | 3295079936 | 3194940928 | -100139008 |
| chunked_dynamic | default | 8192 | 3877137920 | 3411570176 | -465567744 |
| chunked_dynamic | default | 32768 | 6217952768 | 4317408768 | -1900544000 |
| static_cache | default | 512 | 3146681856 | 3147206144 | 524288 |
| static_cache | default | 2048 | 3295079936 | 3303468544 | 8388608 |
| static_cache | default | 8192 | 3877137920 | 4011355648 | 134217728 |
| workspace_control | default | 512 | 3146681856 | 3138162176 | -8519680 |
| workspace_control | expandable | 512 | 3144715776 | 3136196096 | -8519680 |

Diagnostics change execution semantics or workspace configuration; their points are excluded from primary fits.
In pinned HF 4.51.3, the static-cache path builds an explicit query-by-capacity causal mask. For full prefill this adds an L-by-L mask even with efficient SDPA. Chunked queries bound the mask query dimension. These comparisons therefore include mask behavior and cannot isolate cache-storage policy alone. [Pinned Qwen2 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py).

## Remaining scientific criteria

- Paired prospective held-out validation at L=114688.
- Actual OOM boundary unobserved; conservative safe-budget censoring is not an OOM threshold.
- Transferred HF peak thresholds remain unvalidated; published comparison checks KV bytes only.

Technical review was delegated by the user; Google Drive stayed disconnected. No independent second-person reproduction or full cloud-session restart is claimed.

## Sources

- [Original formal prediction](https://github.com/DevanshSharma351/KV-Cache-Scaling/blob/7baff5121ed643f27ba06959065855a8913f8e8c/prereg/formal-2026-10-18-t4/prediction.md)
- [Executed notebook](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS)
- [vLLM published KV block table](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html)
- [PyTorch 2.6 cuBLAS workspace source](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp)
- Official pinned transfer config URLs and hashes: configs/transfer/provenance.json.


## Deliverable guide

| Required output | Where to show it |
| --- | --- |
| 1. Derivation with symbols | Section 1; docs/DERIVATION.md; config-derived weights and KV equation. |
| 2. Timestamped advance prediction | Section 2; prereg/formal-2026-10-18-t4/prediction.json and original Git commit. |
| 3. Checkpointing sweep harness | Section 3; src/kv_cache_scaling/sweep.py, probe.py and coordinator_store.py; raw checkpoint capsule. |
| 4. Measured / analytical plots | Sections 4-6; six standalone PNG/PDF figures in artifacts/figures/project-results/. |
| 5. Quantified discrepancy | Section 7; data/processed/project-results/discrepancy.csv; signed components, closure and unaccounted bytes. |
| 6. Cross-model / cross-GPU predictions | Section 8; data/processed/project-results/transfer/; four scenarios and published KV-only verification. |

## Five-phase project flow

| Phase | What we built |
| --- | --- |
| 1. Derive and preregister | Read pinned architecture; count tied weights and GQA cache; commit the prediction before model work. |
| 2. Build isolated probes | Use one cold prefill per fresh process, explicit cache/backend/logits policy and synchronized peak counters. |
| 3. Coordinate and preserve | Checkpoint every worker, verify external copies, enforce safe admission and a single OOM allowance. |
| 4. Explain the measurements | Fit inspected KV separately from total peaks; plot both; close signed discrepancy tables with explicit unknowns. |
| 5. Generalize and present | Predict two unrun models on two unrun GPUs; compare published KV figures; provide a CPU-replayable notebook and report. |

## Original five-length commitment

Original Git commit: 7baff5121ed643f27ba06959065855a8913f8e8c. Original commit timestamp: 2026-10-18T19:54:52+05:30.
Tensor envelope excludes fixed headroom; admission budget includes it. Values come from the unchanged original JSON, not fitted observations.

| L tokens | KV bytes | Central tensor peak bytes | Central budget bytes |
| ---: | ---: | ---: | ---: |
| 512 | 14680064 | 3142216704 | 3679087616 |
| 1024 | 29360128 | 3197004800 | 3733875712 |
| 2048 | 58720256 | 3306580992 | 3843451904 |
| 4096 | 117440512 | 3525733376 | 4062604288 |
| 8192 | 234881024 | 3964038144 | 4500909056 |

## Representative byte-level discrepancy

Default allocator, L=114688, repeat=1. Signed peak gap=-1339293184 bytes; closure=0 bytes; absolute unaccounted=0 bytes.

| Component | Signed bytes | Signed MiB | Evidence |
| --- | ---: | ---: | --- |
| unique_weight_difference | 0 | 0.000000 | Deduplicated parameter storage vs config-derived weights |
| loaded_model_buffer_storage | 256 | 0.000244 | Unique loaded model storage excluding parameters |
| postload_allocator_internal_padding | 917760 | 0.875244 | Same-snapshot allocated minus requested bytes |
| unaccounted_postload_storage | 0 | 0.000000 | Unclassified post-load allocation; not assumed to be framework overhead |
| controlled_cublas_workspace | 8519680 | 8.125000 | Source-supported fixed-size inference from same-allocator L=512 controls; not a direct intervention at this length; pinned PyTorch 2.6 fixed non-Hopper workspace; controls: 258f7320-7005-421e-a768-900f82727dcc,84065451-393c-4ffa-9aea-3e0385da962d,ab101abe-710e-4861-88c9-f7fa22042a87 |
| unaccounted_postforward_persistent_growth | 0 | 0.000000 | Measured released-minus-postload allocation; physical cause unconfirmed |
| cache_lifetime_difference | 0 | 0.000000 | Controlled cache release delta minus analytical retained KV |
| output_and_input_lifetime | 2139136 | 2.040039 | Controlled output/input release delta |
| transient_envelope_difference | -1350870016 | -1288.290039 | Exact peak excess above synchronized endpoint minus assumed envelope; descriptive, not pure-activation attribution |

## Presenting the deliverables

Open notebooks/FINAL_PRESENTATION.ipynb or artifacts/presentation/PROJECT_PRESENTATION.html to show all six deliverables, figures, tables and downloads. Run All in the clean notebook regenerates saved-data results on CPU. See docs/LIVE_DEMO.md and docs/DECISIONS_AND_PHASE_GUIDE.md.
The saved cohort contains 80 successful workers: 48 primary sweep, 28 diagnostics and four held-out validations. Two original expandable held-out repeats remain absent from the core allocation; the final boundary attempt was completed independently. No deliberate OOM was dispatched in the core cohort; the independent continuation below completed the one permitted attempt. The maximum successful context is a lower bound, not a completed OOM threshold.
