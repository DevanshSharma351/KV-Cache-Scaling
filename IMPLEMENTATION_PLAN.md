# Implementation plan: KV-cache scaling

Measurement design: `prefill-v1`, organized into one intensive week and
[five phases](PHASES.md). This document preserves the technical design; the
[decision guide](docs/DECISIONS_AND_PHASE_GUIDE.md) explains actual execution.
The software and 80-worker saved-data pipeline are integrated. Colab quota still
blocks final GPU validation/boundary evidence; completion is judged by evidence,
not the calendar or a passing software test alone. Original preregistration and
raw observations remain unchanged.

## 1. Research question and operational definition

For a fixed model, batch size, cache, attention kernel, allocator and available
VRAM, predict the largest input that a **single prefill forward pass** can
process. Compare its measured peak with the prediction and explain the gap.

L is the number of input tokens in `input_ids.shape == (1, L)`. Every point starts
with an empty explicit `DynamicCache()`, calls the model once, and retains the
resulting L-token cache for inspection. There is no call to `generate()`, no
decoding loop, no sampling, and no padding. This measures the memory required to
ingest a prompt, which includes both the retained KV and transient activations.
It does not estimate a generation-to-L limit.

Use fixed-seed uniformly sampled valid token IDs. Token content is irrelevant to
the tensor shapes; no tokenizer or text-quality benchmark is needed. Force FP16
despite the checkpoint's BF16 metadata because T4 is the target. Set `eval()` and
`torch.inference_mode()`, `use_cache=True`, `output_attentions=False`,
`output_hidden_states=False`, `return_dict=True`, `logits_to_keep=1`, and no labels.
The older alias is `num_logits_to_keep`; our pinned version uses the new spelling.
Check the forward signature and resulting `(1, 1, vocab_size)` logits; never
silently retry with full logits.

Justification: single prefill gives a precise, useful prompt-admission limit and
is feasible on free GPUs. Reduced logits prevent the large vocabulary from
dominating for an avoidable reason. Peak memory is still not pure KV memory.

## 2. Model, attention and cache decisions

**Use Qwen2.5-1.5B base**, not an unresolved choice between Qwen and Llama. It is
ungated, has an Apache-2.0 model license, fits a T4 in FP16, and has GQA and tied
embeddings that make the analytical checks informative. Llama-3.2-3B would add a
gated download and a different weight/cache budget; it is outside measured scope.

Pin model revision `8faed761d45a263340a0528343f099c05c9a4323`. The exact config
snapshot and its SHA-256 are committed. Re-fetch and compare before formal
preregistration, without replacing the snapshot silently.

| Config property | Value |
| --- | ---: |
| Layers / hidden size / intermediate size | 28 / 1536 / 8960 |
| Query heads / KV heads / head dimension | 12 / 2 / 128 |
| Vocabulary / tied embeddings | 151936 / true |
| Config position ceiling | 131072 |
| Model-card full context length | 32768 |
| Unique parameters / FP16 bytes | 1543714304 / 3087428608 |
| Correct / naive KV bytes per token, batch 1 | 28672 / 172032 |

The card/config discrepancy matters: report lengths through 32768 separately
from synthetic memory stress at 32769–131072. The latter tests memory only and
does not demonstrate useful language quality at those positions. Never exceed
the pinned config ceiling. If memory still fits there, the result is
right-censored: `L_memory >= 131072`, not an invented OOM boundary. Also report
`L_supported = min(32768, L_memory)` with the appropriate inequality.

Load with HF `attn_implementation="sdpa"` and wrap the forward in
`sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`. Only the memory-efficient backend
is enabled. A missing kernel is a preflight failure; no automatic math fallback.
Profile one small, post-preregistration diagnostic process to confirm the kernel.
Standard FlashAttention-2 targets newer GPUs and is not supported on T4; separate
Turing implementations are outside scope. Eager attention and SDPA's math path
can materialize quadratic scores. Do not use them in the primary curve.

HF's pinned SDPA adapter repeats grouped K/V to query-head width for attention.
Those temporary tensors belong in activation memory; the retained cache still
has only two KV heads. Do not assume that efficient SDPA removes all activations.

Explicit `DynamicCache` is appropriate for one empty-cache prefill because no
repeated growth loop is involved. In the chunked diagnostic it concatenates
old/new cache tensors, creating transient copies. A small `StaticCache` diagnostic
uses capacity exactly equal to L and includes its allocation in the measured
window. Never pool static/dynamic or prefill/chunked results into one fit.

## 3. Analytical model and preregistration

Use bytes internally; label decimal GB and binary GiB separately. Define:

```text
k(B) = 2 * B * layers * kv_heads * head_dim * cache_element_bytes
K(L,B) = k(B) * L
P_tensor(L) = W + K(L,B) + A_peak(L,B,backend,cache,logits)
P_budget(L) = P_tensor(L) + allocator/workspace allowance
V_usable = driver_total - other_process_usage - context_baseline
L_memory(V) = max integer L with P_budget(L) <= V_usable
```

`V_usable` is the budget before model load. The context is already deducted;
do not subtract it again in `P_budget`. Report actual `mem_get_info()`/NVML
observations and deductions. A T4 may expose around 15 GiB driver capacity and
roughly 14.5 GiB after context, but these are illustrative, not a fixed constant.
The user's approximate “14.5 GB” must not be silently treated as 14.5 GiB.
Compare max allocated to the tensor prediction, and budget usage to the budget
prediction; they are different quantities.

Compute W from architecture, including Q/K/V biases, two RMSNorm vectors per
layer and the final norm. Count the tied `vocab * hidden` matrix once. Check
against unique `model.parameters()` counts/storage after load, before forward.
Buffer and allocator-rounding differences are separate terms. The duplicated
embedding error would be 466747392 bytes = 0.467 GB = 0.435 GiB.

Before GPU forward passes, commit:

1. Derivation with all symbols, config/revision/hash and parameter count.
2. Predicted memory at L=512,1024,2048,4096,8192; predicted limits as functions of
   V; at least a 12/13/14/14.5 GiB sensitivity table.
3. Optimistic, central and conservative activation/headroom assumptions and
   resulting limit range. State that the initial range is an assumption band,
   not a statistical confidence interval. Use the shape envelope in
   [DERIVATION.md](docs/DERIVATION.md); revise reasoning before freezing if needed.
4. Mode, dtype, logits, kernel, cache, cold/no-warmup policy, domain, allocator
   variants, repeats, safety policy, OOM budget and acceptance criteria.
5. Environment-only GPU identity and usable-budget snapshot: context allocation
   is allowed, but no model loading, model forward, calibration or sweep yet.

Push this prediction commit before any such measurements. Store the remote
commit URL, SHA, observed push evidence/time and both reviewers in a later
receipt. Local Git timestamps alone are not an independent or immutable proof.
Use a GitHub release/CI receipt or push event while available; record the evidence
honestly. Never claim the initial scaffold is a completed preregistration.
Preserve the frozen version; changed assumptions require a dated new protocol
and new preregistration, with old measurements kept separate.

## 4. Implement the measurement worker

The exact lifecycle and pseudocode are in
[MEASUREMENT_PROTOCOL.md](docs/MEASUREMENT_PROTOCOL.md). Key requirements:

- One new interpreter per (protocol, allocator, L, repeat). No fork after CUDA
  initialization. Cache weights on disk to avoid repeated downloads.
- Check exact torch/transformers versions, GPU identity, native allocator,
  model SHA, dtype, cache type, logits API and enforced SDPA backend.
- Measure context-only and post-load baselines, then clear unused load buffers.
  Cold forward is the primary policy: no warmup. Record load peak separately.
- Synchronize and reset peaks before input/cache allocation and the forward;
  after synchronization record max allocated/reserved, retained allocated,
  reserved/free and timestamped driver observations. Never substitute resting
  `memory_allocated()` for the peak.
- Keep cache while deleting output/logits, inputs, positions and other references.
  Measure retained state, count unique K/V storage, then release cache and record
  another snapshot. Release all views too. Assert the cache contains L tokens.
- Catch CUDA OOM distinctly from timeout, unsupported kernel, API error or runtime
  termination. Exit promptly after OOM; no second forward in that process.
- Peak-counter sums and NVML samples are not synchronous by default. Keep their
  scopes/timestamps and use the accounting rules below.

All three repeats use fresh processes. Default-vs-expandable pairs are run on
the same GPU UUID, within one allocation where possible, with comparable idle
free memory. Alternate pair order by repeat to reduce time/order bias.

## 5. Coordinator, persistence and safety

The coordinator does not import torch. It sets
`PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False` or `...:True`
in the child's environment before the interpreter starts. It removes conflicting
allocator aliases from the child env. PyTorch 2.6 documents expandable segments;
check the actual wheel, backend, warning output and a small allocation on the
target GPU. Version support does not prove platform support. Unsupported runs
are labelled unsupported, never silently counted as expandable.

For Colab: restart after package changes. If any notebook cell itself owns CUDA,
restart before changing allocator setting; `empty_cache()` does not reconfigure
the allocator. Prefer a CPU-only notebook launching workers, where fresh workers
already supply the required restart. No in-session toggle experiment.

Fresh-process isolation deliberately removes fragmentation inherited from earlier
points. This comparison therefore estimates allocator behavior **within a cold
probe**, not benefits in a long-lived serving workload. A small or zero effect is
a valid finding. A separate repeated-size stress workload would need its own
protocol and is outside this project's agreed research scope.

Write a durable start event before launch and a result/error event after each
worker. Append JSONL with flush+fsync to a persistent mounted path; use one writer
lock and bounded stdout/stderr files. Save manifests and raw records before
analysis. On resume validate protocol/model/environment fingerprints, preserve
failures and repeats, and handle a torn final JSONL line explicitly. Record an
orphan start as interrupted; do not treat it as measured success.

Coarse ladder: required points first, then 16384,32768,65536,98304,131072 only as
admitted by the safety gate. Three independent repeats at required points and
selected boundary points. Complete both diagnostic grids from the config with
three independent repeats per point; keep them separate from the primary fit.

Safety gate: a candidate's conservative estimated budget must be below 90% of
the current context-adjusted available budget. Start from the preregistered upper
envelope; after safe points, use an explicitly labelled calibrated upper envelope
with held-out residual allowance. Check free-memory drift before every launch.
Do not delete or overwrite the original prediction. If the initial envelope
underpredicts, halt escalation, investigate and record the deviation. This
reduces risk but cannot guarantee absence of OOM.

Bisect the **safe budget crossing** to a 256-token interval using admissible
points. This produces `L_safe`, not a proven hardware failure threshold. Estimate
the hardware crossing separately, including uncertainty. Never call untested
points failures. Keep `largest_success`, `smallest_observed_oom`,
`predicted_boundary_interval`, and `safe_budget_interval` distinct.

After safe data, at most one deliberate model-forward OOM attempt is allowed
globally across variants and all resumes. Persistently consume its token before
launch, including on timeout/interruption. Choose default allocator, B=1, primary
mode and a defensible L above the estimated boundary but within the domain. The
attempt must not be an unrelated giant allocation. If it succeeds, do not retry;
report “0 observed OOMs, 1 attempt.” If the domain ceiling fits, make no artificial
attempt and report right-censoring. Any accidental OOM halts the experiment and
consumes the remaining OOM allowance pending review; never conceal or relabel it.

The source's “exactly one OOM” and “exact thresholds for both allocators” cannot
both be guaranteed with a single global OOM budget. Preferred outcome: one
deliberate OOM; defensible exception: censored/interval result and an explicit
requirement deviation. Both members approve the frozen resolution. A partial
success/one failure bracket is not an exact threshold and cannot establish an
exact allocator improvement. Report a safe-limit delta and estimated intervals.

## 6. Complete experiment scope

| Experiment | Purpose | Scope |
| --- | --- | --- |
| Primary single prefill, dynamic, efficient SDPA | Prompt-admission peak and retained cache | B=1, required grid + safe escalation, both allocators |
| Chunked prefill, dynamic, chunks of 512 | Cache growth/copy diagnostic | Required: 512,2048,8192,32768; default allocator; three fresh-process repeats per point |
| Single prefill, static capacity=L | Cache policy sensitivity | Required: 512,2048,8192; default; three fresh-process repeats; include preallocation in peak window |
| Eager or full logits | Conceptual activation comparison | Analytical only; no extra OOM exposure |

Primary retained-cache inspection validates KV independently; the chunked and
static diagnostics explain how execution/cache policy changes the peak. Complete
both as part of the one-week delivery. Prepare all five figure families:
peak-vs-L with preregistered bands, retained-cache slope, same-time allocator
slack, allocator safe-limit intervals and discrepancy accounting. Include a
dynamic/chunked/static comparison table. No training, quantization,
offloading, device_map auto placement, compilation, serving engine or batch-size
sweep in the primary project. Any changed B is a separate versioned extension.

## 7. Analysis and honest error accounting

**KV slope:** regress directly inspected cache storage bytes against L; verify
28672 bytes/token. Also regress `(retained_allocated - post_load_allocated)` after
inputs/logits/positions have been deleted. Deduplicate storage aliases and record
logical-vs-storage differences. Fit each mode/cache/allocator separately. Report
slope, intercept, R², bootstrap/repeat uncertainty, absolute and percentage error.
Use at least five lengths. Passing slope tolerance: 3%; cache-byte formula
tolerance: 1%. Peak-vs-L regression models activations plus KV; it is not the KV
slope in FR5.

**Residual:** at the retained snapshot report
`(retained_allocated - measured_weights - measured_cache_bytes) / retained_allocated`.
Separately report the transient excess `peak_allocated - retained_allocated`,
which is an excess over retained state, not an exact sum of all activation sizes.
Estimate activation phases from tensor shapes and diagnostic profiling. Avoid
assuming every layer's full activations coexist in inference.

**Same-time identity:** for a synchronized snapshot t only:

```text
driver_used(t) = allocated(t) + [reserved(t)-allocated(t)]
                + [driver_used(t)-reserved(t)]
```

The last term includes context, other GPU users and non-PyTorch allocations; only
attribute it to this process after isolation/baseline evidence. Reserved minus
allocated is allocator slack/cached capacity, not proof of fragmentation.
Inspect inactive split bytes and allocation retries; corroborate with snapshots
at selected small points. Do not infer fragmentation solely from a large gap.

`max_reserved - max_allocated` is NOT same-time allocator slack, and
`driver_used_after - peak_allocated` is NOT peak overhead. Sampled NVML maxima are
lower-bound observations at a stated interval (e.g. 10 ms), not exact transient
peaks. Report these as separate series. Profiling/memory-history runs are
labelled diagnostics because instrumentation can perturb memory.

Build discrepancy tables at fixed L (8192 and a safe large L), comparing the same
quantity: measured peak allocated vs the preregistered tensor-peak prediction.
Attribute weight error, measured KV error and independently supported activation/
workspace corrections. Allocator slack and external overhead belong to a
separate budget/driver table, not PyTorch allocated memory. If peak-time
attribution lacks synchronized evidence, leave that portion unaccounted.

Signed deltas must close within `max(1 MiB, 0.1% of measured bytes)`. Require
`abs(unaccounted) / abs(total_gap) <= 0.10` when gap >=10 MiB. Also disclose
`sum(abs(attributed_deltas))` to expose cancelling explanations. Below 10 MiB use
an absolute residual tolerance of 1 MiB and label the ratio ill-conditioned.
Do not force closure by naming the algebraic remainder “activations.” If evidence
cannot account for 90%, the criterion fails visibly and the report explains why.

Translate model error into token-space intervals using the monotonic memory
model, separately from byte-space tables. Report the central pre-registered
prediction, measured successes, observed OOM (if any), uncertainty and censoring.

## 8. Generalization

Predict Qwen2.5-3B and Qwen2.5-7B on A100-40GB and RTX 4090-24GB (four cells).
Read and pin each model config without loading its weights. The 7B is a
config-only exercise and does not violate the measured-model 1B–3B constraint.
Check embedding ties and bias rules per architecture. Use usable VRAM as input;
do not transfer a single T4 “usable fraction” to every GPU. Give low/central/high
budgets and activation assumptions. New backends/hardware widen uncertainty.

Find a published number with identifiable model, precision, GPU, cache policy,
mode, backend and memory setting. Compare like quantities. A vLLM “KV token
capacity” can validate the KV-capacity submodel only, not our HF single-prefill
peak boundary. Compute percentage error on the comparable quantity, cite the
exact page/table/log and retrieval date, and list mismatches. If no compatible
end-to-end figure exists, mark that validation unmet; do not substitute an
unrelated generation throughput benchmark.

## 9. Delivery order and gates

Day 1 is the team's chosen start day; the schedule covers seven consecutive
workdays. Task ownership and percentage shares are in
[TASK_ALLOCATION.md](TASK_ALLOCATION.md). Book the primary GPU window and a backup
Colab/Kaggle option on day 1. Reserve GPU access across days 3–5 and keep durable
capture automated. Adjust run ordering to the admitted grid, reloads and
long-context kernels while independent analysis and documentation proceed.

| Day | Milestone | Gate |
| --- | --- | --- |
| 1 | Finalize config/derivation, environment-only inventory, VRAM scenarios; review and push prediction/receipt; begin worker design | Formal preregistration completed before any model memory measurement |
| 2 | Implement worker/telemetry and journal; freeze analysis columns; start external-source search | Peak lifecycle, backend/cache checks and persistence contract ready for integration |
| 3 | Finish worker, integrate coordinator/journal, run CPU fault tests and post-preregistration GPU smoke checks; build plot templates | Backend/support check and crash/resume/OOM-ledger tests pass before bulk sweeps |
| 4 | Finish safety/boundary integration; collect paired primary sweeps; begin retained-cache regression and report methods | Same GPU UUID; durable raw records; admitted required grid collected or explicit stop reason |
| 5 | Complete both diagnostic grids and boundary repeats; one final OOM attempt if justified; analyze discrepancy; regenerate figures | Measurements locked by end of day; same-metric closure/coverage and deviations reviewed |
| 6 | Finish transfer predictions/comparison, conclusions, report assembly and clean reproduction | Every required result, source, figure and limitation in the draft; reproduction issues listed |
| 7 | Protected recovery/integration block, final reproduction, PDF visual review, repository cleanup and release | Full definition of done reviewed; unresolved failures remain explicitly open |

Critical path: day-1 preregistration -> day-3 smoke-tested harness -> days-4/5 locked
data -> day-6 complete draft -> day-7 reproduced final delivery. B2 persistence,
B3 fixture-based plots, report structure and published-source discovery proceed
alongside the worker implementation. Start the source search on day 2 rather
than waiting for measurements; finalize comparable calculations on day 6.

Use a brief start/end-of-day handoff: task IDs, completed evidence, blockers and
the next shared interface. Only one coordinator owns a GPU at a time. If day-3
integration slips, use day 7's protected block and move independent documentation,
plot templates and config-only work earlier. If GPU access fails, use the backup
provider and a new manifest when hardware changes; never merge unmatched
allocator pairs. Preserve the single global OOM-attempt ledger across providers.

Do not drop diagnostics, repeats, external verification or acceptance checks to
meet the date. If an external constraint remains after recovery, mark the affected
deliverable incomplete and record a concrete recovery plan; do not claim the
project is fully finished. Unsupported hardware or domain censoring can be valid
documented outcomes under the existing protocol, not fabricated measurements.

Final report structure: setup; derivation/preregistration; method; measured
curves/cache slope; allocator comparison; discrepancy; generalization;
limitations/deviations; reproduction instructions. Export a PDF only after results
exist; this request delivers the plan and scaffold, not invented experiment data.
