# Project decisions and five-phase walkthrough

## What are we building?

We are building an experiment that predicts how much GPU memory a transformer
needs for a long input, measures the actual peak and explains the difference.
The product is a memory model, a reliable experiment harness and a reproducible
set of results. It is not a chatbot or a text-quality benchmark.

The useful question is: **“For this exact model and workload, how long an input
can fit in the memory actually available?”** GPU model names alone are not enough.
Weights occupy a fixed amount. KV storage grows with token count. Other tensors,
workspaces, allocator behavior and attention masks can make a request fail earlier.

## The experimental decisions

| Decision | Why we made it | Consequence |
| --- | --- | --- |
| Qwen2.5-1.5B base | Ungated model in the required size range, fits FP16 weights on T4, exposes GQA/tied-embedding issues | We can study sequence-dependent memory instead of nearly filling the GPU with weights |
| One cold prefill of L tokens | Clearly defined and repeatable; generation has a different lifetime/cache-copy pattern | All primary observations describe this one workload, not arbitrary chat generation |
| Batch 1 measured | Matches the requested preregistration | Batch scaling in the explorer is analytical, not another measured curve |
| FP16 weights and cache | Matches the problem constraints | Every KV element contributes two bytes |
| `logits_to_keep=1` | Avoids allocating vocabulary logits for every token | The reduced-logit policy is part of the prediction, not an optional optimization |
| Forced efficient SDPA | Avoids eager attention's score-matrix materialization; T4 does not support FlashAttention-2 | Actual SM75 kernel evidence is saved; unsupported fallback fails |
| DynamicCache primary | Makes the default HF cache policy explicit | Static and chunked modes remain separate controls, never silently substituted |
| Peak reset + max counters | Memory after a forward misses tensors that have already been released | The primary metric is the forward allocated peak, with a separate load peak |
| Fresh process per point | Eliminates prior tensors/allocator state | Allocator settings can be applied before CUDA without toggling them mid-session |
| Context-only usable VRAM | Nominal capacity is not the available experiment budget | The T4 budget is measured in bytes after context initialization |
| Original prediction first | Prevents fitting the answer and calling it a prediction | Git preserves the original numbers, timestamp and subsequent receipt |
| Checkpoint before advance | Colab sessions can vanish | Raw data is externally verified/fsynced after every worker; Drive stayed disconnected |
| One-use deliberate attempt | A failure is evidence, repeated crashes are not a search method | The claim must be consumed before dispatch; no uncertain retry is allowed |

## Phase 1 — Turn the architecture into equations

We read architecture values from a pinned config, rather than a blog. The model
has 28 layers, 12 query heads, two KV heads and head dimension 128. Keys and
values both need storage, so at FP16 and batch 1:

```text
KV bytes = 2 × L × 28 × 2 × 128 × 2 = 28,672 × L
```

Using query heads would predict six times too much KV memory. The input and output
embeddings share weights, so they must be counted once. The resulting unique
parameter count is 1,543,714,304 and FP16 weight storage is 3,087,428,608 bytes.
Counting the tied embedding twice would add 466,747,392 bytes incorrectly.

The total prediction adds an activation envelope and fixed budget allowance.
Those were assumptions before execution, not measured constants. We show three
scenarios and make the context formula depend on usable VRAM and batch size.
The original central commitment was 111,232 tokens, with a 76,024–131,072
assumption range. The range is not a statistical confidence interval.

The original commit is `7baff5121ed643f27ba06959065855a8913f8e8c`, timestamp
`2026-10-18T19:54:52+05:30` (`2026-10-18T14:24:52+00:00`). A later receipt records
remote observation before the first model measurement. The demo verifies the
prediction bytes against that original Git blob and displays both timestamps.

## Phase 2 — Measure a real peak, not an idle snapshot

Each worker loads the model, measures its weights before any forward and checks
that the actual unique storage matches the config derivation. It then resets
peak statistics before allocating the inputs/cache, executes one prefill,
synchronizes and records the peak. We retain and release outputs, input tensors
and cache separately to distinguish persistent and temporary allocations.

PyTorch allocated, reserved and driver memory answer different questions.
Allocated memory tracks tensors. Reserved memory includes cached allocator
capacity. NVML sees driver/device use, and our NVML maximum is sampled rather
than an exact high-water mark. We do not subtract unrelated maxima to invent
a simultaneous fragmentation measurement.

Separate diagnostics test chunking, StaticCache and cuBLAS workspace. The pinned
HF version creates an explicit L×L mask in the full-prefill StaticCache path even
with efficient SDPA. That diagnostic therefore changes masks as well as cache
storage; its difference cannot be attributed solely to static versus dynamic KV.

## Phase 3 — Keep evidence when a runtime disappears

The coordinator controls worker creation and safety; workers only measure and
return structured output. The shared persistence store locks a run, stores
immutable artifacts and mirrors fsynced journal records before scheduling another
point. The public journal API delegates to that same implementation. Recovery
tests exercise missing local files, torn lines, orphans and mirror failures.

An actual result schema validates the saved worker objects and coordinator result
envelopes. Additional code checks ordering, identity, hashes and scientific
compatibility. The presentation's live checkpoint demo deletes only a temporary
local demo copy, restores it from its temporary mirror and compares the bytes.
It never produces a fake GPU observation or touches the global attempt allowance.

Runtime loss happened during the project. Earlier and current GPU UUIDs remain
separate cohorts. We preserved every externally verified checkpoint and did not
pretend an unfinished journal had reached its last event.

## Phase 4 — Compare predictions with real observations

The current cohort contains eight primary lengths from 512 to 65,536, three
repeats per allocator: 48 workers. All 28 registered diagnostics completed.
Four held-out workers at 114,688 tokens succeeded: three default and one
expandable. Directly inspected cache storage has slope 28,672 bytes/token,
zero error from theory and R² = 1 for both allocators.

The total allocated peak grows faster than KV alone. We compare the original
tracked-tensor prediction to matching allocated peaks, then construct signed
component tables. Every row is in bytes; the presentation additionally shows
MiB (1 MiB = 1,048,576 bytes). An explicit unaccounted row is retained. Arithmetic
closure and the supported-coverage criterion are checked separately.

The zero-workspace intervention supports the exact 8,519,680-byte persistent
workspace at L=512. Attribution at other lengths is labelled fixed-size source
inference. Lifetime differences are useful accounting evidence, but they do not
prove a pure causal split between activations and fragmentation.

A later fit was frozen externally before held-out validation. Its error at
114,688 is about −0.010449% for default and 0% for the one expandable observation.
This validates the later fit at that point; it does not replace the original
prediction or identify the hardware OOM threshold.

**What is unfinished:** Colab disconnected and then refused GPU access due to its
usage limit. Two expandable repeats and the single registered boundary attempt
remain unperformed/unverified. The next unsaved worker's outcome is unknown.
The deliberate allowance was not consumed. We report 114,688 as a successful
lower bound and leave the OOM-threshold error unmeasured.

## Phase 5 — Generalize and make the result presentable

Pinned configs for Qwen2.5-3B and Qwen2.5-7B provide four predictions across an
A100-40GB and RTX 4090-24GB. We did not load those checkpoints or run those GPUs.
Their usable-VRAM budgets and activation envelopes are stated assumptions.
A published vLLM KV-block table provides a compatible check of the KV equation.
The source's MB convention is ambiguous, so both binary and decimal percentage
errors are shown. This is not verification of the unrun HF peak-memory limits.

The CPU reproduction command validates the raw capsule and regenerates tables,
six PNG/PDF figure pairs, transfer outputs and the report. The clean notebook
then presents all six deliverables separately, includes a VRAM/batch explorer,
and offers optional individual files or one complete deliverables ZIP. It can
be shown from cached outputs or regenerated live on a CPU runtime. A standalone
HTML export provides another presentation option.

## How to explain our conclusion

“We predicted memory from the architecture before measuring it. Retained KV
storage agrees exactly with the GQA-aware equation. Actual prefill peaks also
include non-KV memory, which we quantify and investigate with separate controls.
Both allocators reached 114,688 tokens. We preserve the original commitment and
all uncertainty; Colab quota prevented the final failed upper-bound observation.”

The software and saved-data deliverables are integrated. Full research acceptance
must wait for the remaining GPU evidence; the report and notebook say so openly.
See [remaining work](REMAINING_WORK.md) for the valid continuation procedure.
