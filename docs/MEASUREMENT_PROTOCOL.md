# Measurement protocol: prefill-v1

Implementation contract for A2/A3, with B4 owning diagnostic execution paths.
This is a specification, not an already-tested
GPU worker. The primary policy is cold forward, batch 1, FP16, one empty-cache
prefill, reduced logits and efficient SDPA.

## Preflight and baselines

1. The CPU coordinator checks the frozen prediction receipt, acquires the run
   lock, commits/fsyncs a start event and passes a complete request to a fresh
   interpreter. No parent-owned CUDA context or inherited live CUDA tensors.
2. Before importing torch, the child verifies allocator env against the request;
   conflicting aliases are rejected. Record executable, platform, Python,
   torch/CUDA/transformers versions, allocator backend, env and package manifest.
3. Bind logical `cuda:0` to its NVML physical GPU UUID (honor
   `CUDA_VISIBLE_DEVICES`; do not assume NVML index 0 is that same GPU). Initialize
   CUDA, create/release a tiny allocation, synchronize and `empty_cache()`.
   Snapshot context-only allocated/reserved, CUDA free/total and NVML used/free/
   total. Record other-process evidence where permissions allow. If attribution
   is unavailable, label driver overhead as combined/unknown.
4. Confirm expected version and `native` allocator. Capture warnings from the
   expandable test; `expandable_segments not supported` invalidates that variant.
   `get_allocator_backend()` confirms native, not that expansion actually works.
5. Load the exact model SHA with safetensors, `trust_remote_code=False`,
   `torch_dtype=torch.float16`, `attn_implementation="sdpa"`; move the whole model
   to the single GPU. No offload/quantization/compile/autocast/automatic placement.
6. Verify unique parameter count, actual bytes/dtypes and tied input/output
   storage. Snapshot load peak separately, release temporary loader references,
   `gc.collect()`, `empty_cache()`, synchronize, then snapshot post-load state.
   There has been no model forward or warmup in this worker.

## Timed peak window

Illustrative pseudocode (implement error paths and schema, do not paste as a
complete worker):

```python
model.eval()
torch.cuda.synchronize()
post_load = snapshot()
torch.cuda.reset_peak_memory_stats()
start_nvml_sampler(interval_ms=10)  # CPU thread; observations are sampled only

with torch.inference_mode():
    ids = seeded_token_ids(batch=1, length=L, device="cuda:0")
    positions = torch.arange(L, device="cuda:0")
    cache = DynamicCache()
    # No attention_mask for unpadded single prefill; avoids an explicit L x L mask.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = model(
            input_ids=ids,
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    after_forward = snapshot()  # same-time values, not necessarily peak time
    verify_logits_dtype_and_shape(out.logits, (1, 1, vocab))
    cache = out.past_key_values
    verify_cache_length_and_shapes(cache, L)
    cache_bytes = inspect_unique_storage_bytes(cache)
    del out, ids, positions  # release logits and every other output/input reference
    torch.cuda.synchronize()
    retained = snapshot()   # weights + cache + explicitly recorded live buffers
    del cache              # ensure no inspection views hold cache alive
    torch.cuda.synchronize()
    released = snapshot()

stop_nvml_sampler()
```

The reset includes input/position/cache allocations and all first-forward
transients; weights are already live at reset. Record inputs and retained logits
bytes so no term is confused with KV. Do not reset after input/cache allocation.
Do not call `empty_cache()` during the measured window or between after-forward
and retained/released snapshots. All primary repeats follow this lifecycle.

Inspect storage using unique device + untyped-storage pointer/size identifiers;
separately sum logical `numel()*element_size()`. A pointer reused after a tensor
is deleted is not evidence of aliasing across times. Assert cache FP16, layers,
two heads, head dimension 128 and sequence length L. Keep only plain CPU numeric
metadata from inspection. Report loaded buffers explicitly.

## Counters and what they establish

| Field | Interpretation |
| --- | --- |
| `peak_allocated_bytes` | Exact PyTorch tracked allocation high-water mark within this reset scope |
| `peak_reserved_bytes` | Independent reservation high-water mark; not necessarily at allocated peak |
| `allocated_bytes` at a snapshot | Live PyTorch storage at that instant |
| `reserved_bytes - allocated_bytes` | Same-time cached/unused allocator capacity |
| `inactive_split_bytes` | Native allocator split-block diagnostic; not a complete fragmentation measurement |
| CUDA `free_bytes` / `total_bytes` | Driver-reported device budget at that snapshot |
| NVML sampled maximum | Observed driver usage maximum at recorded sample interval; may miss transient peaks |
| `nvml_used_bytes - reserved_bytes` | Same-time external/context/non-PyTorch term, including other users |

Snapshot fields are acquired sequentially around a synchronize; save a monotonic
start/end timestamp and describe this finite acquisition window. An external
process can change driver counters inside it. Profile tensor lifetimes in
separate diagnostic workers if peak attribution needs stronger evidence. Never
subtract unrelated maxima and call the result a same-time identity.

## Failures and coordinator rules

On CUDA OOM, save whatever counters/error stage are available, mark incomplete
metrics null, classify the planned/accidental status from the coordinator request,
and exit. Do not allocate recovery tensors or continue the model. An exit signal,
timeout, unsupported kernel, bad API and ordinary Python error are distinct
statuses; lack of output is not automatically CUDA OOM.

The parent validates JSON result/probe ID, appends a durable result/error event,
and mirrors it to the persistent path. Bound stderr/stdout logs; include their
hash/path in the manifest. Fresh subprocess exit releases its context. Recheck
idle driver memory before launching another worker. If memory has not returned
near baseline or another GPU user appears, pause run execution and record it.

The deliberate-attempt budget is global, not per allocator, mode, notebook or
file. Its durable token is consumed before launch. Interrupted deliberate attempts
are not retried. The source allows one OOM; the implementation must not promise
that a chosen attempt necessarily fails. See the boundary limitations in the plan.

## Diagnostics, always separated

- **Chunked dynamic:** same total L, 512-token chunks, one retained cache across
  chunks; reset once before all chunks. Pass advancing cache positions and valid
  causal masking. Delete old outputs before each next call. Its mask may be
  chunk-by-context and its concatenations cause extra peaks. The output is not
  a generation benchmark or interchangeable with primary single prefill.
- **Static:** allocate a new FP16 `StaticCache` with capacity L inside the reset
  window. The preallocation costs memory before any token is written. Masking
  and capacity must be recorded. Fit capacity, not just populated length.
- **Kernel/profiler:** after preregistration, small diagnostic workers may check
  the efficient operator trace. Profiling does not share a process with primary
  points and its counters are not merged with uninstrumented results.

## Required GPU implementation checks

Run after the preregistration gate: L=512 worker smoke check; verify peak >= live
after-forward allocation and reserved >= allocated at each snapshot; inspect
cache bytes; confirm efficient SDPA trace; test both allocator envs; test clean
exit and persistent write. Use CPU synthetic workers for crash/resume, timeout,
token consumption and schema tests so test development does not consume OOMs.
Actual GPU acceptance is not covered by the current CPU CI.
