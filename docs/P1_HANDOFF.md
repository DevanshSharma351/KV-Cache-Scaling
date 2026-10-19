# Person 1 delivery and remaining GPU acceptance

**19 October 2026. Implementation delivered; research acceptance incomplete.**
Person 1's five phases are integrated. Colab removed the T4 runtime after the
80th externally verified worker and then refused GPU access due to its usage
limit. This is the remaining execution blocker; passing CPU tests is not used
as a substitute for missing GPU observations.

## What works

- The config-derived predictor counts tied embeddings once, uses two KV heads
  rather than twelve query heads, and accepts usable VRAM and batch size.
- The same probe/coordinator path produced 48 primary sweep workers and all
  28 diagnostic workers, each isolated and checkpointed externally before the
  next dispatch. The original bounded acceptance also proved checkpoint restore.
- Analysis validates archive hashes, journal chains and raw references, separates
  compatible cohorts, fits retained KV independently of total peaks, and writes
  explicit discrepancy/unaccounted rows.
- The CPU pipeline generates six figure pairs, four transfer predictions,
  published KV-block comparisons and an evidence-linked Markdown/PDF report.
- All **109 CPU tests pass** locally, in Colab and in CI on Python 3.10–3.13.
  Both Colab and CI reproduce the saved final-cohort data.
  The executed Colab notebook records source checkout, commands and outputs.

## Actual results

| Quantity | Observed or committed value |
| --- | --- |
| Model / primary workload | Qwen2.5-1.5B FP16; one cold B=1 prefill; DynamicCache; `logits_to_keep=1`; forced efficient SDPA |
| T4 context-only usable memory | 15,527,116,800 bytes = 14.460754 GiB |
| Original prediction | 111,232 tokens; assumption range 76,024–131,072; original commit unchanged |
| Unique FP16 weights | 3,087,428,608 bytes; matches config-derived tied-weight count |
| Retained KV slope | 28,672 bytes/token; 0% formula/slope error; R² = 1 for both allocators |
| Sweep coverage | Eight lengths 512–65,536, three repeats per allocator = 48 workers |
| Diagnostics | 12 chunked + 9 static + 6 workspace + 1 actual SM75 kernel trace = 28 workers |
| Held-out L=114,688 | Default n=3; expandable n=1; all four succeeded |
| Allocated peak at held-out L | Default 14,020,668,928 bytes; expandable 14,019,751,424 bytes |
| Reserved peak at held-out L | Default 15,319,695,360 bytes; expandable 14,644,412,416 bytes |
| Frozen post-sweep fit error | Default −0.010449%; expandable 0% at the observed held-out point |
| Curve/discrepancy checks | Required paired five-length grid, cache slope, coverage and closure all pass |
| Hardware OOM threshold | **Unobserved**; 114,688 is a successful lower bound for both allocators |
| Deliberate attempt | Not dispatched; external and internal allowances unconsumed in saved evidence |

The 675,282,944-byte difference in reserved peaks is an allocator observation,
not a proven gain in maximum context. Independent allocated/reserved maxima
are not same-time fragmentation measurements. Conservative safe-limit delta
is 0 tokens with a [−256,+256] search-bracket interval; this is policy-censored.

Matched zero-workspace controls remove the exact 8,519,680-byte persistent
allocation at L=512. Attribution at other lengths is explicitly fixed-size source
inference. Signed discrepancy closure and lifetime labels do not establish a
pure activation/fragmentation causal decomposition of an unobserved OOM gap.
The StaticCache diagnostic includes HF 4.51.3's L×L causal-mask allocation;
it cannot isolate cache-storage policy alone.

L above the model-card 32,768-token quality domain is synthetic memory stress.
Transfer estimates for Qwen2.5-3B/7B on A100-40GB/RTX4090-24GB are config-only.
Published vLLM block sizes validate the compatible KV submodel, with explicit
unit/dtype caveats; they do not validate HF peak limits on those GPUs.

## Reproduce and inspect

```bash
python -m pip install -e . -r requirements-analysis.txt
python -m unittest discover -s tests -v
python scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip \
  --output /tmp/kv-reproduction
```

Use Python 3.11–3.13 for pinned report dependencies; the output path must be new.
No model load, CUDA allocation or GPU is needed. Raw ZIP bytes remain unchanged.
Report status is intentionally **RESEARCH_CRITERIA_PENDING**. The strict
`scripts/verify_p1_delivery.py` must reject this partial cohort; do not weaken its
83-worker, complete-journal and final-boundary requirements to obtain a pass.

[Raw evidence](../artifacts/experiments/2026-10-19-t4-final/README.md) ·
[PDF](../artifacts/reports/p1-2026-10-19/report.pdf) ·
[processed tables](../data/processed/p1-2026-10-19/) ·
[executed notebook](../notebooks/03_p1_executed_t4_2026-10-19.ipynb).

## Resume only with a valid GPU allocation

1. Preserve this checkpoint and its unfinished validation journal. The next
   worker may have started before disconnect; its outcome is unknown. Never
   infer success, mark its journal complete, or duplicate a live worker.
2. Reconcile GPU UUID, runtime versions, source and context-only usable memory
   before loading a model. The old receipt is tied to the lost allocation;
   it must not authorize another GPU silently.
3. If the exact runtime survives, inspect/resolve pending dispatch and resume
   from verified state under the gate. If it does not, prospectively register a
   distinct continuation cohort, disclose the 80 observations, and define its
   own reference/admission evidence. Do not pool different UUIDs into one fit.
4. Complete two additional expandable held-out repeats only under a compatible
   registered continuation. Then finalize the validation journal.
5. Consume the durable external one-use claim before the **single** final
   L=131,072 default attempt, following the frozen policy. Never retry an
   uncertain dispatch. Preserve either actual success or a typed CUDA OOM;
   no model measurements follow a confirmed OOM. A success is censoring,
   not permission for a second attempt.
6. Regenerate analysis/report and rerun strict delivery checks. Claim final
   research acceptance only when the actual recorded outcome supports it.

Drive remained disconnected as requested. One contributor review was authorized
and delegated on behalf of Person 1. Independent Person 2 work/reproduction is
not claimed. The anonymous **55/45** allocation and seven-day plan remain intact.
