# Phases 4-5 integration and evidence policy

The analysis pipeline is CPU-only and consumes verified, immutable checkpoint
capsules. It does not run the GPU experiment or manufacture missing observations.

```bash
python -m pip install -e . -r requirements-analysis.txt
python scripts/reproduce_results.py \
  --capsule artifacts/acceptance/2026-10-18-t4/gpu-acceptance-complete.zip \
  --output /tmp/kv-reproduction
```

Use Python 3.11-3.13 for the pinned plotting dependencies. Output directories must
be new. The command records subprocess commands, times, exit codes and log hashes.
It produces validated tables, six figure pairs, four transfer predictions and a
Markdown/PDF report with explicit requirement status. The original L=512 capsule
alone cannot establish a cache slope, full curve or hardware context limit.

## Analysis decisions

- Verify capsule members, journal ordering and referenced raw-result hashes.
  Deduplicate identical probe IDs; reject divergent histories.
- Separate runtime, GPU, model, policy, cache type, allocator and measurement mode.
  Use per-length medians for regression; show repeat ranges separately.
- Fit directly inspected KV storage and retained allocation separately from peak
  memory. Do not interpret the total-peak slope as KV.
- Report allocator slack from synchronized snapshots. Independent allocated and
  reserved maxima do not define a same-time gap.
- Close discrepancy tables arithmetically, then assess explanatory coverage
  independently. Unsupported residuals remain unaccounted even when tables close.
- Distinguish a successful point, conservative safe-budget crossing, actual OOM
  and config-domain censoring. A safe-budget bracket is not an OOM threshold.
- Transfer predictions use pinned configs without loading target checkpoints.
  Usable VRAM ranges are explicit assumptions. Published KV-block agreement
  validates only that submodel, not an HF peak-memory or context-limit prediction.

## Remaining GPU integration

Execution changes require a new implementation amendment, delegated review,
pushed prediction and later receipt before measurements. Preserve the original
advance prediction and disclose the earlier L=512 observations. Never rewrite the
original numerical commitment as if informed estimates preceded those checks.

Primary semantics stay one cold prefill with DynamicCache and reduced logits.
Separate diagnostics add chunked prefill, capacity-equals-length StaticCache,
and an L=512 cuBLAS workspace intervention. They must never enter primary fits.
`CUBLAS_WORKSPACE_CONFIG=:0:0` is child-process-only for that intervention; normal
workers remove inherited overrides. Requested allocator bytes are an optional
additional counter, not a substitute for allocated peak.

Drive remains disconnected. Every coordinator call may stop after one probe with
`--checkpoint-each-probe`. Copy, verify and fsync the complete capsule outside
Colab before resuming. An ephemeral `/content` mirror alone is not durable.
Any GPU allocation, version, UUID or usable-budget change requires reconciliation
before model work. No independent Person 2 reproduction is claimed.
