# Phases 4-5 integration and evidence policy

The analysis pipeline is CPU-only and consumes verified, immutable checkpoint
capsules. It does not run the GPU experiment or manufacture missing observations.

```bash
python -m pip install -e . -r requirements-analysis.txt
python scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip \
  --output /tmp/kv-reproduction
```

Use Python 3.11-3.13 for the pinned plotting dependencies. Output directories must
be new. The command records subprocess commands, times, exit codes and log hashes.
It produces validated tables, six figure pairs, four transfer predictions and a
Markdown/PDF report with explicit requirement status. The final saved capsule contains 80 successful workers, including the paired
sweep and complete diagnostics. Its report deliberately remains
`RESEARCH_CRITERIA_PENDING`: two expandable held-out repeats and the deliberate
OOM boundary are absent after Colab runtime loss/quota refusal. See
[P1 handoff](P1_HANDOFF.md). The original L=512 capsule alone cannot establish
a cache slope, full curve or hardware context limit.

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

## GPU integration and current interruption

Execution changes require a new implementation amendment, delegated review,
pushed prediction and later receipt before measurements. Preserve the original
advance prediction and disclose the earlier L=512 observations. Never rewrite the
original numerical commitment as if informed estimates preceded those checks.

Primary semantics stay one cold prefill with DynamicCache and reduced logits.
Separate diagnostics add chunked prefill, capacity-equals-length StaticCache,
and paired L=512 cuBLAS workspace interventions. They must never enter primary fits.
`CUBLAS_WORKSPACE_CONFIG=:0:0` is child-process-only for that intervention; normal
workers remove inherited overrides. Requested allocator bytes are an optional
additional counter, not a substitute for allocated peak.

Drive remains disconnected. Every coordinator call may stop after one probe with
`--checkpoint-each-probe`. Copy, verify and fsync the complete capsule outside
Colab before resuming. An ephemeral `/content` mirror alone is not durable.
Any GPU allocation, version, UUID or usable-budget change requires reconciliation
before model work. No independent Person 2 reproduction is claimed.

## Prospective boundary extension

The legacy 90% conservative sweep stays unchanged. A separate validation mode
can schedule one held-out primary length after a finished paired sweep. It
rechecks source hashes, raw references, repeat coverage and idle evidence. Its
peak fit requires at least five lengths, R-squared >=0.995 and maximum relative
fit error <=2%. Admission inflates the fitted token slope by 5% and requires
the estimate to fit within 95% of the measured usable budget. These are
engineering assumptions, not confidence bounds. The length and policy need a
prospective pushed amendment before execution.

Confirmation-only mode launches one final deliberate attempt using a compatible
finished sweep or validation run, without rerunning safe points. The durable
one-use authority is a local external dispatch claim created with
`scripts/claim_external_attempt.py` before browser dispatch. Its SHA-256 is
required by confirmation-only and recorded in the run manifest. The claim stays
consumed even if dispatch or the cloud session is lost. Never create another
claim/root or retry an uncertain dispatch. Colab also consumes its mirrored
ledger before the GPU worker. Finish all other GPU work first: observed OOM
stops model execution globally.

Same-allocator L=512 zero-workspace controls directly establish the fixed cuBLAS
allocation. Attribution at other lengths is labelled source-supported inference:
pinned PyTorch 2.6 uses a length-independent T4 workspace, and each target's
persistent delta must equal that exact source size. Unsupported allocators,
runtimes or unmatched deltas remain unaccounted.
