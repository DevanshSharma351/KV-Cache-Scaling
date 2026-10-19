# KV-Cache Scaling

**Predict the context limit before measuring it, then explain the difference.**
We built a config-derived memory model, a checkpointing GPU harness and a
reproducible analysis/presentation pipeline in [five phases](PHASES.md).

**Measured evidence:** 80 successful T4 workers; both allocators reached
114,688 tokens; inspected KV storage grows by exactly 28,672 bytes/token.
**Research acceptance remains incomplete:** Colab's GPU quota blocked two final
validation repeats and the single boundary attempt. No hardware OOM threshold
is claimed. [Current status](docs/PROJECT_STATUS.md) · [Remaining work](docs/REMAINING_WORK.md).

## Present the project

[Open the clean presentation notebook in Colab](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/FINAL_PRESENTATION.ipynb).
Choose a **CPU runtime** and **Run all**. It regenerates the saved-data results
and presents each required deliverable; no GPU, model download or Drive mount
is needed. Change usable VRAM/batch size for an analytical live demo, or run the
temporary-filesystem checkpoint recovery demo. Downloads are optional.

- [Executed presentation notebook](notebooks/FINAL_PRESENTATION.ipynb)
- [Standalone presentation HTML](artifacts/presentation/PROJECT_PRESENTATION.html)
- [Project report PDF](artifacts/reports/project-report/report.pdf)
- [Easy decision and phase guide](docs/DECISIONS_AND_PHASE_GUIDE.md)
- [Live-demo instructions and questions](docs/LIVE_DEMO.md)

## The six deliverables

| Deliverable | Where to inspect it directly |
| --- | --- |
| Derivation with every symbol defined | [Derivation](docs/DERIVATION.md); presentation section 1 |
| Original prediction and commit timestamp | [Unchanged commitment](prereg/formal-2026-10-18-t4/prediction.md), [original commit](https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/7baff5121ed643f27ba06959065855a8913f8e8c); section 2 |
| Checkpointing sweep harness | [Coordinator](src/kv_cache_scaling/sweep.py), [persistence API](src/kv_cache_scaling/journal.py), [contract](docs/DATA_CONTRACT.md); section 3 |
| Memory curves with analytical overlays | [Separate PNG/PDF figures](artifacts/figures/p1-2026-10-19/); section 4 |
| Discrepancy table in bytes / MiB | [CSV](data/processed/p1-2026-10-19/discrepancy.csv); section 5. Fixed-length peak accounting is available; OOM-threshold error is unmeasured |
| Cross-model / cross-GPU predictions with verification | [Transfer table](data/processed/p1-2026-10-19/transfer/transfer.md); section 6. Published comparison validates KV blocks, not unrun HF peak limits |

Some immutable evidence paths retain their historical names. Current docs and
presentation use the unified project flow; frozen predictions, receipts and audit
exports retain their original bytes and review attribution.

## Reproduce on CPU

Use Python 3.11–3.13 for the pinned plotting/report packages and a new output path.

```bash
python -m pip install -e . -r requirements-analysis.txt
python -m unittest discover -s tests -v
python scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip \
  --output /tmp/kv-reproduction
```

For local notebook execution/HTML export, install `requirements-demo.txt`.
[Reproduction guide](docs/REPRODUCTION.md) · [Architecture and scaffold](PROJECT_FLOW.md).
Generated figures/tables/report carry hashes; the original ZIP is preserved.

## Frozen measurement decisions

| Setting | Decision |
| --- | --- |
| Model | Pinned ungated `Qwen/Qwen2.5-1.5B`; GQA and tied embeddings; appropriate T4 weight footprint |
| L | One cold prefill of exactly L tokens, B=1; no generation |
| Dtype/logits | FP16 weights/cache; eval + inference mode; `logits_to_keep=1`; no labels/loss |
| Attention | HF SDPA with forced PyTorch efficient attention; actual SM75 trace; no eager fallback or FA2 |
| Cache | Fresh DynamicCache; chunked/static paths are separate diagnostics |
| Isolation/peaks | Fresh process per point/repeat; allocator env before CUDA; peak reset and max counters |
| Budget | Context-adjusted usable VRAM; parametric prediction with assumption range |
| Safety | Conservative scheduling, external checkpoint before advance, one-use final attempt gate |

Changing a measurement decision requires prospective registration before new
model work. An archived receipt cannot authorize another GPU silently. See
[measurement protocol](docs/MEASUREMENT_PROTOCOL.md), [requirements](docs/REQUIREMENTS.md)
and [evidence](artifacts/experiments/2026-10-19-t4-final/README.md).
