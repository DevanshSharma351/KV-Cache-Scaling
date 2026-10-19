# Project status

The project is organized into [five phases](../PHASES.md). Implementation,
saved-data analysis, report generation and presentation are integrated.
**Research acceptance remains incomplete because Colab GPU access is quota-blocked.**

| Phase | Verified result | Remaining criterion |
| --- | --- | --- |
| 1 — Derive and preregister | Unique weights/GQA math; original prediction and timestamped receipt preserved | None for original analytical commitment |
| 2 — Isolated probe | Real weight/cache/logits checks, peak/reset lifecycle, both allocators and SM75 efficient-SDPA trace | Reconcile any new GPU before new measurements |
| 3 — Checkpointing and safety | Shared journal API, mirrored fsync, locks/recovery tests, result schema and external per-worker acknowledgements | Do not mark the interrupted validation journal finished |
| 4 — Curves and discrepancy | 48 sweep + 28 diagnostic + 4 held-out workers; exact KV slope; all four curve/accounting checks pass | Two expandable validation repeats and final deliberate boundary attempt; hardware allocator delta remains unidentified |
| 5 — Transfer and presentation | Four transfer cells, compatible published comparison, report, figures, tables, CPU reproduction and clean demo | Final research signoff waits on the missing hardware evidence |

Both allocators succeeded at 114,688 tokens. The original prediction remains
111,232 tokens, with assumption range 76,024–131,072. The successful length is
a lower bound, not a measured OOM threshold. The later fit was frozen before
held-out measurements and does not replace the original commitment.

[Saved evidence](../artifacts/experiments/2026-10-19-t4-final/README.md) ·
[Presentation notebook](../notebooks/FINAL_PRESENTATION.ipynb) ·
[Decision guide](DECISIONS_AND_PHASE_GUIDE.md) · [Remaining work](REMAINING_WORK.md).

Different GPU UUIDs are not pooled. The user authorized a delegated technical
review and one contributor review; no independent human approval is invented.
Drive remained disconnected. No deliberate attempt was dispatched or consumed.
Historical review labels survive only in immutable evidence and Git history.
