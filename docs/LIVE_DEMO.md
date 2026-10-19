# Presenting the project

## The reliable demo route

1. Open [FINAL_PRESENTATION.ipynb in Colab](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/FINAL_PRESENTATION.ipynb).
2. Select a **CPU** runtime. Leave Drive disconnected. Use **Run all**.
3. Setup checks out the notebook's pinned source commit, verifies the original
   GPU capsule and regenerates every plot/table/report. It does not load model
   weights, import CUDA packages or run an OOM attempt.
4. Present the six numbered sections in order. The table of contents follows the
   deliverables rather than the historical debugging trail.
5. Download the PDF, a CSV or one complete deliverables ZIP only if needed. The
   download cell defaults to `none`, so replay does not scatter downloads.

The committed notebook has executed outputs for immediate viewing. For a browser
presentation without a runtime, open [the standalone HTML](../artifacts/presentation/PROJECT_PRESENTATION.html).
Images are embedded. Re-executing the notebook demonstrates reproducibility of
saved data; it is not a new GPU benchmark.

## What to show

| Section | Demonstration | Point to explain |
| --- | --- | --- |
| 1 — Derivation | Equation, symbols, architecture table and unique-weight formula | GQA uses KV heads; tied embeddings count once |
| 2 — Prediction | Original commit timestamp, later receipt and five-length table | Prediction preceded measurements; assumptions are not fitted results |
| Live explorer | Change usable GiB or batch size and rerun that cell | Context capacity depends on usable memory and workload; no GPU probe runs |
| 3 — Harness | Real coverage/hash table and temporary recovery demo | Evidence survives loss; the next worker waits for a durable checkpoint |
| 4 — Curves | Peak vs original analytical envelope, direct KV slope, allocator slack and boundary chart | Peak slope is not KV slope; successful lower bound is not OOM |
| 5 — Discrepancy | Byte/MiB component rows plus both waterfalls | Closure, explanatory support and causal claims are different |
| 6 — Transfer | Four model/GPU cases, published comparison and percentage errors | The compatible comparison verifies KV blocks, not unrun HF peak limits |
| Conclusions | Observed facts and open hardware criteria | State the quota limitation clearly; do not improvise a threshold |

## Easy questions and answers

**What does sequence length mean?** One cold forward over L random tokens,
empty DynamicCache, B=1, FP16 and `logits_to_keep=1`. We do not generate to L.

**Why not use FlashAttention-2?** The allocated T4 is not supported. We force
PyTorch's efficient SDPA backend and preserve an actual SM75 kernel trace.

**Why is measured peak larger than KV alone?** The forward also holds activations,
inputs, output/workspace and temporary copies. StaticCache diagnostics additionally
include this HF version's explicit causal mask. We measure KV storage directly.

**Did you find the exact context limit?** No. Both allocators succeeded at
114,688. GPU quota prevented a failed upper-bound observation. This is explicitly
incomplete hardware evidence, despite a working prediction/replay pipeline.

**What happens if Colab has no GPU during the demo?** Nothing essential breaks:
the clean demo runs on CPU from the immutable saved measurements. Use the cached
notebook outputs or standalone HTML if a runtime/network is unavailable.

**Can we run a GPU example live?** Only after a valid allocation has been
reconciled and prospectively registered. An ad-hoc model forward would violate
the original workflow and could consume the remaining resources or contaminate
an experiment. The presentation's live calculations and checkpoint demo are safe
CPU operations; the historical GPU audit is the measurement trail.

## Historical audit versus presentation

The supplied Drive notebook remains the detailed audit trail. Its final added
presentation section exposes the six deliverables; run only that section when
using the historical notebook. Do not Run all its old measurement cells. The
clean presentation notebook is the recommended public demo because every cell
is designed for safe CPU replay. Historical exports retain original attribution
and names as provenance; current presentation/docs use the unified five-phase flow.

The optional all-deliverables ZIP packages display artifacts, frozen prediction
files, pinned configs and source snapshots. For execution with historical Git
provenance, use the cloned repository/pinned notebook checkout rather than
treating that artifact ZIP as a replacement Git history.
