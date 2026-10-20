# Presenting the project

## The reliable demo route

1. Open [FINAL_PRESENTATION.ipynb in Colab](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/FINAL_PRESENTATION.ipynb).
2. For immediate presentation, scroll to **Final presentation board — no runtime needed**. Its seven pinned graph images and tables are ordinary notebook Markdown. No connection or execution is needed.
3. For an optional reproduction demo, select a **CPU** runtime, leave Drive disconnected and use **Run all**.
4. Setup checks out the notebook's pinned source commit, verifies the original
   GPU capsule and regenerates every plot/table/report. It does not load model
   weights, import CUDA packages or run an OOM attempt.
5. Present the six numbered sections in order. The table of contents follows the
   deliverables rather than the historical debugging trail.
6. Download the PDF, a CSV or one complete deliverables ZIP only if needed. The
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
| Conclusions | Observed facts and open hardware criteria | Explain the final resource limit; do not improvise a threshold |

## Easy questions and answers

**What does sequence length mean?** One cold forward over L random tokens,
empty DynamicCache, B=1, FP16 and `logits_to_keep=1`. We do not generate to L.

**Why not use FlashAttention-2?** The allocated T4 is not supported. We force
PyTorch's efficient SDPA backend and preserve an actual SM75 kernel trace.

**Why is measured peak larger than KV alone?** The forward also holds activations,
inputs, output/workspace and temporary copies. StaticCache diagnostics additionally
include this HF version's explicit causal mask. We measure KV storage directly.

**Did you find OOM?** Yes: 114,688 tokens succeeded and 131,072 raised typed forward CUDA OOM on the same final T4. The exact maximum lies in an interval, assuming monotonicity. At 131,072 the pinned HF masking path tries to allocate an FP16 [L,L+1] causal mask (32.00024 GiB). This is a framework mask cliff; the KV cache remains linear. See docs/OOM_BOUNDARY.md.

**What happens if Colab has no GPU during the demo?** Nothing essential breaks:
the clean demo runs on CPU from the immutable saved measurements. Use the cached
notebook outputs or standalone HTML if a runtime/network is unavailable.

**Can we run a GPU example live?** The final demo is deliberately CPU-only.
The one permitted OOM allowance is consumed. Any future independent experiment requires new
explicitly authorized, prospectively registered continuation. The presentation's live calculations and checkpoint demo are safe
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

## If Colab cannot render outputs

During the final browser check, Colab reported “Could not load the JavaScript
files needed to display output” and referenced Google login access or third-party
cookies. The backend run still completed: its downloaded notebook verifies all
the then-current test suite and six scientific image outputs. This browser-rendering problem is
separate from the GPU quota limit.

Use the standalone HTML immediately for presentation: it embeds the graphs,
tables and individual downloads and has no Google login dependency. Its content
was verified in the browser. Cached notebook outputs are also in GitHub. Do not
change security/cookie settings merely to demonstrate the science. A CPU
re-execution regenerates the artifacts, but cannot itself repair browser login
or blocked frontend scripts.


## Completed boundary check

Separate T4 boundary confirmation: one cold prefill succeeded at 114,688 tokens and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at 131,072 tokens. The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. The original 111,232-token prediction error is -15.14% to -3.01% using (prediction-limit)/limit. This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.

The final board and report show this independent allocation separately from the six core-study figures. The seventh figure shows the measured success/OOM interval. The CPU demo never repeats the consumed OOM attempt.
