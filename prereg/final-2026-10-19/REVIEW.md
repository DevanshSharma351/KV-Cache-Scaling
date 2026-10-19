# Prospective final-execution amendment review

Codex reviewed this package on behalf of Person 1 under the user's explicit
standing delegation. This is one contributor review, not independent human
review. The allocation remains anonymous Person 1 55% / Person 2 45%.

The original numerical commitment (7baff51) is preserved: 111232 tokens, with
76024–131072 assumption range for 15527116800 usable bytes. Seven original
acceptance workers and 43 later primary workers already exist. The later runtime
was lost after primary-043; its original packet was recovered and verified.
Those results inform this amendment and are disclosed, not represented as unseen.
The new GPU UUID requires its own measurement cohort and reference sweep.

Reviewed code: cold FP16 prefill, B=1, DynamicCache, logits_to_keep=1, forced
memory-efficient SDPA on SM75; exact tied-weight and GQA arithmetic; reset peak
counters before probe allocations; fresh child processes; child-only allocator
configuration; no eager/FlashAttention-2 fallback. The context-only baseline uses
torch 2.6.0+cu124, Transformers 4.51.3 and an otherwise idle T4. 104 CPU tests
pass, including receipt/source drift rejection, external checkpoints, held-out
admission and exclusive final dispatch. CPU tests do not establish GPU results.

Prospective execution order, all in the existing notebook:

1. Complete the paired conservative primary sweep (normally 48 workers: eight
   lengths through 65536, three repeats per allocator). The fixed original
   90% conservative admission and 256-token safe bisection remain unchanged.
2. Separate diagnostics: chunked DynamicCache at 512/2048/8192/32768 (three each),
   StaticCache at 512/2048/8192 (three each), zero-cuBLAS-workspace controls at
   512 (three per allocator), and one separate kernel trace: 28 workers total.
3. Hold out L=114688 with three primary repeats per allocator (six workers).
   The completed new-GPU primary sweep is the sole reference. Its immutable
   manifest/journal and fitted coefficients are captured before dispatch.
   Admission requires >=5 lengths, R-squared >=0.995, maximum relative fit error
   <=2%, a 5% slope inflation and <=95% of usable VRAM. Stop if it fails.
4. After every other GPU experiment, dispatch L=131072, default allocator,
   exactly once. Consume the durable local external claim before browser
   dispatch and also the Colab ledger before the worker. No retry after an
   uncertain dispatch. A success is reported as success, never called OOM;
   only a typed CUDA OutOfMemoryError establishes the failed upper point.

All stages pause after each ordinary probe. Before advancing, export the full
capsule, verify transport/member/raw-reference hashes and fsync outside Colab.
Drive stays disconnected. Use the same experiment identity and one-use allowance
across runtime loss; no old OOM allowance was consumed. New failures stop the
workflow and remain visible. No model probes follow the final deliberate attempt.

Analysis separates primary, chunked, static, profiler and workspace cohorts.
Direct same-allocator controls can support L512 workspace attribution. Larger
lengths require labelled inference from pinned PyTorch source and an exact
matching persistent delta. Unsupported bytes stay unaccounted. Arithmetic
closure is not causal attribution; apply the >=90% explanatory acceptance rule.
Hardware-success bounds, safe-budget censoring, actual OOM and config/quality
caps remain distinct. Do not claim an exact allocator hardware-limit delta if
one allowed OOM cannot identify it. Transfer verification checks published KV
block bytes only, not unmeasured HF hardware thresholds. Person 2 independent
reproduction remains explicitly outside this delegated P1 review.
