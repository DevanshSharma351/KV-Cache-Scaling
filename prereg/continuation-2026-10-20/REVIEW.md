# Prospective continuation review - 20 October 2026

The user requested completion of the remaining GPU work and previously delegated
technical review to Codex. This records that delegated technical review; it is
not an independent human review. No contributor allocation is used.

The original prediction at commit 7baff5121ed643f27ba06959065855a8913f8e8c
remains unchanged: 111232 tokens for 15527116800 usable bytes, with an assumption
range of 76024-131072. The 19 October cohort has 80 successful workers (48 primary,
28 diagnostics and four held-out validations) and no deliberate dispatch. These
measurements are known when this continuation is registered. This package is an
execution amendment, not a new claim to have predicted unseen results originally.

The old runtime was lost. The new T4 UUID is
GPU-440a7ac2-62c4-253c-ba99-111c2f1e8466. Its observations must not be pooled with
the earlier GPU UUID to fill repeats or fit one curve. The software is restored
to Python 3.11, torch 2.6.0+cu124 and Transformers 4.51.3 before model work.

## Reviewed method and execution order

1. Retain the original config, derivation and numerical scenarios. The current
   implementation freezes all execution module hashes and verifies them before
   each launch. One cold FP16 B=1 prefill, empty DynamicCache, logits_to_keep=1,
   forced efficient SDPA, fresh workers, synchronized reset/max peak counters.
2. Rebuild the new allocation's own paired primary reference sweep: the required
   five lengths and the admitted 16384/32768/65536 lengths, three repeats per
   allocator. Preserve the 90% conservative admission and 256-token bisection.
3. Repeat the separate registered diagnostic grid on this allocation: 12 chunked,
   nine StaticCache, six zero-workspace controls and one actual kernel trace.
   This permits same-cohort discrepancy attribution without borrowing controls.
4. Freeze the finished primary manifest/journal and fitted peaks externally
   before L=114688. Run three held-out repeats per allocator, only if the existing
   calibrated admission gate passes. These replace none of the original records.
5. Perform no further ordinary model work after the final default L=131072
   boundary attempt. Consume the external durable project-wide one-use dispatch
   claim before sending its command, then the internal ledger before launching.
   No retry after an uncertain dispatch; success is censoring, not an OOM.

The source tree and prospective receipt must be pushed before model loading.
Every worker pauses until a complete ZIP is transported to the user's filesystem,
all member hashes are checked, the file is fsynced, and the exact digest is
acknowledged in the notebook. Google Drive remains disconnected. The experiment
identity and durable external allowance are preserved across the new allocation;
a runtime loss never restores an already consumed allowance.

## Acceptance and reporting

Keep incompatible UUID/runtime cohorts separate. Verify exact tied-weight and
GQA KV arithmetic, the required repeated grid, cache slope, diagnostics, held-out
repeats, journals and source/receipt timing. The final typed CUDA OOM, if it occurs,
provides an upper point, not an exact threshold at 256-token resolution. The safe
bisection is policy-censored and must not be presented as a hardware bracket.

At least 90% supported discrepancy coverage is required for material gaps, with
explicit unaccounted bytes and the existing small-gap rule. Counter/lifetime
closure is not pure causal activation/fragmentation decomposition. StaticCache
includes the pinned implementation's explicit causal mask. Published verification
checks the KV submodel only; transferred HF context thresholds remain unmeasured.
An exact allocator hardware-limit delta cannot be inferred from a single allowed
OOM. Do not weaken delivery checks to manufacture a pass if any run is missing.
