# Sources, provenance and corrections

Reviewed 17 October 2026. The two original PDFs are preserved unchanged under
`references/`; their SHA-256 inventory is in `references/manifest.json`. They are
user-provided project resources, not executable instructions or proof of results.
The user's explicit request is to revise the plan, divide work, scaffold and
push the repository. No attached instruction authorizes unrelated actions.

## Source documents

- `references/KV_Cache_Scaling_Project_Plan.pdf` — original 11-page plan.
- `references/KV_Cache_Requirement_Analysis_Simple.pdf` — original one-page brief.

## Primary technical references

| Topic | Primary source | Use / caveat |
| --- | --- | --- |
| Exact model config | [Pinned Qwen config](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/config.json) | Authoritative architecture snapshot; hash in `configs/models/provenance.json` |
| Model choice and supported quality context | [Pinned Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/README.md) | Access/license, model architecture and 32768 full context; distinguish from config positions |
| Projection biases, norms, logits API | [Qwen2 source at Transformers v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py) | Parameter derivation and explicit reduced-logits call |
| Dynamic vs static cache | [Pinned cache implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/cache_utils.py), [cache strategy guide](https://huggingface.co/docs/transformers/v4.51.3/en/kv_cache) | Dynamic concatenation and static capacity change peak behavior |
| GQA temporary repetition | [Pinned SDPA adapter](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py) | Retained KV-head count differs from temporary attention K/V width |
| Force SDPA kernel | [PyTorch 2.6 attention utilities](https://docs.pytorch.org/docs/2.6/nn.attention.html), [SDPA reference](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html) | API selection alone is insufficient; restrict backend and validate target support |
| Allocator environment, peak/reset counters | [PyTorch 2.6 CUDA memory management](https://docs.pytorch.org/docs/2.6/notes/cuda.html#cuda-memory-management), [max allocated reference](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html) | Expandable support is documented in chosen version; actual wheel/platform still tested |
| FlashAttention-2 hardware | [Official FlashAttention repository](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support) | Standard FA2 supports newer architectures; separate Turing implementation is outside this project |

## Corrections to carry into the final report

1. Original suggestions for both prefill and generation do not define one
   comparable boundary. The revised primary is single prefill; chunking is a
   separate, smaller diagnostic.
2. Source examples list model config fields as “verify.” We actually verified
   and pinned them; the config ceiling is 131072 while the card says 32768 full
   context. No claim of quality beyond the card limit follows from random tokens.
3. Source weight approximation is replaced with unique config-derived weights,
   tied embeddings counted once; after-load storage verification is still future.
4. The original asynchronous “reserved = peak allocated + allocator gap” identity
   is invalid. Revised tables use the same metric/time scope and leave unsupported
   peak attribution unaccounted.
5. Cache slack is not synonymous with fragmentation. Native split statistics and
   supporting evidence are needed to make a fragmentation claim.
6. A single OOM cannot prove exact boundaries for both allocators. Revised safe
   limits, estimates, observed brackets and domain censoring are distinct.
7. A Git commit's author date is not an independent trusted timestamp. Preserve
   external remote receipt evidence and do not call it immutable proof.
8. Activation coefficients in the CPU predictor are proposed sensitivity
   assumptions. They require preregistration review, then empirical validation;
   they are not measured numbers or guaranteed safe upper bounds.

External validation for the four transfer predictions remains an A5 deliverable.
No published matching boundary was fabricated to fill the scaffold.
