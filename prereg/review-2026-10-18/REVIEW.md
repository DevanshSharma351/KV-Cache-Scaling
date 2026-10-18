# Delegated Person 1 review

Reviewer: Codex acting on behalf of Person 1, explicitly requested by the user
on 18 October 2026: “please review everything is correct and working properly
on my behalf”. This is an agent technical review, not an independent human
review or an invented Person 2 approval. The user separately authorized a
one-contributor review policy. No names or work-hour estimates are recorded.

Scope: scientific inputs, architecture/weight/KV math, numerical commitment,
probe lifecycle, request/gate checks, isolated process coordinator, bounded
acceptance scheduling, checkpoint/recovery behavior, runtime/API evidence and
remaining empirical acceptance. Base source 0876cf6; corrected source 9718890.

## Finding resolved before preregistration

A worker's NVML process query could return unavailable attribution without
stopping model load. Fixed to reject unavailable attribution before loading;
a regression test proves rejection. CPU fixture behavior remains explicitly
synthetic. The numerical prediction and measurement protocol are unchanged.

## Findings and evidence

- Read the actual pinned config and independently recomputed the parameter
  expression: 1543714304 unique parameters, 3087428608 FP16 bytes, tied input/
  output storage counted once. Naive duplicate error is 466747392 bytes.
- Independently recomputed 2*28*2*128*2 = 28672 KV bytes/token at B=1 and 6x
  naive query-head overestimate. Batch/length and element-size scaling are tested.
- Independently inverted the integer budget for the fresh 15527116800-byte
  context-adjusted budget: raw optimistic/central/conservative limits are
  146740/111232/76024; config-capped optimistic is 131072. Central required-length
  budgets match the concrete candidate. The range reflects assumptions, not
  empirical/statistical uncertainty or guaranteed admission bounds.
- Runtime metadata and unwrapped installed source/API evidence match the target
  Python 3.11 / torch 2.6.0+cu124 / Transformers 4.51.3. The new GPU UUID is
  recorded separately from historical evidence. No old baseline was relabelled.
- Primary probes start empty DynamicCache, use one cold prefill, FP16,
  logits_to_keep=1, and forced efficient SDPA. Peaking starts before input/cache
  construction; retained and released snapshots do not retain inspected tensors.
- Fresh subprocesses set allocator policy before import. Parent performs metadata
  and NVML checks; worker checks identity, driver, context budget and code hashes.
  Non-OOM errors cannot be inferred as typed OOM from a text log.
- Receipt tests require a pushed prediction ancestor, subsequent pushed receipt,
  unchanged frozen code/input bytes, and real review metadata. A receipt alone
  cannot bypass durable starts or duplicate claims. Human review authenticity
  cannot be established cryptographically by Git.
- CPU tests exercise synthetic worker faults, mirror failure, torn journals,
  orphan recovery, duplicate keys, bounded logs and the global attempt token.
  Actual Linux forced process exit, volatile-directory deletion, torn recovery
  and synthetic-token refusal also passed in the recorded notebook. These are
  process/filesystem checks, not proof of cloud durability or GPU correctness.
- User requested no Drive mount. Acceptance pauses after every new probe and
  requires an externally fsynced, verified local checkpoint before continuation.
  Colab staging is ephemeral. Unattended bulk execution is outside this method.
- Kernel-profile records have a separate request/point identity and are excluded
  from primary fits. Acceptance is limited to seven fresh L=512 workers and
  cannot consume a deliberate token or claim a context boundary.

## Decision and empirical conditions

Approve the unchanged numerical commitment and bounded acceptance protocol for
formal freeze and subsequent receipt. This approval permits testing the GPU
implementation; it does not assert that unrun GPU/backend/allocator checks pass.
Preserve failures and deviations. Any correction to frozen code requires a new
prediction/receipt before further model measurements. No deliberate OOM is
approved for this acceptance run.

Local verification: 83 CPU tests passed; source compilation and diff checks pass.
Colab verification: the corrected 83-test suite passed in the pinned runtime
(notebook cell 24), before any model execution. The formal receipt records
the actual completed-review timestamp.
