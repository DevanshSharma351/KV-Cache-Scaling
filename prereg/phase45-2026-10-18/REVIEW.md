# Implementation amendment review: Phases 4-5

Review performed by Codex on behalf of Person 1 under the user's explicit
standing delegation. This is not independent human review.

The original prediction in commit 7baff51 remains the advance numerical
commitment. Seven successful L=512 workers and one preserved preflight failure
already exist. This amendment is prospective only for the extended implementation
and the newly allocated T4; it does not claim to precede the earlier acceptance.
The new context-only baseline has the same exact usable budget, so the original
central 111232-token estimate and 76024-131072 assumption range are unchanged.

Reviewed: unique tied weights, GQA heads, byte units, input/cache lifetime,
reduced logits, forced efficient SDPA, current execution-module hashes,
child-only allocator and workspace configuration, diagnostic scope separation,
fresh-process isolation, conservative admission and 256-token safe bisection.
97 CPU tests pass. Historical acceptance verification uses the original frozen
Git blobs rather than silently replacing them with new implementation bytes.

The required grid is 512/1024/2048/4096/8192 with three repeats for each allocator.
The existing conservative protocol governs larger points and may censor the
hardware boundary. Such censoring is a reported limitation, never an OOM result.
Separate default-allocator diagnostics use registered chunked/static grids plus
three L=512 zero-cuBLAS-workspace controls and one kernel trace. New diagnostic
paths still require real-GPU validation; mocked tests do not establish it.

Drive stays disconnected. Every newly executed point returns through the
checkpoint barrier. Before any next point, transfer the full state capsule to the
user's local filesystem, validate its manifest and raw references, fsync the file
and directory, and record its SHA-256 in the next visible notebook command.
The old acceptance successfully tested external restoration; continuous cloud
persistence and a full cloud-session restart are not claimed. Colab staging is
ephemeral. Do not advance on a failed export, probe or changed environment.

No deliberate OOM is scheduled by this amendment. The global one-attempt
allowance remains unused; its ledger and all historical raw records must be
preserved. Unsupported explanations remain unaccounted; arithmetic closure alone
cannot satisfy the 90% explanatory criterion. No Person 2 work is represented as
an independent completed reproduction.
