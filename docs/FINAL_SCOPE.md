# Final scope and measurement limits

**20 October 2026 - final delivery with measurement limitations.**

The user requested that we stop spending resources on repeated GPU probes and
finish the report and final presentation using the working implementation and
available evidence. This supersedes the planned full repeat sweep on the new T4.
No additional GPU work is planned under this delivery scope. The five project
phases remain intact; contributor names and percentage allocations are absent.

## What is delivered

All six requested artifact types are present: the symbol-defined derivation;
original numerical prediction and original Git timestamp; checkpointing harness;
six scientific figures with analytical overlays; byte-level discrepancy tables;
and two-model/two-GPU predictions with a published KV-block comparison. The
clean final notebook and standalone HTML present each item independently of the
PDF and regenerate the saved-data analysis on CPU.

The core study retains 80 successful workers from the 19 October T4 allocation:
48 primary sweep, 28 diagnostics and four held-out checks. Its paired short-length
curves, exact weights/cache accounting, slope checks and fixed-length discrepancy
coverage pass. Both allocators succeeded at 114688 tokens. The original central
prediction was 111232 tokens, so an observed successful input exceeds it by 3456
tokens; this is not an exact OOM-threshold error.

On 20 October, a different T4 was allocated. We restored the pinned environment,
registered its execution amendment before model work, and completed **two fresh
L=512 default-allocator replication workers**. Both match the core study's weight,
cache and allocated-peak bytes. They are a separate reproducibility check and
are not appended to the old curve or used to fill its missing repeats.

The loop was stopped while waiting for checkpoint acknowledgement after worker
2. Both results had exited and passed idle checks; no model worker was interrupted.
The final closure cell verified no live GPU processes and exported the complete
capsule. Its transport and every member hash were checked and the ZIP was fsynced
on the user's filesystem. Google Drive remained disconnected.

## What the results do not establish

- No deliberate OOM attempt was dispatched. The project-wide allowance is unused.
- 114688 is a demonstrated successful lower bound, not the exact maximum context.
- The original cohort has default n=3 and expandable n=1 at L=114688. Additional
  repeats were deferred; observations from different GPU UUIDs are not pooled.
- Without a failed upper point, exact OOM-threshold error and a quantitative
  decomposition of an OOM-limit gap cannot be measured. The discrepancy tables
  instead explain tracked tensor-peak differences at fixed lengths.
- Allocator slack/reserved-peak differences do not establish pure fragmentation
  or an exact gain in hardware context capacity.
- The published comparison verifies KV block arithmetic, not the transferred
  Hugging Face peak-context limits on unrun models/GPUs.

This is a completed **resource-bounded evidence delivery**, not a claim that every
original research objective has been experimentally satisfied. The original
strict completion verifier and missing-evidence checks are preserved. They must
not be weakened or reported as passing for this data. A separate final-scope
verifier checks the delivered evidence and these disclosures.

## Where to present it

- `notebooks/FINAL_PRESENTATION.ipynb`: safe CPU Run All; six figures and separate downloads.
- `artifacts/presentation/PROJECT_PRESENTATION.html`: standalone offline presentation.
- `artifacts/reports/project-report/report.pdf`: integrated final report.
- `docs/DECISIONS_AND_PHASE_GUIDE.md`: plain-language decisions and five-phase story.
- `configs/delivery.json`: machine-readable evidence sources and scope limits.

Optional future research would require a separately authorized prospective
continuation. It is not part of the current delivery or a reason to keep running
GPU workers now. Historical predictions, receipts and raw data remain unchanged.
