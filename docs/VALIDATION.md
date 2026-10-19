# Validation and evidence

Validation is separated into software correctness, saved-data integrity and
research acceptance. Passing one does not imply the others.

## Software and saved-data checks

- CPU tests cover architecture/GQA/tied weights, preregistration, receipt checks,
  worker lifecycle/errors, counters, sampler limits, locks, fsync/mirrors,
  interruption/recovery, duplicate claims and the global one-use allowance.
- The public journal API reuses the actual coordinator store and propagates
  mirror failures. Its tests cannot bypass lock/identity rules.
- JSON Schema tests validate all 80 actual worker records and all 80 result
  envelopes. Negative bytes and incomplete successful results are rejected.
- Presentation tests verify all six deliverables, original commit proof,
  parametric VRAM/batch calculations, tampering rejection, and CPU-only imports.
- The executed demo must run every code cell, emit six scientific PNG outputs,
  regenerate manifests and export standalone HTML. No GPU measurement is run.

```bash
python -m pip install -e . -r requirements-demo.txt
python -m unittest discover -s tests -v
python scripts/execute_presentation.py --output /tmp/kv-presentation
```

GitHub's Python 3.10–3.13 jobs run the CPU suite. The Python 3.12 presentation job
installs the optional notebook/schema dependencies, runs the complete suite and
executes/exports the presentation. JSON Schema tests are explicitly skipped in
minimal environments without the optional dependency and run in presentation CI.

## Actual GPU evidence

The saved current cohort has 80 distinct successful worker PIDs: 48 paired sweep,
28 diagnostic and four held-out. Hashes, source/config/protocol fingerprints,
request/result identities and chronological references are verified. The original
prediction, later receipt and held-out reference freeze precede their relevant
measurements. Different GPU UUIDs are not pooled.

The original bounded acceptance also tested restoration after both Colab copies
were removed, using an externally preserved verified checkpoint. That is a bounded
restore test, not a claim of full cloud-session recovery on the same GPU.

## Research checks still open

The paired short-length grid, direct KV slope and fixed-length discrepancy
coverage/closure checks pass. Colab quota prevents the remaining two expandable
held-out repeats and final boundary attempt. The validation journal remains
unfinished. A successful lower bound is not a hardware OOM threshold. Strict final
research verification must reject the current partial cohort.

[Current status](PROJECT_STATUS.md) · [Remaining work](REMAINING_WORK.md) ·
[Historical raw evidence](../artifacts/experiments/2026-10-19-t4-final/README.md).
