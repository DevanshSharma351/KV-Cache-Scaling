# Interim T4 checkpoint: 42 primary probes

This is a partial, externally saved checkpoint, not the final project result.
The original ZIP bytes and member manifest are preserved. Its SHA-256 is
`3f78bba590aef474d7e638b6d66c2e23c921d992e338b37504a00685b53bb33c`.
`verification.json` records measured coverage, fits and unfinished gates.

Default allocation has eight lengths (512–65536), each with three repeats.
Expandable allocation has six lengths (512–16384), each with three repeats.
Both directly inspected cache fits have slope 28672 bytes/token and R-squared 1.
The signed discrepancy tables close, but explanatory coverage remains incomplete
until the matched workspace controls are measured. No OOM was observed or
deliberately dispatched in this checkpoint. The default safe-budget bracket
65536–65792 is an analytical admission boundary, not a hardware OOM threshold.

Probe primary-043 was dispatched in cell 94 of the existing audit notebook.
Its result has not been retrieved. Inspect that cell before resuming; do not
blindly dispatch it again. Browser control was blocked by an automatic approval
review service capacity error, not by a scientific test failure or a denial of
user authorization. Google Drive remains disconnected.

CPU reproduction (use a new output directory):

```bash
python scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-interim/primary-042-checkpoint.zip \
  --output /tmp/kv-primary042
```

The 104-test CPU suite and complete analysis/plot/transfer/report pipeline pass.
The primary run still needs the remaining expandable points and finalization.
Diagnostic controls, prospective held-out checks and the one-use final boundary
attempt must follow their reviewed execution amendment; none is claimed here.
The original numerical prediction remains unchanged.
