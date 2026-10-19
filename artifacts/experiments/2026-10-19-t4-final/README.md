# Saved final-cohort evidence — interrupted at 80 workers

**Valid checkpoint; research execution incomplete.** Colab disconnected after
`final-validation-004` and refused GPU reconnection due to its usage limit.
[Quota screenshot](colab-gpu-quota.png), [interruption record](interruption.json).

- [Original checkpoint ZIP](gpu-checkpoint-080.zip): 7,066,557 bytes;
  SHA-256 `cd0dedd8185a876e1a600b2c3ec8a1b95ac3701f8e22be57d2bab82f51c9c389`.
- All 1,090 archive entries and manifest/referenced raw records are verified;
  80 distinct GPU worker PIDs succeeded. ZIP bytes are unchanged from transport.
- GPU `GPU-89f781af-1ec0-992b-3dc3-46ef2d0941de`, Tesla T4; torch 2.6.0+cu124,
  Transformers 4.51.3; context-only usable bytes 15,527,116,800.
- Execution source `ce227af40476676238cbef9126807fd1d42be76b` and
  [prospective receipt](../../../prereg/final-2026-10-19/receipt.json).
- Primary run `c6b37c71-23e8-4209-a278-2c9f65b6b763`: completed, 48 workers.
- Diagnostics `6e6aeb55-a540-43bc-a4e7-e881ec3d29f2`: completed, 28 workers.
- Held-out `0cb5db86-c7be-41f1-99cf-c3fb73430756`: interrupted, four saved
  workers (default n=3, expandable n=1 at L=114688). Frozen reference and parent
  external acknowledgement precede validation; both are in `audit/`.
- The next worker's outcome was not checkpointed and is unknown. Do not label
  it success or OOM. The original validation journal is preserved unchanged.
- No final confirmation run exists; neither an external dispatch claim nor the
  internal deliberate-attempt ledger was consumed. No OOM threshold is observed.

This cohort is analyzed separately from the original acceptance and earlier lost
T4 allocation. Its complete command logs, versions, source/receipt hashes, raw
telemetry, coordinator journals and external checkpoint acknowledgements are in
`audit/`, `state/` and `local/` inside the ZIP. Drive remained disconnected.

[Final report](../../reports/project-report/report.pdf) ·
[delivery and continuation](../../../docs/REMAINING_WORK.md).

## Final CPU verification

All 109 tests, checkpoint integrity and complete saved-data reproduction passed
in Colab Python 3.11. [Actual command audit](colab-cpu-verification.json) includes
timestamps, return codes, log hashes and the preserved venv setup failure.
The subsequent system-Python path passed. Local clean-commit reproduction and
GitHub Actions also passed. [Archived executed notebook](../../../notebooks/03_p1_executed_t4_2026-10-19.ipynb).

The original delivery-manifest.json describes the publication snapshot at commit
`71660a51d6259dd80cd50e49c1b69bf9e305afb2`; renamed/updated current docs should be
verified against that historical Git snapshot, not mistaken for raw-data changes.
The GPU capsule itself remains byte-identical. Current presentation provenance
is separate from that original delivery manifest.
