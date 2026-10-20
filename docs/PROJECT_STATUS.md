# Final project status

**20 October 2026: final resource-bounded delivery with measurement limitations.**
The user closed further GPU probing to conserve resources. Implementation,
saved-data analysis, report generation and the six-deliverable presentation work
together. No additional GPU run is planned. This does not claim that every
original research objective was experimentally completed.

| Phase | Delivered result | Scientific limit |
| --- | --- | --- |
| 1 - Derive and preregister | Unique tied weights/GQA formula; original prediction and timestamp | Original assumptions are preserved, not relabelled fitted results |
| 2 - Isolated probe | Fresh workers, exact storage, peak counters and SM75 trace | Primary workload is single prefill; diagnostics change semantics |
| 3 - Checkpoint and recover | Journals, fsync/mirroring, raw schemas and verified external capsules | Original interrupted sweep journals are not fabricated as finished |
| 4 - Measure and explain | 80 core workers; six figures; exact KV slope; byte-level discrepancy accounting | Exact hardware OOM and allocator-context gain were not measured |
| 5 - Generalize and present | Four transfer cases, KV publication check, PDF, CPU notebook and HTML | Published comparison validates KV bytes, not unrun HF peak limits |

Two new L=512 default-allocator workers on 20 October reproduce the core peak
and cache bytes exactly. They are a separate cohort, not two extra points in
the old curve. Total preserved successful workers: **82 across two allocations**.
The bulk loop stopped at its input prompt after both workers exited and passed
idle checks; the final closure verified no live GPU process. No deliberate OOM
attempt was dispatched or consumed. Drive stayed disconnected.

The [final scope](FINAL_SCOPE.md) records the user's change and all limitations.
The strict original full-experiment verifier is preserved and is not claimed to
pass. `python scripts/verify_final_delivery.py` verifies the agreed final evidence
scope and explicitly reports that all original research objectives are not met.

[Five phases](../PHASES.md) · [Decision guide](DECISIONS_AND_PHASE_GUIDE.md) ·
[Presentation notebook](../notebooks/FINAL_PRESENTATION.ipynb) ·
[Report](../artifacts/reports/project-report/report.pdf) · [Live demo](LIVE_DEMO.md).

The clean notebook executes 12 CPU code cells and embeds six scientific figures.
The standalone HTML is the fallback for Colab's intermittent output-renderer
login/cookie error. A CPU replay demonstrates saved-data reproducibility and
requires no model load, CUDA allocation or Drive mount.
