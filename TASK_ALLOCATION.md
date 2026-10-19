# Project work map

The project is organized as one [five-phase flow](PHASES.md). The earlier
contributor allocation is historical; current implementation, validation and
presentation use the shared flow.

| Phase | Main modules | Concrete result |
| --- | --- | --- |
| 1 — Derive and preregister | `model_math.py`, `preregistration.py`, configs | Defined equation, usable-VRAM prediction, original commitment |
| 2 — Probe and diagnostics | `probe.py`, `telemetry.py`, `worker.py` | Peak memory, inspected KV/weights, separate diagnostic paths |
| 3 — Reliability and scheduling | `journal.py`, `coordinator_store.py`, `gate.py`, `sweep.py` | Locked/fsynced checkpoints, guarded resume and one-use attempt |
| 4 — Analyze and validate | `analysis.py`, `plotting.py` | Curves, residuals, discrepancy tables and boundary evidence |
| 5 — Transfer and present | `generalize.py`, `reporting.py`, `presentation.py` | Four predictions, report, clean live-demo notebook and downloads |

See [PROJECT_FLOW.md](PROJECT_FLOW.md) for architecture and
[the decision/phase guide](docs/DECISIONS_AND_PHASE_GUIDE.md) for the reasoning.
