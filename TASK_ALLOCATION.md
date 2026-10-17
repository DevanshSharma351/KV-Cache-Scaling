# One-week, two-person task allocation

**Person 1 handles 55% of the total project work; Person 2 handles 45%.**
The schedule covers seven consecutive days, starting on the team's chosen day 1.
The percentages below are shares of the complete project, based on deliverable
complexity, implementation responsibility and validation effort. They are
planning allocations, not measured contributions.

All original requirements, both diagnostic grids, three-repeat measurements,
external verification and final reproduction remain in scope. Reserve GPU access
across days 3–5 and identify a backup provider on day 1.

## Person 1 — 55%: scientific model and core measurement implementation

| ID | Work package | Project share | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| A1 | Derivation, final assumptions, VRAM sensitivity and preregistration | 8% | `model_math.py`, `prereg/`, `docs/DERIVATION.md`; remote prediction receipt | B1 context-only inventory and B6 review | Five-length/VRAM predictions and unique weights/GQA checks reviewed and pushed on day 1 before model measurements |
| A2 | Primary probe, telemetry and backend/cache verification | 13% | `probe.py` primary path, `telemetry.py`, GPU smoke evidence; interface for B4 diagnostic paths | A1 before GPU execution; shared B2 schema | Fresh-process reset/peak lifecycle, efficient backend, primary shapes/dtypes and cache bytes verified |
| A3 | Coordinator, safety gate and boundary estimation | 13% | `sweep.py`, CLI, fault-injection integration with B2 | A1, A2, B2 for integrated checks | Resume, pairing, timeouts, admission gate and global attempt token work; safe/observed/estimated limits separated before bulk runs |
| A4 | Cache regression, residual and discrepancy analysis | 12% | `analysis.py`, processed tables; scientific results supplied to B3/B5 | A2 column contract; B4 data | Separate primary/diagnostic fits; slope <=3% error; gap closure/90% evidence and residual fractions checked |
| A5 | Transfer calculations and comparable external validation | 5% | `generalize.py`, four uncertainty-aware predictions; source/config evidence from B6 | B6 source preparation; A4 for final calibrated values | Two-model/two-GPU predictions and compatible published comparison with percentage error |
| A6 | Scientific integration, conclusions and final technical review | 4% | Technical conclusions, integration fixes, scientific review of B5 report | Draft structure starts early; finalization after A3–A5/B5 | Scientific claims match evidence; limitations/deviations reviewed; final fixes integrated |
| | **Person 1 total** | **55%** | | | |

## Person 2 — 45%: reliability, diagnostics, presentation and reproduction

| ID | Work package | Project share | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| B1 | Linux/Colab setup, driver inventory and runtime lock | 5% | `notebooks/`, dependency capture, context-only manifest, primary/backup GPU plan | Scaffold | Inventory ready on day 1; remaining setup by day 2; persistence/restart instructions tested |
| B2 | Journal, schema validation and recovery tests | 9% | `journal.py`, schema, persistence tests; integration evidence for A3 | Contract agreed with A2/A3 | Start/result fsync, lock, torn-line recovery, duplicate keys and crash-safe attempt ledger tested |
| B3 | Reproducible figures, comparison tables and visual checks | 9% | `plotting.py`, all five figure families and dynamic/chunked/static comparison table | A4 column contract; labelled fixtures before data | Metrics, units, bands, uncertainty and censoring correctly labelled; figures regenerate from processed data |
| B4 | Diagnostic execution paths, paired runs and data coverage | 8% | Chunked/static diagnostic helpers in `probe.py`; primary and diagnostic journals/manifests | A2 primary probe interface; A1 gate and A3/B2 checks before bulk runs | Both diagnostic paths implemented and checked; paired sweeps/repeats complete; statuses and any justified single attempt preserved |
| B5 | Report assembly, independent reproduction and final packaging | 10% | Consolidated report, clean-environment transcript, README, PDF, figure/table regeneration and hashes | Report structure starts early; final results after A4/A5/B3 | Complete report assembled from scientific inputs, outputs independently reproduced, PDF visually reviewed and delivery packaged |
| B6 | Source/config preparation and independent scientific review | 4% | Pinned transfer-config/source inventory, day-1 prereg review, final evidence checklist | A1 initially; coordinate with A5; A2/A4 at final review | Compatible published source located early; config provenance and final scientific arithmetic checked |
| | **Person 2 total** | **45%** | | | |

## Daily milestones and handoffs

| Day | Person 1 — primary responsibilities | Person 2 — primary responsibilities | End-of-day evidence |
| --- | --- | --- | --- |
| 1 | A1 derivation/predictions and preregistration; A2 worker design | B1 context-only inventory, persistent storage and GPU plan; B6 prereg review | Predictions and remote receipt pushed before model measurements; interfaces agreed |
| 2 | A2 primary worker/telemetry; define analysis columns and A5 source requirements | B1 runtime capture; B2 journal/schema; B6 transfer configs/source discovery; start B5 report structure | Worker ready for final integration; schema shared; plausible compatible external source identified |
| 3 | Complete A2; integrate A3 coordinator with B2 | Finish B2 fault tests; B3 fixture plots; B4 diagnostic helpers using the primary-probe interface | GPU smoke/backend checks and recovery tests pass; basic isolated harness and diagnostic smoke paths work |
| 4 | Complete A3 safety/boundary checks; begin A4 retained-cache analysis | B4 paired primary runs and diagnostic checks; B3 measured-data wiring | Integrated gates pass before bulk sweeps; paired required-grid data durable; cache slope checked early |
| 5 | Complete A4 discrepancy/residual analysis; prepare A5 transfer calculations | B4 complete diagnostic grids/repeats and data audit; B3 figures/comparison table; B5 report assembly | Raw data locked; diagnostics complete; justified OOM attempt/censoring recorded; discrepancy and figures reviewed |
| 6 | Complete A5 transfer/verification calculations; A6 technical conclusions and scientific review | B5 clean reproduction and complete report/PDF draft; B6 source/claim checks | Four predictions and compatible comparison included; complete report draft; independently regenerated tables/figures |
| 7 | A6 recovery, scientific fixes and final technical sign-off | B5 final reproduction/PDF review and packaging; B6 evidence checklist | Every requirement mapped to evidence; final repository/report pushed; unresolved criteria explicitly recorded |

## Responsibility boundaries

Person 1 owns the primary measurement and prediction logic. Person 2 owns the
chunked/static diagnostic execution paths, durable data handling, run operation,
report assembly and independent reproduction. This gives Person 2 substantive
implementation and verification responsibilities alongside presentation work.

For `probe.py`, Person 1 lands the primary-probe interface first; Person 2 adds
separate diagnostic helpers against that interface and Person 1 reviews their
measurement semantics. Agree function boundaries before editing the same file.
Person 2 supplies transfer-config provenance and published-source evidence;
Person 1 owns the transfer equations and final comparison calculations.
Person 2 assembles the full report; Person 1 supplies and reviews the scientific
methods, results and conclusions. Each work package has one accountable owner;
reviews are already included in its project share.

Person 2 can build journal tests and plots on labelled synthetic fixtures before
GPU access. Only one coordinator owns the GPU at a time. Run operations follow
A3's frozen controls; they do not introduce a second independent harness. Start
source discovery on day 2 so incompatible evidence is found before final analysis.

## Completion controls and recovery

- Review task IDs, evidence, blockers and the next handoff at the start/end of
  each day. Keep one coverage checklist for primary, diagnostic, repeat, figure
  and report deliverables.
- Day 1: identify primary/backup GPU access and persistent storage. Day 3:
  escalate integration blockers. Day 5: audit coverage before locking inputs.
- Protect day 7 for recovery, integration and final review. A recovery run still
  follows preregistration, GPU identity checks and the global OOM-attempt budget.
  Regenerate affected analyses and figures after any new data.
- Keep both diagnostics, all planned repeats, scientific checks, external
  verification and clean reproduction. Automate data flow and move independent
  work earlier rather than silently dropping deliverables.
- If an external constraint prevents evidence collection after recovery, record
  the incomplete criterion and concrete recovery steps. Do not manufacture
  thresholds, waive the accounting target without disclosure or use an
  incompatible published result as verification.
