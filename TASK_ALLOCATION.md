# One-week, two-person task allocation

Planning budget: **80 active hours over seven consecutive days**. Devansh:
**52 h (65%)**, averaging about 7.4 h/day; teammate: **28 h (35%)**, averaging
4 h/day. These are planning assumptions for the extra work requested, not logged
contributions. Day 1 is the team's chosen start day. GPU waiting/download time
is additional and can overlap CPU tasks; reserve initial GPU windows across days
3–5 and a backup provider on day 1.

All original requirements, both diagnostic grids, three-repeat measurements,
external validation and final reproduction remain in scope. Hours include reviews
and the protected day-7 recovery/finalization block; do not count those twice.

## Devansh: core implementation and scientific ownership

| ID | Work package | Hours | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| A1 | Derivation, final assumptions, VRAM sensitivity and preregistration | 6 | `model_math.py`, `prereg/`, `docs/DERIVATION.md`; remote prediction receipt | B1 context-only inventory and B6 review; other B1 work can continue | Five-length/VRAM predictions and unique weights/GQA checks reviewed and pushed on day 1 before model measurements |
| A2 | Probe + telemetry + kernel/cache verification | 11 | `probe.py`, `telemetry.py`, smoke evidence, both cache/mode diagnostics | A1 before GPU execution; shared B2 schema for integration | Fresh-process reset/peak lifecycle correct; efficient backend, shapes, dtype, cache bytes and diagnostic modes verified |
| A3 | Subprocess coordinator, safety gate, boundary estimation | 11 | `sweep.py`, CLI, fault-injection tests with B2 | A1, A2, B2 for integrated tests | Resume, pairing, timeouts, admission gate and global attempt token work; safe/observed/estimated limits separated before bulk runs |
| A4 | Cache regression, residual and discrepancy analysis | 10 | `analysis.py`, processed tables, report methods/results | A2 column contract; B4 data for final results | Separate primary/diagnostic fits; slope <=3% error; gap closure/90% evidence and residual fractions checked |
| A5 | Two-model/two-GPU transfer and published comparison | 6 | `generalize.py`, pinned config-only snapshots, compatible cited comparison | Source/config work starts early; A4 needed for final calibrated values | Four uncertainty-aware predictions and comparable published quantity with percentage error |
| A6 | Integration, report, final review and recovery | 8 | Complete technical report, integration fixes, reviews of B2/B3 | Draft structure starts early; finalization after A3–A5/B5 | Three hours of day-6 report integration plus five protected day-7 hours for recovery, final fixes and release; evidence/deviations reviewed |
| | **Total** | **52** | | | |

## Teammate: data reliability, plots and reproducibility

| ID | Work package | Hours | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| B1 | Linux/Colab setup, driver inventory and runtime lock | 4 | `notebooks/`, requirements capture, environment-only manifest, primary/backup GPU plan | Scaffold | Context-only inventory ready on day 1; remaining setup/lock work by day 2; persistence/restart instructions tested |
| B2 | Journal, schema validator and resume fault tests | 6 | `journal.py`, schema, persistence tests | Contract agreed with A2/A3 | Start/result fsync, lock, torn-line recovery, duplicate keys and crash-safe attempt ledger tested before bulk runs |
| B3 | Reproducible plots and labelled tables | 6 | `plotting.py`, all five figure families and cache/mode comparison table | A4 column contract; labelled fixtures before data | No mixed metrics; frozen prediction band, uncertainties, censoring and diagnostic labels preserved |
| B4 | Paired run operation and data checks | 4 | Durable primary + both diagnostic journals/manifests; coverage checklist | A1 gate and A2/A3/B2 integration checks | Safe paired sweeps, repeats, diagnostics and any justified single attempt saved; same UUID within comparisons; unresolved points flagged |
| B5 | Independent reproduction, README and final packaging | 6 | Clean-environment transcript, table/figure regeneration, report PDF, hashes | Prepare checklist early; final checks after A4/A5/B3 | Results regenerate independently; PDF visually reviewed; raw/derived hashes and exact commands packaged |
| B6 | Review preregistration and final scientific evidence | 2 | One hour day-1 prereg review and one hour day-7 evidence review | A1 initially; A2/A4 at final review | Tied weights, peak scope, frozen assumptions, gap arithmetic and all requirements independently checked |
| | **Total** | **28** | | | |

## Daily hours, milestones and handoffs

| Day | Devansh: hours and work | Teammate: hours and work | End-of-day evidence |
| --- | --- | --- | --- |
| 1 | **8 h:** A1 6 h; A2 design 2 h | **4 h:** B1 3 h; B6 review 1 h | Protocol, predictions and remote receipt pushed; context-only inventory/persistent path/GPU windows agreed |
| 2 | **8 h:** A2 worker/telemetry/diagnostics 7 h; A5 published-source/config discovery 1 h | **5 h:** B1 remaining 1 h; B2 journal/schema 4 h | Worker ready for final integration; schema shared, runtime capture ready; plausible compatible external source identified early |
| 3 | **8 h:** A2 final 2 h; A3 coordinator 6 h | **4 h:** B2 final 2 h; B3 fixture plots 2 h | GPU smoke/efficient-backend checks; journal fault tests; basic end-to-end isolated harness works |
| 4 | **8 h:** A3 final safety/boundary integration 5 h; A4 initial analysis 3 h | **5 h:** B4 primary run operations 3 h; B3 plot wiring 2 h | Integrated safety gates pass before bulk sweeps; paired required-grid data durable; cache slope checked early |
| 5 | **8 h:** A4 final analysis 7 h; A5 transfer preparation 1 h | **4 h:** B4 final diagnostics/repeats 1 h; B3 final figures 2 h; B5 preparation 1 h | Raw data locked, both diagnostic grids complete, one justified OOM attempt/censoring recorded; discrepancy/figures reviewed |
| 6 | **7 h:** A5 remaining transfer/verification 4 h; A6 report integration 3 h | **4 h:** B5 clean reproduction, export and issue list | Four predictions + compatible source; complete report draft; independently regenerated tables/figures |
| 7 | **5 h:** A6 protected recovery, final fixes and release | **2 h:** B5 final reproduction/PDF check 1 h; B6 scientific review 1 h | All requirement evidence reviewed; final PDF/repository packaged and pushed; open issues explicitly recorded |
| **Total** | **52 h** | **28 h** | **80 active hours, 65/35 split** |

The teammate can implement journal tests and plotting on labelled synthetic
fixtures before GPU access. Devansh defines scientific columns before the plotting
handoff. Only one person coordinates GPU runs at a time to avoid memory
contamination. B4 is execution under A3's frozen controls, not a second harness.

Source/config discovery for A5 starts with its allocated hour on day 2;
do not wait until day 6 to discover a published
comparison is incompatible. Day-4/5 run operations can span GPU waiting windows;
their four B4 hours count active supervision/data checks, not all worker elapsed
time. Keep raw-data capture automated and durable; never depend on manual copying.

## Completion controls and recovery

- Review task IDs, evidence, blockers and the next handoff at the start/end of
  each day. Keep one authoritative coverage checklist for primary, diagnostic,
  repeat, figure and report deliverables.
- Day 1: reserve primary/backup GPU access and identify persistent storage.
  Day 3: escalate harness blockers immediately. Day 5: review raw-data coverage
  before locking inputs and regenerating final tables.
- Use the five protected Devansh hours on day 7 for integration or recovery. A
  necessary recovery run still follows preregistration, identity checks and the
  global OOM budget. Re-run affected analyses and figures after any new data.
- Keep both diagnostics, all planned repeats, scientific checks, external
  verification and clean reproduction. Improve scheduling, automate data flow
  and move independent CPU/report tasks earlier instead of silently dropping work.
- If GPU access or another external constraint prevents evidence collection after
  recovery, record the exact incomplete criterion and recovery steps. Never
  manufacture a threshold, waive the 90% accounting target without disclosure,
  or present an incompatible published result as verification.
