# Two-person task allocation

Planning budget: **48 active hours over two weeks**. Devansh: **31.2 h (65%)**;
teammate: **16.8 h (35%)**. GPU waiting/download time is additional. Hours include
the listed reviews; do not count them again. This is a workload allocation for
future implementation, not a claim of completed contributions.

## Devansh: core implementation and scientific ownership

| ID | Work package | Hours | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| A1 | Derivation, final assumptions, VRAM sensitivity and preregistration | 5.0 | `model_math.py`, `prereg/`, `docs/DERIVATION.md`; signed-off remote prediction receipt | B1 environment inventory | Weights/GQA hand checks pass; predictions at five lengths and VRAM range pushed before forward passes |
| A2 | Probe + telemetry + kernel/cache verification | 7.0 | `probe.py`, `telemetry.py`, GPU smoke evidence | A1, shared schema from B2 | Fresh-process peak/reset lifecycle correct; output/cache shape and backend verified; no fallback |
| A3 | Subprocess coordinator, safety gate, boundary estimation | 7.0 | `sweep.py`, CLI integration, fault-injection tests with B2 | A1, A2, B2 | Resume, paired env isolation, timeout and one-attempt ledger verified; safe vs observed boundary separated |
| A4 | Cache regression, residual and discrepancy analysis | 5.5 | `analysis.py`, processed tables, report methods/results | A2 schema samples, B4 data | Slope within 3%; same-metric table closes; 90% attribution supported or visibly fails |
| A5 | Two-model/two-GPU transfer and published comparison | 3.5 | `generalize.py`, config-only snapshots, cited comparison | A4 calibrated terms | Four predictions with uncertainty; compatible external quantity and percentage error |
| A6 | Integration, technical report and final review/buffer | 3.2 | Report conclusions, integration fixes, review of B2/B3 | A3–A5, B5 | No false completion claims; reproduction evidence and all deviations included |
| | **Total** | **31.2** | | | |

## Teammate: data reliability, plots and reproducibility

| ID | Work package | Hours | Files / concrete handoff | Depends on | Done when |
| --- | --- | ---: | --- | --- | --- |
| B1 | Linux/Colab setup, driver inventory and runtime lock | 2.5 | `notebooks/`, requirements capture, environment-only manifest | Scaffold | Exact versions/GPU recorded; restart and persistent-path instructions tested; no premature model forward |
| B2 | Journal, schema validator and resume fault tests | 3.5 | `journal.py`, `docs/measurement.schema.json`, persistence tests | Shared contract; coordinate with A3 | Start/result fsync, lock, torn-line recovery, duplicate IDs and OOM-token recovery tested |
| B3 | Reproducible plots and labelled tables | 3.5 | `plotting.py`, figures from processed files | A4 column contract; test fixtures may precede data | Peak bands, cache slope, allocator slack and boundary intervals plotted without mixing metrics |
| B4 | Paired run operation and data checks | 3.0 | Durable raw journals/manifests; missing-point list | A2/A3 and preregistration gate | Default/expandable pairs use same UUID; all statuses saved; no unapproved repeated OOM attempts |
| B5 | Independent reproduction, README and final packaging | 3.0 | Reproduction transcript, dependency lock, figure/report export | A4/A5/B3 | Clean setup regenerates tables/figures; final PDF reviewed; raw/result hashes recorded |
| B6 | Review preregistration and probe/analysis evidence | 1.3 | Review checklist and deviation sign-off | A1, A2, A4 | Independently spot-check tied weights, peak window, frozen assumptions and gap arithmetic |
| | **Total** | **16.8** | | | |

## Milestones and ownership handoffs

| Day | Devansh focus | Teammate focus | Shared checkpoint |
| --- | --- | --- | --- |
| 1 | A1 config and analytical checks | B1 environment inventory | Choose persistent output path; resolve source ambiguities |
| 2–3 | Finish A1; push formal preregistration | B2 contract/journal; B6 prereg review | Both record remote receipt; no GPU forwards before this |
| 4–6 | A2 then A3 | Finish B2 tests; prepare plotting interfaces | Probe record validated; simulated worker kill/resume passes |
| 7–9 | A3 supervision; begin A4 | B4 paired run execution | Store data after each point; stop on accidental OOM |
| 10–11 | A4 accounting and validation | B3 measured plots | 90% attribution criterion checked with evidence |
| 12 | A5 transfer model | B5 independent regeneration | Compatible published comparison identified |
| 13–14 | A6 report and integration buffer | B5 packaging; B6 final evidence review | Final definition of done/deviations signed off |

The teammate can implement journal tests and plotting on labelled synthetic
fixtures before GPU access. Devansh defines scientific columns before the plotting
handoff. Only one person coordinates GPU runs at a time to avoid memory
contamination. B4 is execution under A3's frozen controls, not a second harness.

## If time is tight

Preserve all primary requirements: preregistration, accurate peak measurement,
retained-cache validation, allocator comparison, honest discrepancy accounting
and transfer predictions. Trim optional chunked/static diagnostic grids first.
Use the final integration/reproduction time as buffer; extend the calendar if
GPU access fails. Record unmet criteria explicitly rather than manufacturing
thresholds, 90% coverage or an external validation match.
