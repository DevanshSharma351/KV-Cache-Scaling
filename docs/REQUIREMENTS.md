# Requirements, feedback traceability and acceptance

The PDFs provide project requirements and suggestions. The user's current
feedback and this revised protocol resolve conflicts. Status below describes
acceptance for the future implementation, not results already obtained.

Delivery target: one intensive week, 80 active hours, with the 65/35 contribution
split in `TASK_ALLOCATION.md`. The shorter calendar does not reduce scope,
repetitions, scientific acceptance thresholds or reproduction requirements.
Both chunked-dynamic and static-cache diagnostic grids are required, with three
independent repeats per point and a comparison table separate from primary fits.

## Functional requirements

| ID | Requirement | Acceptance evidence | Owner |
| --- | --- | --- | --- |
| FR1 | Read real architecture; distinguish GQA heads | Pinned config SHA/hash; verified fields; CPU math checks give 28 KiB/token and 6x naive error | A1 |
| FR2 | Derive weights + KV + activation + budget terms | Defined symbols/units; tied embedding counted once; post-load parameter count agrees exactly; any buffer/rounding difference itemized | A1/A2 |
| FR3 | Predict before measurement | Five required lengths; L(V) and assumption range; pushed prediction SHA and external receipt precede any model memory probe; frozen original remains unchanged | A1/B6 |
| FR4 | Measure peaks; isolate/persist/resume safely | Every point/repeat fresh process; reset before allocations/forward; synchronized peaks and snapshots; durable start/result; simulated kill loses at most in-flight point; all errors classified | A2/A3/B2 |
| FR5 | Separate retained cache slope and residual | >=5 lengths; direct cache bytes within 1% of formula; slope error <=3%; retained residual fraction, intercept and R² shown; peak fit separately labelled | A4 |
| FR6 | Compare allocator variants | Same GPU UUID/stack and policy; env set before CUDA; capability/warnings checked; same-time slack vs L; paired safe-limit delta with uncertainty; unsupported is an explicit outcome | A3/B4 |
| FR7 | Explain committed-prediction error | Same metric at fixed L; signed rows close within max(1 MiB, 0.1% measured); >=90% supported attribution for gaps >=10 MiB; explicit unaccounted row; separate budget table | A4 |
| FR8 | Generalize to two unrun models and two unrun GPUs | Four config-derived cells, usable-budget/activation ranges; >=1 compatible published quantity with cited assumptions and percentage error, or explicitly unmet validation | A5 |

FR4's boundary deliverable is an honest safe/observed/predicted interval, not an
unjustified exact maximum. Target <=256 tokens for the safe-budget bracket if
resources permit; do not use that resolution to imply a similarly precise
hardware boundary. Repeat required points and selected boundary points three
times; report median and range. If spread exceeds 5%, investigate environment
drift before comparing allocators. Do not discard unfavorable repeats silently.

The original “exactly one deliberate OOM” is a target, constrained by a global
maximum of one deliberate attempt and by the configured domain. One observed
deliberate OOM meets the target. Zero because of censoring, successful attempt,
or interrupted launch is a documented deviation, not a pass disguised as one.
An accidental OOM is a protocol failure and stops escalation. See the plan for
the incompatibility between repeated failure bisection and this OOM budget.

## Feedback checklist

| Feedback | Resolution | Where to inspect |
| --- | --- | --- |
| Define sequence length; state logits behavior | One empty-cache prefill of L tokens; `logits_to_keep=1`; no generation | Plan sections 1–2; experiment config |
| Measure peak, not resting memory | Synchronized reset/max counters; distinct retained snapshot | Measurement protocol timed window |
| Pin attention; avoid false KV interpretation | HF SDPA + forced efficient backend; fail on fallback; T4 FA2 limitation; retained-cache regression | Plan sections 2,7 |
| Specify dynamic vs static cache | Explicit dynamic primary; static capacity=L diagnostic; growth copies explained | Plan sections 2,6 |
| Isolate each measurement | New OS interpreter per point/repeat and post-load cleanup/reset | Measurement protocol; A3 |
| Set expandable segments before CUDA | Child environment at process creation; notebook restart guidance; pinned 2.6 docs + runtime checks | Plan section 5; notebook guide |
| Avoid tied-weight double count | Exact config-derived unique count; verify storage and post-load allocation | Derivation; CPU tests; A2 |
| Parametric VRAM prediction with uncertainty | L(V), multiple usable budgets, three assumption scenarios, explicit GB/GiB | Derivation; prereg template |
| Add quantitative acceptance | FR criteria and gap coverage/closure with no forced attribution | This document; plan section 7 |
| Choose and justify model | Pinned ungated Qwen2.5-1.5B base; FP16; manageable T4 footprint | Plan section 2 |

## Non-functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| NFR1 | Reproducibility | Model/code/config hashes, Python + full runtime dependency lock, CUDA/driver/GPU UUID, actual kernel, seed, cache/mode/logits/allocator in manifests |
| NFR2 | Persistence and recovery | fsync + persistent mirror; single writer; tested torn-line/timeout/restart; global OOM ledger survives a simulated crash |
| NFR3 | Honest reporting | Drafts vs frozen predictions vs calibrated fits separated; missing/unsupported/censored outcomes retained; uncertainty/sampling limits labelled |
| NFR4 | Practical resource use | B=1 primary; weights cached on disk; no extra 7B/8B measured model; worker timeout; GPU windows booked early; all primary/diagnostic work retained in seven-day schedule |
| NFR5 | Independent use | Teammate reproduces CPU tools and regenerates all tables/figures from preserved data using documented commands |

## Definition of done

- [ ] Final protocol/assumptions reviewed by both contributors.
- [ ] Pushed formal prediction and remote receipt precede first model measurement.
- [ ] CPU tests plus worker/journal fault-injection checks pass.
- [ ] Actual GPU preflight verifies efficient SDPA, versions and allocator support.
- [ ] Default and expandable raw records/manifests preserved with complete statuses.
- [ ] Both diagnostic grids completed with three independent repeats per point; dynamic/chunked/static comparison table included.
- [ ] Peak curves, direct-cache slope, residual fractions and allocator slack plotted.
- [ ] Safe limit and observed/predicted boundary intervals correctly distinguished.
- [ ] One deliberate OOM observed, or the exact unmet target/deviation documented.
- [ ] Fixed-L discrepancy tables meet closure/coverage, or failure is reported explicitly.
- [ ] Four transfer predictions and compatible published verification completed.
- [ ] Clean reproduction transcript, input hashes, final report PDF and limitations reviewed.
- [ ] Seven-day delivery review maps every requirement to evidence; any unresolved criterion is explicitly open rather than waived for the deadline.

**Scaffold delivery is complete independently of these experiment milestones.**
Only the CPU math/prediction/plan preview exists as working implementation today.
No unchecked experiment milestone should be marked complete because a template
or module has been created.
