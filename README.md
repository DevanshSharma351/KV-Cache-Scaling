# KV-Cache Scaling

Predict the context memory limit before measuring it, then explain the error with
reproducible GPU measurements.

**Status: Phase 1 foundation, Phase 2 probe and Phase 3 coordinator implemented and CPU-tested.**
The analytical draft, auditable Colab preparation, GPU inventory and raw evidence
are preserved. The probe/telemetry engine is tested with CPU doubles; real GPU
acceptance is pending runtime reconciliation, independent review, formal
preregistration and real persistence/GPU acceptance checks. No model-memory experiment
or OOM test has run. See [Person 1 progress](docs/P1_PROGRESS.md) and the
[Phase 2 contract](docs/P2_PROBE_CONTRACT.md) and [Phase 3 contract](docs/P3_COORDINATOR_CONTRACT.md).

## Start here

1. Read [the implementation plan](IMPLEMENTATION_PLAN.md).
2. Follow [the project flow and scaffold decisions](PROJECT_FLOW.md).
3. Pick work from [the 55/45 task allocation](TASK_ALLOCATION.md).
4. Review [the measurement protocol](docs/MEASUREMENT_PROTOCOL.md) and
   [acceptance criteria](docs/REQUIREMENTS.md) before implementing the harness.
5. Follow [Person 1's five-phase plan](DevP1_phases.md) for the 55% contribution.
6. Use [the auditable Phase 1 Colab notebook](notebooks/02_p1_phase1_colab.ipynb)
   for recorded commands and evidence export. Check the
   [runtime reconciliation note](docs/COLAB_RUNTIME_RECONCILIATION.md) first.

## Frozen design proposal

| Setting | Decision |
| --- | --- |
| Measured model | `Qwen/Qwen2.5-1.5B`, pinned Hub revision in `configs/experiment.json` |
| Meaning of L | One forward pass over exactly L input tokens, starting with an empty cache |
| Runtime | FP16 weights/cache; batch 1; eval + inference mode; no generation |
| Attention | HF `sdpa`, forced PyTorch `SDPBackend.EFFICIENT_ATTENTION`; fail on unsupported kernels |
| Logits | Explicit `logits_to_keep=1`; no labels or loss |
| Cache | Explicit `DynamicCache()` per point; static/chunked are labelled diagnostics |
| Isolation | New OS process for every point and repeat |
| Primary memory metric | `max_memory_allocated()` after a synchronized peak reset |
| Prediction | Function of usable VRAM with a declared assumption range |
| Team | One intensive week: Person 1 handles 55% of the work; Person 2 handles 45% |

The seven-day schedule preserves all project requirements and includes both
chunked and static-cache diagnostics, independent reproduction and the final PDF.
See [daily responsibilities and handoffs](TASK_ALLOCATION.md). Reserve GPU windows
on days 3–5 and use independent CPU tasks while probes run. Day 7 is protected
for recovery, integration, final reproduction and review.

These decisions become frozen only when the formal prediction is reviewed and
pushed. See [sources and corrections](docs/SOURCES.md).

## Run the working CPU scaffold

From the repository root, with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
kvcache predict --usable-vram-gib 14.5
kvcache plan --experiment configs/experiment.json
kvcache validate-probe --request configs/probe.example.json
kvcache prepare-prereg --output /tmp/kvcache-review-draft
```

`predict` outputs hypothetical sensitivity scenarios, explicitly marked DRAFT.
Its VRAM input is **GiB after context/other-process deductions, before weights**.
`prepare-prereg` verifies the pinned inputs and emits exact-byte prediction
tables, component counts, VRAM sensitivity and integrity hashes into a **new**
directory. It never overwrites existing output or authorizes GPU measurements.
Review the [committed Phase 1 draft](prereg/drafts/prefill-v1-phase1/prediction.md).
`predict --batch-size B` explores theoretical batch scaling; the primary draft
and future measured protocol remain B=1.

`plan` lists candidate points and never touches CUDA. Run these commands from the
repo root, or supply explicit config paths. `validate-probe` checks a primary
request and its scientific hashes without importing GPU libraries. It never
authorizes execution. `check-receipt` verifies committed/pushed evidence without
GPU work. `sweep` and the internal `probe` require a real completed receipt;
workers also require durable launch evidence. The current template is incomplete
and cannot authorize a run. `analyze` and `generalize` remain scaffolded.

## GPU setup target

Use a Linux NVIDIA environment, preferably a Colab/Kaggle T4, with Python 3.11.
Do not install CUDA wheels on a Mac for the CPU checks above.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

Restart the runtime after changing installed torch/transformers. Validate the
driver, wheel and efficient SDPA support on the actual T4 after preregistration.
Direct pins are provided; Person 2 must capture a successful Linux runtime's
full `pip freeze`, driver/CUDA versions and platform in the run manifest. No GPU
compatibility claim is made by the CPU CI checks.

The parent coordinator sets allocator variables before starting each worker.
For notebook-owned CUDA, set the variable in the first cell and restart the
runtime between allocator variants. See [the Colab guide](notebooks/README.md).

## Expected outputs

Formal prediction + remote receipt under `prereg/`; immutable probe journals
under `data/raw/`; derived tables under `data/processed/`; figures and the final
report under `artifacts/`. Generated data is ignored by default. Publish only
reviewed, bounded result files deliberately, with `git add -f` if appropriate;
never commit model weights, tokens or cloud credentials.

The supplied PDFs are preserved under `references/` for traceability. Their
embedded recommendations are source material; the revised plan governs this
project. CI checks CPU math, request/storage validation and mocked probe lifecycle; it does not establish real GPU correctness.
