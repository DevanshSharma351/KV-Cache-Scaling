# Preregistration gate

No formal prediction has been frozen yet. This folder contains templates and
the [Phase 1 CPU prediction draft](drafts/prefill-v1-phase1/prediction.md).
Scaffold setup and CPU-only analytical calculations do not count as GPU
measurements. No model-loading memory check, model forward, calibration or sweep
may precede the formal prediction push.

The working `kvcache prepare-prereg --output NEW_DIRECTORY` command checks pinned
scientific inputs and generates a draft JSON/Markdown package plus local hashes.
Its default VRAM budgets are hypothetical. It does not collect GPU data, freeze
a prediction, record remote evidence or implement the future execution gate.
Keep the committed draft unchanged; regenerate into a new directory to review
changes. See `docs/P1_PROGRESS.md` for the remaining inventory/review handoffs.

1. A1 completes `prediction.template.md` under a new dated name and includes the
   exact protocol/config/revision/hash, numerical predictions and uncertainties.
2. B1 records the actual GPU identity and context-only usable budget without a
   model load/forward. The prediction remains a function of usable VRAM even
   when this hardware observation is supplied.
3. Both contributors review, commit and push the final prediction before model
   measurements. Do not fill in a fake SHA or preregister this template as results.
4. Add a later receipt based on `receipt.template.json`, linking the prediction
   commit and remote push/release/CI evidence. A commit cannot contain its own
   SHA; the subsequent receipt solves that without editing the prediction.
5. The implemented coordinator will require a completed receipt and check hashes
   before execution. That enforcement is an A3 task; it does not exist yet.

Use a new protocol/prediction for scientific changes after freezing; retain the
old files. Repository history and remote receipts improve auditability but do
not constitute an unforgeable timestamping service.
