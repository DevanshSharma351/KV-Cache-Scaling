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
3. At least one contributor reviews (user-authorized amendment, 18 October 2026), then commit and push the final prediction before model
   measurements. Do not fill in a fake SHA or preregister this template as results.
4. Add a later receipt based on `receipt.template.json`, linking the prediction
   commit and remote push/release/CI evidence. A commit cannot contain its own
   SHA; the subsequent receipt solves that without editing the prediction.
5. The Phase 3 coordinator now verifies completed receipt evidence, frozen
   scientific/code hashes, pushed Git state and durable launches before execution.
   `kvcache check-receipt --receipt prereg/receipt.template.json` must reject the
   current incomplete template. See [the formal contract and remaining acceptance](../docs/P3_COORDINATOR_CONTRACT.md).
   The new gate does not turn any draft or template into real preregistration.

Use a new protocol/prediction for scientific changes after freezing; retain the
old files. Repository history and remote receipts improve auditability but do
not constitute an unforgeable timestamping service.
