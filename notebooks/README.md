# Notebook trail

[Executed T4 audit notebook](03_p1_executed_t4_2026-10-19.ipynb) records the real
commands, 80-worker checkpoint, errors and final CPU checks. Its final saved
cell passes all 109 tests, verifies the checkpoint and regenerates the report.
The original cloud notebook remains at
[the supplied Colab URL](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS).

The archived file preserves every source cell and all scientific outputs/errors.
Only 50 duplicate ZIP-transfer base64 blocks are omitted; archive hashes and
checkpoint receipts remain. [Provenance](03_p1_executed_t4_2026-10-19.provenance.json)
records the original and archived hashes. The untouched 81.7 MB export is retained
in the local hidden checkpoint folder. This is a historical execution archive,
not a notebook to run top-to-bottom on a different GPU: receipts and point guards
are intentionally tied to the original allocation.

Colab's attempted `venv` setup failed at ensurepip. The next recorded cell used
its existing Python 3.11 and completed tests, integrity verification and CPU
reproduction. That failure is preserved. GPU research remains incomplete due
to quota; see [remaining work](../docs/P1_HANDOFF.md).

---

# Colab / Kaggle execution guide

The starter notebook is a CPU-only orientation/preview; GPU execution is pending
A2/A3. It intentionally does not contain a hidden sweep or an OOM cell.

Use [02_p1_phase1_colab.ipynb](02_p1_phase1_colab.ipynb) for the auditable Phase 1
workflow. It records every worker command and output, source commit, context-only
inventory, installed API metadata, conditional calculations and an exportable ZIP.
It starts unexecuted; restart the runtime before running so the parent notebook
owns no CUDA context. Save the executed `.ipynb` and ZIP together.

[Open the preparation notebook in Colab](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/02_p1_phase1_colab.ipynb).

The latest reported runtime differs from the original pins. Follow
[the reconciliation note](../docs/COLAB_RUNTIME_RECONCILIATION.md) before any
GPU-package change or formal prediction. The notebook performs no automatic
GPU-stack installation and contains no model forward or OOM command.

1. Clone this repository and change into its root. Use a Linux GPU runtime for
   the later measurements; inspect the actual GPU, driver and Python version.
2. Install the exact target versions using the README commands. Restart the
   runtime after changing torch/transformers. If Python/driver constraints prevent
   the target, choose and document a new complete stack **before** preregistration.
3. Mount a persistent Drive/output folder and select an absolute run path before
   any probe. Colab `/content` is ephemeral. Confirm the mirror/fsync behavior.
4. Run CPU tests and analytical predictions. Complete/push the formal prediction
   and its remote receipt before model-loading memory checks or any forward.
5. Keep the notebook/coordinator CPU-only. Each point is a new OS process with
   allocator env set before interpreter start. Do not import torch/CUDA into the
   parent merely to preview a point.
6. If the notebook itself runs CUDA: set `os.environ['PYTORCH_CUDA_ALLOC_CONF']`
   in the first cell before importing torch. Restart the runtime between default
   and expandable variants; setting it after CUDA initialization is invalid.
   `empty_cache()` and a peak reset do not change allocator configuration.
7. After A2/A3 exist, run post-preregistration backend/support diagnostics, then
   paired safe sweeps on the same GPU UUID. Compare manifests after any reconnect.
   A new GPU or stack starts a new run rather than merging incompatible points.
8. Preserve the one global deliberate-attempt ledger across runtime restarts.
   Do not rerun an OOM cell manually. Save raw results before generating plots.

PyTorch 2.6 documents expandable segments, but capability warnings and the actual
wheel/platform must be checked. Standard FA2 is outside the T4 protocol. Do not
silently change to eager or allow SDPA math fallback to make a notebook run.

## Executed Phases 1–3 acceptance

The [same live audit notebook](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS)
now includes cells 17–43 for the pinned runtime, real preregistration gate,
83 tests, preserved preflight correction, seven T4 workers and external
checkpoint restoration. See the [cell map and evidence](../artifacts/acceptance/2026-10-18-t4/README.md).
Google Drive stayed disconnected; checkpoint archives were verified/fsynced
locally before each next worker. The executed notebook remains saved at that
link. Local notebook templates are not relabelled as executed outputs.
Before replaying this Colab 2025.07 environment, remove its unused torchao 0.10.0
with a visible pip uninstall cell and verify cold imports, as recorded in the
preflight correction. Keep the scientific pins and never reuse a mismatched GPU
identity or bypass the pushed receipt.
