# Colab / Kaggle execution guide

The starter notebook is a CPU-only orientation/preview; GPU execution is pending
A2/A3. It intentionally does not contain a hidden sweep or an OOM cell.

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
