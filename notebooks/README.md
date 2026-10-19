# Project notebooks

## Present the project

Use [FINAL_PRESENTATION.ipynb](FINAL_PRESENTATION.ipynb), or
[open it in Colab](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/FINAL_PRESENTATION.ipynb).
Run all on a CPU runtime. Every code cell replays saved evidence or performs CPU
math/recovery; none loads a model or consumes an OOM attempt. Cached executed
outputs show all six deliverables immediately.

The notebook exposes the original prediction timestamp, equations, graphs,
byte/MiB discrepancy tables, four transfer cases, a VRAM/batch explorer and a
temporary-filesystem checkpoint demo. Downloads default to none. The standalone
[HTML presentation](../artifacts/presentation/PROJECT_PRESENTATION.html) embeds
plots and individual downloadable files. [Live-demo guide](../docs/LIVE_DEMO.md).

## Historical audit

[The archived executed audit](03_p1_executed_t4_2026-10-19.ipynb) retains all
original source cells, scientific outputs and errors. Only duplicate archive
base64 transport strings were omitted, as recorded in its provenance sidecar.
Earlier names in immutable historical files are preserved as evidence.

The supplied Drive notebook is now named **KV_Cache_Project_Audit.ipynb** and
retains its [original URL](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS).
Its final presentation section is safe CPU replay; do not Run all the old GPU
measurement cells. For a public demo, prefer the clean presentation notebook.

The current research gap is GPU access, not presentation access. See
[remaining work](../docs/REMAINING_WORK.md). Drive remains disconnected.
