# Presentation deliverables

[Clean executed notebook](../../notebooks/FINAL_PRESENTATION.ipynb) ·
[Standalone HTML](PROJECT_PRESENTATION.html) ·
[Demo instructions](../../docs/LIVE_DEMO.md).

The HTML embeds the six plots and separate downloadable files; it can be opened
without a notebook runtime. The clean notebook regenerates every result on CPU
and includes a VRAM/batch explorer and temporary checkpoint recovery demo.

Presentation verification records the pinned source commit, executed code-cell
count, image count and file hashes. No model/CUDA calls run in the presentation.
The core study has 80 successful workers and the separate allocation has two
successful L=512 replication checks. The user closed further GPU probing on
20 October. This is a final resource-bounded delivery with an unmeasured exact
OOM threshold; the original full experiment is not claimed complete. See
[final scope](../../docs/FINAL_SCOPE.md).
