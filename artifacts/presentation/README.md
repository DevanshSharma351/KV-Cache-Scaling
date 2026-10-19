# Presentation deliverables

[Clean executed notebook](../../notebooks/FINAL_PRESENTATION.ipynb) ·
[Standalone HTML](PROJECT_PRESENTATION.html) ·
[Demo instructions](../../docs/LIVE_DEMO.md).

The HTML embeds the six plots and separate downloadable files; it can be opened
without a notebook runtime. The clean notebook regenerates every result on CPU
and includes a VRAM/batch explorer and temporary checkpoint recovery demo.

Presentation verification records the pinned source commit, executed code-cell
count, image count and file hashes. No model/CUDA calls run in the presentation.
The saved data has 80 successful workers. Colab quota leaves the last two held-out
repeats and the single boundary attempt unfinished; no OOM threshold is claimed.
