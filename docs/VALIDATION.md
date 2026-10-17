# Scaffold validation

Validated locally on 17 October 2026, macOS, Python 3.14.7. The GPU execution
target remains Linux/Python 3.11; these CPU checks do not establish T4 runtime
compatibility.

| Check | Result |
| --- | --- |
| Editable install in a fresh local virtual environment | Passed |
| `python -m unittest discover -s tests -v` | 8 tests passed |
| `python -m compileall -q src scripts` | Passed |
| Installed `kvcache predict --usable-vram-gib 14.5` | Valid JSON, three explicit draft scenarios, correct hand-derived values |
| Installed `kvcache plan --experiment configs/experiment.json` | 60 candidate point/repeat records; GPU execution false |
| All JSON files and orientation notebook parse | Passed |
| Local Markdown file links resolve | Passed |
| Original PDF hashes match copied resources | Passed |
| Pinned model config SHA-256 and model revision | Checked by test |
| GPU-analysis direct pins resolve for Linux x86_64 / CPython 3.11 | pip cross-platform binary-only dry-run passed |

The dependency dry-run checks `requirements-gpu.txt`; torch's CUDA wheel is
installed separately per README and was not installed or executed on this Mac.
No GPU measurements, expandable-segment behavior, efficient SDPA execution,
boundary search, deliberate OOM or empirical discrepancy analysis was tested.
Those checks belong to the implementation milestones and formal preregistration.
GitHub's CPU matrix is configured for Python 3.10–3.12; its remote result is
separate from this local evidence.
