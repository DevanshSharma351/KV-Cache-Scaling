# Colab inventory, runtime reconciliation and execution trail

Updated 18 October 2026. This is a compatibility/preparation record, not a formal
prediction freeze or a successful model-memory measurement report.

## Evidence received

The [user-reported context snapshot](../prereg/environment/t4-context-2026-10-18.user.json)
contains Tesla T4 UUID `GPU-68cbea81-a683-22be-800d-74ff2d51155e`, driver
580.82.07, zero live allocated/reserved tensor bytes and only its own Python PID
in the reported process list. Its timestamp is 2026-10-18T12:14:03.473613+00:00.
This is caller-supplied evidence, not an independently authenticated observation.

| Quantity | Value |
| --- | ---: |
| Context-initialized CUDA free bytes | 15527116800 |
| Same-observation CUDA total bytes | 15637086208 |
| Usable budget expressed in GiB | 14.46075439453125 |
| CUDA total minus free bytes | 109969408 |

Use free bytes directly; context is already deducted. The total-minus-free value
is a same-observation driver difference, not proof of a pure context allocation.
The earlier `nvidia-smi` total was 15360 MiB; do not replace the newer CUDA free
reading with that older total or force equality between different API/time views.

Applying the **original hypothetical shape scenarios**, without validating them
on this runtime, gives:

| Scenario | Uncapped memory estimate L | Config-capped L |
| --- | ---: | ---: |
| Optimistic | 146740 | 131072 |
| Central | 111232 | 111232 |
| Conservative | 76024 | 76024 |

These are conditional theory, not measured context limits or a registered
GPU-specific prediction. The [machine-readable calculation](../prereg/environment/t4-context-2026-10-18.conditional-prediction.json)
includes source/config hashes, runtime differences and the closed experiment gate.
Its calculation was run locally on CPU. The original Phase 1 draft is unchanged.

## Runtime mismatch and source review

| Setting | Original proposal | Observed |
| --- | --- | --- |
| Python minor version | 3.11 | 3.13 |
| PyTorch | 2.6.0 | 2.11.0+cu130 |
| torch CUDA runtime | 12.4 | 13.0 |
| Transformers | 4.51.3 | 5.18.0 |

This commit does not silently replace the original direct pins. Inspect the
installed stack, choose a defensible runtime/adapter policy, document any changes,
obtain independent review and collect a new context baseline after any restart.
Never reuse an old stack's baseline as a new stack's measured budget.

Source review identifies a specific compatibility concern: the
[Transformers 5.18 adapter](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/sdpa_attention.py)
can request native GQA for an unmasked prefill, whereas our
[4.51.3 adapter reference](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py)
expands K/V heads before SDPA. [PyTorch 2.11's SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
lists native CUDA GQA support for flash/math paths. This raises a conflict with
the forced efficient-backend policy; it is not a successful T4 kernel test.

Cache/forward signatures have also changed: inspect the actual installed
[Qwen2 implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py)
and [cache implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/cache_utils.py)
before implementing legacy attribute/position assumptions. In PyTorch 2.11,
[`PYTORCH_ALLOC_CONF` is the primary allocator variable](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)
and the older CUDA-named variable is an alias; record both and reject conflicts.

No source review authorizes an unregistered model measurement or a math-backend
fallback. A runtime change must be represented in the reviewed scientific inputs.

## Auditable notebook

Use [02_p1_phase1_colab.ipynb](../notebooks/02_p1_phase1_colab.ipynb). It contains:

1. Run directory and append-only command journal with fsync.
2. Explicit Git clone/ref resolution/detached checkout; exact source SHA recorded.
3. CPU tests, full package inventory and driver observation.
4. Context-only inventory in a fresh child process; no parent-owned CUDA context.
5. Installed API signatures/source hashes; no model instantiation or attention run.
6. Exact-byte conditional predictions with mismatch reporting and a closed gate.
7. Generic theoretical draft regeneration into a new directory.
8. Evidence manifest, ZIP export and download. Save the executed notebook too.

Every worker/shell command appears in a cell and is logged with its arguments,
timestamps, return status and full output files. Child Python source is visible
in the same cell. Failed and timed-out commands remain recorded; labels/output
files cannot be silently reused. The ZIP omits the checkout but records its SHA.
Use that SHA instead of `main` for an exact replay.

The notebook does not install/downgrade GPU packages, load weights, run a model,
test an attention kernel, attempt OOM or push from Colab. No Colab execution
counts or outputs have been manufactured. Relevant future package/model/sweep
commands must be added as explicit cells before execution, not run off-notebook.

## Access and commit policy

Direct Chrome access subsequently succeeded for the supplied notebook. The
repository was made public by the user, and the retry cloned and resolved
`cf50e692ac52cbdc85824fe39252507f0f0e73c5` successfully. The earlier private-clone
failure remains recorded. The [observed execution record](../prereg/environment/colab-20261018T123226Z-ad6f415d/README.md)
reports actual preparation counts/results, 36 passing CPU tests and a fresh
isolated context snapshot. The mirrored snapshot matches the SHA-256 printed
by Colab; its usable budget equals the earlier user-supplied value.

The installed API inspection succeeded, but its decorated forward source hash
refers to `generic.py`, not the full Qwen implementation. Backend execution and
allocator intervention are still untested. No package versions were changed.

The notebook created a checksummed evidence ZIP. Browser downloads failed;
a fresh Chrome tab restored page control, and an explicit text-transfer cell
successfully recovered the original ZIP. The [committed archive](../prereg/environment/colab-20261018T123226Z-ad6f415d/evidence.zip)
matches its printed SHA-256 and all 32 manifest entries were verified locally.
It preserves full logs and reports through archive creation, including the
failed private clone. The later export/transfer cells remain in the saved
Colab notebook. The executed `.ipynb` has not been retrieved as a local file;
use the notebook link in the run record for that complete audit trail.

Group changes by working capability: this environment validator, tested notebook,
inventory, conditional report and documentation form one coherent commit. Record
later compatibility decisions or validated runtime executions in substantive
commits with evidence. Avoid a separate commit for every status message or command;
preserve existing research artifacts and published history.

## Fresh pinned-runtime reconciliation, 18 October 2026

Selected Colab runtime 2025.07 with a T4. The native image provides Python
3.11.13 and PyTorch 2.6.0+cu124; recorded installation of requirements-gpu.txt
applies Transformers 4.51.3. The [fresh audit](../prereg/environment/colab-20261018T141242Z-a337a6c3/README.md)
archives 77 passing CPU tests, package inventory and unwrapped API inspection.
The GPU UUID differs from the historical session; the new baseline is kept
separately. The stack now matches the implemented policy; no torch reinstall
was necessary. Backend execution remains an empirical acceptance check.

Google documents the older runtime selector in its
[runtime version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html).
The raw installed-package inventory, rather than that documentation, establishes
the actual versions used here. Google Drive remains disconnected at user request.
