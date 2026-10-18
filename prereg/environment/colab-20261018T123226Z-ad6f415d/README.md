# Observed Colab preparation run

Run ID: `20261018T123226Z-ad6f415d`, 18 October 2026.

Notebook: [KV_Cache_P1_Audit.ipynb](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS).
Resolved source: `cf50e692ac52cbdc85824fe39252507f0f0e73c5`.

This record describes cells actually operated and read through Chrome. It is
preparation evidence, not a formal preregistration receipt or model-memory run.
The original two user inventory cells and outputs remain in the notebook. A
session restart preceded the preparation cells to release the parent CUDA context.

## Recorded execution

| Execution count | Action | Observed outcome |
| --- | --- | --- |
| 1 | Create isolated run directory and append-only command journal | Succeeded |
| 2 | Clone repository while private | Failed, exit 128; original stderr retained |
| 3 | Print the failure journal and stderr | Succeeded |
| 4 | Retry public clone, resolve SHA, detached checkout | All three commands exited 0 |
| 5 | CPU test suite, package freeze, `nvidia-smi` | All exited 0; 36 tests passed; no GPU processes before collection |
| 6 | Fresh child-process context inventory, then `nvidia-smi` | Both exited 0; no GPU processes after worker exit |
| 7 | Inspect installed forward/cache/SDPA signatures and source hashes | Exited 0; no inspection error; no model instantiated |
| 8 | Conditional prediction from exact free-byte snapshot | Exited 0; runtime mismatch visible; measurement gate closed |
| 9 | Generate generic theory in a new directory | Exited 0; draft only |
| 10 | Verify saved JSON and full test output | Assertions passed |
| 11 | Create evidence manifest and ZIP | Archive path and checksum printed |
| 12 | Download existing ZIP through Colab download helper | Cell completed; browser download was not captured |
| 13 | Print existing ZIP as base64 for visible text transfer | Succeeded; transferred bytes verified locally |

The failure-diagnostic cell appears before the failed checkout cell in document
order; execution counts and the journal preserve the actual chronology. This
executed notebook is an audit record, not a clean `Run all` template. Use the
repository's unexecuted preparation template for a fresh run.

## Snapshot provenance

[`context-inventory.json`](context-inventory.json) was mirrored from the fresh
child-process JSON shown in the notebook, not substituted from the earlier user
snapshot. Its exact bytes match the snapshot SHA-256 independently printed by
the conditional-prediction cell:

`6591af2e6877538b07b11f0970a73ccfc4dc6ed26953613df1607826e7d5a4f1`.

Collector PID was 11894. Allocated and reserved tensor bytes were both zero;
the process list contained only that collector. CUDA free bytes were
15527116800 (14.46075439453125 GiB), matching the earlier user observation.
`sm_75` appears in the installed torch build's architecture list. This proves
neither attention-backend support nor expandable-segment support.

Installed versions were Python 3.13.15, torch 2.11.0+cu130, CUDA runtime 13.0,
Transformers 5.18.0. Inspection confirmed `logits_to_keep` in the forward
signature and `enable_gqa` in the adapter source. The first signature inspection
hashed the decorated forward's `generic.py` wrapper, not the Qwen implementation;
it must not be represented as a complete implementation-source audit.

The conditional central limit was 111232 tokens, with scenario limits 76024 to
131072 after the config cap. These remain original-assumption theory, not a
measured or formally registered context limit.

## Export status

Colab created:

`/content/kvcache-evidence/20261018T123226Z-ad6f415d.zip`

Printed archive SHA-256:

`9c96fd1de0f4b51df2309589da1f2a681150094f1549d29cf9e3add86fd6939c`.

The initial browser download attempt timed out and notebook control returned
`Debugger unattached`. A fresh Chrome tab restored control. Further browser
exports were unreliable; Colab also reported output-script and reCAPTCHA
connection problems. No browser security settings were changed.

The existing archive was then printed as base64 in an explicit notebook cell.
That visible output was read, decoded locally and saved as
[`evidence.zip`](evidence.zip). Its SHA-256 matches the original printed value
above. **All 32 manifest entries were verified**, the archive contains exactly
those entries plus its manifest, and the archived context JSON is byte-identical
to the separately mirrored snapshot. No source code, old outputs or archive
members were regenerated to produce this evidence.

The ZIP includes the full command journal, successful/failed command outputs,
package freeze, API inspection, conditional prediction, generic draft and summary
through archive creation. Export attempts, the text-transfer cell and the final
Markdown summary came later and remain recorded in the live notebook; they are
not retroactively included in that original archive.

The executed `.ipynb` itself has not been retrieved as a file. Its saved Colab
link above remains the complete notebook audit trail. The repository preparation
template stays unexecuted; this record does not claim that template was executed
in place. A direct GitHub notebook-copy dialog was cancelled without making a
repository commit. The verified raw ZIP is preserved here instead.

To inspect and independently verify the archived evidence from the repository
root (CPU only):

```python
import hashlib, json, zipfile
from pathlib import Path
archive = Path("prereg/environment/colab-20261018T123226Z-ad6f415d/evidence.zip")
assert hashlib.sha256(archive.read_bytes()).hexdigest() == "9c96fd1de0f4b51df2309589da1f2a681150094f1549d29cf9e3add86fd6939c"
with zipfile.ZipFile(archive) as bundle:
    manifest = json.loads(bundle.read("evidence-manifest.json"))
    assert set(bundle.namelist()) == set(manifest["files_sha256"]) | {"evidence-manifest.json"}
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == expected, name
```

No package changes, checkpoint downloads, model loads, forwards, attention-kernel
tests, allocator intervention or OOM attempts were performed. Runtime
reconciliation, independent review, formal prediction push and a subsequent
receipt remain open.
