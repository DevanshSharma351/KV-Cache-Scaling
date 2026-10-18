# Phases 1–3 acceptance execution

## Policy amendments authorized on 18 October 2026

The user explicitly accepted one contributor review. The gate requires at least
one real, timestamped contributor review; duplicate/unknown roles are rejected.
The earlier two-review policy remains visible in history. Policy permission does
not itself assert that a review happened.

The user requested that Drive remain disconnected. No Drive mount or new Drive
permission is needed. A Colab `/content` directory is ephemeral staging, including
any mirror inside it. It must never be described as cloud-durable storage.

## Bounded GPU acceptance

After the pushed formal prediction and subsequent reviewed receipt:

```bash
python -m kv_cache_scaling sweep --receipt prereg/<actual-receipt>.json \
  --local-root /content/kvcache-local --run-id <same-uuid> \
  --acceptance-only --checkpoint-each-probe
```

Each invocation launches at most one new fresh worker and returns
`awaiting_external_checkpoint`. Invoke the same command only after the complete
state, immutable artifacts, raw results and logs have been transferred through
an explicit notebook cell to the user's local disk, SHA-256 checked and fsynced.
The local copy is the persistent checkpoint. Keep its path/hash and notebook
cell reference in a local append-only transfer ledger. Save the notebook too.

Seven workers are scheduled: three default L=512 primary probes, three expandable
L=512 primary probes, and one default L=512 kernel diagnostic. The diagnostic
uses `torch.profiler`; it must record efficient SDPA and is excluded from primary
fits. Changing acceptance/full-sweep scope on resume is rejected. These points
produce no OOM threshold, no safe-budget bisection and no allocator context-limit
delta. The deliberate-attempt ledger must remain absent.

An external operator enforces the export barrier; the CLI cannot verify a file
on another computer. It is unsuitable for unattended bulk sweeps. No automatic
cloud durability, distributed locking or full runtime-restart guarantee is
inferred from a successful POSIX write in Colab.

## Recovery and limits

Before advancing, verify the archive members and recorded hashes locally. Test
recovery by deleting the volatile local run directory and restoring it from the
staged mirror; also test loss of both Colab copies by restoring the externally
persisted checkpoint through a visible notebook cell. Resume must skip completed
point keys and preserve the original run ID, immutable logs and claim files.
A new process must observe existing attempt state, including a consumed synthetic
token in a separate test-only root. Never create a synthetic token in the real
experiment root.

This proves recovery from the tested loss/restart cases using the external
checkpoint. A complete machine/session-loss test requires reconnecting and
restoring in a new session. A process restart is not a machine restart. Report
which was actually tested. Bulk execution requires Person 2's continuous durable
storage integration or a separately validated continuous external checkpoint
controller; do not remove acceptance stepping for an ephemeral root.
