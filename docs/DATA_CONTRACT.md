# Data and resume contract

Schema version: 1. B2 owns validation/persistence; A2/A3 own scientific fields and
status classification; B4 implements diagnostic-path fields against that contract.
The JSON Schema is a starter for result events; expand it
with tests before actual measurements. It does not validate start/receipt events.

## Run files

```text
data/raw/<run_id>/manifest.json
data/raw/<run_id>/events.jsonl
data/raw/<run_id>/oom_ledger.json
data/raw/<run_id>/stderr/<probe_id>.txt
data/raw/<run_id>/driver_samples/<probe_id>.jsonl
```

`run_id` is a UUID; `probe_id` is a UUID. A deterministic point key combines
protocol hash, environment fingerprint, allocator, mode, cache/capacity, L, B,
seed, logits policy, warmup policy and repeat index. Timestamp alone is not a key.
Only the CPU coordinator writes the journal. Workers return a structured result;
the coordinator stores partial output even if validation fails.

The manifest contains protocol/config/model hashes, frozen preregistration SHA
and remote receipt, current code SHA and dirty status, full dependency lock,
Python/executable/OS, torch wheel/CUDA/driver versions, GPU name/UUID/capability,
logical-to-physical mapping, context-only budget, creation time in UTC and all
deviations. Hash the scientific environment separately from irrelevant host paths.
Do not store credentials or tokens.

## Events

- `probe_started`: durable requested point, deliberate-attempt flag and monotonic/
  UTC timestamps. If deliberate, consume the run-global OOM token first.
- `probe_result`: status, request, environment fingerprint and available metrics;
  reference schema `measurement.schema.json`.
- `probe_interrupted`: orphaned start on recovery; preserves partial stdout/stderr.
- `run_deviation`: reason, affected points, action taken and reviewer acknowledgement.

Metrics include load peak separately from forward peak; context/post-load/
after-forward/retained/released snapshots; direct parameter/storage and cache
logical/storage byte counts; independent max allocated/max reserved; sampled
NVML maximum and sampling interval; elapsed seconds. Missing failed-stage metrics
are null, never zero. All memory is integer bytes and lengths are token counts.

## Persistence and resume behavior

1. Acquire one process/file lock for a run and reject concurrent writers.
2. Append one complete JSON line with flush+fsync; use an atomic replace + parent
   directory fsync for manifest/ledger updates. Mirror to durable storage before
   advancing the next point; log mirror failures and stop.
3. On resume, verify protocol/preregistration/model/environment fingerprints.
   A different GPU UUID or stack starts a new run, not an automatic continuation.
4. Parse journal line by line. If the last line is truncated, retain it in a
   recovery artifact, truncate only that suffix under the lock and add a recovery
   event. Interior corrupt lines are errors requiring investigation.
5. Successful point keys are complete; failed/interrupted keys are not silently
   overwritten. A retry gets a new probe ID and references its prior attempt.
6. Resume normal failed points only after error review. Never retry the consumed
   deliberate OOM token. An observed accidental OOM stops further escalation.

OOM ledger states: available -> consumed_before_launch -> completed_success /
completed_oom / interrupted. All post-consumption states forbid a second attempt.
For this small project, exactly-once *attempt launch across crashes* favors
consuming the token early: a crash can forfeit the attempt, which is acceptable
and must be reported. Logs spanning sessions/variants share this single ledger.

## Derived outputs

`data/processed/` contains normalized probe rows, cache regression, peak fit,
safe/observed/predicted boundary intervals, allocator paired differences,
fixed-L discrepancy tables and transfer predictions. Include input file hashes,
analysis code SHA, units and schema version. Never edit raw records to fix a
plot. Test fixtures belong under `tests/fixtures/`, labelled synthetic, and must
not be loaded into a report as observations.

## Phase 3 reference adapter

A3 supplies `coordinator_store.py` while B2's `journal.py` remains a stub.
[The coordinator contract](P3_COORDINATOR_CONTRACT.md) describes tested POSIX
locking, mirrored fsynced events, orphan/torn recovery, immutable artifacts and
the shared global ledger. A3 emits `coordinator-v1` run manifests and events with
an `event` discriminator; `probe_result.metrics` carries the `primary-probe-v1`
worker object. This is not a claim that the starter JSON Schema above already
validates those wrappers. B2 must review/map the wrapper/schema and validate the
actual persistent mount before bulk GPU use. CPU synthetic results are temporary
test fixtures and are not scientific observations.
