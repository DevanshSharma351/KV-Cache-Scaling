"""Prospectively registered two-point boundary continuation; never a sweep retry.

Uses the existing guarded worker, immutable journals and project-wide OOM token.
One same-allocation success at 114688 followed by one deliberate 131072 attempt.
Each invocation launches at most one fresh worker, for external checkpointing.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from uuid import uuid4, UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kv_cache_scaling.gate import verify_receipt, digest, json_bytes
from kv_cache_scaling.coordinator_store import RunStore, AttemptLedger, now
from kv_cache_scaling.probe import ProbeRequest
from kv_cache_scaling.processes import worker_env
from kv_cache_scaling.sweep import DriverObserver, collect_process, point_key, IDLE_TOLERANCE, calibrated_admission


def choose_step(results, external_sha):
    """Fail closed after any failed/uncertain attempt; no automatic retries."""
    if not results:
        if external_sha is not None:
            raise ValueError('First save the same-allocation success externally')
        return 114688, 'safe'
    if len(results) == 1 and results[0]['status'] == 'success' and results[0]['request']['length'] == 114688:
        if external_sha is None or len(external_sha) != 64 or any(c not in '0123456789abcdef' for c in external_sha):
            raise ValueError('A consumed external dispatch claim is required')
        return 131072, 'deliberate'
    raise ValueError('Scope complete or failed; no further model probes permitted')


def run(args):
    v = verify_receipt(ROOT, args.receipt)
    prediction = json.loads((ROOT / json.loads((ROOT / args.receipt).read_text())['prediction_path']).read_text())
    if prediction['input_sha256'].get('scripts/targeted_boundary.py') != digest(Path(__file__).read_bytes()):
        raise ValueError('Targeted coordinator must be prospectively frozen')
    observer = DriverObserver(v)
    ledger = AttemptLedger(v.state_root, v.experiment_id)
    run_id = str(UUID(args.run_id))
    identity = dict(schema_version='targeted-boundary-v1', receipt_sha256=v.receipt_sha256,
                    code_commit=v.code_commit, prediction_commit=v.prediction_commit,
                    environment_fingerprint=v.environment_fingerprint, gpu_uuid=v.gpu_uuid,
                    experiment_id=v.experiment_id, execution_scope='targeted_boundary',
                    interpretation='Transferred scheduling fit; independent allocation, no pooling',
                    created_at_utc=now())
    with RunStore(args.local_root, v.state_root, run_id) as store:
        if not (store.local / 'manifest.json').exists():
            store.manifest(identity)
        else:
            previous = json.loads((store.local / 'manifest.json').read_text())
            if {k: x for k, x in previous.items() if k != 'created_at_utc'} != {k: x for k, x in identity.items() if k != 'created_at_utc'}:
                raise ValueError('Resume identity drift')
        if store.orphaned() or any(x['event'] in ('run_finished', 'probe_interrupted', 'run_deviation') for x in store.events):
            raise ValueError('Finished/interrupted scope must never be redispatched')
        results = [x for x in store.events if x['event'] == 'probe_result']
        idle = {x['probe_id'] for x in store.events if x['event'] == 'idle_verified'}
        if any(x['probe_id'] not in idle for x in results):
            raise ValueError('Previous worker lacks idle verification')
        length, kind = choose_step(results, args.external_attempt_sha256)
        if ledger.path.exists():
            raise ValueError('Project-wide token already consumed or blocked')
        if kind == 'safe' and not calibrated_admission(prediction['boundary_continuation']['transferred_fit'], length, v.usable_bytes)['admitted']:
            raise ValueError('Transferred fit does not admit this length with registered margins')
        before = observer.read()
        request = ProbeRequest(str(uuid4()), length, v.gpu_uuid, v.protocol_sha256, v.config_sha256, attempt_kind=kind)
        key = point_key(v, length, 'default', 1, kind)
        if kind == 'deliberate':
            store.artifact('external_dispatch_sha256.txt', (args.external_attempt_sha256 + '\n').encode())
            ledger.consume(run_id, request.probe_id, key)
        store.artifact(f'requests/{request.probe_id}.json', json_bytes(asdict(request)) + b'\n')
        store.append('probe_started', probe_id=request.probe_id, point_key=key, request=asdict(request), idle_before=before)
        def command(output):
            return [sys.executable, '-m', 'kv_cache_scaling.worker', '--root', str(ROOT), '--receipt', args.receipt,
                    '--run-id', run_id, '--request', str(store.local / 'requests' / (request.probe_id + '.json')), '--output', str(output)]
        try:
            record = collect_process(store, request, command, worker_env('default', ROOT), ROOT, 1200)
        except BaseException:
            store.append('probe_interrupted', probe_id=request.probe_id, point_key=key, reason='No automatic retry')
            if kind == 'deliberate':
                ledger.finish(request.probe_id, 'interrupted')
            raise
        if kind == 'deliberate':
            ledger.finish(request.probe_id, record['status'])
        elif record['status'] == 'cuda_oom':
            ledger.block_accidental(run_id, request.probe_id)
        store.append('probe_result', probe_id=request.probe_id, point_key=key, request=asdict(request), **record)
        after = observer.read()
        if any(after[k] != before[k] for k in ('uuid', 'driver_version', 'total_bytes')) or abs(after['used_bytes'] - before['used_bytes']) > IDLE_TOLERANCE:
            store.append('run_deviation', probe_id=request.probe_id, idle_after=after)
            raise RuntimeError('GPU did not return to idle baseline')
        store.append('idle_verified', probe_id=request.probe_id, idle_after=after)
        if kind == 'deliberate':
            store.append('run_finished', status=record['status'], largest_success_tokens=114688,
                         failed_oom_upper_tokens=length if record['status'] == 'cuda_oom' else None,
                         exact_threshold_measured=False, no_further_gpu_probes=True)
        print(json.dumps(dict(run_id=run_id, length=length, status=record['status'],
                              peak_allocated_bytes=(record.get('metrics') or {}).get('peak_allocated_bytes'))))
        if record['status'] not in ('success', 'cuda_oom') or (kind == 'safe' and record['status'] != 'success'):
            raise RuntimeError('Worker failure; scope stopped')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--local-root', required=True)
    p.add_argument('--external-attempt-sha256')
    run(p.parse_args())
