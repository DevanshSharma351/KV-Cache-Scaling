"""Verify the complete final project cohort and its external one-use dispatch proof."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kv_cache_scaling.analysis import read_capsules, write_analysis
from kv_cache_scaling.reporting import evidence_gaps


def require(condition,message):
    if not condition:raise ValueError(message)


def verify(capsule,external_intent,external_ledger):
    records,runs,sources=read_capsules([capsule])
    expected={'primary_sweep':48,'diagnostics':28,'calibrated_validation':6,'boundary_confirmation':1}
    require(Counter(r['manifest']['execution_scope'] for r in runs.values())==Counter({k:1 for k in expected}),
            'Exactly four completed execution scopes are required')
    all_records=[r['record'] for r in records]
    require(len(all_records)==83,'The final cohort requires 83 distinct GPU workers')
    require(len({r['pid'] for r in all_records})==83,'GPU workers must have distinct process IDs')
    identities={(r['manifest']['gpu_uuid'],r['manifest']['environment_fingerprint'],r['manifest']['usable_bytes']) for r in runs.values()}
    require(len(identities)==1,'Do not combine different GPU/runtime/budget cohorts')
    for run_id,run in runs.items():
        events=run['events'];meta=run['manifest']
        require(events[-1]['event']=='run_finished','Unfinished run: '+run_id)
        require(not any(e['event'] in ('run_stopped','run_deviation','probe_interrupted') for e in events),'Interrupted/deviating run: '+run_id)
        results=[e for e in events if e['event']=='probe_result']
        require(len(results)==expected[meta['execution_scope']],'Unexpected scope worker count')
        idle={e['probe_id'] for e in events if e['event']=='idle_verified'}
        require(all(e['probe_id'] in idle for e in results),'A worker lacks process-exit/idle verification')
        reference=meta.get('reference_sweep')
        if reference:
            prefix='state/runs/'+reference['run_id']+'/'
            with zipfile.ZipFile(capsule) as z:
                for filename,key in [('manifest.json','manifest_sha256'),('events.jsonl','journal_sha256')]:
                    require(hashlib.sha256(z.read(prefix+filename)).hexdigest()==reference[key],
                            'Prospective reference changed after dispatch')
    failures=[r for r in all_records if r['status']!='success']
    require(len(failures)==1,'Exactly one typed final OOM, and no other failed probe, is required')
    oom=failures[0]
    require((oom['status'],oom['attempt_kind'],oom['oom_classification'],oom['error']['type'],oom['length'],oom['allocator'])
            ==('cuda_oom','deliberate','deliberate_request','OutOfMemoryError',131072,'default'),
            'Only the registered final deliberate CUDA OOM may fail')
    require(all(r['finished_unix_ns']<=oom['started_unix_ns'] for r in all_records if r is not oom),
            'A GPU model probe followed the final attempt')
    intent_bytes=Path(external_intent).read_bytes();intent=json.loads(intent_bytes)
    ledger=json.loads(Path(external_ledger).read_text())
    confirmation_id,confirmation=next((run_id,r) for run_id,r in runs.items() if r['manifest']['execution_scope']=='boundary_confirmation')
    meta=confirmation['manifest'];intent_sha=hashlib.sha256(intent_bytes).hexdigest()
    require(meta['external_attempt_sha256']==intent_sha,'External dispatch proof does not match confirmation')
    for key in ('experiment_id','gpu_uuid','config_sha256','protocol_sha256'):
        require(intent[key]==meta[key],'External dispatch scientific identity mismatch: '+key)
    require(intent['run_id']==confirmation_id and intent['length']==131072,'External run/length mismatch')
    require(ledger['run_id']==intent['run_id'] and ledger['experiment_id']==intent['experiment_id']
            and ledger['point_key']==intent_sha and ledger['state']=='completed_oom'
            and ledger['result_status']=='cuda_oom','External allowance was not consumed and completed exactly once')
    require(datetime.fromisoformat(ledger['consumed_at_utc']).timestamp()<oom['started_unix_ns']/1e9,
            'External allowance was not consumed before GPU launch')
    with zipfile.ZipFile(capsule) as z:
        internal=json.loads(z.read('state/global_deliberate_attempt.json'))
        frozen_reference=json.loads(z.read('audit/heldout-reference.json'))
        acknowledgements=[json.loads(line) for line in z.read('audit/external-checkpoint-receipts.jsonl').splitlines()]
    require(internal['state']=='completed_oom' and internal['probe_id']==oom['probe_id']
            and internal['experiment_id']==ledger['experiment_id']
            and internal['run_id']==ledger['run_id'],'Internal and external attempt ledgers disagree')
    validation=next(r for r in runs.values() if r['manifest']['execution_scope']=='calibrated_validation')
    require(validation['manifest']['reference_sweep']==frozen_reference['reference'],'Held-out fit changed after its external freeze')
    require(any(r['sha256']==frozen_reference['parent_checkpoint_sha256'] for r in acknowledgements),
            'Held-out reference parent lacks an external checkpoint receipt')
    heldout=[r['record'] for r in records if r['manifest']['execution_scope']=='calibrated_validation']
    require(datetime.fromisoformat(frozen_reference['frozen_at_utc']).timestamp()
            <min(r['started_unix_ns'] for r in heldout)/1e9,'Held-out reference was not frozen prospectively')
    code_commits={r['manifest']['code_commit'] for r in runs.values()}
    require(len(code_commits)==1,'Execution source commit changed within the final cohort')
    code_commit=next(iter(code_commits))
    def blob(commit,path):
        return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])
    receipt_bytes=blob(code_commit,intent['receipt_path']);receipt=json.loads(receipt_bytes)
    require(all(hashlib.sha256(receipt_bytes).hexdigest()==r['manifest']['receipt_sha256'] for r in runs.values())
            and hashlib.sha256(receipt_bytes).hexdigest()==intent['receipt_sha256'],'Frozen receipt changed')
    prediction_bytes=blob(receipt['prediction_commit_sha'],receipt['prediction_path'])
    require(hashlib.sha256(prediction_bytes).hexdigest()==receipt['prediction_sha256'],'Frozen prediction hash mismatch')
    prediction=json.loads(prediction_bytes)
    for path,expected_hash in prediction['execution_sha256'].items():
        require(hashlib.sha256(blob(code_commit,path)).hexdigest()==expected_hash,'Execution source differs from preregistration')
    expected_sources={name:prediction['execution_sha256']['src/kv_cache_scaling/'+name] for name in ('probe.py','telemetry.py')}
    require(all(r['code_sha256']==expected_sources for r in all_records),'Worker source fingerprint mismatch')
    observed=datetime.fromisoformat(receipt['remote_receipt_observed_at_utc']).timestamp()
    require(all(r['started_unix_ns']/1e9>observed for r in all_records),'A model worker predates its prospective receipt')
    with tempfile.TemporaryDirectory() as d:
        output=Path(d)/'analysis';write_analysis([capsule],output)
        a=json.loads((output/'analysis.json').read_text())
    require(all(a['acceptance'].values()),'One or more empirical acceptance criteria failed')
    require(not evidence_gaps(a),'Missing scientific evidence: '+'; '.join(evidence_gaps(a)))
    trace=next(r for r in all_records if r['measurement']=='kernel_trace')
    require(any('sm75' in name.lower() and ('fmha' in name.lower() or 'cutlass' in name.lower())
                for name in trace['kernel_diagnostic']['event_names']),'Missing actual SM75 efficient-attention kernel')
    return dict(status='PASS',scope='Final project delivery; independent human reproduction not claimed',
                verified_at_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
                gpu_uuid=next(iter(identities))[0],unique_gpu_worker_pids=83,successful_workers=82,
                deliberate_oom_workers=1,execution_scopes=expected,acceptance=a['acceptance'],
                hardware_brackets=a['hardware_brackets'],held_out_validation=a['held_out_validation'],
                external_intent_sha256=intent_sha,limitations=a['limitations'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',required=True);p.add_argument('--external-intent',required=True)
    p.add_argument('--external-ledger',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    try:result=verify(args.capsule,args.external_intent,args.external_ledger)
    except (ValueError,KeyError,OSError,zipfile.BadZipFile) as error:p.error(str(error))
    with Path(args.output).open('x') as h:h.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
