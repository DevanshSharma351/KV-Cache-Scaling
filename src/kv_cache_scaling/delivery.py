"""Verify final resource-bounded delivery without claiming the original OOM gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from .analysis import read_capsules

ROOT=Path(__file__).resolve().parents[2]


def _require(condition,message):
    if not condition:raise ValueError(message)


def verify_final_scope(root=ROOT):
    root=Path(root).resolve();config=json.loads((root/'configs/delivery.json').read_text())
    _require(config['status']=='FINAL_WITH_MEASUREMENT_LIMITATIONS','Unexpected final scope')
    _require(config['pool_cohorts'] is False and config['additional_gpu_runs_planned'] is False,
             'Final scope must keep cohorts separate and stop GPU work')
    _require(config['exact_oom_threshold_measured'] is False and config['all_original_research_objectives_met'] is False,
             'Do not promote bounded delivery to full experimental completion')
    parsed={};sources={}
    for scope in ('primary','replication'):
        relative=Path(config[scope+'_capsule'])
        _require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe capsule path')
        capsule=root/relative
        _require(hashlib.sha256(capsule.read_bytes()).hexdigest()==config[scope+'_sha256'],'Capsule hash mismatch')
        records,runs,_=read_capsules([capsule]);raw=[r['record'] for r in records]
        _require(len(raw)==config['expected_'+scope+'_workers'] and all(r['status']=='success' for r in raw),'Unexpected worker coverage or failure')
        identities={r['manifest']['gpu_uuid'] for r in runs.values()}
        _require(len(identities)==1,'A scope mixes GPU identities')
        for run in runs.values():
            events=run['events'];results=[e for e in events if e['event']=='probe_result']
            _require({e['probe_id'] for e in results}=={e['probe_id'] for e in events if e['event']=='probe_started'},'Unresolved started worker')
            idle={e['probe_id'] for e in events if e['event']=='idle_verified'}
            _require(all(e['probe_id'] in idle for e in results),'Missing worker exit/idle evidence')
        with zipfile.ZipFile(capsule) as z:
            _require('state/global_deliberate_attempt.json' not in z.namelist(),'Unexpected deliberate-attempt state')
        parsed[scope]=(raw,runs,capsule)
        sources[scope]={'path':str(relative),'sha256':config[scope+'_sha256'],'workers':len(raw),'gpu_uuid':next(iter(identities))}
    _require(sources['primary']['gpu_uuid']!=sources['replication']['gpu_uuid'],'Replication must remain a separate allocation')
    raw,runs,capsule=parsed['replication']
    _require({(r['length'],r['allocator'],r['repeat']) for r in raw}=={(512,'default',1),(512,'default',2)},'Replication scope differs from recorded two probes')
    reference=[r for r in parsed['primary'][0] if (r['measurement'],r['length'],r['allocator'])==('primary',512,'default')]
    rows=[]
    for r in raw:
        _require(r['weights']['unique_parameters']==1543714304 and r['cache']['unique_storage_bytes']==14680064,'Replication weight/cache mismatch')
        _require(all(r['peak_allocated_bytes']==p['peak_allocated_bytes'] for p in reference),'Replication peak differs; report it rather than claim exact agreement')
        rows.append({k:r[k] for k in ('probe_id','length','allocator','repeat','peak_allocated_bytes','peak_reserved_bytes')})
    with zipfile.ZipFile(capsule) as z:closure=json.loads(z.read('audit/scope-closure.json'))
    _require(closure['status']=='CLOSED_AT_USER_RESOURCE_LIMIT' and closure['live_gpu_processes']==[],'Scope was not closed idle')
    _require(closure['new_successful_gpu_workers']==len(raw) and closure['deliberate_attempt_consumed'] is False and closure['cohorts_pooled'] is False,'Closure contradicts saved evidence')
    # Check the new workers against their actual pushed execution receipt and source.
    meta=next(iter(runs.values()))['manifest'];receipt_path='prereg/continuation-2026-10-20/receipt.json'
    blob=lambda commit,path:subprocess.check_output(['git','-C',str(root),'show',commit+':'+path])
    receipt_bytes=blob(meta['code_commit'],receipt_path);receipt=json.loads(receipt_bytes)
    _require(hashlib.sha256(receipt_bytes).hexdigest()==meta['receipt_sha256'],'Replication receipt mismatch')
    prediction_bytes=blob(receipt['prediction_commit_sha'],receipt['prediction_path']);prediction=json.loads(prediction_bytes)
    _require(hashlib.sha256(prediction_bytes).hexdigest()==receipt['prediction_sha256'],'Replication prediction mismatch')
    for name in ('probe.py','telemetry.py'):
        digest=prediction['execution_sha256']['src/kv_cache_scaling/'+name]
        _require(all(r['code_sha256'][name]==digest for r in raw),'Worker execution fingerprint mismatch')
    return {'status':config['status'],'bounded_delivery_complete':True,'all_original_research_objectives_met':False,
            'sources':sources,'total_successful_workers':sum(s['workers'] for s in sources.values()),
            'cohorts_pooled':False,'replication':rows,'closure':closure,'additional_gpu_runs_planned':False,
            'exact_oom_threshold_measured':False,'original_prediction_commit':config['original_prediction_commit'],
            'limitations':['Hardware OOM threshold and its exact prediction error are unobserved.',
                           'Original L=114688 validation remains default n=3, expandable n=1.',
                           'Published transfer verification checks KV bytes, not unrun HF context limits.']}
