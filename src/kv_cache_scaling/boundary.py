"""Verify a separate two-point boundary capsule without pooling its allocation."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]

def verify_boundary(root=ROOT):
    root=Path(root);config=json.loads((root/'configs/delivery.json').read_text())
    if 'boundary_capsule' not in config:return None
    capsule=root/config['boundary_capsule']
    require=lambda ok,msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require(hashlib.sha256(capsule.read_bytes()).hexdigest()==config['boundary_sha256'],'Boundary capsule hash mismatch')
    with zipfile.ZipFile(capsule) as z:
        names=z.namelist();manifest=json.loads(z.read('MANIFEST.json'))
        require(len(names)==len(set(names)) and set(names)==set(manifest)|{'MANIFEST.json'},'Incomplete capsule')
        for n,h in manifest.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Member hash mismatch')
        run=json.loads(z.read('audit/boundary-run.json'));prefix='state/runs/'+run['run_id']+'/'
        meta=json.loads(z.read(prefix+'manifest.json'));events=[json.loads(l) for l in z.read(prefix+'events.jsonl').splitlines()]
        require([e['sequence'] for e in events]==list(range(1,len(events)+1)),'Journal sequence mismatch')
        starts=[e for e in events if e['event']=='probe_started'];results=[e for e in events if e['event']=='probe_result']
        idle={e['probe_id'] for e in events if e['event']=='idle_verified'}
        require(len(starts)==len(results)==2 and {e['probe_id'] for e in starts}=={e['probe_id'] for e in results}==idle,'Unresolved boundary worker')
        require([(e['request']['length'],e['request']['attempt_kind'],e['status']) for e in results]==[(114688,'safe','success'),(131072,'deliberate','cuda_oom')],'Boundary outcomes differ; cannot claim registered bracket')
        records=[]
        for e in results:
            data=z.read(prefix+e['process']['raw_result']['path'])
            require(hashlib.sha256(data).hexdigest()==e['process']['raw_result']['sha256'],'Raw result mismatch')
            r=json.loads(data);records.append(r)
            require(r==e['metrics'] and r['gpu_uuid']==meta['gpu_uuid'],'GPU/result mismatch')
        require(records[1]['error']['type']=='OutOfMemoryError' and records[1]['stage']=='forward','No typed forward CUDA OOM')
        ledger=json.loads(z.read('state/global_deliberate_attempt.json'))
        require(ledger['state']=='completed_oom' and ledger['probe_id']==results[1]['probe_id'] and ledger['run_id']==run['run_id'],'No completed one-use token')
        require(events[-1]['event']=='run_finished','Boundary scope not finished')
        blob=lambda commit,path:subprocess.check_output(['git','-C',str(root),'show',commit+':'+path])
        receipt_bytes=blob(meta['code_commit'],run['receipt']);receipt=json.loads(receipt_bytes)
        require(hashlib.sha256(receipt_bytes).hexdigest()==meta['receipt_sha256'],'Boundary receipt mismatch')
        pred=blob(receipt['prediction_commit_sha'],receipt['prediction_path'])
        require(hashlib.sha256(pred).hexdigest()==receipt['prediction_sha256'],'Boundary registration mismatch')
        prediction=json.loads(pred)
        require(prediction['input_sha256']['scripts/targeted_boundary.py']==hashlib.sha256(blob(meta['code_commit'],'scripts/targeted_boundary.py')).hexdigest(),'Coordinator was not frozen')
        for r in records:
            for name in ('probe.py','telemetry.py'):
                require(r['code_sha256'][name]==prediction['execution_sha256']['src/kv_cache_scaling/'+name],'Boundary worker source mismatch')
            require(r['execution_policy']['mode']=='single_prefill' and r['execution_policy']['logits_to_keep']==1 and r['allocator']=='default','Boundary workload changed')
        from datetime import datetime
        require(records[0]['started_unix_ns']/1e9>datetime.fromisoformat(receipt['remote_receipt_observed_at_utc']).timestamp(),'Boundary measured before receipt')
        intent_data=z.read('audit/external_dispatch_intent.json');intent=json.loads(intent_data)
        require(hashlib.sha256(intent_data).hexdigest()==z.read(prefix+'external_dispatch_sha256.txt').decode().strip(),'External one-use dispatch mismatch')
        authority=json.loads(z.read('audit/external_authority_ledger.json'))
        require(authority['state']=='consumed_before_launch' and authority['point_key']==hashlib.sha256(intent_data).hexdigest() and authority['run_id']==run['run_id'],'External allowance not consumed before dispatch')
        require(intent['run_id']==run['run_id'] and intent['gpu_uuid']==meta['gpu_uuid'] and intent['length']==131072,'External dispatch identity mismatch')
    prior=root/'artifacts/experiments/2026-10-20-boundary-anchor/safe-checkpoint.zip'
    require(hashlib.sha256(prior.read_bytes()).hexdigest()=='8c799e0d089f3c9b971ad35a17aef7747282f2cf1868a269726b8b422535ab34','Prior anchor capsule mismatch')
    with zipfile.ZipFile(prior) as z:
        pm=json.loads(z.read('MANIFEST.json'))
        require(set(z.namelist())==set(pm)|{'MANIFEST.json'},'Prior anchor manifest incomplete')
        for n,h in pm.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Prior anchor member mismatch')
        pr=json.loads(z.read('audit/boundary-run.json'));pe=[json.loads(x) for x in z.read('state/runs/'+pr['run_id']+'/events.jsonl').splitlines()]
        require([x['event'] for x in pe]==['probe_started','probe_result','idle_verified'] and pe[1]['status']=='success' and pe[1]['request']['length']==114688,'Prior anchor is not one completed success')
        require('state/global_deliberate_attempt.json' not in z.namelist(),'Prior anchor consumed an attempt')
        require(pe[1]['metrics']['gpu_uuid']!=meta['gpu_uuid'],'Prior anchor allocation was incorrectly pooled')
    with zipfile.ZipFile(capsule) as z:
        masking=json.loads(z.read('audit/post-oom-source-inspection.stdout.txt'))
    cfg=json.loads((root/'configs/models/qwen2.5-1.5b.json').read_text())
    require(cfg['sliding_window']==131072 and cfg['use_sliding_window'] is False,'Mask edge config changed')
    require('key_value_length < sliding_window' in masking['AttentionMaskConverter._ignore_causal_mask_sdpa']['source'],'Mask edge source differs')
    require('past_seen_tokens + sequence_length + 1' in masking['Qwen2Model._update_causal_mask']['source'],'Mask target length source differs')
    require('(sequence_length, target_length)' in masking['Qwen2Model._prepare_4d_causal_attention_mask_with_cache_position']['source'],'Mask allocation source differs')
    low,high=114688,131072
    return dict(status='OBSERVED_FORWARD_CUDA_OOM_BRACKET',gpu_uuid=meta['gpu_uuid'],run_id=run['run_id'],
                successful_length=low,failed_oom_length=high,largest_fitting_integer_interval=[low,high-1],
                original_prediction=111232,prediction_error_percent_interval=[100*(111232-(high-1))/(high-1),100*(111232-low)/low],
                exact_threshold_measured=False,successful_workers=2,current_allocation_successful_workers=1,prior_allocation_anchor_successes=1,deliberate_oom_attempts=1,cohorts_pooled=False,
                success_peak_allocated_bytes=records[0]['peak_allocated_bytes'],success_peak_reserved_bytes=records[0]['peak_reserved_bytes'],
                failure_counters=records[1].get('failure_counters'),oom_error=records[1]['error'],
                mask_edge={'length':131072,'shape':[131072,131073],'dtype':'float16','requested_tensor_bytes':2*131072*131073,'GiB':2*131072*131073/1024**3,'evidence':'Installed pinned HF source and matching rounded OOM request; source-supported attribution, not a separate intervention'},
                assumption='Monotone fit/failure boundary for this fixed cold-prefill workload; interval, not exact maximum',
                limitations=['One deliberate attempt bounds the threshold; no repeated OOM search.','Allocator hardware-limit delta is unmeasured.','Failed-forward counters are partial peaks, not the peak of a completed forward.'],
                capsule=str(capsule.relative_to(root)),sha256=config['boundary_sha256'])


def explanation(b):
    if not b:return ''
    lo,hi=b['prediction_error_percent_interval']
    return (f"Separate T4 boundary confirmation: one cold prefill succeeded at {b['successful_length']:,} tokens "
            f"and the one deliberate attempt raised torch.cuda.OutOfMemoryError during forward at {b['failed_oom_length']:,} tokens. "
            f"The largest fitting integer context is bracketed at 114,688–131,071 tokens, assuming monotonicity. "
            f"The original 111,232-token prediction error is {lo:.2f}% to {hi:.2f}% using (prediction-limit)/limit. "
            "This is an interval, not an exact maximum. The failed forward has partial peak counters, not a completed-forward peak. "
            "At L=131,072, the pinned HF mask-elision condition fails at config.sliding_window=131072. The DynamicCache path requests an FP16 [L,L+1] causal mask: 34,360,000,512 bytes (32.00024 GiB), matching the rounded 32.00-GiB OOM request. This is a framework mask cliff before efficient SDPA, not an eager attention score matrix or evidence that the KV cache alone filled VRAM. The attribution is supported by installed source; no separate GPU intervention was run. The new allocation is not pooled into the core curve. No model probes followed the deliberate attempt; "
            "the allocator hardware-limit gain and causal split of the threshold error remain unmeasured.")


def plot_boundary(b,path):
    """Threshold interval only: a failed forward never contributes a completed peak."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.3))
    ax.axvspan(114688,131072,color='#e4e9ef',label='Unresolved interval (monotone boundary assumption)')
    ax.axvline(111232,color='#7d4dba',linestyle='--',label='Original advance prediction: 111,232')
    ax.scatter([114688],[1],color='#198754',s=105,zorder=3,label='Successful cold prefill: 114,688')
    ax.scatter([131072],[1],color='#bc3444',marker='X',s=120,zorder=3,label='Typed forward CUDA OOM: 131,072')
    ax.plot([114688,131072],[1,1],color='#526375',linewidth=3)
    ax.set(xlim=(106000,135500),ylim=(.65,1.45),yticks=[],xlabel='Prefill length (tokens)',title='Observed OOM boundary — separate T4 allocation')
    ax.legend(loc='upper left',fontsize=9,frameon=True)
    ax.text(114688,.89,'Fits',ha='center',color='#126b44');ax.text(131072,.89,'Fails',ha='center',color='#9f2938')
    fig.text(.5,.025,b['gpu_uuid']+' | FP16, B=1, DynamicCache, efficient SDPA, logits_to_keep=1',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
