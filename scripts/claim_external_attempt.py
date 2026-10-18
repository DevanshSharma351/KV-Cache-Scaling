"""Consume the external one-use dispatch allowance before any Colab OOM launch.

The local filesystem is the durable authority when Drive is disconnected. Never
delete this state or create another state root to retry an uncertain dispatch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID, uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kv_cache_scaling.coordinator_store import AttemptLedger,atomic_bytes


def reserve(receipt_path, run_id, length, state_root):
    receipt_path=Path(receipt_path).resolve();state_root=Path(state_root).resolve()
    receipt=json.loads(receipt_path.read_text());run_id=str(UUID(run_id))
    if receipt.get('status')!='COMPLETE_PREREGISTRATION_RECEIPT':
        raise ValueError('A completed prospective receipt is required')
    if type(length) is not int or not 1<=length<=131072:raise ValueError('Invalid confirmation length')
    prediction=json.loads((ROOT/receipt['prediction_path']).read_text())
    if hashlib.sha256((ROOT/receipt['prediction_path']).read_bytes()).hexdigest()!=receipt['prediction_sha256']:
        raise ValueError('Prediction differs from receipt')
    inventory=json.loads((ROOT/prediction['context_inventory_path']).read_text())
    intent={'scope':'External one-use dispatch claim; not an internal GPU-worker claim',
            'state':'consumed_before_external_dispatch','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'experiment_id':receipt['experiment_id'],'run_id':run_id,'length':length,
            'gpu_uuid':inventory['gpu_identity_csv'].split(',')[0],
            'receipt_path':str(receipt_path.relative_to(ROOT)),
            'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'prediction_commit':receipt['prediction_commit_sha'],
            'config_sha256':receipt['config_sha256'],'protocol_sha256':receipt['protocol_sha256'],
            'retry_policy':'Never dispatch another deliberate attempt, even after uncertain or interrupted UI execution'}
    data=(json.dumps(intent,sort_keys=True,indent=2)+'\n').encode()
    digest=hashlib.sha256(data).hexdigest()
    ledger=AttemptLedger(state_root,receipt['experiment_id'])
    ledger.consume(run_id,str(uuid4()),digest)
    atomic_bytes(state_root/'external_dispatch_intent.json',data)
    return {'status':'EXTERNAL_ATTEMPT_CONSUMED_BEFORE_DISPATCH','external_attempt_sha256':digest,
            'intent_path':str(state_root/'external_dispatch_intent.json'),'ledger_path':str(ledger.path),
            'run_id':run_id,'length':length}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--length',type=int,required=True);p.add_argument('--state-root',required=True)
    a=p.parse_args()
    print(json.dumps(reserve(a.receipt,a.run_id,a.length,a.state_root),indent=2))
