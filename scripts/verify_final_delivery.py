"""Verify the agreed final evidence scope; never substitute for the original OOM gate."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from kv_cache_scaling.delivery import verify_final_scope
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');args=p.parse_args()
    result=verify_final_scope()
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:
        with Path(args.output).open('x') as h:h.write(payload)
    print(payload)
