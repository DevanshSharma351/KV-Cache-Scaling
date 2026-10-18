"""CPU-only report regeneration; no CUDA or model loading."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kv_cache_scaling.reporting import build_report
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--analysis',required=True);p.add_argument('--transfer',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
try:result=build_report(a.analysis,a.transfer,a.output)
except (ValueError,OSError,KeyError) as e:p.error(str(e))
print(result['status']);print(result['report'])
