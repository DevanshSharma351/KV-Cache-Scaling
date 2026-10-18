"""One command regenerates analysis, figures, transfers and report from saved data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capsule',action='append',required=True)
p.add_argument('--run-id',action='append')
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output).resolve()
if out.exists():p.error('Refusing to overwrite a reproduction directory')
out.mkdir(parents=True)
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
commands=[[sys.executable,'-m','kv_cache_scaling','analyze','--output',str(out/'analysis'),'--figures']]
for c in a.capsule:commands[0]+=['--capsule',str(Path(c).resolve())]
for r in a.run_id or []:commands[0]+=['--run-id',r]
commands += [[sys.executable,'-m','kv_cache_scaling','generalize','--output',str(out/'transfer')],
             [sys.executable,str(ROOT/'scripts/build_report.py'),'--analysis',str(out/'analysis'),'--transfer',str(out/'transfer'),'--output',str(out/'report')]]
with (out/'reproduction.jsonl').open('x') as log:
 for i,command in enumerate(commands):
  started=datetime.now(timezone.utc).isoformat()
  result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
  (out/f'{i+1}.stdout.txt').write_text(result.stdout);(out/f'{i+1}.stderr.txt').write_text(result.stderr)
  log.write(json.dumps(dict(command=command,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest()))+'\n')
  log.flush();os.fsync(log.fileno())
  if result.returncode:raise SystemExit(f'Reproduction stopped: {command}; see preserved logs in {out}')
print('Reproduced saved evidence:',out)
