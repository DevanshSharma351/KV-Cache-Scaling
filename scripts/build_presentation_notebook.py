"""Generate a clean CPU presentation notebook; execution is a separate audit step."""
import argparse
import json
from pathlib import Path
import subprocess
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit):
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'id':uuid4().hex[:8],'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    def md(text):add('markdown',text)
    def code(text):add('code',text)
    md('# KV-Cache Scaling — final deliverables and live presentation\n\n**Run all on a CPU runtime.** This notebook regenerates and presents real saved GPU evidence. It never loads model weights, allocates CUDA memory or consumes an OOM attempt. All six requested deliverables appear as separate sections.\n\n**Research status:** 80 successful workers are saved. Colab GPU quota blocked two validation repeats and the final boundary attempt. Successful context is a lower bound; no OOM threshold is claimed.\n\nThe five-phase implementation and decision guide is `docs/DECISIONS_AND_PHASE_GUIDE.md`. Historical raw evidence and the original prediction remain unchanged.')
    code(f'''# Reproducible setup: no GPU package imports or Google Drive mount.
import os, sys, json, subprocess, tempfile, hashlib, shutil
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from IPython.display import display, Markdown, HTML
SOURCE_COMMIT={source_commit!r}
REPOSITORY_URL='https://github.com/DevanshSharma351/KV-Cache-Scaling.git'
ROOT=Path(os.environ.get('KV_CACHE_ROOT',Path.cwd())).resolve()
if not (ROOT/'src/kv_cache_scaling/presentation.py').is_file():
    ROOT=Path('/content')/('kv-cache-presentation-'+SOURCE_COMMIT[:8])
    if not ROOT.exists():
        subprocess.run(['git','clone','--quiet',REPOSITORY_URL,str(ROOT)],check=True)
        subprocess.run(['git','-C',str(ROOT),'checkout','--quiet','--detach',SOURCE_COMMIT],check=True)
actual_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
assert actual_commit==SOURCE_COMMIT,'Use the pinned source checkout; do not silently present changed code.'
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=normal'],text=True).strip(),'Use a clean source checkout; preserve edits before running the demo.'
expected={{'matplotlib':'3.10.1','reportlab':'4.4.4','pypdf':'5.9.0'}}
def installed(name):
    try:return version(name)
    except PackageNotFoundError:return None
if any(installed(k)!=v for k,v in expected.items()):
    subprocess.run([sys.executable,'-m','pip','install','--quiet','-e',str(ROOT),'-r',str(ROOT/'requirements-analysis.txt')],check=True)
sys.path.insert(0,str(ROOT/'src'))
print('Presentation source:',actual_commit)
print('Runtime:',sys.version.split()[0],'; CPU replay only; Drive disconnected')
''')
    code('''# Regenerate every graph/table/report from the immutable GPU capsule.
RESULTS=Path(os.environ['KV_CACHE_RESULTS']).resolve() if os.environ.get('KV_CACHE_RESULTS') else Path(tempfile.mkdtemp(prefix='kv-cache-presentation-'))/'results'
command=[sys.executable,str(ROOT/'scripts/reproduce_results.py'),'--capsule',str(ROOT/'artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip'),'--output',str(RESULTS)]
print('Recorded command:',json.dumps(command))
completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=1200)
print(completed.stdout)
if completed.returncode:print(completed.stderr);raise RuntimeError('Reproduction failed; inspect saved logs before presenting.')
from kv_cache_scaling.presentation import Presentation,checkpoint_demo,table
presentation=Presentation(ROOT,RESULTS)
print('All generated manifests, raw capsule hash and original prediction bytes verified.')
presentation.show('overview')
''')
    for section,label in [('derivation','1. Derivation'),('prediction','2. Preregistered prediction'),('checkpointing','3. Checkpointing harness'),('curves','4. Analytical overlays and measured curves'),('discrepancy','5. Discrepancy table'),('transfer','6. Cross-model / cross-GPU verification')]:
        md('## '+label)
        code(f"presentation.show({section!r})\n")
        if section=='prediction':
            code('''# Change these inputs during a live demo; only analytical arithmetic runs.
USABLE_VRAM_GIB = 14.46075439453125 #@param {type:"number"}
BATCH_SIZE = 1 #@param {type:"integer"}
presentation.show_explorer(USABLE_VRAM_GIB,BATCH_SIZE)
''')
        if section=='checkpointing':code("demonstration=checkpoint_demo()\ndisplay(HTML(table(['Temporary CPU recovery check','Result'],list(demonstration.items()))))\n")
    md('## Conclusions and separate downloads\nA CPU replay demonstrates reproducibility of saved measurements. It does not claim new GPU observations or finish the quota-blocked experiment.')
    code("presentation.show('conclusions')\npresentation.show_downloads()\n")
    code('''# Optional download: default does not download anything.
DOWNLOAD_ARTIFACT = "none" #@param ["none", "report", "discrepancy_csv", "all_deliverables_zip"]
if DOWNLOAD_ARTIFACT=='none':
    print('Files remain available in RESULTS. Select an artifact above only when you want to download it.')
else:
    if DOWNLOAD_ARTIFACT=='report':selected=RESULTS/'report/report.pdf'
    elif DOWNLOAD_ARTIFACT=='discrepancy_csv':selected=RESULTS/'analysis/discrepancy.csv'
    elif DOWNLOAD_ARTIFACT=='all_deliverables_zip':
        export_root=RESULTS.parent/'all-deliverables'
        shutil.copytree(RESULTS,export_root,dirs_exist_ok=True)
        for relative in ['docs/DERIVATION.md','docs/DECISIONS_AND_PHASE_GUIDE.md','PHASES.md','prereg/formal-2026-10-18-t4/prediction.json','prereg/formal-2026-10-18-t4/prediction.md','prereg/formal-2026-10-18-t4/receipt.json','requirements-analysis.txt']:
            target=export_root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
        shutil.copytree(ROOT/'src',export_root/'src',dirs_exist_ok=True)
        shutil.copytree(ROOT/'configs',export_root/'configs',dirs_exist_ok=True)
        shutil.copyfile(ROOT/'artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip',export_root/'gpu-checkpoint-080.zip')
        (export_root/'ORIGINAL_COMMIT.json').write_text(json.dumps({'commit':presentation.receipt['prediction_commit_sha'],'original_git_timestamp':presentation.commit_timestamp,'prediction_sha256':presentation.receipt['prediction_sha256']},indent=2))
        selected=Path(shutil.make_archive(str(RESULTS.parent/'kv-cache-deliverables'),'zip',export_root))
    else:raise ValueError('Unknown download choice')
    try:
        from google.colab import files
    except ImportError:print('Local artifact:',selected)
    else:files.download(str(selected))
''')
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'KV_Cache_Final_Presentation.ipynb','toc_visible':True},'project_source_commit':source_commit,'execution_mode':'CPU replay of verified measurements; no model/GPU calls'},'cells':cells}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit');p.add_argument('--output',required=True);a=p.parse_args()
    commit=a.source_commit or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    out=Path(a.output)
    if out.exists():p.error('Refusing to overwrite a notebook')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(build(commit),ensure_ascii=False,indent=1)+'\n');print(out)
