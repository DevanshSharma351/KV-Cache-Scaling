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
    md('# KV-Cache Scaling — final deliverables and live presentation\n\n**Present without execution:** scroll to the final presentation board for saved graphs, tables and all six deliverables. **Run all on a CPU runtime** only when demonstrating reproducibility. This notebook regenerates and presents real saved GPU evidence. It never loads model weights, allocates CUDA memory or consumes an OOM attempt. All six requested deliverables appear as separate sections.\n\n**Final scope:** 80 core-study workers and two independent replication checks are saved. On 20 October the user closed further GPU probing to conserve resources. This resource-bounded delivery includes all six artifact types; successful context is a lower bound and the exact OOM threshold is unmeasured. No additional GPU run is planned.\n\nThe five-phase implementation and decision guide is `docs/DECISIONS_AND_PHASE_GUIDE.md`. Historical raw evidence and the original prediction remain unchanged.')
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
    md('## Conclusions and separate downloads\nA CPU replay demonstrates reproducibility of saved measurements. It does not claim new GPU observations or claim completion of the deferred original OOM experiment.')
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
        for relative in ['docs/DERIVATION.md','docs/DECISIONS_AND_PHASE_GUIDE.md','PHASES.md','prereg/formal-2026-10-18-t4/prediction.json','prereg/formal-2026-10-18-t4/prediction.md','prereg/formal-2026-10-18-t4/receipt.json','requirements-analysis.txt','docs/FINAL_SCOPE.md','configs/delivery.json','artifacts/experiments/2026-10-20-continuation/verification.json','docs/OOM_BOUNDARY.md','artifacts/experiments/2026-10-21-boundary/verification.json']:
            target=export_root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
        shutil.copytree(ROOT/'src',export_root/'src',dirs_exist_ok=True)
        shutil.copytree(ROOT/'configs',export_root/'configs',dirs_exist_ok=True)
        shutil.copyfile(ROOT/'artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip',export_root/'gpu-checkpoint-080.zip')
        shutil.copyfile(ROOT/'artifacts/experiments/2026-10-20-continuation/closed-scope-checkpoint.zip',export_root/'replication-checkpoint.zip')
        shutil.copytree(ROOT/'artifacts/experiments/2026-10-21-boundary',export_root/'boundary-evidence')
        shutil.copytree(ROOT/'artifacts/experiments/2026-10-20-boundary-anchor',export_root/'prior-anchor-evidence')
        (export_root/'ORIGINAL_COMMIT.json').write_text(json.dumps({'commit':presentation.receipt['prediction_commit_sha'],'original_git_timestamp':presentation.commit_timestamp,'prediction_sha256':presentation.receipt['prediction_sha256']},indent=2))
        selected=Path(shutil.make_archive(str(RESULTS.parent/'kv-cache-deliverables'),'zip',export_root))
    else:raise ValueError('Unknown download choice')
    try:
        from google.colab import files
    except ImportError:print('Local artifact:',selected)
    else:files.download(str(selected))
''')
    # Ordinary Markdown avoids Colab's authenticated rich-output iframe renderer.
    # Keep this board pinned to committed plots; CPU replay above remains independent.
    base='https://github.com/DevanshSharma351/KV-Cache-Scaling/blob/'+source_commit+'/'
    raw='https://raw.githubusercontent.com/DevanshSharma351/KV-Cache-Scaling/'+source_commit+'/'
    primary=ROOT/'data/processed/project-results'
    import hashlib
    expected=json.loads((primary/'manifest.json').read_text())['analysis.json']
    if hashlib.sha256((primary/'analysis.json').read_bytes()).hexdigest()!=expected:
        raise ValueError('Static presentation data differs from its generated manifest')
    analysis=json.loads((primary/'analysis.json').read_text())
    prediction=json.loads((ROOT/'prereg/formal-2026-10-18-t4/prediction.json').read_text())
    board=['## Final presentation board — no runtime needed',
        'Present this section directly. These are ordinary notebook text, tables and pinned saved images. Running the CPU cells above is optional and demonstrates regeneration, not a new benchmark.',
        '**Final scope:** 80 core-study workers plus two separate L=512 replication checks; no additional GPU run is planned. Exact OOM threshold is unmeasured. Different GPU allocations are not pooled.',
        '### 1. Derivation',
        '`M(L,B) = W + 2BLNKDs + A(L,B) + H0`. L is one cold prefill in tokens, B is batch size, N=28 layers, K=2 KV heads, D=128, s=2 FP16 bytes. W=3,087,428,608 unique weight bytes; KV=28,672BL bytes. Primary uses DynamicCache, efficient SDPA and logits_to_keep=1. Query heads would overestimate KV by 6×; tied embeddings count once.',
        '[Full symbol-defined derivation]('+base+'docs/DERIVATION.md).',
        '### 2. Original preregistration',
        'Original commit: `7baff5121ed643f27ba06959065855a8913f8e8c`; original timestamp **2026-10-18T19:54:52+05:30**. Central limit **111232 tokens**, assumption range **76024–131072**, at **15527116800 usable bytes**. Original bytes are unchanged.',
        '| L tokens | KV bytes | Central tensor peak bytes | Central admission budget bytes |',
        '| ---: | ---: | ---: | ---: |']
    for point in prediction['memory_predictions']:
        c=next(x for x in point['scenarios'] if x['name']=='central')
        board.append('| '+' | '.join(str(v) for v in (point['length_tokens'],point['kv_bytes'],c['tensor_peak_bytes'],c['budget_bytes']))+' |')
    board += ['### 3. Checkpointing harness',
        'Fresh worker per point; synchronized reset/max peak counters; checkpoint, member-hash verification and fsync before the next dispatch. Saved scopes: 48 primary + 28 diagnostics + four held-out workers. Two new-allocation L=512 checks matched peak and KV bytes exactly. No deliberate OOM attempt was dispatched.',
        '[Harness source]('+base+'src/kv_cache_scaling/sweep.py) · [Final scope and decisions]('+base+'docs/FINAL_SCOPE.md).',
        '### 4. Measured curves and analytical overlays',
        'Both allocators succeeded at **114688 tokens**. Direct cache slope is **28672 bytes/token**, R²=1. Total peak slope includes transient memory. Reserved-minus-allocated slack is not a pure fragmentation measurement. The context result is a successful lower bound.']
    for filename,title in [('01_peak_vs_length','Peak memory versus original analytical envelope'),('02_cache_scaling','Direct KV scaling'),('03_allocator_slack','Allocator slack'),('04_boundary_status','Successful lower bounds and unobserved OOM boundary')]:
        board.append('!['+title+']('+raw+'artifacts/figures/project-results/'+filename+'.png)')
    selected=max((p for p in analysis['probes'] if p['measurement']=='primary' and p['allocator']=='default'),key=lambda p:(p['length'],-p['repeat']))
    gap=next(g for g in analysis['discrepancy_tables'] if g['probe_id']==selected['probe_id'])
    board += ['### 5. Discrepancy accounting',
        'Representative default L=114688, repeat 1. Positive bytes increase measured peak relative to the original tensor envelope; negative bytes decrease it. This is fixed-length peak accounting, not an unmeasured OOM-limit gap.',
        '| Component | Signed bytes | Evidence status |','| --- | ---: | --- |']
    board += ['| '+r['component'].replace('_',' ')+' | '+str(r['signed_bytes'])+' | '+('Supported accounting' if r['supported'] else 'Explicit unaccounted row')+' |' for r in gap['rows']]
    board += ['Signed gap: **'+str(gap['signed_gap_bytes'])+' bytes**; closure: **'+str(gap['closure_error_bytes'])+' bytes**; unaccounted: **'+str(gap['unaccounted_absolute_bytes'])+' bytes**. Counter/lifetime closure does not establish a causal activation/fragmentation decomposition. Workspace at large L is labelled fixed-size inference from L=512 controls.',
        '![Default discrepancy components]('+raw+'artifacts/figures/project-results/05_discrepancy_default.png)',
        '![Expandable discrepancy components]('+raw+'artifacts/figures/project-results/05_discrepancy_expandable.png)',
        '### 6. Cross-model / cross-GPU predictions and verification',
        (primary/'transfer/transfer.md').read_text(),
        '**Interpretation:** the published comparison verifies KV block arithmetic only; the target HF context thresholds were not measured.',
        '### Final deliverable files',
        '[Report PDF]('+base+'artifacts/reports/project-report/report.pdf) · [All figures]('+base+'artifacts/figures/project-results/) · [Discrepancy CSV]('+base+'data/processed/project-results/discrepancy.csv) · [Transfer table]('+base+'data/processed/project-results/transfer/transfer.md) · [Decision guide]('+base+'docs/DECISIONS_AND_PHASE_GUIDE.md).',
        'Five phases: derive/preregister → isolated probes → checkpoint/safety → measured accounting → transfer/presentation. The resource-bounded delivery is complete; original exact-OOM objectives are explicitly unfulfilled. For offline display, use the standalone HTML with embedded images.']
    from kv_cache_scaling.boundary import verify_boundary,explanation
    boundary=verify_boundary(ROOT)
    if boundary:
        board=[text.replace('Exact OOM threshold is unmeasured.','The exact token threshold remains interval-valued; a typed OOM bracket is now measured.').replace('No deliberate OOM attempt was dispatched.','One deliberate forward CUDA OOM was later recorded on a separate allocation.').replace('original exact-OOM objectives are explicitly unfulfilled','exact per-token OOM objectives remain interval-valued') for text in board]
        board[2]='**Final scope:** core curve and replication are preserved separately; the completed targeted OOM check is below. No more model probes follow the deliberate attempt.'
        board[3:3]=['### Completed boundary experiment',explanation(boundary),
            '| Length tokens | Outcome | Allocation |','| ---: | --- | --- |',
            '| 114688 | Successful cold prefill | '+boundary['gpu_uuid']+' |',
            '| 131072 | torch.cuda.OutOfMemoryError during forward | Same allocation |',
            '![Observed same-allocation OOM bracket]('+raw+'artifacts/figures/project-results/06_observed_oom_boundary.png)']
        cells[0]['source']=['# KV-Cache Scaling — final deliverables\n\nThe final presentation board below shows all six deliverables without a runtime. A separate targeted check now records a successful 114,688-token prefill and typed forward CUDA OOM at 131,072. The original core curve remains separate. Run All on CPU only to regenerate the evidence.\n']
    board_text=''
    previous=''
    for block in board:
        if board_text:board_text += '\n' if block.startswith('|') and previous.startswith('|') else '\n\n'
        board_text += block
        previous=block
    md(board_text)
    cells[-1]['id']='a9d9306e'
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'KV_Cache_Final_Presentation.ipynb','toc_visible':True},'project_source_commit':source_commit,'execution_mode':'CPU replay of verified measurements; no model/GPU calls'},'cells':cells}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit');p.add_argument('--output',required=True);a=p.parse_args()
    commit=a.source_commit or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    out=Path(a.output)
    if out.exists():p.error('Refusing to overwrite a notebook')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(build(commit),ensure_ascii=False,indent=1)+'\n');print(out)
