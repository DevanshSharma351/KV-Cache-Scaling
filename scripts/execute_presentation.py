"""Execute the clean CPU demo and export a standalone HTML presentation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from build_presentation_notebook import build,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--source-commit');a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists():p.error('Use a new output directory')
    commit=a.source_commit or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    out.mkdir(parents=True)
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    notebook=nbformat.reads(json.dumps(build(commit)),as_version=4);original_sources=[c.source for c in notebook.cells]
    os.environ['KV_CACHE_ROOT']=str(ROOT);os.environ['KV_CACHE_RESULTS']=str(out/'reproduction')
    # Use this environment's Python kernel, not a globally registered executable.
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir=out/'kernels'/'kv-presentation';kernel_dir.mkdir(parents=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'KV CPU presentation','language':'python'}))
    ksm=KernelSpecManager(kernel_dirs=[str(out/'kernels')])
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='kv-presentation',kernel_spec_manager=ksm)
    client=NotebookClient(notebook,timeout=1200,kernel_name='kv-presentation',resources={'metadata':{'path':str(ROOT)}},km=km)
    try:client.execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    assert [c.source for c in notebook.cells]==original_sources
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(isinstance(c.execution_count,int) and not any(o.output_type=='error' for o in c.outputs) for c in code)
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
    assert images==7,images
    notebook_path=out/'FINAL_PRESENTATION.ipynb';nbformat.write(notebook,notebook_path)
    exporter=HTMLExporter(exclude_markdown=True,exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True)
    body,_=exporter.from_notebook_node(notebook)
    style='<style>body{font-family:system-ui,sans-serif}main{max-width:1160px;margin:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ccd5e1;text-align:left;vertical-align:top}th{background:#edf3fa}img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>'
    body=body.replace('</head>',style+'</head>')
    assert body.count('data:image/png;base64,')>=6
    html_path=out/'PROJECT_PRESENTATION.html';html_path.write_text(body)
    from kv_cache_scaling.delivery import verify_final_scope
    delivery=verify_final_scope(ROOT)
    provenance={'status':'FINAL_CPU_PRESENTATION_VERIFIED_WITH_MEASUREMENT_LIMITATIONS','executed_at_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,'source_tree_dirty':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),'python':sys.version,'executed_code_cells':len(code),'embedded_scientific_figures':images,'model_or_cuda_calls':False,'original_prediction_commit':'7baff5121ed643f27ba06959065855a8913f8e8c','saved_gpu_workers':delivery['total_successful_workers'],'observed_boundary':delivery.get('boundary'),'primary_study_workers':80,'separate_replication_workers':2,'additional_gpu_runs_planned':False,'missing_gpu_evidence':['Two original-allocation expandable held-out repeats','Paired allocator hardware-limit gain','Exact per-token threshold and causal threshold-error decomposition'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (notebook_path,html_path)}}
    (out/'presentation-verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2));print('Results:',out)


if __name__=='__main__':main()
