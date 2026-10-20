"""CPU-only presentation of verified saved evidence; no model or CUDA imports."""
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from uuid import uuid4

from .journal import RunStore, append_event
from .model_math import Architecture, prefill_scenarios, predict_limit, vram_gib_to_bytes
from .reporting import evidence_gaps

CAPSULE='artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip'
CAPSULE_SHA='cd0dedd8185a876e1a600b2c3ec8a1b95ac3701f8e22be57d2bab82f51c9c389'
PREDICTION_COMMIT='7baff5121ed643f27ba06959065855a8913f8e8c'
SECTIONS=('overview','derivation','prediction','checkpointing','curves','discrepancy','transfer','conclusions')


def table(headers,rows):
    """Escaped display table; values cannot inject markup into a notebook."""
    head=''.join('<th>'+escape(str(v))+'</th>' for v in headers)
    body=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return '<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'


def verify_manifest(directory):
    directory=Path(directory)
    for name,digest in json.loads((directory/'manifest.json').read_text()).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe generated manifest path')
        if hashlib.sha256((directory/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Generated artifact changed: '+name)


class Presentation:
    """Render the six required deliverables independently of the report PDF."""
    def __init__(self,root,results):
        self.root=Path(root).resolve();self.results=Path(results).resolve()
        from .delivery import verify_final_scope
        self.delivery=verify_final_scope(self.root)
        if hashlib.sha256((self.root/CAPSULE).read_bytes()).hexdigest()!=CAPSULE_SHA:
            raise ValueError('Saved GPU capsule differs from the verified checkpoint')
        for name in ('analysis','analysis/figures','transfer','report'):
            verify_manifest(self.results/name)
        self.analysis=json.loads((self.results/'analysis/analysis.json').read_text())
        self.transfer=json.loads((self.results/'transfer/transfer.json').read_text())
        if {s['sha256'] for s in self.analysis['provenance']['sources']}!={CAPSULE_SHA}:
            raise ValueError('Presentation analysis has a different evidence source')
        self.receipt=json.loads((self.root/'prereg/formal-2026-10-18-t4/receipt.json').read_text())
        frozen=subprocess.check_output(['git','-C',str(self.root),'show',PREDICTION_COMMIT+':'+self.receipt['prediction_path']])
        if hashlib.sha256(frozen).hexdigest()!=self.receipt['prediction_sha256'] or (self.root/self.receipt['prediction_path']).read_bytes()!=frozen:
            raise ValueError('Original prediction differs from its historical commitment')
        self.prediction=json.loads(frozen)
        self.commit_timestamp=subprocess.check_output(['git','-C',str(self.root),'show','-s','--format=%cI',PREDICTION_COMMIT],text=True).strip()
        self.architecture=Architecture.from_config(json.loads((self.root/'configs/models/qwen2.5-1.5b.json').read_text()))

    def explore(self,usable_gib=14.46075439453125,batch_size=1):
        """Analytical sensitivity only. Changing inputs never launches a GPU probe."""
        usable=vram_gib_to_bytes(usable_gib)
        return [(s.name,predict_limit(self.architecture,usable,s,batch_size=batch_size),
                 predict_limit(self.architecture,usable,s,cap=self.architecture.max_positions,batch_size=batch_size))
                for s in prefill_scenarios(self.architecture)]

    def content(self,section):
        if section not in SECTIONS:raise ValueError('Unknown presentation section')
        a=self.analysis;blocks=[]
        def md(text):blocks.append(('markdown',text))
        def html(headers,rows):blocks.append(('html',table(headers,rows)))
        def image(name):blocks.append(('image',str(self.results/'analysis/figures'/name)))
        if section=='overview':
            md('# KV-Cache Scaling\n## Predicting the context limit before you hit it\nWe built an analytical predictor, a checkpointing GPU harness, and a reproducible evidence pipeline. This presentation exposes all six required deliverables separately from the PDF.')
            html(['Evidence','Actual result'],[['Core study / separate replication workers',str(len(a['probes']))+' / '+str(len(self.delivery['replication']))],['Primary model','Qwen2.5-1.5B, FP16, batch 1'],['Largest successful prefill','114,688 tokens with both allocators'],['KV storage slope','28,672 bytes/token; exact agreement with theory'],['Final scope','Resource-bounded delivery; exact OOM threshold unmeasured'],['Demo mode','CPU replay of real saved measurements; no model load or OOM attempt']])
            md('The user closed further GPU work on **20 October 2026** to conserve resources. **82 successful workers** are preserved across two separate allocations. No additional GPU run is planned. The original full OOM experiment is not claimed complete.')
            html(['Separate replication allocator','L','Repeat','Peak allocated bytes'],[[r['allocator'],r['length'],r['repeat'],r['peak_allocated_bytes']] for r in self.delivery['replication']])
            md('The two new-allocation L=512 checks exactly match the core study. They are excluded from its curve, fit and repeat counts; GPU UUIDs are not pooled.')
            identity=next(g['identity'] for g in a['groups'] if g['identity']['measurement']=='primary')
            html(['Measured environment','Recorded value'],[['GPU UUID',identity['gpu_uuid']],['Driver',identity['driver']],['PyTorch / CUDA runtime',identity['torch']+' / '+identity['cuda']],['Transformers',identity['transformers']],['Context-adjusted usable bytes',identity['usable_bytes']],['Model revision',identity['policy']['model_revision']]])
            md('### Five phases followed\n1. Derive and preregister the prediction.\n2. Implement and verify the isolated probe.\n3. Add checkpointing, recovery and safe scheduling.\n4. Measure curves and account for discrepancies.\n5. Generalize, reproduce and present the results.')
        elif section=='derivation':
            md('## Deliverable 1 — Derivation and defined symbols\nSequence length **L** means one cold forward pass over exactly L tokens, starting with an empty DynamicCache. It is not generation to L. We use `logits_to_keep=1`, FP16 and forced efficient SDPA.\n\n```text\nM(L,B) = W + KV(L,B) + A(L,B) + H0\nKV(L,B) = 2 × B × L × N × K × D × s\nW = s × unique_parameter_count\n```')
            html(['Symbol','Meaning','Value / units'],[['L','Input prefill sequence length','tokens'],['B','Batch size','1 measured; variable in theory'],['N','Transformer layers',28],['Q / K','Query / KV heads','12 / 2'],['D','Head dimension',128],['s','Bytes per FP16 element',2],['W','Unique tied FP16 weights','3,087,428,608 bytes'],['A','Length-dependent activation/workspace envelope','assumed before measurement; diagnostics assess it'],['H0','Fixed budget allowance','536,870,912 bytes in central scenario']])
            md('**Result:** `KV(L,B) = 28,672 × B × L bytes`. Using Q instead of K gives a **6× overestimate**. Counting tied input/output embeddings twice adds **466,747,392 bytes**. Architecture values come from the pinned model config.\n\nUnique parameter formula (all biases/norms included):\n```text\nP = V×H + N×[H×q + 2H×k + q×H + q + 2k + 3H×I + 2H] + H\nq = Q×D; k = K×D; H = 1536; I = 8960; V = 151936\nP = 1,543,714,304; W = 2P\n```\nThe full derivation and lifetime assumptions are in `docs/DERIVATION.md`.')
        elif section=='prediction':
            utc=datetime.fromisoformat(self.commit_timestamp).astimezone(timezone.utc).isoformat()
            md('## Deliverable 2 — Original preregistered prediction\n**Committed before model measurements.** The original bytes and timestamp are preserved; later measurements do not replace the commitment.\n\n- Commit: `'+PREDICTION_COMMIT+'`\n- Original Git commit timestamp: **'+self.commit_timestamp+'**\n- Same instant in UTC: **'+utc+'**\n- Later remote receipt observation: **'+self.receipt['remote_receipt_observed_at_utc']+'**\n- Usable VRAM: **15,527,116,800 bytes**, after context, before weights.\n- Central prediction: **111,232 tokens**; assumption range **76,024–131,072**.\n- Model-card quality domain: 32,768; higher lengths are synthetic memory stress.\n\n[Inspect the original commit](https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/'+PREDICTION_COMMIT+').')
            rows=[]
            for p in self.prediction['memory_predictions']:
                central=next(s for s in p['scenarios'] if s['name']=='central')
                rows.append([p['length_tokens'],p['kv_bytes'],central['tensor_peak_bytes'],central['budget_bytes']])
            html(['L tokens','KV bytes','Predicted tensor peak bytes','Predicted budget bytes incl. H0'],rows)
            md('Parametric prediction: `L_max(U,B) = max(0, floor((U − W − H0) / (B × (28,672 + a))))`, capped by the config when reported. Central `a = 78,336 bytes/token`. U is context-adjusted usable VRAM, not the GPU label. The range is sensitivity to assumptions, not a confidence interval.')
        elif section=='checkpointing':
            md('## Deliverable 3 — Checkpointing sweep harness\nEach probe uses a new process. Allocator variables are set before CUDA initialization; load and forward peaks are measured separately with a peak reset. Worker output is saved, fsynced, mirrored and verified before another dispatch.\n\nThe immutable ZIP contains the real raw records, journals, command logs and external checksum acknowledgements. This is how the 80 core measurements survived Colab runtime loss. Two separate replication workers were also verified and saved before the user closed further GPU work.')
            html(['Scope','Saved workers','Journal state'],[['Primary paired sweep',48,'Finished'],['Diagnostics',28,'Finished'],['Held-out validation',4,'Partial: default 3, expandable 1; further repeats deferred'],['Final deliberate boundary attempt',0,'Deferred under final scope; allowance unused']])
            md('**Saved checkpoint SHA-256:** `'+CAPSULE_SHA+'`\n\nLive CPU recovery demonstration below uses a temporary directory. Its events are explicitly demo events, never GPU measurements. No OOM token is consumed.')
        elif section=='curves':
            md('## Deliverable 4 — Measured curves with analytical overlays\nPoints are actual observations; bands are advance assumptions. Repeats use fresh workers. The total-peak slope includes transient memory and must not be called the KV slope.')
            image('01_peak_vs_length.png');image('02_cache_scaling.png')
            html(['Allocator','Observed held-out L','Repeats','Peak allocated bytes','Frozen-fit prediction error %'],[[x['allocator'],x['length'],x['repeats'],x['observed_peak_median_bytes'],round(x['prospective_fit_error_percent'],6)] for x in a['held_out_validation']])
            md('The later fit was frozen before held-out execution. Its small error validates that later fit at this point; it does not turn the original commitment into an exact OOM prediction.')
            image('03_allocator_slack.png');image('04_boundary_status.png')
            md('Reserved minus allocated at the same snapshot is allocator slack, not a pure fragmentation measurement. **114,688 is a successful lower bound; no failed upper bound was observed.** The safe-policy delta cannot establish the hardware-context benefit of expandable segments.')
        elif section=='discrepancy':
            md('## Deliverable 5 — Discrepancy accounting, down to bytes\nThe table compares matching tracked tensor peaks at a fixed L. Fixed headroom H0 belongs to the budget model and is not silently compared with allocated tensor memory. Positive rows increase measured memory; negative rows reduce it.')
            candidates=[p for p in a['probes'] if p['measurement']=='primary' and p['allocator']=='default']
            selected=max(candidates,key=lambda p:(p['length'],-p['repeat']))
            gap=next(g for g in a['discrepancy_tables'] if g['probe_id']==selected['probe_id'])
            md('Shown point: **default, L='+str(selected['length'])+', repeat '+str(selected['repeat'])+'**. Full repeat-level tables are downloadable below.')
            html(['Component','Signed bytes','Signed MiB','Evidence / interpretation'],[[r['component'],r['signed_bytes'],round(r['signed_bytes']/1024**2,6),r['evidence']] for r in gap['rows']])
            html(['Check','Result'],[['Signed measured − predicted tensor peak, bytes',gap['signed_gap_bytes']],['Arithmetic closure error, bytes',gap['closure_error_bytes']],['Explicit unaccounted bytes',gap['unaccounted_absolute_bytes']],['Coverage criterion',gap['coverage_pass']]])
            html(['Separate planning budget','Bytes / interpretation'],[['Original tensor-peak prediction',gap['predicted_tensor_peak_bytes']],['Original fixed allowance H0',gap['budget_table']['fixed_headroom_bytes']],['Original total budget prediction',gap['budget_table']['predicted_budget_bytes']],['Caution',gap['budget_table']['interpretation']]])
            image('05_discrepancy_default.png');image('05_discrepancy_expandable.png')
            md('Lifetime/counter attribution is not a pure causal activation-versus-fragmentation decomposition. Matched zero-workspace controls support 8,519,680 bytes at L=512; other lengths use labelled fixed-size source inference. The true OOM-threshold gap remains unmeasured. StaticCache controls also include the pinned HF version’s L×L mask behavior.')
        elif section=='transfer':
            md('## Deliverable 6 — Cross-model / cross-GPU predictions and verification\nThe targets were not run: Qwen2.5-3B and Qwen2.5-7B on A100-40GB and RTX 4090-24GB. The measured model remains the required 1.5B model. Usable VRAM, activations, config caps and uncertainty are explicit assumptions.')
            md((self.results/'transfer/transfer.md').read_text())
            md('The published serving-engine comparison verifies the compatible KV-block formula, with both MB interpretations reported. It does not independently verify HF peak context limits on the unrun GPUs.')
        elif section=='conclusions':
            md('## What we can defend in a live demo\n- Unique weights and GQA-aware KV storage match the config equation.\n- Both allocators successfully processed 114,688 tokens.\n- All required short-length curves and diagnostic grids are saved.\n- Original prediction, later validation and transfer estimates remain separate.\n- CPU replay regenerates the plots, tables and report without a GPU.\n\n## Measurement limitations under the final scope\n'+ '\n'.join('- '+m for m in evidence_gaps(a))+'\n\nThe original run lost GPU access; on 20 October the user closed further probing after two independent L=512 replication checks. The two original expandable held-out repeats and final boundary attempt were deferred. The resource-bounded delivery is complete, while these original research objectives remain unmet. No OOM threshold, completed validation journal or independent human review is invented.')
            md('### Presentation and reproduction\nUse **Runtime → Run all** in the clean presentation notebook on a CPU runtime. This replays saved measurements; it does not rerun the historical GPU sweep. For actual new GPU measurements, reconcile identity/runtime/usable memory and freeze a new prospective execution package first. Drive remains disconnected.')
        return blocks

    def show(self,section):
        from IPython.display import display, Markdown, HTML, Image
        for kind,value in self.content(section):
            display({'markdown':Markdown,'html':HTML,'image':Image}[kind](value) if kind!='image' else Image(filename=value))

    def show_explorer(self,usable_gib=14.46075439453125,batch_size=1):
        from IPython.display import display,HTML,Markdown
        display(Markdown('### Live analytical VRAM / batch explorer\nChange the two inputs and rerun this cell. These are theoretical scenario predictions; no GPU measurement runs.'))
        display(HTML(table(['Scenario','Uncapped predicted tokens','Config-capped predicted tokens'],self.explore(usable_gib,batch_size))))

    def show_downloads(self):
        import base64
        import mimetypes
        from IPython.display import display,Markdown,HTML
        display(Markdown('### Separate deliverable files\nThe links embed their files for standalone HTML use. In Colab, the optional download cell below is the recommended way to retrieve a file or one complete ZIP.'))
        paths=[self.results/relative for relative in ['report/report.pdf','report/report.md','analysis/probes.csv','analysis/discrepancy.csv','analysis/hardware_brackets.csv','analysis/held_out_validation.csv','transfer/transfer.json','transfer/transfer.md']]
        paths += [self.root/relative for relative in ['docs/DERIVATION.md','prereg/formal-2026-10-18-t4/prediction.json','prereg/formal-2026-10-18-t4/prediction.md','src/kv_cache_scaling/sweep.py','src/kv_cache_scaling/journal.py','src/kv_cache_scaling/coordinator_store.py','docs/FINAL_SCOPE.md','configs/delivery.json','artifacts/experiments/2026-10-20-continuation/verification.json']]
        for name in ('01_peak_vs_length','02_cache_scaling','03_allocator_slack','04_boundary_status','05_discrepancy_default','05_discrepancy_expandable'):
            paths += [self.results/'analysis/figures'/(name+suffix) for suffix in ('.png','.pdf')]
        links=[]
        for path in paths:
            mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            payload=base64.b64encode(path.read_bytes()).decode('ascii')
            links.append('<li><a download="'+escape(path.name,quote=True)+'" href="data:'+mime+';base64,'+payload+'">'+escape(path.name)+'</a></li>')
        display(HTML('<ul>'+''.join(links)+'</ul>'))


def checkpoint_demo():
    """Real filesystem recovery in temporary CPU-only demo storage, not a probe."""
    with tempfile.TemporaryDirectory(prefix='kv-checkpoint-demo-') as directory:
        root=Path(directory);run_id=str(uuid4())
        with RunStore(root/'local',root/'mirror',run_id) as store:
            store.artifact('demo.json',b'{"label":"CPU durability demonstration only"}\n')
            append_event(store,{'event':'demo_checkpoint','note':'not a GPU observation'})
            (store.local/'demo.json').unlink()  # Simulate loss of the local copy only.
        with RunStore(root/'local',root/'mirror',run_id) as resumed:
            restored=(resumed.local/'demo.json').read_bytes()==(resumed.mirror/'demo.json').read_bytes()
            equal=(resumed.local/'events.jsonl').read_bytes()==(resumed.mirror/'events.jsonl').read_bytes()
            return {'mode':'CPU temporary-filesystem demonstration; no model or OOM ledger',
                    'restored_missing_local_artifact':restored,'mirrored_journals_match':equal,'recovered_events':len(resumed.events)}
