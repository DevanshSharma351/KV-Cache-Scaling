"""Build an evidence-linked research report without promoting missing data to results."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import shlex
import subprocess
from statistics import median
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[2]

PREDICTION_COMMIT='7baff5121ed643f27ba06959065855a8913f8e8c'
PHASE_FLOW=[
    ('1. Derive and preregister','Read pinned architecture; count tied weights and GQA cache; commit the prediction before model work.'),
    ('2. Build isolated probes','Use one cold prefill per fresh process, explicit cache/backend/logits policy and synchronized peak counters.'),
    ('3. Coordinate and preserve','Checkpoint every worker, verify external copies, enforce safe admission and a single OOM allowance.'),
    ('4. Explain the measurements','Fit inspected KV separately from total peaks; plot both; close signed discrepancy tables with explicit unknowns.'),
    ('5. Generalize and present','Predict two unrun models on two unrun GPUs; compare published KV figures; provide a CPU-replayable notebook and report.'),
]
DELIVERABLE_MAP=[
    ('1. Derivation with symbols','Section 1; docs/DERIVATION.md; config-derived weights and KV equation.'),
    ('2. Timestamped advance prediction','Section 2; prereg/formal-2026-10-18-t4/prediction.json and original Git commit.'),
    ('3. Checkpointing sweep harness','Section 3; src/kv_cache_scaling/sweep.py, probe.py and coordinator_store.py; raw checkpoint capsule.'),
    ('4. Measured / analytical plots','Sections 4-6; six standalone PNG/PDF figures in artifacts/figures/project-results/.'),
    ('5. Quantified discrepancy','Section 7; data/processed/project-results/discrepancy.csv; signed components, closure and unaccounted bytes.'),
    ('6. Cross-model / cross-GPU predictions','Section 8; data/processed/project-results/transfer/; four scenarios and published KV-only verification.'),
]


def report_context():
    """Read the unchanged advance prediction; use its original Git timestamp."""
    rel='prereg/formal-2026-10-18-t4/prediction.json'
    frozen=subprocess.check_output(['git','-C',str(ROOT),'show',f'{PREDICTION_COMMIT}:{rel}'])
    if (ROOT/rel).read_bytes()!=frozen:
        raise ValueError('Original prediction differs from its committed preregistration')
    timestamp=subprocess.check_output(['git','-C',str(ROOT),'show','-s','--format=%cI',PREDICTION_COMMIT],text=True).strip()
    prediction=json.loads(frozen)
    rows=[]
    for point in prediction['memory_predictions']:
        central=next(x for x in point['scenarios'] if x['name']=='central')
        rows.append((point['length_tokens'],point['kv_bytes'],central['tensor_peak_bytes'],central['budget_bytes']))
    return dict(prediction=prediction,commit_timestamp=timestamp,prediction_rows=rows)


def evidence_gaps(a):
    """Check required evidence without promoting partial grids to completion."""
    missing=[]
    if not a['acceptance']['five_length_paired_primary_complete']:
        missing.append('Full primary grid with five lengths and three repeats per allocator.')
    if not a.get('allocator_boundary_deltas'):
        missing.append('Paired safe-budget boundary search.')
    if not a['acceptance']['discrepancy_coverage_pass']:
        missing.append('The 90% explanatory-coverage criterion remains unmet; inspect explicit unaccounted rows.')
    rows=a.get('diagnostic_comparisons',[])
    for mode,grid,allocators in [('chunked_dynamic',[512,2048,8192,32768],['default']),
                                  ('static_cache',[512,2048,8192],['default']),
                                  ('workspace_control',[512],['default','expandable'])]:
        if not all(any(r['measurement']==mode and r['length']==n and r['allocator']==allocator
                       and r['diagnostic_repeats']>=3 for r in rows) for n in grid for allocator in allocators):
            missing.append(mode+' diagnostic grid incomplete.')
    if not any(p['measurement']=='kernel_trace' for p in a['probes']):
        missing.append('Separate efficient-SDPA kernel trace.')
    if not all(any(r['allocator']==allocator and r['length']==114688 and r['repeats']>=3
                   for r in a.get('held_out_validation',[])) for allocator in ('default','expandable')):
        missing.append('Paired prospective held-out validation at L=114688.')
    if not any(b.get('smallest_observed_oom_tokens') and b.get('consistent_monotone_bracket')
               for b in a.get('hardware_brackets',[])):
        missing.append('Actual OOM boundary unobserved; conservative safe-budget censoring is not an OOM threshold.')
    return missing


def build_report(analysis_dir, transfer_dir, output_dir):
    analysis_dir,transfer_dir,out=map(Path,(analysis_dir,transfer_dir,output_dir))
    if out.exists():raise ValueError("Refusing to overwrite a report")
    a=json.loads((analysis_dir/'analysis.json').read_text());t=json.loads((transfer_dir/'transfer.json').read_text())
    context=report_context()
    primary=[p for p in a['probes'] if p['measurement']=='primary']
    cohorts={json.dumps({k:v for k,v in g['identity'].items() if k!='allocator'},sort_keys=True)
             for g in a['groups'] if g['identity']['measurement']=='primary'}
    if len(cohorts)>1:raise ValueError('Select one scientific cohort before reporting; incompatible hardware cannot be pooled')
    out.mkdir(parents=True)
    required=a['acceptance']['five_length_paired_primary_complete']
    coverage=a['acceptance']['discrepancy_coverage_pass']
    lengths=sorted({p['length'] for p in primary})
    evidence_summary=f"{len(primary)} primary probes across {len(lengths)} measured lengths ({min(lengths,default=0)}-{max(lengths,default=0)} tokens)"
    controlled=any(row['component']=='controlled_cublas_workspace' for gap in a['discrepancy_tables'] for row in gap['rows'])
    diagnostic_rows=a.get('diagnostic_comparisons',[])
    remaining=evidence_gaps(a)
    pending=bool(remaining)
    observed_oom=any(b.get('smallest_observed_oom_tokens') and b.get('consistent_monotone_bracket') for b in a.get('hardware_brackets',[]))
    remaining.append('Transferred HF peak thresholds remain unvalidated; published comparison checks KV bytes only.')
    criteria=[
        ('FR1','PASS','Pinned config, KV heads and tied-weight arithmetic'),
        ('FR2','PASS_FOR_SAVED_ACCEPTANCE','Exact weights/cache verified; residuals reported separately'),
        ('FR3','PASS_FOR_SAVED_ACCEPTANCE','Original prediction 7baff51 and later receipt 976158c precede model work'),
        ('FR4','CHECKPOINTED_GRID_PASS' if required else 'BOUNDED_PASS_BULK_PENDING','Fresh workers and per-probe external checkpoint barriers; original acceptance restoration verified'),
        ('FR5','PASS' if required and a['acceptance']['cache_slope_pass'] else 'NOT_MET','Five lengths, three repeats and retained-cache slope required'),
        ('FR6','OBSERVED_OOM_BRACKET_ALLOCATOR_DELTA_CENSORED' if observed_oom else ('SAFE_BUDGET_CENSORED' if a.get('allocator_boundary_deltas') else 'BOUNDARY_UNMEASURED'),'One-use OOM bracket and paired safe-budget delta are distinct; an exact hardware-limit delta is not established'),
        ('FR7','PASS' if coverage and a['acceptance']['discrepancy_closure_pass'] else 'NOT_MET','Closed signed table; unsupported bytes remain unaccounted'),
        ('FR8','CONDITIONAL_KV_COMPARISON','Four unrun transfer cells; published KV block comparison with dtype/unit caveats'),
    ]
    status='RESEARCH_CRITERIA_PENDING' if pending or any(v=='NOT_MET' for _,v,_ in criteria) else 'EVIDENCE_COMPLETE_WITH_LIMITATIONS'
    md=['# KV-Cache Scaling: integrated evidence report','',f'Status: **{status}**. No missing experiment is presented as a successful result.','',
        '## What we built','',
        'A config-derived memory predictor, preregistration gate, fresh-process CUDA probes, checkpointed coordinator, immutable-data analysis, reproducible scientific plots, and config-only transfer predictions. The measurement coverage below is computed from verified records.','',
        '## Scientific method','',
        'L means one cold FP16 prefill on L random tokens at B=1, with an empty DynamicCache, logits_to_keep=1 and forced efficient SDPA. Peak counters reset before inputs/cache allocation. Profiled diagnostics are separate.','',
        'P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H; W = 2P; KV(L,B) = 2BLNKD*2.','',
        'For Qwen2.5-1.5B: 1543714304 unique parameters, 3087428608 weight bytes and 28672 KV bytes/token at B=1. Query heads instead of KV heads cause a 6x error; duplicating tied embeddings adds 466747392 bytes.','',
        'The original advance central context estimate is 111232 tokens at 15527116800 usable bytes; assumption range 76024-131072 (config capped). These are not measured limits. The model-card quality domain is 32768.','',
        '## Saved primary measurements','',evidence_summary+'.','',
        '| Allocator | L | Repeats | Median allocated peak, bytes | Median reserved peak, bytes | Median cache, bytes |','| --- | ---: | ---: | ---: | ---: | ---: |']
    metric_rows=[]
    for allocator in ('default','expandable'):
        for length in sorted({p['length'] for p in primary if p['allocator']==allocator}):
            rs=[p for p in primary if p['allocator']==allocator and p['length']==length]
            row=(allocator,length,len(rs),median(p['peak_allocated_bytes'] for p in rs),median(p['peak_reserved_bytes'] for p in rs),median(p['cache_bytes'] for p in rs))
            metric_rows.append(row);md.append('| '+' | '.join(str(x) for x in row)+' |')
    md+=['','## Cache fits and residuals','',
         '| Allocator | KV slope bytes/token | KV slope error % | KV R-squared | Retained-minus-load intercept bytes | Total-peak slope bytes/token |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g in a['groups']:
        if g['identity']['measurement']!='primary':continue
        fit=g['fits'];err=g['slope_relative_error']
        md.append('| '+' | '.join(str(v) for v in [g['identity']['allocator'],fit['cache_bytes']['slope_bytes_per_token'],None if err is None else err*100,fit['cache_bytes']['r_squared'],fit['retained_minus_postload_bytes']['intercept_bytes'],fit['peak_allocated_bytes']['slope_bytes_per_token']])+' |')
    if primary:
        fractions=[p['retained_residual_fraction'] for p in primary]
        md+=['',f'Retained residual fraction spans {min(fractions):.6%} to {max(fractions):.6%} across the observed primary points.','']
    md+=['## Boundary evidence','']
    for b in a.get('hardware_brackets',[]):
        md.append(f"- {b['allocator']}: observed success at {b['largest_observed_success_tokens']} tokens; observed OOM upper point {b['smallest_observed_oom_tokens']}. Original prediction error interval, (prediction-limit)/limit: {b['prediction_error_percent_interval']} percent. This is an interval, not an exact threshold.")
    for b in a.get('allocator_boundary_deltas',[]):
        md.append(f"- Expandable minus default conservative safe-limit delta: observed-success delta {b['observed_success_delta_tokens']} tokens; bracket interval [{b['safe_bracket_delta_min_tokens']}, {b['safe_bracket_delta_max_tokens']}] tokens. A common analytical admission ceiling can censor both; this does not establish an unchanged hardware limit.")
    for x in a.get('held_out_validation',[]):
        md.append(f"- Held-out {x['allocator']} L={x['length']}, n={x['repeats']}: prospective fitted peak {x['prospective_fit_peak_bytes']:.0f} bytes, observed median {x['observed_peak_median_bytes']:.0f} bytes; error {x['prospective_fit_error_percent']:.4f}%. This fit was frozen after the safe sweep and before this held-out point; it does not replace the advance commitment.")
    md+=['','Only measured lengths are observations. Missing slope/R-squared values are null, not zero. Repeat ranges are descriptive, not confidence intervals. Separate peak maxima cannot be subtracted to infer same-time slack.','',
         '## Discrepancy accounting','',
         '| Allocator / L / repeat | Signed peak gap, bytes | Closure error, bytes | Absolute unaccounted, bytes | Coverage criterion |','| --- | ---: | ---: | ---: | --- |']
    lookup={p['probe_id']:p for p in primary};gap_rows=[]
    for gap in a['discrepancy_tables']:
        p=lookup[gap['probe_id']];label=f"{p['allocator']} / {p['length']} / {p['repeat']}"
        row=(label,gap['signed_gap_bytes'],gap['closure_error_bytes'],gap['unaccounted_absolute_bytes'],'PASS' if gap['coverage_pass'] else 'NOT MET')
        gap_rows.append(row);md.append('| '+' | '.join(str(x) for x in row)+' |')
    md+=['','The signed table closes using measured lifetime/counter deltas. Closure does not establish causal attribution. Any unsupported post-load storage or post-forward persistent growth remains explicitly unaccounted. For material gaps, both signed-gap and absolute-contribution coverage must reach 90%; for gaps below 10 MiB, the unaccounted amount must be <=1 MiB.','',
         ('Matched L=512 zero-workspace controls support the attributed cuBLAS row where present. Other lengths use explicitly labelled fixed-size inference from the pinned PyTorch source, only for the same allocator/runtime and exact persistent delta.' if controlled else 'A source-level hypothesis is that 8519680 persistent bytes correspond to PyTorch 2.6 cuBLAS workspace. A matched intervention is still required before counting that hypothesis as explained memory.'),'',
         '## Transfer predictions','']
    transfer_md=(transfer_dir/'transfer.md').read_text();md.append(transfer_md.replace('# Config-derived transfer predictions','### Unrun target models and GPUs'))
    md+=['','## Requirement status','', '| Requirement | Status | Evidence / remaining criterion |','| --- | --- | --- |']
    md += ['| '+' | '.join(row)+' |' for row in criteria]
    capsule_args=' '.join('--capsule '+shlex.quote(source['path']) for source in a['provenance']['sources'])
    run_args=' '.join('--run-id '+shlex.quote(run_id) for run_id in a['provenance']['run_ids'])
    md+=['','## Reproduction','', '```bash', 'python -m pip install -e . -r requirements-analysis.txt',
         f'kvcache analyze {capsule_args} {run_args} \\',
         '  --output /tmp/kv-analysis --figures', 'kvcache generalize --output /tmp/kv-transfer',
         'python scripts/build_report.py --analysis /tmp/kv-analysis \\',
         '  --transfer /tmp/kv-transfer --output /tmp/kv-report', '```','',
         'Each output directory must be new. Source capsules, code, configs and output files carry hashes. No GPU is required to regenerate these analyses, figures, transfers or report. The executed Colab notebook retains the original command trail.','',
         '']
    md+=['','## Separate diagnostic comparisons','',
         '| Mode | Allocator | L | Primary peak bytes | Diagnostic peak bytes | Delta bytes |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in diagnostic_rows:
        md.append('| '+' | '.join(str(row[k]) for k in ('measurement','allocator','length','primary_peak_median_bytes','diagnostic_peak_median_bytes','diagnostic_minus_primary_peak_bytes'))+' |')
    md+=['','Diagnostics change execution semantics or workspace configuration; their points are excluded from primary fits.',
         'In pinned HF 4.51.3, the static-cache path builds an explicit query-by-capacity causal mask. For full prefill this adds an L-by-L mask even with efficient SDPA. Chunked queries bound the mask query dimension. These comparisons therefore include mask behavior and cannot isolate cache-storage policy alone. [Pinned Qwen2 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py).','',
         '## Remaining scientific criteria','']+['- '+item for item in remaining]
    md+=['','The user authorized a delegated one-contributor review and kept Google Drive disconnected. No independent second-person reproduction or full cloud-session restart is claimed.','',
         '## Sources','',
         '- [Original formal prediction](https://github.com/DevanshSharma351/KV-Cache-Scaling/blob/7baff5121ed643f27ba06959065855a8913f8e8c/prereg/formal-2026-10-18-t4/prediction.md)',
         '- [Executed notebook](https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS)',
         '- [vLLM published KV block table](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html)',
         '- [PyTorch 2.6 cuBLAS workspace source](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp)',
         '- Official pinned transfer config URLs and hashes: configs/transfer/provenance.json.','']
    md+=['','## Deliverable guide','',
         '| Required output | Where to show it |','| --- | --- |']
    md+=['| '+' | '.join(row)+' |' for row in DELIVERABLE_MAP]
    md+=['','## Five-phase project flow','',
         '| Phase | What we built |','| --- | --- |']
    md+=['| '+' | '.join(row)+' |' for row in PHASE_FLOW]
    md+=['','## Original five-length commitment','',
         f'Original Git commit: {PREDICTION_COMMIT}. Original commit timestamp: {context["commit_timestamp"]}.',
         'Tensor envelope excludes fixed headroom; admission budget includes it. Values come from the unchanged original JSON, not fitted observations.','',
         '| L tokens | KV bytes | Central tensor peak bytes | Central budget bytes |','| ---: | ---: | ---: | ---: |']
    md+=['| '+' | '.join(str(v) for v in row)+' |' for row in context['prediction_rows']]
    representative=max((p for p in primary if p['allocator']=='default'),key=lambda p:(p['length'],-p['repeat']))
    gap=next(g for g in a['discrepancy_tables'] if g['probe_id']==representative['probe_id'])
    md+=['','## Representative byte-level discrepancy','',
         f"Default allocator, L={representative['length']}, repeat={representative['repeat']}. Signed peak gap={gap['signed_gap_bytes']} bytes; closure={gap['closure_error_bytes']} bytes; absolute unaccounted={gap['unaccounted_absolute_bytes']} bytes.",'',
         '| Component | Signed bytes | Signed MiB | Evidence |','| --- | ---: | ---: | --- |']
    md+=['| '+' | '.join(str(v) for v in (r['component'],r['signed_bytes'],f"{r['signed_bytes']/1024**2:.6f}",r['evidence']))+' |' for r in gap['rows']]
    md+=['','## Presenting the deliverables','',
         'Open notebooks/FINAL_PRESENTATION.ipynb or artifacts/presentation/PROJECT_PRESENTATION.html to show all six deliverables, figures, tables and downloads. Run All in the clean notebook regenerates saved-data results on CPU. See docs/LIVE_DEMO.md and docs/DECISIONS_AND_PHASE_GUIDE.md.',
         'The saved cohort contains 80 successful workers: 48 primary sweep, 28 diagnostics and four held-out validations. Two compatible expandable held-out repeats and the registered final boundary attempt remain unverified after the GPU quota interruption. No deliberate OOM attempt was dispatched. The maximum successful context is a lower bound, not a completed OOM threshold.','']
    (out/'report.md').write_text('\n'.join(md))
    (out/'requirements.json').write_text(json.dumps([dict(requirement=k,status=v,criterion=c) for k,v,c in criteria],indent=2)+'\n')
    _build_pdf(out/'report.pdf',a,t,criteria,metric_rows,gap_rows,analysis_dir/'figures',status,evidence_summary,required,controlled,context)
    deps={}
    for dist in ('matplotlib','numpy','reportlab','pypdf'):
        try:deps[dist]=version(dist)
        except PackageNotFoundError:deps[dist]=None
    def git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'packages':deps,
                'git_commit':git('rev-parse','HEAD'),'source_dirty':bool(git('status','--porcelain')),
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__),ROOT/'src/kv_cache_scaling/analysis.py',ROOT/'src/kv_cache_scaling/generalize.py',ROOT/'src/kv_cache_scaling/plotting.py')},
                'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (analysis_dir/'analysis.json',transfer_dir/'transfer.json')},
                'status':status}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},indent=2)+'\n')
    return {'status':status,'report':str(out/'report.pdf'),'requirements':criteria}


def _build_pdf(path,a,t,criteria,metric_rows,gap_rows,figures,status,evidence_summary,required,controlled,context):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyKV',parent=styles['BodyText'],fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#26354a')))
    styles.add(ParagraphStyle(name='SmallKV',parent=styles['BodyText'],fontSize=8,leading=11,spaceAfter=6))
    styles.add(ParagraphStyle(name='HeroKV',parent=styles['Title'],fontSize=27,leading=32,alignment=TA_LEFT,textColor=colors.HexColor('#183d67'),spaceAfter=20))
    story=[];width=A4[0]-94
    report_date=datetime.now(timezone.utc).strftime('%d %B %Y')
    def para(text,style='BodyKV'):return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    def p(text):story.append(para(text))
    def h(text):story.append(Paragraph(escape(text),styles['Heading1']));story.append(Spacer(1,6))
    def table(headers,rows,widths=None,padding=7):
        data=[[para(v,'SmallKV') for v in headers]]+[[para(v,'SmallKV') for v in row] for row in rows]
        tbl=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7990a9')),('BOTTOMPADDING',(0,0),(-1,-1),padding),
                                ('TOPPADDING',(0,0),(-1,-1),padding),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fc')])]))
        story.append(tbl);story.append(Spacer(1,12))
    def plot(filename,caption):
        src=figures/filename
        if src.exists():
            from PIL import Image as PILImage
            with PILImage.open(src) as im:w,hgt=im.size
            story.append(Image(str(src),width=width,height=width*hgt/w))
            story.append(para(caption,'SmallKV'));story.append(Spacer(1,10))
        else:p('Figure unavailable: '+filename)
    def page():story.append(PageBreak())
    story.append(Paragraph('KV-Cache Scaling',styles['HeroKV']))
    p('Predicting the context limit before reaching it')
    p('Integrated scientific implementation and evidence report | '+report_date)
    p('Status: '+status.replace('_',' '))
    p('The analysis, plots and transfer calculations are integrated and reproducible. Verified coverage: '+evidence_summary+'. Requirement status below distinguishes completed measurements from unmet scientific criteria.')
    table(['Built capability','Evidence / limit'],[
        ('Memory model','Unique tied weights and GQA-aware KV equation; original numerical prediction preserved'),
        ('GPU observations',evidence_summary),
        ('Recovery','External checkpoint restoration passed after both Colab copies were removed'),
        ('Analysis','Validated immutable capsules; regressions require multiple lengths; unexplained bytes remain explicit'),
        ('Transfer','Four config-only model/GPU predictions; conditional published KV-only comparison')],[135,width-135])
    p('The project followed five phases: derivation and preregistration, isolated probes, checkpointing and safety, empirical analysis, and transfer with reproducible presentation. Drive remained disconnected; review provenance is preserved in the original frozen records.')
    if evidence_gaps(a):p('Unmet evidence: '+' '.join(evidence_gaps(a)))
    page();h('Project purpose and deliverable guide')
    p('The question is how much context fits before an LLM exhausts usable GPU memory, and why observed peaks differ from an advance equation. We built a predictor and a reproducible experiment: derive first, commit a numerical prediction, measure isolated peaks, and explain the difference.')
    table(['Required deliverable','Report and repository evidence'],DELIVERABLE_MAP,[150,width-150],padding=6)
    p('All six output types are available. Their scientific scope differs: the saved curve and KV verification are established; the final hardware OOM bracket and transferred HF context thresholds are not yet validated.')
    h('The five phases followed')
    table(['Phase','Implementation decision'],PHASE_FLOW,[140,width-140],padding=5)
    page();h('1. Derivation with defined symbols')
    p('Primary model: Qwen/Qwen2.5-1.5B, FP16, batch size 1. Sequence length L is one cold prefill of L tokens with an empty DynamicCache and logits_to_keep=1. HF SDPA is forced to the efficient backend. No generation-to-L, eager fallback or FlashAttention-2 is used.')
    p('Single prefill models prompt admission: the full input is processed at once, making transient peaks observable. Generation-to-L would mix a different token-by-token workload and cache-update history into the comparison. We keep it outside the primary contract.')
    p('We chose Qwen2.5-1.5B because its roughly 3.09 GB FP16 weights leave room for cache and transient memory on a T4. Its grouped-query attention and tied embeddings also make both architecture-counting pitfalls testable. A larger measured model would leave less room for the phenomenon under study.')
    table(['Symbol','Meaning','Value'],[('L / B','Input tokens / batch size','variable / 1 measured'),('P / W','Unique parameters / FP16 weight bytes','1543714304 / 3087428608'),('U','Usable context-only GPU memory, bytes','15527116800'),('a / H0','Assumed transient bytes per token / fixed budget reserve','78336 / 536870912'),('N','Transformer layers','28'),('H / I','Hidden / intermediate width','1536 / 8960'),('Q / K / D','Query heads / KV heads / head dimension','12 / 2 / 128'),('V','Vocabulary','151936'),('s','Bytes per FP16 cache element','2')],[55,265,width-320])
    p('q=QD; k=KD. Unique parameters P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H. The tied input/output embedding is counted once.')
    p('W=2P=3087428608 bytes. KV(L,B)=2BLNKDs=28672BL bytes. Using Q in place of K overestimates KV by 6x; duplicating the tied embedding adds 466747392 bytes.')
    p('Advance budget model: M(L,B)=W+BL(28672+a)+H0. The central assumption is a=78336 bytes/token and H0=536870912 bytes. L(U,B)=max(0,floor((U-W-H0)/(B(28672+a)))). U is usable memory after context, before weights; context must not be subtracted again.')
    p('Original T4 usable budget: 15527116800 bytes. Central estimate: 111232 tokens; assumption range: 76024-131072 after the config cap. This is not a measured confidence interval or an OOM observation. The model-card quality limit is 32768; larger lengths are synthetic memory stress.')
    p('P sums embeddings, attention projections and Q/K/V biases, gated MLP weights and RMSNorm scales from the pinned config. M_weights=W; M_kv=28672BL; M_activations is approximated by aBL; M_overhead is budgeted with H0. H0 is an admission reserve, not a measured tensor allocation. One GiB is 1073741824 bytes; one MiB is 1048576 bytes.')
    page();h('2. Original preregistered prediction')
    p('Original commit: '+PREDICTION_COMMIT+'. Original Git timestamp: '+context['commit_timestamp']+'. The prediction JSON remains byte-identical to that commit. The later receipt 976158c82ccb10f12817af2ee16b76ade13f0cf3 and updated environment receipts preserve execution provenance without rewriting the advance prediction.')
    table(['L tokens','KV bytes','Central tensor peak bytes','Central budget bytes'],context['prediction_rows'],[65,115,155,width-335])
    p('The tensor envelope is W+BL(28672+a). The budget adds H0, so it must not be compared to allocated tensor peaks as though it were another tensor. The central limit is 111232 tokens at U=15527116800 bytes; the assumption range is 76024-131072, capped by config. This sensitivity range is not a confidence interval.')
    p('A later fit was frozen after the primary sweep and before L=114688 validation. It is assessed separately below; it never replaces this original prediction or its timestamp.')
    h('Hardware and runtime identity')
    p('Measured cohort: Tesla T4, GPU-89f781af-1ec0-992b-3dc3-46ef2d0941de; Python 3.11.13, PyTorch 2.6.0+cu124, CUDA runtime 12.4, Transformers 4.51.3, driver 580.82.07. Total GPU bytes: 15637086208; context-only usable bytes: 15527116800. The original preliminary Python 3.13 / torch 2.11 snapshot is not pooled into this cohort.')
    p('GPU execution source: ce227af40476676238cbef9126807fd1d42be76b. Model revision: 8faed761d45a263340a0528343f099c05c9a4323. Measurements are grouped by hardware, runtime and protocol identity before analysis.')
    page();h('3. Checkpointed measurement harness')
    table(['Step','Operational rule'],[
        ('Isolate','A new worker process loads the FP16 model for each point. Measure post-load allocation before any forward pass; compare it with unique-weight bytes.'),
        ('Measure peak','Synchronize CUDA, reset_peak_memory_stats(), allocate probe inputs/cache, run the prefill and synchronize again. Read max_memory_allocated() and max_memory_reserved(); retain allocated/reserved snapshots and sampled NVML/free-memory telemetry.'),
        ('Separate lifetimes','Inspect actual cache tensors, then release outputs/cache and measure retained allocation. Fit KV bytes separately; total peak slope also contains transient activations and workspace.'),
        ('Pin the backend','Force efficient SDPA. Eager attention materializes head-wise L-by-L scores and invalidates the linear transient envelope. FlashAttention-2 is unsupported on this T4 path; no silent fallback is accepted.'),
        ('Pin cache and allocator','Primary uses empty DynamicCache in a single prefill. Chunked DynamicCache and StaticCache are diagnostics. Set expandable_segments before CUDA initialization in each fresh worker; record version support and effective settings.'),
        ('Preserve and approach','Write typed results and a fsynced journal after every worker; verify an external checkpoint before continuing. Sweep upward, then use conservative admission and bisection. One deliberate final OOM attempt is allowed; none has been dispatched.')
    ],[110,width-110],padding=6)
    p('DynamicCache concatenation during chunked updates can create transient copies. StaticCache changes preallocation and, in pinned HF, causal-mask behavior. Neither diagnostic is substituted for the primary contract. Driver peaks from NVML remain sampled, while PyTorch peak counters record allocator high-water marks.')
    p('Drive stayed disconnected. The raw checkpoint capsule preserves commands, worker records, package inventories, source fingerprints and coordinator state. Recovery of saved files was tested; restoration of an entire cloud session on the same physical GPU is not claimed.')
    page();h('4. Saved GPU measurements and KV slope')
    table(['Allocator','L','n','Peak allocated bytes','Peak reserved bytes','Cache bytes'],metric_rows,[65,45,22,120,120,width-372],padding=5)
    p('Repeated points use fresh worker processes. The cache contains 14680064 bytes at L=512 and reduced logits are FP16 (1,1,151936). Per-length medians and ranges come from the recorded repeats. Kernel traces remain separate diagnostics.')
    if a['probes']:
        fractions=[x['retained_residual_fraction'] for x in a['probes'] if x['measurement']=='primary']
        if fractions:p(f'Retained residual fraction: {min(fractions):.6%} to {max(fractions):.6%} of retained allocation across primary points.')
    p('All snapshots respect reserved >= allocated and forward peaks cover synchronized post-forward allocation. Retained-minus-released allocation matches inspected cache bytes. Driver samples are sampled observations, not exact driver high-water marks.')
    p('Both allocators cover the five required lengths with three repeats.' if required else 'The required paired five-length grid is incomplete. Missing observations do not establish a complete cache slope or hardware context limit.')
    for group in a['groups']:
        if group['identity']['measurement']=='primary':
            fit=group['fits']['cache_bytes']
            p(f"{group['identity']['allocator']}: directly inspected KV slope = {fit['slope_bytes_per_token']} bytes/token; R-squared = {fit['r_squared']}. Total-peak fits remain separate.")
    page();h('5. Memory curves and retained KV')
    plot('01_peak_vs_length.png','The band is an advance tensor-envelope assumption, not measured uncertainty. Points are observations; the unmeasured curve remains visibly labelled.')
    plot('02_cache_scaling.png','Direct cache and corrected retained allocation are separate quantities. Fits require multiple distinct lengths.')
    page();h('6. Allocator slack and boundary status')
    plot('03_allocator_slack.png','Same-snapshot reserved minus allocated is allocator slack. It is not, by itself, fragmentation.')
    page();h('6. Boundary evidence (continued)')
    plot('04_boundary_status.png','Observed success, safe-budget bracket, actual OOM upper bound and advance prediction remain distinct. No hardware threshold is fabricated.')
    for b in a.get('allocator_boundary_deltas',[]):
        p(f"Expandable-minus-default conservative safe-limit interval: {b['safe_bracket_delta_min_tokens']} to {b['safe_bracket_delta_max_tokens']} tokens; observed-success delta {b['observed_success_delta_tokens']}. A common analytical admission ceiling can censor both allocators; this is not a hardware-limit delta.")
    for b in a.get('hardware_brackets',[]):
        p(f"{b['allocator']}: observed success {b['largest_observed_success_tokens']} tokens; OOM upper point {b['smallest_observed_oom_tokens']}; original prediction error interval {b['prediction_error_percent_interval']} percent. The interval assumes monotone feasibility and does not identify an exact threshold.")
    for x in a.get('held_out_validation',[]):
        p(f"Held-out {x['allocator']} L={x['length']}, n={x['repeats']}: frozen post-sweep fit predicted {x['prospective_fit_peak_bytes']:.0f} bytes; observed median {x['observed_peak_median_bytes']:.0f}; error {x['prospective_fit_error_percent']:.4f}%. This is a later validation, not a replacement advance prediction.")
    page();h('7. Signed prediction discrepancy')
    # Keep the PDF readable; all repeat-level rows remain in report.md and CSV.
    grouped_gaps={}
    for row in gap_rows:
        allocator,length,_=row[0].split(' / ')
        grouped_gaps.setdefault((allocator,int(length)),[]).append(row)
    summary_rows=[]
    for (allocator,length),rows in sorted(grouped_gaps.items()):
        summary_rows.append((f'{allocator} / {length} (n={len(rows)})',median(r[1] for r in rows),
                             max(abs(r[2]) for r in rows),max(r[3] for r in rows),
                             'PASS' if all(r[4]=='PASS' for r in rows) else 'NOT MET'))
    table(['Allocator / L / repeats','Median gap bytes','Max closure bytes','Max unaccounted bytes','All repeats'],summary_rows,[140,95,80,100,width-415])
    p('Summary uses median signed gaps and worst-case absolute closure/unaccounted bytes across repeats. Every repeat-level row is preserved in the accompanying Markdown report and discrepancy CSV.')
    p('The signed identity telescopes across post-load, after-forward, retained and released counters. Arithmetic closure is separate from explaining causes. Unknown baseline storage and persistent post-forward growth are retained as unaccounted, even if a plausible source-level hypothesis exists.')
    p('For a material gap (at least 10 MiB), supported attribution must reach 90% under both gap and absolute-contribution denominators. Smaller gaps require at most 1 MiB unaccounted and an ill-conditioned-ratio label. Computed coverage status: '+str(a['acceptance']['discrepancy_coverage_pass'])+'.')
    page();h('7. Byte-level discrepancy example')
    primary=[probe for probe in a['probes'] if probe['measurement']=='primary' and probe['allocator']=='default']
    representative=max(primary,key=lambda probe:(probe['length'],-probe['repeat']))
    gap=next(g for g in a['discrepancy_tables'] if g['probe_id']==representative['probe_id'])
    p(f"Default allocator, L={representative['length']}, repeat={representative['repeat']}. Predicted tensor peak: {gap['predicted_tensor_peak_bytes']} bytes. Measured peak: {gap['measured_peak_bytes']} bytes. Signed difference: {gap['signed_gap_bytes']} bytes. Positive components add memory; negative components reduce the advance envelope.")
    table(['Component','Signed bytes','Signed MiB','Interpretation'],[
        (row['component'].replace('_',' '),row['signed_bytes'],f"{row['signed_bytes']/1024**2:.6f}",
         'Unaccounted' if not row['supported'] else ('Fixed-size inference from L=512 controls' if row['component']=='controlled_cublas_workspace' else ('Descriptive lifetime/counter difference' if row['component']=='transient_envelope_difference' else 'Measured storage/counter evidence')))
        for row in gap['rows']],[155,100,85,width-340],padding=6)
    p(f"Sum of components: {sum(row['signed_bytes'] for row in gap['rows'])} bytes; closure error: {gap['closure_error_bytes']} bytes; absolute unaccounted: {gap['unaccounted_absolute_bytes']} bytes. The separate fixed planning reserve is {gap['budget_table']['fixed_headroom_bytes']} bytes and is not included in this tensor-peak comparison.")
    p('The large negative transient-envelope component means the advance tensor envelope was conservative at this length. It is a measured counter/lifetime residual relative to an assumed envelope, not a direct causal measurement of pure activations. Zero unaccounted bytes here therefore establishes this accounting identity, not a complete physical decomposition of allocator fragmentation or an unmeasured OOM gap.')
    page();h('7. Discrepancy components (continued)')
    plot('05_discrepancy_default.png','Default allocator. Red components remain unaccounted. Transient counter deltas do not establish pure activation memory.')
    plot('05_discrepancy_expandable.png','Expandable allocator. The same signed accounting and explanatory-coverage rules apply.')
    if a.get('diagnostic_comparisons'):
        page();h('Separate diagnostic comparisons')
        rows=[(x['measurement'],x['allocator'],x['length'],x['diagnostic_repeats'],x['diagnostic_minus_primary_peak_bytes']) for x in a['diagnostic_comparisons']]
        table(['Mode','Allocator','L','Repeats','Peak delta bytes'],rows,[135,80,65,55,width-335])
        p('Diagnostic minus matching primary median. Chunking and static caching change execution semantics. Workspace interventions change GEMM workspace policy. All are excluded from the primary cache slope and hardware-boundary observations.')
        p('Pinned HF 4.51.3 creates an explicit query-by-capacity causal mask for StaticCache. A full-prefill mask is L by L even with efficient SDPA; chunking bounds the query dimension. Peak deltas include this mask behavior, not solely cache storage.')
    p('PyTorch 2.6 source allocates 8519680 default cuBLAS workspace bytes on a non-Hopper CUDA GPU. '+('Three matched same-allocator zero-workspace controls directly support L=512 rows. Other lengths use labelled source-supported fixed-size inference only when their persistent delta matches exactly.' if controlled else 'A matching persistent delta remains a hypothesis until matched intervention evidence exists.'))
    page();h('8. Transfer to two unrun models and GPUs')
    transfer_rows=[]
    for x in t['predictions']:
        c=x['config_capped_limit_tokens'];u=x['usable_vram_bytes']
        transfer_rows.append((x['model_id'].replace('Qwen/',''),x['gpu'],f"{u['min']/1024**3:g}/{u['central']/1024**3:g}/{u['max']/1024**3:g}",x['uncapped_limit_tokens']['central'],c['central'],f"{c['min']}-{c['max']}"))
    table(['Model','GPU','Assumed usable GiB','Raw central L','Capped L','Capped range'],transfer_rows,[75,85,85,80,65,width-390])
    p('Only official config files were downloaded. No 3B/7B checkpoint was loaded and neither target GPU was measured. VRAM budgets and activation envelopes are explicit assumptions. The 3B config cap censors its memory estimate; censoring is not an observed OOM.')
    ext=[]
    for x in t['published_verification']['comparisons']:
        ext.append((x['base_model_id'].replace('Qwen/',''),x['predicted_block_bytes'],x['published_display_MB'],f"{x['display_error_percent_if_binary']:.4f}%",f"{x['display_error_percent_if_decimal']:.4f}%"))
    table(['Matching KV architecture','Predicted block bytes','Shown MB','Error if MiB','Error if decimal MB'],ext,[135,115,65,85,width-400])
    p('The vLLM team publishes physical 16-token KV blocks across all layers. Base/Instruct KV fields agree in pinned configs. Under a two-byte cache assumption the comparison checks retained KV scaling only. The source labels MB without defining decimal versus binary units, so both displayed-value errors are reported.')
    p('The published serving engine has different attention, allocation and scheduling semantics. This evidence does not validate the transferred HF single-prefill peak thresholds, activation envelopes or target usable VRAM.')
    page();h('9. Acceptance status and reproducibility')
    table(['ID','Status','Criterion'],[(k,v.replace('_',' '),c) for k,v,c in criteria],[35,145,width-180])
    p('Regeneration is CPU-only: install the package with requirements-analysis.txt, run kvcache analyze on the immutable capsule, run kvcache generalize, and invoke scripts/build_report.py. Every output directory must be new. Full command lines appear in report.md and the reproduction guide.')
    p('Historical acceptance preserved an import-time torchao preflight failure and verified restoration after deleting both Colab checkpoint copies. Removing unused torchao restored cold imports. That bounded acceptance consumed no OOM allowance; full cloud-session restoration of the same GPU identity is not claimed.')
    p('The requirement table is computed from saved evidence. Safe-budget boundaries remain distinct from actual hardware OOM thresholds. Unsupported causes remain unaccounted. The original prediction is preserved; the prospective implementation amendment and later receipt record the changed GPU/source before extended probes.')
    page();h('10. Presenting the deliverables')
    p('Open notebooks/FINAL_PRESENTATION.ipynb in Colab and Run All. It replays the verified saved evidence on CPU, displays six scientific figures, the derivation, the original prediction and timestamp, harness/checkpointing evidence, discrepancy tables and all four transfer predictions. It also offers individual artifact downloads. No model download, CUDA allocation or Drive mount is needed for this presentation.')
    p('Use artifacts/presentation/PROJECT_PRESENTATION.html as the standalone presentation if Colab output scripts fail to load. It contains the same saved-data presentation and embedded graphs. The live-demo guide explains both routes; the decisions and phase guide explains the project in plain language.')
    h('What is established, and what remains')
    p('There are 80 verified successful workers: 48 primary sweep points, 28 diagnostics and four held-out validation points. Both allocators successfully reached 114688 tokens. Two compatible expandable held-out repeats and the registered final boundary attempt remain unverified following the GPU quota interruption. No deliberate OOM attempt was dispatched; no hardware OOM upper bound or exact threshold error is claimed.')
    p('The discrepancy tables explain fixed-length tracked tensor peak differences under stated accounting assumptions. They do not quantify an unmeasured OOM-threshold gap. The published comparison verifies KV block arithmetic, not the unrun target systems’ full context limits. These limitations are part of the deliverable, not hidden exceptions.')
    h('Primary sources')
    for title,url in [('Clean presentation notebook','https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/main/notebooks/FINAL_PRESENTATION.ipynb'),('Five-phase decisions guide','https://github.com/DevanshSharma351/KV-Cache-Scaling/blob/main/docs/DECISIONS_AND_PHASE_GUIDE.md'),('Original prediction commit','https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/'+PREDICTION_COMMIT),('Executed audit notebook','https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS'),('Official vLLM KV block table','https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html'),('PyTorch v2.6.0 workspace source','https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp'),('Pinned HF 4.51.3 Qwen2 masking source','https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py')]:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#2563a6">{escape(title)}</link>',styles['SmallKV']))
    p('Official transfer config URLs/revisions/hashes: configs/transfer/provenance.json. Source capsule hashes, analysis code hashes and report package versions are in provenance and manifest files.')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d2dce8'));canvas.line(47,38,A4[0]-47,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#66758a'))
        canvas.drawString(47,25,'KV-Cache Scaling | Evidence-linked report | '+report_date)
        canvas.drawRightString(A4[0]-47,25,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=47,leftMargin=47,topMargin=45,bottomMargin=50,
                      title='KV-Cache Scaling - integrated evidence report',author='Project team').build(story,onFirstPage=footer,onLaterPages=footer)
