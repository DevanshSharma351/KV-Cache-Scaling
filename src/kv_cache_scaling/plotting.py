"""Regenerable scientific plots. Missing data gets a visible label, never fake dots."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import textwrap

from .model_math import Architecture, peak_tensor_bytes, prefill_scenarios

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2
COLORS = {"default":"#2563a6", "expandable":"#db7b25"}


def build_figures(processed_data, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    if isinstance(processed_data, (str, Path)):
        processed_data = json.loads(Path(processed_data).read_text())
    out=Path(output_dir)
    if out.exists():raise ValueError("Refusing to overwrite figures")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "figure.dpi":130, "savefig.facecolor":"white"})
    arch=Architecture.from_config(json.loads((ROOT/"configs/models/qwen2.5-1.5b.json").read_text()))
    primary=[p for p in processed_data["probes"] if p["measurement"]=="primary"]
    identities=[g["identity"] for g in processed_data["groups"] if g["identity"]["measurement"]=="primary"]
    cohorts={json.dumps({k:v for k,v in identity.items() if k!="allocator"},sort_keys=True) for identity in identities}
    if len(cohorts)>1:
        raise ValueError("Select one scientific cohort before plotting; incompatible runs cannot be pooled")
    lengths=sorted({p["length"] for p in primary})
    x=sorted(set(lengths)|{512,1024,2048,4096,8192})
    sparse=len(lengths)<5
    figures=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","pdf"):
            path=out/(name+"."+ext);fig.savefig(path,bbox_inches="tight");figures.append(path)
        plt.close(fig)
    def by_allocator_points(metric):
        for allocator,color in COLORS.items():
            grouped=defaultdict(list)
            for p in primary:
                if p["allocator"]==allocator:grouped[p["length"]].append(p[metric]/MIB)
            ns=sorted(grouped)
            if ns:
                mid=[median(grouped[n]) for n in ns]
                yield allocator,color,ns,mid,[[mid[i]-min(grouped[n]) for i,n in enumerate(ns)],
                                          [max(grouped[n])-mid[i] for i,n in enumerate(ns)]]
    fig,ax=plt.subplots(figsize=(8,4.7))
    scenarios=prefill_scenarios(arch)
    ax.fill_between(x,[peak_tensor_bytes(arch,n,scenarios[0])/MIB for n in x],
                    [peak_tensor_bytes(arch,n,scenarios[2])/MIB for n in x],color="#c8d8e9",alpha=.6,label="Advance tensor-envelope assumptions")
    ax.plot(x,[peak_tensor_bytes(arch,n,scenarios[1])/MIB for n in x],color="#425466",ls="--",label="Advance central tensor peak")
    for allocator,color,ns,mid,err in by_allocator_points("peak_allocated_bytes"):
        ax.errorbar(ns,mid,yerr=err,fmt="o",color=color,capsize=4,label=allocator+" median / repeat range")
    ax.set(xlabel="Single-prefill length L (tokens)",ylabel="Tracked forward peak (MiB)",title="Forward peak allocation versus the advance model")
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    if sparse:ax.text(.02,.98,f"Only {len(lengths)} measured length(s); full curve pending",transform=ax.transAxes,va="top",fontsize=9)
    save(fig,"01_peak_vs_length")
    fig,ax=plt.subplots(figsize=(8,4.7));ax.plot(x,[arch.kv_bytes(n)/MIB for n in x],color="#425466",ls="--",label="Analytical retained KV")
    for allocator,color,ns,mid,err in by_allocator_points("cache_bytes"):
        ax.errorbar(ns,mid,yerr=err,fmt="o",color=color,label=allocator+" inspected cache")
    for allocator,color,ns,mid,err in by_allocator_points("retained_minus_postload_bytes"):
        ax.plot(ns,mid,"s",mfc="none",color=color,label=allocator+" retained minus post-load")
    ax.set(xlabel="L (tokens)",ylabel="Retained growth (MiB)",title="Direct KV storage and corrected retained allocation")
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    if sparse:ax.text(.02,.98,"Insufficient lengths for a validated slope",transform=ax.transAxes,va="top",fontsize=9)
    save(fig,"02_cache_scaling")
    fig,ax=plt.subplots(figsize=(8,4.7))
    for allocator,color in COLORS.items():
        for stage,style in (("after_forward","o-"),("retained","s--")):
            rows=[s for s in processed_data["snapshots"] if s["measurement"]=="primary" and s["allocator"]==allocator and s["stage"]==stage]
            ns=sorted({s["length"] for s in rows})
            vals=[median(s["allocator_slack_bytes"] for s in rows if s["length"]==n)/MIB for n in ns]
            if ns:ax.plot(ns,vals,style,color=color,label=allocator+" / "+stage)
    ax.set(xlabel="L (tokens)",ylabel="Same-snapshot reserved − allocated (MiB)",title="Allocator slack; not a complete fragmentation measurement")
    ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,"03_allocator_slack")
    fig,ax=plt.subplots(figsize=(8,4.7))
    boundaries=processed_data.get("boundaries",[])
    for i,allocator in enumerate(COLORS):
        rows=[b for b in boundaries if b["allocator"]==allocator]
        for b in rows:
            lo,hi=b.get("safe_budget_lower_tokens"),b.get("safe_budget_upper_tokens")
            if lo is not None and hi is not None:ax.plot([lo,hi],[i,i],lw=5,color=COLORS[allocator])
            if b["largest_observed_success_tokens"] is not None:ax.plot(b["largest_observed_success_tokens"],i,"o",color=COLORS[allocator])
            if b["observed_oom_upper_tokens"] is not None:ax.plot(b["observed_oom_upper_tokens"],i,"x",color="red",ms=9)
    ax.axvspan(76024,131072,color="#c8d8e9",alpha=.5,label="Original T4 assumption range")
    ax.axvline(111232,color="#425466",ls="--",label="Original central prediction")
    ax.set_yticks([0,1],list(COLORS));ax.set(xlabel="L (tokens)",title="Observed successes, safe-budget brackets and advance estimates",ylim=(-.7,1.7))
    if not any(b.get("safe_budget_upper_tokens") is not None or b.get("observed_oom_upper_tokens") is not None for b in boundaries):
        ax.text(.02,.96,"No safe-budget bracket or OOM boundary measured",transform=ax.transAxes,va="top",fontsize=9)
    ax.legend(fontsize=8,loc="lower right");ax.grid(axis="x",alpha=.15);save(fig,"04_boundary_status")
    tables={t["probe_id"]:t for t in processed_data["discrepancy_tables"]}
    for allocator in COLORS:
        rows=[p for p in primary if p["allocator"]==allocator]
        if not rows:continue
        length=max(p["length"] for p in rows)
        chosen=[tables[p["probe_id"]] for p in rows if p["length"]==length]
        components=chosen[0]["rows"]
        values=[median(t["rows"][i]["signed_bytes"] for t in chosen)/MIB for i in range(len(components))]
        fig,ax=plt.subplots(figsize=(10,5.3));running=0
        for i,(component,value) in enumerate(zip(components,values)):
            ax.bar(i,value,bottom=running,color="#3d789f" if component["supported"] else "#c34848")
            running+=value
        ax.bar(len(values),running,color="#303f50")
        labels=["\n".join(textwrap.wrap(c["component"].replace("_"," "),18)) for c in components]+["Signed total\ngap"]
        ax.set_xticks(range(len(labels)),labels,fontsize=7);ax.axhline(0,color="black",lw=.6)
        ax.set(ylabel="Signed difference (MiB)",title=f"{allocator}, L={length}: peak discrepancy (red = unaccounted)")
        ax.text(.01,.98,"Counter/lifetime decomposition; descriptive deltas are not causal activation profiling",transform=ax.transAxes,va="top",fontsize=8)
        save(fig,"05_discrepancy_"+allocator)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figures}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
