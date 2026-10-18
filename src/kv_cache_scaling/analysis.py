"""CPU analysis of immutable capsules. Counter identities are not causal proof."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
import zipfile

from .model_math import Architecture, peak_tensor_bytes, peak_budget_bytes, prefill_scenarios

SCHEMA = "memory-analysis-v1"
ROOT = Path(__file__).resolve().parents[2]
MAX_CAPSULE_BYTES = 256 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_capsules(paths):
    """Validate every member and journal reference; deduplicate identical probes."""
    seen, runs, sources = {}, {}, []
    for filename in paths:
        path = Path(filename)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require(sum(e.file_size for e in entries) <= MAX_CAPSULE_BYTES, "Capsule is too large")
            names = archive.namelist()
            require(len(names) == len(set(names)), "Duplicate archive members")
            manifest = json.loads(archive.read("MANIFEST.json"))
            require(set(names) == set(manifest) | {"MANIFEST.json"}, "Incomplete capsule manifest")
            for name, expected in manifest.items():
                require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                        "Unsafe archive path")
                require(sha(archive.read(name)) == expected, "Capsule hash mismatch: " + name)
            sources.append({"path": str(path), "sha256": sha(path.read_bytes()), "verified_members": len(manifest)})
            for name in names:
                if not (name.startswith("state/runs/") and name.endswith("/manifest.json")):
                    continue
                prefix = name[:-len("manifest.json")]
                run_id = prefix.split("/")[2]
                meta = json.loads(archive.read(name))
                require(meta.get("schema_version") == "coordinator-v1", "Unknown run schema")
                journal = [json.loads(line) for line in archive.read(prefix + "events.jsonl").splitlines()]
                starts = {}
                for sequence, event in enumerate(journal, 1):
                    require(event.get("sequence") == sequence and event.get("run_id") == run_id, "Corrupt journal sequence/identity")
                    if event["event"] == "probe_started":
                        require(event["probe_id"] not in starts, "Duplicate durable start")
                        starts[event["probe_id"]] = event
                    if event["event"] != "probe_result":
                        continue
                    pid = event["probe_id"]
                    require(pid in starts and event["request"] == starts[pid]["request"], "Result has no matching start")
                    raw_ref = event.get("process", {}).get("raw_result")
                    if not raw_ref:
                        # Process failures without a worker result stay explicit.
                        record = {**event["request"], "status": event["status"], "error": event.get("error"),
                                  "schema_version": "missing-worker-result", "snapshots": {}}
                        raw = json.dumps(record, sort_keys=True).encode()
                    else:
                        relative = raw_ref["path"]
                        require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts,
                                "Result reference escapes run")
                        raw = archive.read(prefix + relative)
                        require(sha(raw) == raw_ref["sha256"] and len(raw) == raw_ref["bytes"], "Result reference hash/size mismatch")
                        record = json.loads(raw)
                        require(record["probe_id"] == pid and record["status"] == event["status"], "Result status/identity mismatch")
                        for key, value in event["request"].items():
                            if key in record:
                                require(record[key] == value, "Result/request mismatch: " + key)
                    if pid in seen:
                        require(seen[pid]["raw_sha256"] == sha(raw), "Divergent duplicate probe")
                    else:
                        seen[pid] = {"record": record, "run_id": run_id, "manifest": meta,
                                     "raw_sha256": sha(raw), "source_capsule_sha256": sources[-1]["sha256"]}
                previous = runs.get(run_id)
                if previous:
                    require(previous["manifest"] == meta, "Run manifest drift")
                    shorter, longer = sorted((previous["events"], journal), key=len)
                    require(longer[:len(shorter)] == shorter, "Divergent journal histories")
                    journal = longer
                runs[run_id] = {"manifest": meta, "events": journal}
    require(bool(runs), "No measured run manifests found")
    return list(seen.values()), runs, sources


def regression(points):
    """OLS on per-length medians; repeat range is separate, never a fake CI."""
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return {"status": "INSUFFICIENT_LENGTHS", "slope_bytes_per_token": None, "intercept_bytes": None, "r_squared": None}
    xs, ys = zip(*points)
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x-xm)**2 for x in xs)
    slope = sum((x-xm)*(y-ym) for x, y in points) / xx
    intercept = ym - slope*xm
    residual = sum((y-(intercept+slope*x))**2 for x, y in points)
    total = sum((y-ym)**2 for y in ys)
    return {"status": "FIT", "slope_bytes_per_token": slope, "intercept_bytes": intercept,
            "r_squared": 1-residual/total if total else (1.0 if residual == 0 else None),
            "length_count": len(points), "fit_method": "OLS on per-length medians; no statistical CI claimed"}


def checked_success(record, arch):
    require(record.get("schema_version") == "primary-probe-v1", "Unsupported successful worker schema")
    for name in ("length", "repeat", "pid", "peak_allocated_bytes", "peak_reserved_bytes"):
        require(type(record.get(name)) is int and record[name] > 0, "Invalid worker metric: " + name)
    require(record.get("allocator") in ("default", "expandable"), "Unknown allocator")
    policy = record.get("execution_policy", {})
    require(policy.get("dtype") == "float16" and policy.get("batch_size") == 1 and policy.get("logits_to_keep") == 1,
            "Unsupported dtype/batch/logits policy")
    require(policy.get("sdpa_backend") == "EFFICIENT_ATTENTION", "Unsupported attention policy")
    for key in ("context_only", "post_load", "after_forward", "retained", "released"):
        snap = record.get("snapshots", {}).get(key, {})
        require(type(snap.get("allocated_bytes")) is int and type(snap.get("reserved_bytes")) is int
                and 0 <= snap["allocated_bytes"] <= snap["reserved_bytes"], "Invalid snapshot: " + key)
    require(record["peak_allocated_bytes"] >= record["snapshots"]["after_forward"]["allocated_bytes"], "Peak misses live allocation")
    require(record["peak_reserved_bytes"] >= record["snapshots"]["after_forward"]["reserved_bytes"], "Peak misses live reservation")
    require(record["weights"]["unique_parameters"] == arch.parameter_count(), "Unique weight count mismatch")
    expected = arch.kv_bytes(record["length"])
    require(record["cache"]["logical_bytes"] == expected, "Cache logical size mismatch")
    logits = record["logits"]["tensors"]
    require(len(logits) == 1 and logits[0]["shape"] == [1, 1, arch.vocab]
            and logits[0]["dtype"] == "torch.float16", "Wrong logits")
    require(not record.get("sampling", {}).get("error"), "Telemetry failed")


def group_identity(r, manifest):
    rt = r.get("runtime", {})
    return {"protocol_sha256": r.get("protocol_sha256"), "config_sha256": r.get("config_sha256"),
            "gpu_uuid": r.get("gpu_uuid"), "driver": manifest.get("expected_driver"),
            "environment_fingerprint":manifest.get("environment_fingerprint"),"usable_bytes":manifest.get("usable_bytes"),
            "torch": rt.get("torch"), "cuda": rt.get("cuda"), "transformers": rt.get("transformers"),
            "runtime_sources":rt.get("source_files"),
            "package_inventory_sha256":sha(json.dumps(rt.get("package_versions"),sort_keys=True).encode()),
            "execution_sha256": r.get("code_sha256"), "policy": r.get("execution_policy"),
            "measurement": r.get("measurement", "primary"), "allocator": r.get("allocator")}


def discrepancy(r, arch, workspace_evidence=None):
    """Signed, closed peak identity with explicit unsupported attribution.

    Lifetime deltas establish *when* bytes persist, not that every transient is
    an activation or every cached block is fragmentation. Missing requested-byte
    counters and unverified workspace causes remain unaccounted.
    """
    s = r["snapshots"]
    allocated = lambda stage: s[stage]["allocated_bytes"]
    expected = peak_tensor_bytes(arch, r["length"], prefill_scenarios(arch)[1])
    gap = r["peak_allocated_bytes"] - expected
    weights = r["weights"]["parameters"]["unique_storage_bytes"]
    rows = []
    def add(name, amount, supported, evidence):
        rows.append({"component": name, "signed_bytes": amount, "supported": supported, "evidence": evidence})
    add("unique_weight_difference", weights-arch.weight_bytes(), True, "Deduplicated parameter storage vs config-derived weights")
    buffers = r["weights"]["all_model_storage"]["unique_storage_bytes"]-weights
    add("loaded_model_buffer_storage", buffers, True, "Unique loaded model storage excluding parameters")
    baseline_other = allocated("post_load")-weights-buffers
    req = s["post_load"].get("requested_bytes")
    if req is not None:
        padding = allocated("post_load")-req
        add("postload_allocator_internal_padding", padding, True, "Same-snapshot allocated minus requested bytes")
        baseline_other -= padding
    add("unaccounted_postload_storage", baseline_other, False, "Unclassified post-load allocation; not assumed to be framework overhead")
    persistent_growth = allocated("released")-allocated("post_load")
    # Workspace attribution requires an actual separately matched control; a
    # coincidentally matching byte count alone never enables it.
    # Only the analyzer's independently matched records may supply evidence.
    # Raw worker-supplied analysis_evidence labels are deliberately ignored.
    supported_workspace = (workspace_evidence or {}).get("bytes", 0)
    require(type(supported_workspace) is int and 0 <= supported_workspace <= max(0, persistent_growth), "Invalid workspace attribution")
    if supported_workspace:
        add("controlled_cublas_workspace", supported_workspace, True,
            workspace_evidence["scope"]+"; pinned PyTorch 2.6 fixed non-Hopper workspace; controls: "
            + ",".join(workspace_evidence["control_probe_ids"]))
    add("unaccounted_postforward_persistent_growth", persistent_growth-supported_workspace, False,
        "Measured released-minus-postload allocation; physical cause unconfirmed")
    add("cache_lifetime_difference", allocated("retained")-allocated("released")-arch.kv_bytes(r["length"]), True,
        "Controlled cache release delta minus analytical retained KV")
    add("output_and_input_lifetime", allocated("after_forward")-allocated("retained"), True,
        "Controlled output/input release delta")
    activation = r["length"]*prefill_scenarios(arch)[1].activation_bytes_per_token
    add("transient_envelope_difference", r["peak_allocated_bytes"]-allocated("after_forward")-activation, True,
        "Exact peak excess above synchronized endpoint minus assumed envelope; descriptive, not pure-activation attribution")
    accounted = sum(x["signed_bytes"] for x in rows)
    unaccounted_absolute = sum(abs(x["signed_bytes"]) for x in rows if not x["supported"])
    absolute_total = sum(abs(x["signed_bytes"]) for x in rows)
    gap_coverage = max(0.0, 1-unaccounted_absolute/abs(gap)) if gap else (1.0 if unaccounted_absolute == 0 else 0.0)
    contribution_coverage = 1-unaccounted_absolute/absolute_total if absolute_total else 1.0
    closure = gap-accounted
    large = abs(gap) >= 10*1024**2
    coverage_pass = gap_coverage >= .9 and contribution_coverage >= .9 if large else unaccounted_absolute <= 1024**2
    return {"probe_id": r["probe_id"], "metric": "forward_peak_allocated_bytes", "predicted_tensor_peak_bytes": expected,
            "measured_peak_bytes": r["peak_allocated_bytes"], "signed_gap_bytes": gap, "rows": rows,
            "closure_error_bytes": closure, "closure_pass": abs(closure) <= max(1024**2, .001*r["peak_allocated_bytes"]),
            "unaccounted_absolute_bytes": unaccounted_absolute, "gap_explained_fraction": gap_coverage,
            "absolute_contribution_coverage": contribution_coverage, "coverage_pass": coverage_pass,
            "ratio_interpretation": "material gap" if large else "small gap; ratio ill-conditioned; <=1 MiB remainder required",
            "budget_table": {"predicted_budget_bytes": peak_budget_bytes(arch, r["length"], prefill_scenarios(arch)[1]),
                             "fixed_headroom_bytes": prefill_scenarios(arch)[1].fixed_headroom_bytes,
                             "interpretation": "Advance planning allowance; do not subtract unrelated reservation/driver maxima"}}


def matched_workspace_controls(records, arch):
    """Require three successful same-allocator controls and an exact source-size delta.

    L=512 has a directly matched intervention. Other lengths may use explicitly
    labelled source inference because the pinned T4 workspace size is independent
    of L, only if the released-minus-load delta is exactly the same source size.
    The other allocator requires its own three controls.
    """
    valid=[]
    for wrapped in records:
        r=wrapped.get("record",wrapped)
        if r.get("status")=="success":
            checked_success(r,arch);valid.append(r)
    def identity(r):
        rt=r.get("runtime",{})
        policy=dict(r["execution_policy"])
        policy.pop("diagnostic",None);policy.pop("cublas_workspace_config",None)
        return json.dumps([r.get("gpu_uuid"),r.get("protocol_sha256"),r.get("config_sha256"),
                           r.get("allocator"),r.get("code_sha256"),
                           [rt.get(k) for k in ("torch","cuda","transformers","source_files")],policy,
                           r["snapshots"]["post_load"]["allocated_bytes"]],sort_keys=True)
    controls=defaultdict(list)
    for r in valid:
        if (r.get("measurement")=="workspace_control" and r.get("length")==512
                and r.get("runtime",{}).get("cublas_workspace_config")==":0:0"):
            controls[identity(r)].append(r)
    evidence={}
    for r in valid:
        if r.get("measurement")!="primary":continue
        if r.get("gpu_identity",{}).get("name")!="Tesla T4" or not r.get("runtime",{}).get("torch","").startswith("2.6.0"):
            continue
        peers=controls.get(identity(r),[])
        if len({p["repeat"] for p in peers})<3:continue
        growth=lambda p:p["snapshots"]["released"]["allocated_bytes"]-p["snapshots"]["post_load"]["allocated_bytes"]
        # 4096 KiB * 2 + 16 KiB * 8, pinned non-Hopper PyTorch 2.6 source.
        if growth(r)==8519680 and all(growth(p)==0 for p in peers):
            evidence[r["probe_id"]]={"bytes":8519680,"control_probe_ids":sorted(p["probe_id"] for p in peers),
                                      "scope":"Direct L=512 intervention" if r["length"]==512 else
                                              "Source-supported fixed-size inference from same-allocator L=512 controls; not a direct intervention at this length"}
    return evidence


def analyze(records, architecture=None):
    arch = architecture or Architecture.from_config(json.loads((ROOT/"configs/models/qwen2.5-1.5b.json").read_text()))
    records=list(records)
    controls=matched_workspace_controls(records,arch)
    groups, normalized, failures, tables, snapshots = defaultdict(list), [], [], [], []
    for wrapped in records:
        r = wrapped["record"] if "record" in wrapped else wrapped
        meta = {**wrapped.get("manifest", {}), "run_id": wrapped.get("run_id")}
        if r.get("status") != "success":
            failures.append({"probe_id": r.get("probe_id"), "status": r.get("status"), "stage": r.get("stage"),
                             "length":r.get("length"),"allocator":r.get("allocator"),
                             "attempt_kind":r.get("attempt_kind"),"oom_classification":r.get("oom_classification"),"error": r.get("error")})
            continue
        checked_success(r, arch)
        identity = group_identity(r, meta)
        key = sha(json.dumps(identity, sort_keys=True).encode())
        s = r["snapshots"]
        weights = r["weights"]["parameters"]["unique_storage_bytes"]
        cache = r["cache"]["unique_storage_bytes"]
        residual = s["retained"]["allocated_bytes"]-weights-cache
        row = {"probe_id": r["probe_id"], "run_id": wrapped.get("run_id"), "group_id": key,
               "length": r["length"], "repeat": r["repeat"], "allocator": r["allocator"],
               "measurement": r.get("measurement", "primary"), "pid": r["pid"],
               "cache_bytes": cache, "cache_formula_relative_error": abs(cache/arch.kv_bytes(r["length"])-1),
               "cache_lifetime_bytes": s["retained"]["allocated_bytes"]-s["released"]["allocated_bytes"],
               "retained_minus_postload_bytes": s["retained"]["allocated_bytes"]-s["post_load"]["allocated_bytes"],
               "retained_residual_bytes": residual, "retained_residual_fraction": residual/s["retained"]["allocated_bytes"],
               "peak_allocated_bytes": r["peak_allocated_bytes"], "peak_reserved_bytes": r["peak_reserved_bytes"],
               "transient_excess_bytes": r["peak_allocated_bytes"]-s["after_forward"]["allocated_bytes"],
               "raw_sha256": wrapped.get("raw_sha256")}
        normalized.append(row); groups[key].append((identity, row, r))
        for stage, snap in s.items():
            snapshots.append({"probe_id": r["probe_id"], "group_id": key, "length": r["length"], "allocator": r["allocator"],
                              "measurement": row["measurement"], "repeat": r["repeat"], "stage": stage,
                              "allocated_bytes": snap["allocated_bytes"], "reserved_bytes": snap["reserved_bytes"],
                              "allocator_slack_bytes": snap["reserved_bytes"]-snap["allocated_bytes"],
                              "inactive_split_bytes": snap.get("inactive_split_bytes"),
                              "nvml_used_bytes": (snap.get("nvml") or {}).get("used_bytes")})
        if row["measurement"] == "primary":
            tables.append(discrepancy(r, arch, controls.get(r["probe_id"])))
    summaries = []
    for key, items in groups.items():
        identity = items[0][0]
        by_length = defaultdict(list)
        for _, row, _ in items:
            by_length[row["length"]].append(row)
        coverage = {str(n): sorted(r["repeat"] for r in rs) for n, rs in sorted(by_length.items())}
        require(all(len(v) == len(set(v)) for v in coverage.values()), "Duplicate repeats in one scientific group")
        fits = {}
        for metric in ("cache_bytes", "cache_lifetime_bytes", "retained_minus_postload_bytes", "peak_allocated_bytes"):
            fits[metric] = regression([(n, median(r[metric] for r in rs)) for n, rs in sorted(by_length.items())])
        fit = fits["cache_bytes"]
        slope_error = None if fit["slope_bytes_per_token"] is None else abs(fit["slope_bytes_per_token"]/arch.kv_bytes_per_token()-1)
        required_covered = all(coverage.get(str(n)) == [1, 2, 3] for n in (512,1024,2048,4096,8192))
        repeat_stats = [{"length": n, "peak_median_bytes": median(r["peak_allocated_bytes"] for r in rs),
                         "peak_min_bytes": min(r["peak_allocated_bytes"] for r in rs),
                         "peak_max_bytes": max(r["peak_allocated_bytes"] for r in rs),
                         "peak_spread_fraction": (max(r["peak_allocated_bytes"] for r in rs)-min(r["peak_allocated_bytes"] for r in rs))/median(r["peak_allocated_bytes"] for r in rs)}
                        for n, rs in sorted(by_length.items())]
        summaries.append({"group_id": key, "identity": identity, "coverage": coverage, "fits": fits,
                          "run_ids":sorted({row["run_id"] for _,row,_ in items if row["run_id"]}),
                          "repeat_statistics": repeat_stats, "slope_relative_error": slope_error,
                          "required_grid_complete": required_covered,
                          "cache_formula_pass": all(r["cache_formula_relative_error"] <= .01 for _,r,_ in items),
                          "cache_slope_pass": required_covered and slope_error is not None and slope_error <= .03,
                          "repeat_spread_pass": all(x["peak_spread_fraction"] <= .05 for x in repeat_stats),
                          "interpretation": "Peak fit is total tracked memory, not a KV-cache slope; repeat range is not a confidence interval"})
    primary = [g for g in summaries if g["identity"]["measurement"] == "primary"]
    cohorts = defaultdict(dict)
    for group in primary:
        identity = {k:v for k,v in group["identity"].items() if k != "allocator"}
        cohorts[sha(json.dumps(identity,sort_keys=True).encode())][group["identity"]["allocator"]] = group
    paired = [{"cohort_id": key, "allocators": sorted(groups),
               "required_grid_complete": set(groups) == {"default","expandable"} and all(g["required_grid_complete"] for g in groups.values())}
              for key,groups in cohorts.items()]
    comparisons=[]
    def comparable(identity):
        return {k:v for k,v in identity.items() if k not in ("run_id","measurement","policy")}
    for diagnostic in summaries:
        mode=diagnostic["identity"]["measurement"]
        if mode in ("primary","kernel_trace"):continue
        peers=[g for g in primary if comparable(g["identity"])==comparable(diagnostic["identity"])]
        for peer in peers:
            for length in sorted(set(peer["coverage"])&set(diagnostic["coverage"]),key=int):
                p=[r for r in normalized if r["group_id"]==peer["group_id"] and r["length"]==int(length)]
                d=[r for r in normalized if r["group_id"]==diagnostic["group_id"] and r["length"]==int(length)]
                comparisons.append({"measurement":mode,"allocator":diagnostic["identity"]["allocator"],"length":int(length),
                                    "primary_repeats":len(p),"diagnostic_repeats":len(d),
                                    "primary_peak_median_bytes":median(r["peak_allocated_bytes"] for r in p),
                                    "diagnostic_peak_median_bytes":median(r["peak_allocated_bytes"] for r in d),
                                    "diagnostic_minus_primary_peak_bytes":median(r["peak_allocated_bytes"] for r in d)-median(r["peak_allocated_bytes"] for r in p),
                                    "interpretation":"Separate semantics/intervention; not a primary slope or boundary observation"})
    return {"schema_version": SCHEMA, "units": {"memory":"bytes", "length":"tokens"}, "groups": summaries,
            "probes": normalized, "snapshots": snapshots, "failures": failures, "discrepancy_tables": tables,
            "diagnostic_comparisons":comparisons,
            "paired_cohorts": paired,
            "acceptance": {"five_length_paired_primary_complete": bool(paired) and all(c["required_grid_complete"] for c in paired),
                           "cache_slope_pass": bool(primary) and all(g["cache_slope_pass"] for g in primary),
                           "discrepancy_coverage_pass": bool(tables) and all(t["coverage_pass"] for t in tables),
                           "discrepancy_closure_pass": bool(tables) and all(t["closure_pass"] for t in tables)},
            "limitations": ["No fabricated OOM threshold from a safety skip", "Unexplained storage remains unaccounted",
                            "No fragmentation claim from reserved minus allocated alone", "Published serving-cache figures do not validate HF prefill peaks"]}


def boundary_summary(runs):
    result = []
    for run_id, run in runs.items():
        events = run["events"]
        success = [e for e in events if e["event"] == "probe_result" and e["status"] == "success"
                   and e["request"].get("measurement", "primary") == "primary"]
        for allocator in ("default", "expandable"):
            rows = [e for e in success if e["request"]["allocator"] == allocator]
            boundary = [e for e in events if e["event"] == "allocator_boundary" and e["allocator"] == allocator]
            oom = [e["request"]["length"] for e in events if e["event"] == "probe_result" and e["status"] == "cuda_oom"
                   and e["request"]["allocator"] == allocator]
            if rows or boundary or oom:
                result.append({"run_id": run_id, "allocator": allocator, "execution_scope": run["manifest"].get("execution_scope"),
                               "largest_observed_success_tokens": max((e["request"]["length"] for e in rows), default=None),
                               "observed_oom_upper_tokens": min(oom, default=None),
                               "safe_budget_lower_tokens": boundary[-1].get("safe_budget_lower_tokens") if boundary else None,
                               "safe_budget_upper_tokens": boundary[-1].get("safe_budget_upper_tokens") if boundary else None,
                               "status": "SAFE_BUDGET_CENSORED" if boundary and not oom else ("OBSERVED_OOM_BRACKET" if oom else "NO_BOUNDARY_MEASURED")})
    return result


def write_analysis(paths, output, run_ids=None):
    output = Path(output)
    require(not output.exists(), "Refusing to overwrite processed output")
    records, runs, sources = read_capsules(paths)
    if run_ids:
        allowed = set(run_ids); records = [r for r in records if r["run_id"] in allowed]
        runs = {k:v for k,v in runs.items() if k in allowed}
        require(bool(runs), "Requested runs were not found")
    result = analyze(records)
    result["boundaries"] = boundary_summary(runs)
    deltas=[]
    for run_id in runs:
        pair={b["allocator"]:b for b in result["boundaries"] if b["run_id"]==run_id and b["execution_scope"]=="primary_sweep"}
        if set(pair)=={"default","expandable"} and all(b["safe_budget_upper_tokens"] is not None for b in pair.values()):
            d,e=pair["default"],pair["expandable"]
            deltas.append({"run_id":run_id,"observed_success_delta_tokens":e["largest_observed_success_tokens"]-d["largest_observed_success_tokens"],
                           "safe_bracket_delta_min_tokens":e["safe_budget_lower_tokens"]-d["safe_budget_upper_tokens"],
                           "safe_bracket_delta_max_tokens":e["safe_budget_upper_tokens"]-d["safe_budget_lower_tokens"],
                           "interpretation":"Expandable minus default conservative safe-budget limit, not a measured hardware-context improvement"})
    result["allocator_boundary_deltas"]=deltas
    brackets=[]
    for g in result["groups"]:
        if g["identity"]["measurement"]!="primary":continue
        oom=[]
        for wrapped in records:
            r=wrapped["record"]
            if (r.get("status")=="cuda_oom" and r.get("error",{}).get("type")=="OutOfMemoryError"
                    and r.get("attempt_kind")=="deliberate" and r.get("oom_classification")=="deliberate_request"):
                identity=group_identity(r,wrapped["manifest"])
                if identity==g["identity"]:oom.append(r["length"])
        lower=max(int(n) for n in g["coverage"]);upper=min(oom,default=None)
        brackets.append({"group_id":g["group_id"],"allocator":g["identity"]["allocator"],
                         "largest_observed_success_tokens":lower,"smallest_observed_oom_tokens":upper,
                         "original_prediction_tokens":111232,
                         "consistent_monotone_bracket":upper is None or upper>lower,
                         "prediction_error_percent_interval":None if upper is None or upper<=lower else
                             [100*(111232/(upper-1)-1),100*(111232/lower-1)],
                         "interpretation":"Under monotone feasibility, successful lower bound <= true limit < failed upper bound; no exact threshold or confidence interval"})
    result["hardware_brackets"]=brackets
    held_out=[]
    for run_id,run in runs.items():
        meta=run["manifest"]
        if meta.get("execution_scope")!="calibrated_validation":continue
        for allocator in ("default","expandable"):
            points=[p for p in result["probes"] if p["run_id"]==run_id and p["allocator"]==allocator and p["measurement"]=="primary"]
            if not points:continue
            length=meta["validation_length"];fit=meta["reference_sweep"]["peak_fits"][allocator]
            predicted=fit["intercept_bytes"]+fit["slope_bytes_per_token"]*length
            observed=median(p["peak_allocated_bytes"] for p in points)
            held_out.append({"run_id":run_id,"reference_run_id":meta["reference_sweep"]["run_id"],
                             "allocator":allocator,"length":length,"repeats":len(points),
                             "prospective_fit_peak_bytes":predicted,"observed_peak_median_bytes":observed,
                             "prospective_fit_error_percent":100*(predicted-observed)/observed,
                             "interpretation":"Frozen reference fit evaluated on a later held-out length; separate from original advance prediction"})
    result["held_out_validation"]=held_out
    result["provenance"] = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                            "analysis_sha256": sha(Path(__file__).read_bytes()), "run_ids": sorted(runs)}
    output.mkdir(parents=True)
    (output/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for name, rows in (("probes",result["probes"]),("snapshots",result["snapshots"]),("boundaries",result["boundaries"]),("failures",result["failures"]),("diagnostic_comparisons",result["diagnostic_comparisons"]),("allocator_boundary_deltas",deltas),("hardware_brackets",brackets),("held_out_validation",held_out)):
        if rows:
            with (output/(name+".csv")).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    flat = [{"probe_id": t["probe_id"], "signed_gap_bytes": t["signed_gap_bytes"], **row}
            for t in result["discrepancy_tables"] for row in t["rows"]]
    if flat:
        with (output/"discrepancy.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (output/"manifest.json").write_text(json.dumps({p.name:sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()},indent=2)+"\n")
    return result
