"""CPU coordinator. No torch import; primary execution lives only in worker.py."""

from dataclasses import asdict
import json
import re
from importlib.metadata import version
from pathlib import Path
import sys
from statistics import median
from uuid import UUID, uuid4

from .coordinator_store import AttemptLedger, RunStore, file_lock, now
from .gate import digest, json_bytes, verify_receipt
from .model_math import Architecture, peak_budget_bytes, prefill_scenarios
from .probe import EXPECTED_RUNTIME, POLICY, ProbeRequest, RuntimeMismatch, execution_policy
from .processes import launch_worker, worker_env

RESULT_LIMIT = 16 * 1024**2
IDLE_TOLERANCE = 64 * 1024**2


class CheckpointPause(Exception):
    """A bounded acceptance step completed; external controller must persist it."""


class DriverObserver:
    """NVML only: no parent-owned CUDA context."""
    def __init__(self, verified):
        self.verified = verified

    def read(self):
        import pynvml
        from .telemetry import NvmlDevice
        pynvml.nvmlInit()
        try:
            device = NvmlDevice(pynvml, self.verified.gpu_uuid)
            identity, memory, processes = device.identity(), device.memory(), device.processes()
            if not processes["available"] or processes["other_pids"] or processes["pids"]:
                raise ValueError("Idle GPU process attribution unavailable or GPU is occupied")
            if identity["driver_version"] != self.verified.driver_version:
                raise ValueError("Driver drift")
            return {**identity, **memory}
        finally:
            pynvml.nvmlShutdown()


def point_key(verified, length, allocator, repeat, attempt_kind="safe", measurement="primary"):
    return digest(json_bytes({"policy": POLICY, "protocol": verified.protocol_sha256,
                              "config": verified.config_sha256, "environment": verified.environment_fingerprint,
                              "allocator": allocator, "length": length, "repeat": repeat,
                              "attempt_kind": attempt_kind, "measurement": measurement}))


def admission(arch, length, usable_bytes):
    # Immutable advance model; never refit from measured peaks here.
    estimate = peak_budget_bytes(arch, length, prefill_scenarios(arch)[2])
    budget = usable_bytes * 9 // 10
    return {"admitted": estimate <= budget, "conservative_estimate_bytes": estimate,
            "safe_budget_bytes": budget, "model": "original conservative scenario; not a proven bound"}


def next_bisection(lower, upper, resolution=256):
    if upper - lower <= resolution:
        return None
    point = ((lower + upper) // (2 * resolution)) * resolution
    return point if lower < point < upper else None


def validate_result(result, request, arch, root):
    if not isinstance(result, dict) or result.get("probe_id") != request.probe_id or result.get("schema_version") != "primary-probe-v1":
        raise ValueError("Worker result ID/schema mismatch")
    if result.get("length") != request.length:
        raise ValueError("Worker result length mismatch")
    if result.get("status") not in {"success", "cuda_oom", "unsupported_backend", "unsupported_allocator",
                                   "runtime_mismatch", "timeout", "telemetry_error", "error",
                                   "environment_drift", "authorization_error"}:
        raise ValueError("Unknown worker status")
    if result["status"] == "cuda_oom":
        intent = "deliberate_request" if request.attempt_kind == "deliberate" else "accidental"
        if result.get("oom_classification") != intent or result.get("error", {}).get("type") != "OutOfMemoryError":
            raise ValueError("OOM status lacks the matching typed exception/intent")
    if result["status"] == "success":
        if result.get("execution_policy") != execution_policy(request):
            raise ValueError("Execution policy mismatch")
        runtime = result.get("runtime", {})
        if (runtime.get("torch", "").split("+")[0] != EXPECTED_RUNTIME["torch"]
                or runtime.get("cuda") != EXPECTED_RUNTIME["cuda"]
                or runtime.get("transformers") != EXPECTED_RUNTIME["transformers"]):
            raise ValueError("Worker runtime mismatch")
        for name in ("probe.py", "telemetry.py"):
            if result.get("code_sha256", {}).get(name) != digest((Path(root)/"src/kv_cache_scaling"/name).read_bytes()):
                raise ValueError("Worker code fingerprint mismatch")
        for key in ("repeat", "allocator", "protocol_sha256", "config_sha256", "gpu_uuid", "attempt_kind", "measurement"):
            if result.get(key) != getattr(request, key):
                raise ValueError("Successful result request mismatch: " + key)
        if request.measurement == "kernel_trace":
            names = result.get("kernel_diagnostic", {}).get("event_names", [])
            if not any("_scaled_dot_product_efficient_attention" in name for name in names):
                raise ValueError("Diagnostic lacks efficient SDPA trace evidence")
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if type(result.get(name)) is not int or result[name] < 0:
                raise ValueError("Success requires actual integer peaks")
        if set(result.get("snapshots", {})) != {"context_only", "post_load", "after_forward", "retained", "released"}:
            raise ValueError("Success is missing a lifecycle snapshot")
        for snap in result["snapshots"].values():
            for field in ("allocated_bytes", "reserved_bytes", "cuda_free_bytes", "cuda_total_bytes"):
                if type(snap.get(field)) is not int or snap[field] < 0:
                    raise ValueError("Invalid snapshot counter")
            if snap["reserved_bytes"] < snap["allocated_bytes"]:
                raise ValueError("Snapshot reservation invariant failed")
        after = result["snapshots"]["after_forward"]
        if result["peak_allocated_bytes"] < after["allocated_bytes"] or result["peak_reserved_bytes"] < after["reserved_bytes"]:
            raise ValueError("Peak does not cover live counters")
        cache, weights = result.get("cache", {}), result.get("weights", {})
        expected = arch.kv_bytes(request.length)
        if (cache.get("logical_bytes") != expected
                or type(cache.get("unique_storage_bytes")) is not int
                or abs(cache["unique_storage_bytes"] / expected - 1) > 0.01
                or weights.get("unique_parameters") != arch.parameter_count()):
            raise ValueError("Worker cache/weight accounting mismatch")
        if not result.get("cache") or not result.get("weights") or result.get("sampling", {}).get("error"):
            raise ValueError("Success lacks validated cache/weights/telemetry")
    return result


def collect_process(store, request, command, env, root, timeout, runner=launch_worker):
    output = store.local / "worker" / (request.probe_id + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = runner(command(output), env=env, cwd=root, timeout=timeout)
    log_meta = process.pop("logs")
    for name in ("stdout", "stderr"):
        log_meta[name]["artifact"] = store.artifact(f"logs/{request.probe_id}.{name}.txt", process.pop(name))
    process["logs"] = log_meta
    raw = output.read_bytes() if output.exists() and output.stat().st_size <= RESULT_LIMIT else None
    if raw is not None:
        process["raw_result"] = store.artifact(f"results/{request.probe_id}.json", raw)
    elif output.exists():
        with output.open("rb") as handle:
            process["raw_result_prefix"] = store.artifact(f"results/{request.probe_id}.oversized-prefix", handle.read(RESULT_LIMIT))
    if process["process_status"] != "exited":
        return {"status": process["process_status"], "metrics": None, "process": process}
    if process["returncode"] < 0:
        return {"status": "signal", "metrics": None, "process": process}
    try:
        arch = Architecture.from_config(json.loads((Path(root)/"configs/models/qwen2.5-1.5b.json").read_text()))
        result = validate_result(json.loads(raw) if raw is not None else None, request, arch, root)
        if (result["status"] == "success") != (process["returncode"] == 0):
            raise ValueError("Exit status disagrees with result")
    except (TypeError, ValueError) as error:
        return {"status": "invalid_output", "metrics": None, "error": str(error), "process": process}
    return {"status": result["status"], "metrics": result, "process": process}


def confirmation_reference(verified, reference_run_id, _depth=0):
    """Read a completed compatible safe sweep; never manufacture launch events."""
    reference_run_id=str(UUID(reference_run_id))
    directory=verified.state_root/"runs"/reference_run_id
    manifest=json.loads((directory/"manifest.json").read_text())
    scope=manifest.get("execution_scope")
    if scope not in ("primary_sweep","calibrated_validation") or _depth>1:
        raise ValueError("Unsupported or recursive confirmation reference")
    expected={"gpu_uuid":verified.gpu_uuid,
              "environment_fingerprint":verified.environment_fingerprint,
              "usable_bytes":verified.usable_bytes,"protocol_sha256":verified.protocol_sha256,
              "config_sha256":verified.config_sha256,"experiment_id":verified.experiment_id,
              "deliberate_length":None}
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Confirmation reference differs from the verified scientific environment")
    events=[json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    if (not events or events[-1]["event"]!="run_finished"
            or any(e["event"] in ("run_stopped","run_deviation","probe_interrupted") for e in events)):
        raise ValueError("Confirmation requires a finished, uninterrupted safe sweep")
    if [e["sequence"] for e in events]!=list(range(1,len(events)+1)):
        raise ValueError("Confirmation reference journal is corrupt")
    results=[e for e in events if e["event"]=="probe_result"]
    idle={e["probe_id"] for e in events if e["event"]=="idle_verified"}
    required={(a,n,i) for a in ("default","expandable") for n in (512,1024,2048,4096,8192) for i in (1,2,3)}
    parent=None
    if scope=="calibrated_validation":
        parent=confirmation_reference(verified,manifest["reference_sweep"]["run_id"],_depth+1)
        if parent!=manifest["reference_sweep"]:raise ValueError("Validation parent reference changed")
        required={(a,manifest["validation_length"],i) for a in ("default","expandable") for i in (1,2,3)}
    seen=set()
    peak_points={a:{} for a in ("default","expandable")}
    architecture=Architecture.from_config(json.loads((verified.root/"configs/models/qwen2.5-1.5b.json").read_text()))
    starts={e["probe_id"]:e for e in events if e["event"]=="probe_started"}
    for e in results:
        if e["status"]!="success" or e["probe_id"] not in idle:
            raise ValueError("Reference contains a failed or unreconciled worker")
        ref=e["process"]["raw_result"]
        path=Path(ref["path"])
        if path.is_absolute() or ".." in path.parts:raise ValueError("Unsafe reference result")
        data=(directory/path).read_bytes()
        if digest(data)!=ref["sha256"] or len(data)!=ref["bytes"]:raise ValueError("Reference raw result changed")
        r=json.loads(data)
        if (r.get("status")!="success" or r.get("probe_id")!=e["probe_id"]
                or e["probe_id"] not in starts or starts[e["probe_id"]]["request"]!=e["request"]):
            raise ValueError("Reference worker identity/start/status mismatch")
        validate_result(r,ProbeRequest.from_dict(e["request"]),architecture,verified.root)
        if r["code_sha256"]!={name:digest((verified.root/"src/kv_cache_scaling"/name).read_bytes()) for name in ("probe.py","telemetry.py")}:
            raise ValueError("Primary worker semantics changed since reference sweep")
        q=e["request"]
        if q["measurement"]=="primary":
            seen.add((q["allocator"],q["length"],q["repeat"]))
            peak_points[q["allocator"]].setdefault(q["length"],[]).append(r["peak_allocated_bytes"])
    if not required.issubset(seen):raise ValueError("Reference required paired grid is incomplete")
    boundaries=events[-1]["boundaries"]
    if any(not boundaries.get(a,{}).get("largest_success_tokens") for a in ("default","expandable")):
        raise ValueError("Reference lacks paired safe boundaries")
    fits={}
    for allocator,points in peak_points.items():
        data=[(n,median(v)) for n,v in sorted(points.items())]
        if len(data)<5:
            if parent:fits[allocator]=parent["peak_fits"][allocator]
            continue
        xm=sum(x for x,y in data)/len(data);ym=sum(y for x,y in data)/len(data)
        slope=sum((x-xm)*(y-ym) for x,y in data)/sum((x-xm)**2 for x,y in data)
        intercept=ym-slope*xm
        residual=sum((y-intercept-slope*x)**2 for x,y in data);total=sum((y-ym)**2 for x,y in data)
        fits[allocator]={"slope_bytes_per_token":slope,"intercept_bytes":intercept,"length_count":len(data),
                         "r_squared":1-residual/total if total else 1.0,
                         "max_relative_fit_error":max(abs(y-intercept-slope*x)/y for x,y in data)}
    return {"run_id":reference_run_id,"manifest_sha256":digest((directory/"manifest.json").read_bytes()),
            "journal_sha256":digest((directory/"events.jsonl").read_bytes()),"boundaries":boundaries,"peak_fits":fits}


def calibrated_admission(fit, length, usable_bytes):
    """Prospective held-out check; 5% slope allowance plus 5% VRAM reserve.

    These are explicit engineering assumptions, not confidence bounds. They do
    not change the original advance prediction or legacy conservative sweep.
    """
    slope,intercept=fit["slope_bytes_per_token"],fit["intercept_bytes"]
    if (slope<=0 or intercept<0 or fit["length_count"]<5
            or fit.get("r_squared",0)<.995 or fit.get("max_relative_fit_error",1)>.02):
        raise ValueError("No physical multi-length peak fit for calibrated admission")
    upper=intercept+1.05*slope*length
    return {"estimated_upper_peak_bytes":upper,"admission_budget_bytes":usable_bytes*.95,
            "admitted":upper<=usable_bytes*.95,"slope_allowance_fraction":.05,"vram_reserve_fraction":.05,
            "interpretation":"Post-observation scheduling assumptions; not preregistered prediction or proven bounds"}


def _run_sweep(verified, local_root, *, run_id=None, deliberate_length=None, acceptance_only=False,
              checkpoint_each_probe=False, diagnostics_only=False,
              confirmation_only=False, reference_run_id=None,
              validation_only=False, validation_length=None,
              external_attempt_sha256=None,
              _runner=launch_worker, _observer=None):
    """verified must come from verify_receipt in production; injected doubles are tests only."""
    run_id = run_id or str(uuid4())
    if acceptance_only and deliberate_length is not None:
        raise ValueError("Acceptance mode cannot consume a deliberate attempt")
    if diagnostics_only and (acceptance_only or deliberate_length is not None):
        raise ValueError("Diagnostic scope cannot mix acceptance or deliberate intent")
    if confirmation_only and (acceptance_only or diagnostics_only or validation_only or deliberate_length is None or reference_run_id is None):
        raise ValueError("Confirmation-only requires a deliberate length and finished reference sweep")
    if confirmation_only and (not isinstance(external_attempt_sha256,str) or not re.fullmatch(r"[0-9a-f]{64}",external_attempt_sha256)):
        raise ValueError("Confirmation-only requires the consumed external dispatch intent SHA-256")
    if external_attempt_sha256 is not None and not confirmation_only:
        raise ValueError("External dispatch intent applies only to confirmation-only scope")
    if validation_only and (acceptance_only or diagnostics_only or deliberate_length is not None or reference_run_id is None):
        raise ValueError("Validation-only requires a finished reference and no deliberate intent")
    if validation_length is not None and not validation_only:raise ValueError("Validation length needs validation-only scope")
    if reference_run_id is not None and not (confirmation_only or validation_only):
        raise ValueError("A reference sweep needs confirmation/validation-only scope")
    reference=confirmation_reference(verified,reference_run_id) if (confirmation_only or validation_only) else None
    root = verified.root
    protocol = json.loads((root / "configs/experiment.json").read_text())
    arch = Architecture.from_config(json.loads((root / "configs/models/qwen2.5-1.5b.json").read_text()))
    observer = _observer or DriverObserver(verified)
    ledger = AttemptLedger(verified.state_root, verified.experiment_id)
    identity = {"schema_version": "coordinator-v1", "receipt_sha256": verified.receipt_sha256,
                "prediction_commit": verified.prediction_commit, "code_commit": verified.code_commit,
                "environment_fingerprint": verified.environment_fingerprint,
                "gpu_uuid": verified.gpu_uuid, "usable_bytes": verified.usable_bytes,
                "protocol_sha256": verified.protocol_sha256, "config_sha256": verified.config_sha256,
                "experiment_id": verified.experiment_id, "deliberate_length": deliberate_length,
                "execution_scope": "calibrated_validation" if validation_only else ("boundary_confirmation" if confirmation_only else ("diagnostics" if diagnostics_only else ("acceptance_l512" if acceptance_only else "primary_sweep"))),
                "checkpoint_each_probe": checkpoint_each_probe,
                "expected_runtime": EXPECTED_RUNTIME, "expected_driver": verified.driver_version,
                "expected_cuda_total_bytes": verified.total_bytes,
                "actual_package_inventory": "per-worker runtime.package_versions in journal/result artifacts"}
    if reference is not None:identity["reference_sweep"]=reference
    if validation_only:identity["validation_length"]=validation_length
    if confirmation_only:identity["external_attempt_sha256"]=external_attempt_sha256
    with RunStore(local_root, verified.state_root, run_id) as store:
        if deliberate_length is not None and ledger.path.exists() and not (store.local / "manifest.json").exists():
            return {"run_id": run_id, "status": "needs_review", "reason": "deliberate token unavailable", "boundaries": {}}
        if ledger.path.exists() and json.loads(ledger.path.read_text()).get("state") in ("blocked_after_accidental_oom", "completed_oom"):
            return {"run_id": run_id, "status": "needs_review", "reason": "global OOM stop", "boundaries": {}}
        existing = store.local / "manifest.json"
        if existing.exists():
            old = json.loads(existing.read_text())
            if {k: v for k, v in old.items() if k != "created_at_utc"} != identity:
                raise ValueError("Resume fingerprint drift; start a new run")
        else:
            store.manifest({**identity, "created_at_utc": now()})
        if any(row["event"] in ("run_stopped", "run_deviation") for row in store.events):
            return {"run_id": run_id, "status": "needs_review", "boundaries": {}}
        finished = [row for row in store.events if row["event"] == "run_finished"]
        if finished:
            return {"run_id": run_id, "status": "complete", "boundaries": finished[-1]["boundaries"]}
        orphans = store.orphaned()
        for started in orphans:
            store.append("probe_interrupted", probe_id=started["probe_id"], point_key=started["point_key"],
                         reason="orphaned durable start; no automatic retry")
        if orphans:
            store.append("run_stopped", reason="interrupted attempts require review; use a new run after review")
            return {"run_id": run_id, "status": "needs_review", "boundaries": {}}
        if any(row["event"] == "probe_result" and row["status"] not in ("success",) for row in store.events):
            return {"run_id": run_id, "status": "needs_review", "boundaries": {}}
        idle_checked = {row["probe_id"] for row in store.events if row["event"] == "idle_verified"}
        if any(row["event"] == "probe_result" and row["probe_id"] not in idle_checked for row in store.events):
            return {"run_id": run_id, "status": "needs_review", "boundaries": {}}
        complete = {row["point_key"]: row for row in store.events if row["event"] == "probe_result"}
        if any(row["event"] == "probe_interrupted" for row in store.events):
            return {"run_id": run_id, "status": "needs_review", "boundaries": {}}
        boundaries = {}
        def perform(length, allocator, repeat, kind="safe", measurement="primary"):
            key = point_key(verified, length, allocator, repeat, kind, measurement)
            if key in complete:
                return complete[key]
            request = ProbeRequest(str(uuid4()), length, verified.gpu_uuid, verified.protocol_sha256,
                                   verified.config_sha256, allocator, repeat, kind, measurement)
            verified.check_request(request)
            before = observer.read()
            if kind == "deliberate":
                ledger.consume(run_id, request.probe_id, key)
            store.artifact(f"requests/{request.probe_id}.json", json_bytes(asdict(request)) + b"\n")
            store.append("probe_started", probe_id=request.probe_id, point_key=key, request=asdict(request), idle_before=before)
            def command(output):
                return [sys.executable, "-m", "kv_cache_scaling.worker", "--root", str(root),
                        "--receipt", verified.receipt_path, "--run-id", run_id,
                        "--request", str(store.local / "requests" / (request.probe_id + ".json")),
                        "--output", str(output)]
            try:
                record = collect_process(store, request, command, worker_env(allocator, root, measurement), root,
                                         protocol["per_probe_timeout_seconds"], _runner)
            except BaseException:
                store.append("probe_interrupted", probe_id=request.probe_id, point_key=key,
                             reason="launch/collection interrupted; preserve artifacts and review")
                if kind == "deliberate":
                    ledger.finish(request.probe_id, "interrupted")
                raise
            if kind == "deliberate":
                ledger.finish(request.probe_id, record["status"])
            elif record["status"] == "cuda_oom":
                ledger.block_accidental(run_id, request.probe_id)
            row = store.append("probe_result", probe_id=request.probe_id, point_key=key,
                               request=asdict(request), **record)
            complete[key] = row
            if record["status"] not in ("success", "cuda_oom") or (record["status"] == "cuda_oom" and kind != "deliberate"):
                raise RuntimeError("Probe failed; escalation stopped: " + record["status"])
            after = observer.read()
            if (after["uuid"] != before["uuid"] or after["driver_version"] != before["driver_version"]
                    or after["total_bytes"] != before["total_bytes"]
                    or abs(after["used_bytes"] - before["used_bytes"]) > IDLE_TOLERANCE):
                store.append("run_deviation", probe_id=request.probe_id, reason="idle driver memory/identity drift", idle_after=after)
                raise RuntimeError("GPU idle state did not return to baseline")
            store.append("idle_verified", probe_id=request.probe_id, idle_after=after)
            if checkpoint_each_probe and kind!="deliberate":
                raise CheckpointPause()
            return row
        def measure_length(length, allocator):
            safe = True
            for repeat in range(1, protocol["repeats"] + 1):
                row = perform(length, allocator, repeat)
                peak = row["metrics"]["peak_reserved_bytes"]
                safe = safe and peak + prefill_scenarios(arch)[2].fixed_headroom_bytes <= verified.usable_bytes * 9 // 10
            return safe
        try:
            if validation_only:
                if type(validation_length) is not int or not 1<=validation_length<=arch.max_positions:
                    raise ValueError("Validation length exceeds the config domain")
                for allocator in protocol["allocator_variants"]:
                    if validation_length<=reference["boundaries"][allocator]["largest_success_tokens"]:
                        raise ValueError("Validation point must be beyond the reference grid")
                    decision=calibrated_admission(reference["peak_fits"][allocator],validation_length,verified.usable_bytes)
                    if not decision["admitted"]:raise ValueError("Calibrated validation fails its prospective admission gate")
                    for repeat in range(1,protocol["repeats"]+1):perform(validation_length,allocator,repeat)
                    boundaries[allocator]={"largest_success_tokens":validation_length,"observed_oom_upper_tokens":None,
                                           "calibrated_admission":decision,"interpretation":"Held-out primary validation, not an OOM boundary"}
                store.append("run_finished",boundaries=boundaries)
                return {"run_id":run_id,"status":"complete","boundaries":boundaries}
            if confirmation_only:
                if type(deliberate_length) is not int or not 1<=deliberate_length<=arch.max_positions:
                    raise ValueError("Confirmation length exceeds the config domain")
                lower=reference["boundaries"]["default"]["largest_success_tokens"]
                if deliberate_length<=lower:raise ValueError("Confirmation must exceed the observed default success")
                row=perform(deliberate_length,"default",1,"deliberate")
                boundaries={"default":{"largest_success_tokens":deliberate_length if row["status"]=="success" else lower,
                                       "observed_oom_upper_tokens":deliberate_length if row["status"]=="cuda_oom" else None,
                                       "deliberate_outcome":row["status"],"reference_run_id":reference["run_id"],
                                       "interpretation":"One final deliberate attempt; reference safe sweep is separate evidence"}}
                store.append("run_finished",boundaries=boundaries)
                return {"run_id":run_id,"status":"complete","boundaries":boundaries}
            if diagnostics_only:
                for measurement, grid in (("chunked_dynamic",protocol["diagnostics"]["chunked_prefill"]["lengths"]),
                                          ("static_cache",protocol["diagnostics"]["static_cache"]["lengths"]),
                                          ("workspace_control",[512])):
                    for length in grid:
                        if not admission(arch,length,verified.usable_bytes)["admitted"]:
                            raise RuntimeError("Diagnostic fails analytical admission")
                        for allocator in (protocol["allocator_variants"] if measurement=="workspace_control" else ["default"]):
                            for repeat in range(1,protocol["repeats"]+1):
                                perform(length,allocator,repeat,measurement=measurement)
                perform(512,"default",1,measurement="kernel_trace")
                boundaries={"interpretation":"Separate diagnostic grids; never primary boundary observations"}
                store.append("run_finished",boundaries=boundaries)
                return {"run_id":run_id,"status":"complete","boundaries":boundaries}
            if acceptance_only:
                if not admission(arch, 512, verified.usable_bytes)["admitted"]:
                    raise RuntimeError("L=512 fails analytical admission")
                for allocator in protocol["allocator_variants"]:
                    measure_length(512, allocator)
                perform(512, "default", 1, measurement="kernel_trace")
                boundaries = {"interpretation": "L=512 acceptance only; no boundary search or context-limit estimate"}
                store.append("run_finished", boundaries=boundaries)
                return {"run_id": run_id, "status": "complete", "boundaries": boundaries}
            for allocator in protocol["allocator_variants"]:
                lower, upper = 0, None
                for length in sorted(protocol["required_lengths"] + protocol["candidate_lengths"]):
                    decision = admission(arch, length, verified.usable_bytes)
                    if not decision["admitted"]:
                        store.append("probe_skipped", allocator=allocator, length=length, reason="analytical admission envelope", **decision)
                        upper = length
                        break
                    if measure_length(length, allocator):
                        lower = length
                    else:
                        upper = length
                        break
                if upper is not None:
                    while (mid := next_bisection(lower, upper, protocol["boundary_resolution_tokens"])) is not None:
                        if not admission(arch, mid, verified.usable_bytes)["admitted"]:
                            upper = mid
                            store.append("probe_skipped", allocator=allocator, length=mid, reason="bisection admission envelope")
                        elif measure_length(mid, allocator):
                            lower = mid
                        else:
                            upper = mid
                successes = [row["request"]["length"] for row in complete.values()
                             if row["status"] == "success" and row["request"]["allocator"] == allocator]
                boundaries[allocator] = {"largest_success_tokens": max(successes, default=None),
                                         "safe_budget_lower_tokens": lower or None, "safe_budget_upper_tokens": upper,
                                         "observed_oom_upper_tokens": None,
                                         "interpretation": "safe-budget interval, not an OOM threshold; config cap may censor"}
                store.append("allocator_boundary", allocator=allocator, **boundaries[allocator])
            if deliberate_length is not None:
                if type(deliberate_length) is not int or not 1 <= deliberate_length <= arch.max_positions:
                    raise ValueError("Deliberate length exceeds the config domain")
                lower = boundaries["default"]["largest_success_tokens"]
                if lower is None or deliberate_length <= lower:
                    raise ValueError("Deliberate point must exceed the largest successful default point")
                row = perform(deliberate_length, "default", 1, "deliberate")
                boundaries["default"]["deliberate_outcome"] = row["status"]
                if row["status"] == "cuda_oom":
                    boundaries["default"]["observed_oom_upper_tokens"] = deliberate_length
                else:
                    boundaries["default"]["largest_success_tokens"] = deliberate_length
            store.append("run_finished", boundaries=boundaries)
            return {"run_id": run_id, "status": "complete", "boundaries": boundaries}
        except CheckpointPause:
            return {"run_id": run_id, "status": "awaiting_external_checkpoint",
                    "completed_probes": len(complete), "boundaries": boundaries,
                    "required_action": "Persist the complete state and logs outside Colab before invoking the next step"}
        except Exception as error:
            store.append("run_stopped", reason=str(error), boundaries=boundaries)
            return {"run_id": run_id, "status": "stopped", "reason": str(error), "boundaries": boundaries}


def check_parent_runtime():
    """Package metadata only; CUDA runtime itself is checked inside each worker."""
    observed = {"python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch": version("torch").split("+")[0], "transformers": version("transformers")}
    if any(value != EXPECTED_RUNTIME[key] for key, value in observed.items()):
        raise RuntimeMismatch("Coordinator runtime differs from frozen policy: " + str(observed))
    return observed


def run_sweep(verified, local_root, **kwargs):
    """Serialize all sessions/allocator variants of one experiment on POSIX storage."""
    check_parent_runtime()
    with file_lock(verified.state_root / ".project-run.lock"):
        return _run_sweep(verified, local_root, **kwargs)
