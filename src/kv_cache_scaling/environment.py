"""CPU validation of context-only inventory and conditional budget calculations.

This is not an execution gate: a valid inventory never authorizes model probes.
The older proposed runtime is retained until compatibility is reconciled.
"""

import csv
from datetime import datetime
import io
import re
from uuid import UUID

from .model_math import GIB, predict_limit, prefill_scenarios, positive_int

PROPOSED_RUNTIME = {
    "python_minor": "3.11",
    "torch": "2.6.0",
    "torch_cuda_runtime": "12.4",
    "transformers": "4.51.3",
}


def context_prediction(snapshot, architecture):
    """Compute from exact free bytes; retain provenance and runtime uncertainty.

    The result applies the original shape scenarios to a reported budget. It
    does not claim that their execution/activation assumptions hold on a new stack.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a JSON object")
    for field in ("cuda_free_bytes", "cuda_total_bytes", "current_pid"):
        positive_int(snapshot.get(field), field)
    free, total = snapshot["cuda_free_bytes"], snapshot["cuda_total_bytes"]
    if free > total:
        raise ValueError("CUDA free bytes cannot exceed total bytes")
    for field in ("allocated_bytes", "reserved_bytes"):
        if type(snapshot.get(field)) is not int or snapshot[field] != 0:
            raise ValueError("Context-only inventory requires zero allocated/reserved bytes")
    try:
        timestamp = datetime.fromisoformat(snapshot["timestamp_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("snapshot must contain an ISO timestamp with timezone") from error
    if timestamp.utcoffset() is None:
        raise ValueError("snapshot timestamp needs an explicit timezone")
    for field in ("python", "torch", "torch_cuda_runtime", "transformers", "gpu_name"):
        if not isinstance(snapshot.get(field), str) or not snapshot[field].strip():
            raise ValueError(f"Missing runtime field: {field}")
    rows = list(csv.reader(io.StringIO(snapshot.get("gpu_identity_csv", "")), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("Inventory adapter currently requires exactly one identified GPU")
    gpu_uuid, gpu_name, driver = (value.strip() for value in rows[0])
    try:
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError("Missing GPU UUID prefix")
        UUID(gpu_uuid[4:])
    except ValueError as error:
        raise ValueError("Invalid GPU UUID") from error
    if gpu_name != snapshot["gpu_name"] or not re.fullmatch(r"\d+(?:\.\d+)+", driver):
        raise ValueError("GPU identity/name or driver is inconsistent")
    allocator = snapshot.get("allocator_env")
    if not isinstance(allocator, dict) or not all(
            key in allocator and (allocator[key] is None or isinstance(allocator[key], str))
            for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
        raise ValueError("Both allocator environment settings must be recorded")
    process_rows = list(csv.reader(io.StringIO(snapshot.get("gpu_processes", "")), skipinitialspace=True))
    if not process_rows:
        raise ValueError("Expected the context-owning process in the driver observation")
    pids = []
    for row in process_rows:
        if len(row) != 3 or not row[0].strip().isdigit():
            raise ValueError("Cannot interpret GPU process inventory")
        pids.append(int(row[0].strip()))
    if snapshot["current_pid"] not in pids:
        raise ValueError("Context-owning PID is absent from driver process inventory")
    other_pids = sorted(set(pids) - {snapshot["current_pid"]})
    observed = {
        "python_minor": ".".join(snapshot["python"].split()[0].split(".")[:2]),
        "torch": snapshot["torch"].split("+")[0],
        "torch_cuda_runtime": snapshot["torch_cuda_runtime"],
        "transformers": snapshot["transformers"],
    }
    differences = {key: {"proposed": expected, "observed": observed[key]}
                   for key, expected in PROPOSED_RUNTIME.items() if observed[key] != expected}
    limits = []
    for scenario in prefill_scenarios(architecture):
        uncapped = predict_limit(architecture, free, scenario)
        limits.append({
            "scenario": scenario.name,
            "uncapped_limit_tokens": uncapped,
            "config_capped_limit_tokens": min(uncapped, architecture.max_positions),
            "quality_domain_limit_tokens": min(uncapped, 32768),
        })
    return {
        "status": "CONDITIONAL_THEORY_NOT_PREREGISTERED",
        "gpu_measurements_authorized": False,
        "evidence_origin": "caller-supplied context inventory; not independently authenticated",
        "gpu_uuid": gpu_uuid, "gpu_name": gpu_name, "driver_version": driver,
        "snapshot_timestamp": snapshot["timestamp_utc"],
        "usable_vram_bytes": free, "usable_vram_gib": free / GIB,
        "cuda_total_bytes": total, "cuda_total_minus_free_bytes": total - free,
        "budget_policy": "use observed free bytes directly; context is already deducted",
        "other_gpu_pids_observed": other_pids,
        "observed_runtime": observed, "proposed_runtime": dict(PROPOSED_RUNTIME),
        "runtime_matches_proposal": not differences, "runtime_differences": differences,
        "scenario_limits": limits,
        "interpretation": "Original analytical assumptions applied to exact free bytes; not validation of a changed runtime.",
        "remaining_gates": ["runtime/API/backend reconciliation", "independent derivation review",
                            "formal prediction push", "subsequent remote receipt"],
    }
