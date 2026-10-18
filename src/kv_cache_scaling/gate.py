"""CPU-only preregistration verification. Git receipts are auditable, not unforgeable."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from uuid import UUID

from .environment import context_prediction
from .model_math import Architecture, prefill_scenarios, predict_limit, peak_budget_bytes
from .probe import EXPECTED_RUNTIME, ProbeRequest, validate_inputs
from .preregistration import _validate_protocol


class GateError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def git(root, *args):
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)
    if completed.returncode:
        raise GateError("Git evidence unavailable: " + completed.stderr.decode(errors="replace")[:500])
    return completed.stdout


def utc(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as error:
        raise GateError("Expected a timezone-aware timestamp") from error


def contained(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise GateError("Evidence paths must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise GateError("Evidence path escapes repository")
    return path


@dataclass(frozen=True)
class VerifiedRun:
    root: Path
    receipt_path: str
    receipt_sha256: str
    code_commit: str
    prediction_commit: str
    experiment_id: str
    state_root: Path
    gpu_uuid: str
    driver_version: str
    usable_bytes: int
    total_bytes: int
    protocol_sha256: str
    config_sha256: str
    environment_fingerprint: str

    def check_request(self, request):
        if (request.gpu_uuid != self.gpu_uuid or request.protocol_sha256 != self.protocol_sha256
                or request.config_sha256 != self.config_sha256):
            raise GateError("Request differs from verified experiment")
        validate_inputs(request, self.root)


def _verify_receipt(root, receipt_path):
    """No torch, model download or GPU call. Require a clean, currently pushed tree."""
    root = Path(root).resolve()
    receipt_file = contained(root, receipt_path)
    receipt_data = receipt_file.read_bytes()
    receipt = json.loads(receipt_data)
    if receipt.get("status") != "COMPLETE_PREREGISTRATION_RECEIPT":
        raise GateError("Receipt is incomplete; draft/template never authorizes measurements")
    prediction_sha = receipt.get("prediction_commit_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", prediction_sha):
        raise GateError("Invalid prediction commit SHA")
    experiment_id = str(UUID(receipt["experiment_id"]))
    if receipt.get("first_model_measurement_at_utc") is not None:
        raise GateError("Initial receipt must precede the first model measurement")
    observed = utc(receipt["remote_receipt_observed_at_utc"])
    if observed > datetime.now(timezone.utc):
        raise GateError("Receipt observation is in the future")
    roles = set()
    for reviewer in receipt.get("reviewers", []):
        role = reviewer.get("role")
        if role in roles or role not in ("Person 1", "Person 2"):
            raise GateError("Two distinct contributor roles must review")
        if utc(reviewer["reviewed_at_utc"]) > observed:
            raise GateError("Review occurred after claimed remote observation")
        roles.add(role)
    if roles != {"Person 1", "Person 2"}:
        raise GateError("Independent two-contributor review is missing")
    if git(root, "status", "--porcelain", "--untracked-files=normal").strip():
        raise GateError("Code/evidence tree must be clean, including untracked files")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    origin = git(root, "remote", "get-url", "origin").decode().strip()
    match = re.fullmatch(r"(?:https://github.com/|git@github.com:)([\w.-]+/[\w.-]+?)(?:\.git)?", origin)
    if not match:
        raise GateError("Only an explicit GitHub repository origin is supported")
    repo = match[1]
    branch = receipt.get("remote_branch", "main")
    if not isinstance(branch, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", branch) or ".." in branch:
        raise GateError("Invalid remote branch")
    remote = git(root, "ls-remote", "origin", "refs/heads/" + branch).decode().split()
    if len(remote) != 2 or remote[0] != head:
        raise GateError("Current code/receipt commit is not the remote branch head")
    if git(root, "show", f"HEAD:{receipt_path}") != receipt_data:
        raise GateError("Receipt differs from its pushed Git blob")
    receipt_commit = git(root, "log", "-1", "--format=%H", "--", receipt_path).decode().strip()
    if receipt_commit == prediction_sha:
        raise GateError("Receipt must be committed after the prediction")
    git(root, "merge-base", "--is-ancestor", prediction_sha, receipt_commit)
    if receipt.get("remote_commit_url") != f"https://github.com/{repo}/commit/{prediction_sha}":
        raise GateError("Prediction URL does not identify the committed prediction")
    evidence_url = receipt.get("remote_receipt_evidence_url", "")
    if not evidence_url.startswith(f"https://github.com/{repo}/"):
        raise GateError("Remote evidence must link to the same GitHub repository")
    prediction_path = receipt["prediction_path"]
    if not prediction_path.startswith("prereg/") or "/drafts/" in prediction_path:
        raise GateError("Formal prediction must be separately named outside drafts")
    frozen = git(root, "show", f"{prediction_sha}:{prediction_path}")
    if digest(frozen) != receipt.get("prediction_sha256") or contained(root, prediction_path).read_bytes() != frozen:
        raise GateError("Frozen prediction was modified or has the wrong hash")
    prediction = json.loads(frozen)
    if prediction.get("status") != "PREREGISTERED" or prediction.get("missing_evidence") != []:
        raise GateError("Prediction is not a completed formal commitment")
    required_inputs = {"configs/experiment.json", "configs/models/provenance.json",
                       "configs/models/qwen2.5-1.5b.json", "docs/DERIVATION.md"}
    if not required_inputs.issubset(prediction.get("input_sha256", {})):
        raise GateError("Formal prediction is missing scientific input hashes")
    required_code = {"src/kv_cache_scaling/" + name for name in
                     ("probe.py", "telemetry.py", "worker.py", "gate.py", "sweep.py",
                      "coordinator_store.py", "processes.py", "model_math.py", "environment.py", "cli.py", "preregistration.py")}
    if set(prediction.get("execution_sha256", {})) != required_code:
        raise GateError("Formal prediction must freeze all execution modules")
    frozen_inputs = {**prediction["input_sha256"], **prediction["execution_sha256"]}
    for name, expected in frozen_inputs.items():
        data = contained(root, name).read_bytes()
        if digest(data) != expected or git(root, "show", f"{prediction_sha}:{name}") != data:
            raise GateError("Scientific input drift: " + name)
    protocol = (root / "configs/experiment.json").read_bytes()
    config = (root / "configs/models/qwen2.5-1.5b.json").read_bytes()
    for key, actual in (("protocol_sha256", digest(protocol)), ("config_sha256", digest(config))):
        if receipt.get(key) != actual:
            raise GateError("Receipt scientific fingerprint mismatch")
    inventory_path = prediction["context_inventory_path"]
    inventory_data = contained(root, inventory_path).read_bytes()
    if (digest(inventory_data) != prediction.get("context_inventory_sha256")
            or git(root, "show", f"{prediction_sha}:{inventory_path}") != inventory_data):
        raise GateError("Context inventory is not frozen with the prediction")
    arch = Architecture.from_config(json.loads(config))
    experiment = json.loads(protocol)
    _validate_protocol(experiment, arch)
    if experiment["safe_budget_fraction"] != 0.9 or experiment["boundary_resolution_tokens"] != 256:
        raise GateError("Coordinator implements the reviewed 90%/256-token safety policy only")
    inventory = json.loads(inventory_data)
    if any(value is not None for value in inventory.get("allocator_env", {}).values()):
        raise GateError("Frozen context baseline must use the default allocator environment")
    report = context_prediction(inventory, arch)
    if not report["runtime_matches_proposal"] or report["other_gpu_pids_observed"]:
        raise GateError("Runtime mismatch or other GPU users in frozen inventory")
    if prediction.get("runtime") != EXPECTED_RUNTIME:
        raise GateError("Formal runtime policy differs from implemented worker")
    if utc(report["snapshot_timestamp"]) > observed:
        raise GateError("Context snapshot postdates receipt observation")
    scenarios = prefill_scenarios(arch)
    if prediction.get("scenarios") != [asdict(scenario) for scenario in scenarios]:
        raise GateError("Formal prediction must preserve the parametric uncertainty scenarios")
    baseline = prediction.get("gpu_prediction", {})
    expected_range = [predict_limit(arch, report["usable_vram_bytes"], scenarios[2], arch.max_positions),
                      predict_limit(arch, report["usable_vram_bytes"], scenarios[0], arch.max_positions)]
    if baseline.get("limit_range_tokens") != expected_range:
        raise GateError("Formal numerical uncertainty range is missing/inconsistent")
    central = prefill_scenarios(arch)[1]
    required = {str(n): peak_budget_bytes(arch, n, central) for n in (512, 1024, 2048, 4096, 8192)}
    if (baseline.get("usable_vram_bytes") != report["usable_vram_bytes"]
            or baseline.get("central_limit_tokens") != predict_limit(arch, report["usable_vram_bytes"], central, arch.max_positions)
            or baseline.get("central_budget_bytes_at_required_lengths") != required):
        raise GateError("Formal numerical commitment is missing/inconsistent")
    state = Path(receipt["persistent_state_root"])
    if not state.is_absolute() or state.resolve().is_relative_to(root):
        raise GateError("A shared absolute persistent state root outside source is required")
    environment = {"gpu_uuid": report["gpu_uuid"], "driver": report["driver_version"],
                   "runtime": EXPECTED_RUNTIME, "total_bytes": report["cuda_total_bytes"]}
    verified = VerifiedRun(root, receipt_path, digest(receipt_data), head, prediction_sha,
                           experiment_id, state.resolve(), report["gpu_uuid"], report["driver_version"],
                           report["usable_vram_bytes"], report["cuda_total_bytes"], digest(protocol), digest(config),
                           digest(json_bytes(environment)))
    verified.check_request(ProbeRequest("receipt-check", 512, verified.gpu_uuid,
                                       verified.protocol_sha256, verified.config_sha256))
    return verified


def verify_receipt(root, receipt_path):
    try:
        return _verify_receipt(root, receipt_path)
    except GateError:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        raise GateError("Invalid or unavailable preregistration evidence: " + str(error)) from error


def verify_launch(verified, request, run_id):
    """A committed receipt alone cannot bypass durable starts/global attempt state."""
    run_id = str(UUID(run_id))
    run = verified.state_root / "runs" / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("receipt_sha256") != verified.receipt_sha256
            or manifest.get("code_commit") != verified.code_commit
            or manifest.get("environment_fingerprint") != verified.environment_fingerprint):
        raise GateError("Launch manifest differs from verified receipt/code/environment")
    from dataclasses import asdict
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    starts = [row for row in events if row.get("event") == "probe_started" and row.get("probe_id") == request.probe_id]
    if len(starts) != 1 or starts[0].get("request") != asdict(request):
        raise GateError("No matching durable launch event")
    if any(row.get("probe_id") == request.probe_id and row.get("event") in
           ("probe_result", "probe_interrupted") for row in events):
        raise GateError("This worker launch already finished or was interrupted")
    ledger_path = verified.state_root / "global_deliberate_attempt.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    if request.attempt_kind == "deliberate":
        if (ledger is None or ledger.get("state") != "consumed_before_launch"
                or ledger.get("experiment_id") != verified.experiment_id
                or ledger.get("probe_id") != request.probe_id or ledger.get("run_id") != run_id
                or ledger.get("point_key") != starts[0].get("point_key")):
            raise GateError("Deliberate attempt has no matching consumed global token")
    elif ledger and ledger.get("state") in ("blocked_after_accidental_oom", "completed_oom"):
        raise GateError("A recorded OOM blocks further model probes")


def claim_launch(verified, request, run_id):
    """One durable worker claim per started probe; duplicate worker invocations fail."""
    from .coordinator_store import now, sync_dir
    directory = verified.state_root / "runs" / str(UUID(run_id)) / "claims"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (request.probe_id + ".json")
    import os
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"probe_id": request.probe_id, "run_id": run_id,
                                 "receipt_sha256": verified.receipt_sha256, "claimed_at_utc": now()}))
        handle.flush()
        os.fsync(handle.fileno())
    sync_dir(directory)
