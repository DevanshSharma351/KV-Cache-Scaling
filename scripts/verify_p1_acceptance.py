"""CPU-only verification of a saved P1 acceptance capsule; never executes CUDA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kv_cache_scaling.model_math import Architecture
from kv_cache_scaling.probe import ProbeRequest
from kv_cache_scaling.sweep import validate_result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, run_id):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        manifest = json.loads(archive.read("MANIFEST.json"))
        require(set(names) == set(manifest) | {"MANIFEST.json"}, "Manifest coverage mismatch")
        for name, expected in manifest.items():
            require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    "Unsafe archive path")
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, "Hash mismatch: " + name)
        prefix = f"state/runs/{run_id}/"
        rows = [json.loads(line) for line in archive.read(prefix + "events.jsonl").splitlines()]
        require([r["sequence"] for r in rows] == list(range(1, len(rows) + 1)), "Journal sequence gap")
        require(all(r["run_id"] == run_id for r in rows), "Journal identity mismatch")
        require(rows[-1]["event"] == "run_finished", "Acceptance run is not finished")
        meta = json.loads(archive.read(prefix + "manifest.json"))
        require(meta["execution_scope"] == "acceptance_l512" and meta["checkpoint_each_probe"], "Wrong scope")
        require(meta["deliberate_length"] is None, "Unexpected deliberate intent")
        require("state/global_deliberate_attempt.json" not in names, "Real OOM ledger unexpectedly exists")
        receipt = json.loads((ROOT / "prereg/formal-2026-10-18-t4/receipt.json").read_text())
        prediction = json.loads((ROOT / receipt["prediction_path"]).read_text())
        for name, expected in {**prediction["input_sha256"], **prediction["execution_sha256"]}.items():
            blob=subprocess.check_output(["git","-C",str(ROOT),"show",receipt["prediction_commit_sha"]+":"+name])
            require(hashlib.sha256(blob).hexdigest() == expected, "Historical frozen blob drift: " + name)
        arch = Architecture.from_config(json.loads((ROOT / "configs/models/qwen2.5-1.5b.json").read_text()))
        inventory = json.loads((ROOT / prediction["context_inventory_path"]).read_text())
        expected_uuid = inventory["gpu_identity_csv"].split(",")[0]
        observed_receipt = datetime.fromisoformat(receipt["remote_receipt_observed_at_utc"]).timestamp()
        starts = [r for r in rows if r["event"] == "probe_started"]
        results = [r for r in rows if r["event"] == "probe_result"]
        idle = {r["probe_id"] for r in rows if r["event"] == "idle_verified"}
        require(len(starts) == len(results) == len(idle) == 7, "Expected seven completed fresh workers")
        points, pids, reports, kernels = set(), set(), [], []
        for row in results:
            request = ProbeRequest.from_dict(row["request"])
            r = json.loads(archive.read(prefix + f"results/{request.probe_id}.json"))
            with tempfile.TemporaryDirectory() as historical:
                historical_root=Path(historical)
                for name in ("src/kv_cache_scaling/probe.py","src/kv_cache_scaling/telemetry.py","configs/models/qwen2.5-1.5b.json"):
                    target=historical_root/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(subprocess.check_output(["git","-C",str(ROOT),"show",receipt["prediction_commit_sha"]+":"+name]))
                validate_result(r, request, arch, historical_root)
            require(r["status"] == row["status"] == "success", "Unsuccessful acceptance result")
            require(request.gpu_uuid == expected_uuid, "GPU drift")
            require(request.protocol_sha256 == receipt["protocol_sha256"] and
                    request.config_sha256 == receipt["config_sha256"], "Scientific fingerprint drift")
            require(r["started_unix_ns"] / 1e9 > observed_receipt, "Probe predates pushed receipt")
            require(r["weights"]["parameters"]["unique_storage_bytes"] == arch.weight_bytes(), "Weight storage mismatch")
            require(r["cache"]["unique_storage_bytes"] == arch.kv_bytes(512), "Cache mismatch")
            logits = r["logits"]["tensors"]
            require(len(logits) == 1 and logits[0]["shape"] == [1, 1, arch.vocab] and
                    logits[0]["dtype"] == "torch.float16", "Wrong logits")
            require("torchao" not in {k.lower() for k in r["runtime"]["package_versions"]}, "torchao unexpectedly installed")
            require(r["snapshots"]["context_only"]["allocated_bytes"] == 0, "Nonempty context baseline")
            point = (request.measurement, request.allocator, request.repeat)
            require(point not in points and r["pid"] not in pids, "Duplicate point/process")
            points.add(point); pids.add(r["pid"])
            if request.measurement == "kernel_trace":
                kernels = r["kernel_diagnostic"]["event_names"]
            reports.append({"probe_id": r["probe_id"], "pid": r["pid"], "measurement": request.measurement,
                "allocator": request.allocator, "repeat": request.repeat,
                "peak_allocated_bytes": r["peak_allocated_bytes"], "peak_reserved_bytes": r["peak_reserved_bytes"],
                "kv_bytes": r["cache"]["unique_storage_bytes"],
                "retained_minus_released_bytes": r["snapshots"]["retained"]["allocated_bytes"] - r["snapshots"]["released"]["allocated_bytes"]})
        expected_points = {("primary", a, i) for a in ("default", "expandable") for i in (1, 2, 3)} | {("kernel_trace", "default", 1)}
        require(points == expected_points, "Acceptance grid mismatch")
        restore = json.loads(archive.read("audit/external-checkpoint-restore.json"))
        require(restore["status"] == "PASS" and restore["both_colab_checkpoint_trees_removed"], "Missing external restore proof")
        return {"status": "PASS", "scope": "Person 1 Phases 1–3 bounded acceptance; not bulk research results",
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "verified_members": len(manifest),
            "run_id": run_id, "gpu_uuid": expected_uuid, "prediction_commit": receipt["prediction_commit_sha"],
            "code_commit": meta["code_commit"], "unique_worker_pids": len(pids), "deliberate_attempt_used": False,
            "cloud_session_restart_tested": False, "external_restore": restore,
            "kernel_evidence": [n for n in kernels if "efficient_attention" in n or "fmha" in n.lower() or "cutlass" in n.lower()],
            "probes": sorted(reports, key=lambda r: (r["measurement"], r["allocator"], r["repeat"]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--run-id", default="4353f74e-31b3-4574-93b0-7b0dc2a2beb2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.archive, args.run_id)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(rendered)
    print(rendered)
