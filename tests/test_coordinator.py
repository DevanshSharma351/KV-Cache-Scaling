"""Synthetic CPU workers/receipts only. No network, CUDA, models or research data."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from kv_cache_scaling import gate
from kv_cache_scaling.coordinator_store import AttemptLedger, RunStore, file_lock
from kv_cache_scaling.model_math import Architecture, prefill_scenarios, predict_limit, peak_budget_bytes
from kv_cache_scaling.probe import EXPECTED_RUNTIME, POLICY, ProbeRequest, execution_policy
from kv_cache_scaling.processes import launch_worker, worker_env
from kv_cache_scaling.sweep import admission, next_bisection, point_key, run_sweep

ROOT = Path(__file__).resolve().parents[1]
CODE = ("probe.py", "telemetry.py", "worker.py", "gate.py", "sweep.py", "coordinator_store.py",
        "processes.py", "model_math.py", "environment.py", "cli.py", "preregistration.py")
INPUTS = ("configs/experiment.json", "configs/models/provenance.json", "configs/models/qwen2.5-1.5b.json", "docs/DERIVATION.md")


class GitFixture:
    def __init__(self, base):
        self.root, self.state = Path(base) / "source", Path(base) / "persistent"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.root)], check=True)
        self.git("config", "user.name", "Synthetic fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("remote", "add", "origin", "https://github.com/synthetic-fixture/kv-cache.git")
        for name in INPUTS + tuple("src/kv_cache_scaling/" + name for name in CODE):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.inventory = json.loads((ROOT / "prereg/environment/t4-context-2026-10-18.user.json").read_text())
        self.inventory.update(python="3.11.0 synthetic", torch="2.6.0+cu124", torch_cuda_runtime="12.4",
                              transformers="4.51.3", timestamp_utc="2020-01-01T00:00:00+00:00",
                              collection_kind="SYNTHETIC TEST FIXTURE; NOT A GPU OBSERVATION")
        self.write("prereg/context.json", self.inventory)
        arch = Architecture.from_config(json.loads((self.root / INPUTS[2]).read_text()))
        scenario = prefill_scenarios(arch)[1]
        self.prediction = {"status": "PREREGISTERED", "missing_evidence": [], "runtime": EXPECTED_RUNTIME,
                           "scenarios": [asdict(s) for s in prefill_scenarios(arch)],
                           "input_sha256": {name: gate.digest((self.root / name).read_bytes()) for name in INPUTS},
                           "execution_sha256": {"src/kv_cache_scaling/"+name: gate.digest((self.root / "src/kv_cache_scaling" / name).read_bytes()) for name in CODE},
                           "context_inventory_path": "prereg/context.json",
                           "context_inventory_sha256": gate.digest((self.root / "prereg/context.json").read_bytes()),
                           "gpu_prediction": {"usable_vram_bytes": self.inventory["cuda_free_bytes"],
                             "limit_range_tokens": [predict_limit(arch,self.inventory["cuda_free_bytes"],prefill_scenarios(arch)[i],arch.max_positions) for i in (2,0)],
                             "central_limit_tokens": predict_limit(arch, self.inventory["cuda_free_bytes"], scenario, arch.max_positions),
                             "central_budget_bytes_at_required_lengths": {str(n): peak_budget_bytes(arch, n, scenario) for n in (512,1024,2048,4096,8192)}}}
        self.write("prereg/formal.json", self.prediction)
        self.commit("synthetic prediction")
        self.prediction_sha = self.git("rev-parse", "HEAD").decode().strip()
        self.receipt = {"status": "COMPLETE_PREREGISTRATION_RECEIPT", "experiment_id": str(uuid4()),
                        "prediction_path": "prereg/formal.json", "prediction_commit_sha": self.prediction_sha,
                        "prediction_sha256": gate.digest((self.root / "prereg/formal.json").read_bytes()),
                        "persistent_state_root": str(self.state), "remote_branch": "main",
                        "remote_commit_url": "https://github.com/synthetic-fixture/kv-cache/commit/" + self.prediction_sha,
                        "remote_receipt_evidence_url": "https://github.com/synthetic-fixture/kv-cache/actions/runs/123",
                        "remote_receipt_observed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol_sha256": gate.digest((self.root / INPUTS[0]).read_bytes()),
                        "config_sha256": gate.digest((self.root / INPUTS[2]).read_bytes()),
                        "reviewers": [{"role": role, "reviewed_at_utc": "2020-01-01T00:00:00+00:00"} for role in ("Person 1", "Person 2")],
                        "first_model_measurement_at_utc": None}
        self.write("prereg/receipt.json", self.receipt)
        self.commit("synthetic subsequent receipt")

    def git(self, *args):
        return gate.git(self.root, *args)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-q", "-m", message)

    def verify(self, remote_sha=None):
        real_git = gate.git
        head = self.git("rev-parse", "HEAD").decode().strip()
        def synthetic_remote(root, *args):
            if args[0] == "ls-remote":
                return ((remote_sha or head) + "\trefs/heads/main\n").encode()
            return real_git(root, *args)
        with patch.object(gate, "git", side_effect=synthetic_remote):
            return gate.verify_receipt(self.root, "prereg/receipt.json")


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = GitFixture(self.directory.name)
    def tearDown(self): self.directory.cleanup()

    def test_real_git_ancestry_and_frozen_bytes_with_synthetic_remote(self):
        verified = self.fixture.verify()
        self.assertEqual(verified.usable_bytes, self.fixture.inventory["cuda_free_bytes"])
        self.assertNotEqual(verified.code_commit, verified.prediction_commit)

    def test_incomplete_review_and_draft_status_are_rejected(self):
        for changes in ({"status": "INCOMPLETE_TEMPLATE_NOT_A_RECEIPT"}, {"reviewers": []}):
            with self.subTest(changes=changes):
                value = {**self.fixture.receipt, **changes}
                self.fixture.write("prereg/receipt.json", value)
                with self.assertRaises(gate.GateError): self.fixture.verify()

    def test_single_actual_contributor_review_is_allowed(self):
        self.fixture.receipt["reviewers"] = self.fixture.receipt["reviewers"][:1]
        self.fixture.write("prereg/receipt.json", self.fixture.receipt)
        self.fixture.commit("synthetic single review policy")
        self.assertEqual(self.fixture.verify().gpu_uuid, self.fixture.inventory["gpu_identity_csv"].split(",")[0])

    def test_neutral_review_labels_preserve_historical_receipt_compatibility(self):
        self.fixture.receipt["reviewers"]=[{"role":"Project reviewer","reviewed_at_utc":"2020-01-01T00:00:00+00:00"}]
        self.fixture.write("prereg/receipt.json",self.fixture.receipt)
        self.fixture.commit("synthetic project review label")
        self.assertEqual(self.fixture.verify().gpu_uuid,self.fixture.inventory["gpu_identity_csv"].split(",")[0])

    def test_unpushed_head_and_dirty_source_are_rejected(self):
        with self.assertRaises(gate.GateError): self.fixture.verify(remote_sha="0"*40)
        (self.fixture.root / "unexpected.py").write_text("# untracked\n")
        with self.assertRaises(gate.GateError): self.fixture.verify()

    def test_changed_formal_prediction_is_not_refrozen_by_later_commit(self):
        self.fixture.prediction["gpu_prediction"]["central_limit_tokens"] += 1
        self.fixture.write("prereg/formal.json", self.fixture.prediction)
        self.fixture.commit("tampered prediction")
        with self.assertRaises(gate.GateError): self.fixture.verify()

    def test_request_and_launch_need_matching_durable_start(self):
        verified = self.fixture.verify()
        req = ProbeRequest(str(uuid4()), 512, verified.gpu_uuid, verified.protocol_sha256, verified.config_sha256)
        run_id = str(uuid4())
        with RunStore(Path(self.directory.name)/"local", verified.state_root, run_id) as store:
            store.manifest({"receipt_sha256": verified.receipt_sha256, "code_commit": verified.code_commit,
                            "environment_fingerprint": verified.environment_fingerprint})
            with self.assertRaises(gate.GateError): gate.verify_launch(verified, req, run_id)
            store.append("probe_started", probe_id=req.probe_id, point_key="key", request=asdict(req))
            gate.verify_launch(verified, req, run_id)
            gate.claim_launch(verified, req, run_id)
            with self.assertRaises(FileExistsError): gate.claim_launch(verified, req, run_id)
            deliberate = replace(req, probe_id=str(uuid4()), attempt_kind="deliberate")
            store.append("probe_started", probe_id=deliberate.probe_id, point_key="deliberate", request=asdict(deliberate))
            with self.assertRaises(gate.GateError): gate.verify_launch(verified, deliberate, run_id)
            AttemptLedger(verified.state_root, verified.experiment_id).consume(run_id, deliberate.probe_id, "deliberate")
            gate.verify_launch(verified, deliberate, run_id)
            store.append("probe_result", probe_id=deliberate.probe_id)
            with self.assertRaises(gate.GateError): gate.verify_launch(verified, deliberate, run_id)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.run_id = str(uuid4())
    def tearDown(self): self.directory.cleanup()
    def store(self): return RunStore(self.base/"local", self.base/"durable", self.run_id)

    def test_lock_and_immutable_mirror(self):
        with self.store() as store:
            store.manifest({"fixture": True})
            store.artifact("logs/one", b"original")
            with self.assertRaises(ValueError): store.artifact("logs/one", b"replacement")
            with self.assertRaises(ValueError): store.artifact("../escape", b"no")
            with self.assertRaises(RuntimeError):
                with self.store(): pass
            self.assertEqual((store.local/"logs/one").read_bytes(), (store.mirror/"logs/one").read_bytes())

    def test_torn_suffix_is_preserved_and_interior_corruption_is_rejected(self):
        with self.store() as store:
            store.append("first")
            with (store.local/"events.jsonl").open("ab") as handle: handle.write(b'{"torn":')
        with self.store() as store:
            self.assertEqual(store.events[-1]["event"], "journal_recovered")
            self.assertEqual(next((store.mirror/"recovery").iterdir()).read_bytes(), b'{"torn":')
            (store.local/"events.jsonl").write_bytes(b"corrupt\n")
            (store.mirror/"events.jsonl").write_bytes(b"corrupt\n")
        with self.assertRaises(ValueError):
            with self.store(): pass

    def test_volatile_loss_restores_started_event_and_artifacts(self):
        with self.store() as store:
            store.artifact("logs/partial", b"partial")
            store.append("probe_started", probe_id="orphan", point_key="point")
            local = store.local
        shutil.rmtree(local)
        with self.store() as store:
            self.assertEqual(store.orphaned()[0]["probe_id"], "orphan")
            self.assertEqual((store.local/"logs/partial").read_bytes(), b"partial")

    def test_global_token_is_consumed_before_launch_and_shared_across_runs(self):
        experiment = str(uuid4())
        ledger = AttemptLedger(self.base, experiment)
        ledger.consume(self.run_id, "one", "point")
        with self.assertRaises(RuntimeError): AttemptLedger(self.base, experiment).consume(str(uuid4()), "two", "other")
        with self.assertRaises(RuntimeError): AttemptLedger(self.base, str(uuid4())).consume(str(uuid4()), "different-experiment", "other")
        ledger.finish("one", "success")
        with self.assertRaises(RuntimeError): ledger.consume(self.run_id, "three", "third")
        self.assertEqual(json.loads(ledger.path.read_text())["state"], "completed_success")


class ProcessTests(unittest.TestCase):
    def test_fresh_processes_bounded_logs_and_timeout_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            env = worker_env("default", ROOT)
            one = launch_worker([sys.executable, "-c", "import os; print(os.getpid()); print('x'*10000)"], env=env, cwd=directory, timeout=5, log_limit=32)
            two = launch_worker([sys.executable, "-c", "import os; print(os.getpid())"], env=env, cwd=directory, timeout=5)
            self.assertNotEqual(one["pid"], two["pid"])
            self.assertTrue(one["logs"]["stdout"]["truncated"])
            self.assertEqual(len(one["stdout"]), 32)
            timeout = launch_worker([sys.executable, "-c", "import time; time.sleep(10)"], env=env, cwd=directory, timeout=0.05)
            self.assertEqual(timeout["process_status"], "timeout")
            self.assertNotEqual(timeout["returncode"], 0)

    def test_coordinator_import_is_cpu_only_in_a_fresh_interpreter(self):
        command = [sys.executable, "-c", "import sys,json; import kv_cache_scaling.sweep; print(json.dumps({k:k in sys.modules for k in ('torch','pynvml')}))"]
        output = launch_worker(command, env=worker_env("default",ROOT), cwd=ROOT, timeout=5)
        self.assertEqual(json.loads(output["stdout"]), {"torch":False,"pynvml":False})

    def test_allocator_env_is_set_only_in_child_mapping(self):
        with patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": "wrong", "PYTORCH_CUDA_ALLOC_CONF": "wrong"}):
            default = worker_env("default", ROOT)
            expanded = worker_env("expandable", ROOT)
            self.assertNotIn("PYTORCH_ALLOC_CONF", default)
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", default)
            self.assertEqual(expanded["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(os.environ["PYTORCH_ALLOC_CONF"], "wrong")


class SyntheticRunner:
    """Writes CPU-only fake records in TemporaryDirectory for scheduling tests."""
    def __init__(self, status="success"):
        self.status, self.calls = status, []
    def __call__(self, command, **kwargs):
        request = ProbeRequest.from_dict(json.loads(Path(command[command.index("--request")+1]).read_text()))
        self.calls.append(request)
        snap = {"allocated_bytes": 1, "reserved_bytes": 2, "cuda_free_bytes": 10, "cuda_total_bytes": 20}
        arch = Architecture.from_config(json.loads((Path(kwargs["cwd"])/INPUTS[2]).read_text()))
        result = {"schema_version": "primary-probe-v1", **asdict(request), "status": self.status,
                  "execution_policy": execution_policy(request), "runtime": {"torch": "2.6.0+cu124", "cuda": "12.4", "transformers": "4.51.3"},
                  "code_sha256": {name: gate.digest((Path(kwargs["cwd"])/"src/kv_cache_scaling"/name).read_bytes()) for name in ("probe.py", "telemetry.py")},
                  "oom_classification": "accidental", "error": {"type": "OutOfMemoryError"},
                  "snapshots": {key: snap for key in ("context_only", "post_load", "after_forward", "retained", "released")},
                  "peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "cache": {"logical_bytes": arch.kv_bytes(request.length), "unique_storage_bytes": arch.kv_bytes(request.length)},
                  "weights": {"unique_parameters": arch.parameter_count()}, "sampling": {"error": None}}
        if request.measurement == "kernel_trace":
            result["kernel_diagnostic"] = {"event_names": ["aten::_scaled_dot_product_efficient_attention"]}
        Path(command[command.index("--output")+1]).write_text(json.dumps(result))
        return {"process_status": "exited", "returncode": 0 if self.status == "success" else 1,
                "stdout": b"synthetic CPU fixture", "stderr": b"", "logs": {"stdout": {}, "stderr": {}}}


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = GitFixture(self.directory.name)
        self.verified = self.fixture.verify()
        self.observer = type("SyntheticObserver", (), {"read": lambda obj: {"uuid": self.verified.gpu_uuid,
                                  "driver_version": self.verified.driver_version, "total_bytes": 100, "used_bytes": 0}})()
    def tearDown(self): self.directory.cleanup()
    def run_fake(self, runner, **kwargs):
        if kwargs.get('confirmation_only') and 'external_attempt_sha256' not in kwargs:
            kwargs['external_attempt_sha256']='0'*64  # Explicitly synthetic dispatch receipt.
        with patch("kv_cache_scaling.sweep.check_parent_runtime", return_value={"synthetic":True}):
            return run_sweep(self.verified, Path(self.directory.name)/"local", _runner=runner, _observer=self.observer, **kwargs)

    def test_safe_sweep_bisection_and_resume_never_repeat_completed_points(self):
        runner = SyntheticRunner()
        result = self.run_fake(runner)
        self.assertEqual(result["status"], "complete", result)
        self.assertTrue(all(admission(Architecture.from_config(json.loads((self.fixture.root/INPUTS[2]).read_text())), req.length, self.verified.usable_bytes)["admitted"] for req in runner.calls))
        self.assertEqual(len({(r.allocator,r.length,r.repeat) for r in runner.calls}), len(runner.calls))
        for boundary in result["boundaries"].values():
            self.assertLessEqual(boundary["safe_budget_upper_tokens"] - boundary["safe_budget_lower_tokens"], 256)
            self.assertIsNone(boundary["observed_oom_upper_tokens"])
        count = len(runner.calls)
        resumed = self.run_fake(runner, run_id=result["run_id"])
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(count, len(runner.calls))

    def test_acceptance_run_is_bounded_separate_resumable_and_never_uses_token(self):
        runner = SyntheticRunner()
        result = self.run_fake(runner, acceptance_only=True)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(len(runner.calls), 7)
        self.assertTrue(all(r.length == 512 and r.attempt_kind == "safe" for r in runner.calls))
        self.assertEqual(sum(r.measurement == "kernel_trace" for r in runner.calls), 1)
        self.assertEqual(len({point_key(self.verified, r.length, r.allocator, r.repeat,
                                      r.attempt_kind, r.measurement) for r in runner.calls}), 7)
        self.assertFalse(AttemptLedger(self.verified.state_root, self.verified.experiment_id).path.exists())
        self.assertEqual(self.run_fake(runner, run_id=result["run_id"], acceptance_only=True)["status"], "complete")
        self.assertEqual(len(runner.calls), 7)
        with self.assertRaises(ValueError):
            self.run_fake(runner, run_id=result["run_id"])
        with self.assertRaises(ValueError):
            self.run_fake(runner, acceptance_only=True, deliberate_length=131072)

    def test_acceptance_checkpoint_barrier_runs_one_new_probe_per_invocation(self):
        runner = SyntheticRunner()
        run_id = str(uuid4())
        for count in range(1, 8):
            result = self.run_fake(runner, run_id=run_id, acceptance_only=True, checkpoint_each_probe=True)
            self.assertEqual(result["status"], "awaiting_external_checkpoint", result)
            self.assertEqual(result["completed_probes"], count)
            self.assertEqual(len(runner.calls), count)
        result = self.run_fake(runner, run_id=run_id, acceptance_only=True, checkpoint_each_probe=True)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(len(runner.calls), 7)

    def test_bulk_checkpoint_barrier_preserves_completed_keys(self):
        runner=SyntheticRunner();rid=str(uuid4())
        first=self.run_fake(runner,run_id=rid,checkpoint_each_probe=True)
        self.assertEqual(first['status'],'awaiting_external_checkpoint')
        self.assertEqual(len(runner.calls),1)
        second=self.run_fake(runner,run_id=rid,checkpoint_each_probe=True)
        self.assertEqual(second['completed_probes'],2)
        self.assertEqual(len(runner.calls),2)
        self.assertEqual([r.repeat for r in runner.calls],[1,2])
        self.assertFalse(AttemptLedger(self.verified.state_root,self.verified.experiment_id).path.exists())

    def test_diagnostic_grids_have_distinct_scope_and_no_deliberate_token(self):
        runner=SyntheticRunner()
        result=self.run_fake(runner,diagnostics_only=True)
        self.assertEqual(result['status'],'complete',result)
        self.assertEqual(len(runner.calls),28)
        self.assertEqual(sum(r.measurement=='chunked_dynamic' for r in runner.calls),12)
        self.assertEqual(sum(r.measurement=='static_cache' for r in runner.calls),9)
        self.assertEqual(sum(r.measurement=='workspace_control' for r in runner.calls),6)
        self.assertTrue(all(r.attempt_kind=='safe' for r in runner.calls))
        self.assertTrue(all(r.allocator=='default' for r in runner.calls if r.measurement!='workspace_control'))
        self.assertFalse(AttemptLedger(self.verified.state_root,self.verified.experiment_id).path.exists())
        with self.assertRaises(ValueError):self.run_fake(runner,diagnostics_only=True,acceptance_only=True)

    def test_confirmation_only_uses_finished_sweep_and_one_global_attempt(self):
        safe=SyntheticRunner();reference=self.run_fake(safe)
        runner=SyntheticRunner()
        result=self.run_fake(runner,confirmation_only=True,reference_run_id=reference['run_id'],
                             deliberate_length=131072,checkpoint_each_probe=True)
        self.assertEqual(result['status'],'complete',result)
        self.assertEqual(len(runner.calls),1)
        self.assertEqual(runner.calls[0].attempt_kind,'deliberate')
        self.assertEqual(result['boundaries']['default']['deliberate_outcome'],'success')
        again=self.run_fake(runner,confirmation_only=True,reference_run_id=reference['run_id'],deliberate_length=131072)
        self.assertEqual(again['status'],'needs_review');self.assertEqual(len(runner.calls),1)

    def test_confirmation_rejects_incomplete_or_changed_reference(self):
        from kv_cache_scaling.sweep import confirmation_reference
        runner=SyntheticRunner();rid=str(uuid4())
        self.run_fake(runner,run_id=rid,checkpoint_each_probe=True)
        with self.assertRaises(ValueError):confirmation_reference(self.verified,rid)
        reference=self.run_fake(runner)
        directory=self.verified.state_root/'runs'/reference['run_id']
        manifest=json.loads((directory/'manifest.json').read_text());manifest['gpu_uuid']='changed'
        (directory/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):confirmation_reference(self.verified,rid)
        with self.assertRaises(ValueError):self.run_fake(runner,confirmation_only=True,deliberate_length=131072)

    def test_confirmation_oom_finishes_before_external_export_and_blocks_retry(self):
        reference=self.run_fake(SyntheticRunner())
        runner=SyntheticRunner('cuda_oom')
        def deliberate(command,**kwargs):
            outcome=runner(command,**kwargs)
            path=Path(command[command.index('--output')+1]);raw=json.loads(path.read_text())
            raw['oom_classification']='deliberate_request';path.write_text(json.dumps(raw))
            return outcome
        result=self.run_fake(deliberate,confirmation_only=True,reference_run_id=reference['run_id'],
                             deliberate_length=131072,checkpoint_each_probe=True)
        self.assertEqual(result['status'],'complete',result)
        self.assertEqual(result['boundaries']['default']['observed_oom_upper_tokens'],131072)
        events=[json.loads(line) for line in (self.verified.state_root/'runs'/result['run_id']/'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['event'],'run_finished')
        retried=self.run_fake(deliberate,confirmation_only=True,reference_run_id=reference['run_id'],deliberate_length=131072)
        self.assertEqual(retried['status'],'needs_review');self.assertEqual(len(runner.calls),1)

    def test_held_out_validation_is_admitted_from_reference_then_can_anchor_confirmation(self):
        from kv_cache_scaling.sweep import confirmation_reference
        runner=SyntheticRunner()
        def shaped(command,**kwargs):
            outcome=runner(command,**kwargs)
            path=Path(command[command.index('--output')+1]);raw=json.loads(path.read_text())
            raw['peak_allocated_bytes']=3_000_000_000+90_000*raw['length']
            raw['peak_reserved_bytes']=raw['peak_allocated_bytes']+2*1024**2
            path.write_text(json.dumps(raw));return outcome
        reference=self.run_fake(shaped)
        count=len(runner.calls)
        rejected=self.run_fake(shaped,validation_only=True,validation_length=131072,reference_run_id=reference['run_id'])
        self.assertEqual(rejected['status'],'stopped');self.assertEqual(len(runner.calls),count)
        validation=self.run_fake(shaped,validation_only=True,validation_length=114688,reference_run_id=reference['run_id'])
        self.assertEqual(validation['status'],'complete',validation)
        self.assertEqual(len(runner.calls)-count,6)
        self.assertFalse(AttemptLedger(self.verified.state_root,self.verified.experiment_id).path.exists())
        checked=confirmation_reference(self.verified,validation['run_id'])
        self.assertEqual(checked['boundaries']['default']['largest_success_tokens'],114688)
        confirmed=self.run_fake(shaped,confirmation_only=True,deliberate_length=131072,reference_run_id=validation['run_id'])
        self.assertEqual(confirmed['status'],'complete',confirmed)
        self.assertEqual(len(runner.calls)-count,7)

    def test_accidental_oom_stops_escalation_and_blocks_global_attempt(self):
        runner = SyntheticRunner("cuda_oom")
        result = self.run_fake(runner)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(len(runner.calls), 1)
        ledger = AttemptLedger(self.verified.state_root, self.verified.experiment_id)
        self.assertEqual(json.loads(ledger.path.read_text())["state"], "blocked_after_accidental_oom")
        self.assertEqual(self.run_fake(runner, run_id=result["run_id"])["status"], "needs_review")
        self.assertEqual(len(runner.calls), 1)

    def test_interrupted_start_is_reviewed_instead_of_retried(self):
        runner = SyntheticRunner()
        run_id = str(uuid4())
        with RunStore(Path(self.directory.name)/"local", self.verified.state_root, run_id) as store:
            store.append("probe_started", probe_id="interrupted", point_key="orphan")
        result = self.run_fake(runner, run_id=run_id)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(runner.calls, [])

    def test_deliberate_success_consumes_token_and_is_not_called_oom(self):
        runner = SyntheticRunner()
        result = self.run_fake(runner, deliberate_length=131072)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(sum(req.attempt_kind == "deliberate" for req in runner.calls), 1)
        ledger = AttemptLedger(self.verified.state_root, self.verified.experiment_id)
        self.assertEqual(json.loads(ledger.path.read_text())["state"], "completed_success")
        boundary = result["boundaries"]["default"]
        self.assertEqual(boundary["largest_success_tokens"], 131072)
        self.assertIsNone(boundary["observed_oom_upper_tokens"])
        count = len(runner.calls)
        self.assertEqual(self.run_fake(runner, deliberate_length=131072)["status"], "needs_review")
        self.assertEqual(len(runner.calls), count)

    def test_deliberate_token_and_durable_start_precede_launch(self):
        original = SyntheticRunner()
        def inspecting(command, **kwargs):
            req = json.loads(Path(command[command.index("--request")+1]).read_text())
            run_id = command[command.index("--run-id")+1]
            events = [json.loads(line) for line in (self.verified.state_root/"runs"/run_id/"events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "probe_started")
            self.assertEqual(events[-1]["request"], req)
            if req["attempt_kind"] == "deliberate":
                ledger = AttemptLedger(self.verified.state_root, self.verified.experiment_id)
                self.assertEqual(json.loads(ledger.path.read_text())["state"], "consumed_before_launch")
            return original(command, **kwargs)
        self.assertEqual(self.run_fake(inspecting, deliberate_length=131072)["status"], "complete")

    def test_failed_start_mirror_prevents_launch(self):
        from kv_cache_scaling import coordinator_store
        original = coordinator_store.atomic_bytes
        def broken(path, data):
            if str(path).startswith(str(self.verified.state_root)) and b'"event":"probe_started"' in data:
                raise OSError("synthetic mirror outage")
            return original(path, data)
        runner = SyntheticRunner()
        with patch.object(coordinator_store, "atomic_bytes", side_effect=broken):
            with self.assertRaises(OSError): self.run_fake(runner)
        self.assertEqual(runner.calls, [])

    def test_deliberate_oom_is_an_observed_upper_point_not_an_exact_threshold(self):
        runner = SyntheticRunner()
        def deliberate_only(command, **kwargs):
            req = json.loads(Path(command[command.index("--request")+1]).read_text())
            runner.status = "cuda_oom" if req["attempt_kind"] == "deliberate" else "success"
            result = runner(command, **kwargs)
            if runner.status == "cuda_oom":
                path = Path(command[command.index("--output")+1])
                value = json.loads(path.read_text())
                value["oom_classification"] = "deliberate_request"
                path.write_text(json.dumps(value))
            return result
        result = self.run_fake(deliberate_only, deliberate_length=131072)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["boundaries"]["default"]["observed_oom_upper_tokens"], 131072)
        self.assertLess(result["boundaries"]["default"]["largest_success_tokens"], 131072)

    def test_missing_worker_json_is_not_inferred_as_oom(self):
        def no_json(command, **kwargs):
            return {"process_status":"exited", "returncode":1, "stdout":b"CUDA out of memory in a text log", "stderr":b"", "logs":{"stdout":{},"stderr":{}}}
        result = self.run_fake(no_json)
        self.assertEqual(result["status"], "stopped")
        self.assertIn("invalid_output", result["reason"])
        self.assertFalse(AttemptLedger(self.verified.state_root,self.verified.experiment_id).path.exists())

    def test_manifest_identity_change_cannot_resume(self):
        runner = SyntheticRunner()
        result = self.run_fake(runner)
        self.verified = replace(self.verified, environment_fingerprint="changed")
        with self.assertRaises(ValueError): self.run_fake(runner, run_id=result["run_id"])

    def test_point_identity_includes_repeat_allocator_and_intent(self):
        keys = {point_key(self.verified,512,a,r,k) for a in ("default","expandable") for r in (1,2) for k in ("safe","deliberate")}
        self.assertEqual(len(keys), 8)
        self.assertIsNone(next_bisection(512, 768))
