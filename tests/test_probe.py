"""CPU doubles only: these counters are not GPU measurement evidence."""

from contextlib import nullcontext
import hashlib
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
from unittest.mock import patch
import weakref

from kv_cache_scaling import probe
from kv_cache_scaling.model_math import Architecture
from kv_cache_scaling.telemetry import NvmlDevice, NvmlSampler, snapshot, tensor_storage_report

ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-68cbea81-a683-22be-800d-74ff2d51155e"
ARCH = Architecture.from_config(json.loads((ROOT / "configs/models/qwen2.5-1.5b.json").read_text()))


def request(**changes):
    fields = dict(probe_id="cpu-test", length=512, gpu_uuid=UUID,
                  protocol_sha256=hashlib.sha256((ROOT / "configs/experiment.json").read_bytes()).hexdigest(),
                  config_sha256=probe.CONFIG_SHA256)
    fields.update(changes)
    return probe.ProbeRequest(**fields)


class Tensor:
    def __init__(self, shape, pointer=1, size=None, dtype="float16", device="cuda:0"):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.pointer, self.size = pointer, size

    def numel(self):
        from math import prod
        return prod(self.shape)

    def element_size(self):
        return 2 if self.dtype == "float16" else 8

    def untyped_storage(self):
        return NS(data_ptr=lambda: self.pointer,
                  nbytes=lambda: self.size if self.size is not None else self.numel() * self.element_size())

    def to(self, device):
        self.device = device
        return self


class Cache:
    def __init__(self):
        self.key_cache, self.value_cache = [], []

    def get_seq_length(self):
        return self.key_cache[0].shape[2]


class OOM(RuntimeError):
    pass


def fake_runtime(failure=None):
    events, refs = [], {}
    class Cuda:
        initialized = False
        OutOfMemoryError = OOM
        def is_initialized(self): return self.initialized
        def init(self): self.initialized = True; events.append("init")
        def synchronize(self, device=0): events.append("sync")
        def empty_cache(self): events.append("empty_cache")
        def get_allocator_backend(self): return "native"
        def get_device_properties(self, device): return NS(uuid=UUID)
        def reset_peak_memory_stats(self, device): events.append("reset")
        def max_memory_allocated(self, device): return 100
        def max_memory_reserved(self, device): return 120
        def memory_allocated(self, device):
            events.append("snapshot")
            if "out" in refs:
                events.append("out-live" if refs["out"]() else "out-dead")
                events.append("cache-live" if refs["cache"]() else "cache-dead")
            return 80
        def memory_reserved(self, device): return 120
        def mem_get_info(self, device): return (1000, 2000)
        def memory_stats(self, device): return {"inactive_split_bytes.all.current": 4}
    class Model:
        def to(self, device): events.append("move"); return self
        def eval(self): events.append("eval")
        def __call__(self, **kwargs):
            events.append("forward")
            if failure is not None: raise failure
            assert kwargs["logits_to_keep"] == 1 and "attention_mask" not in kwargs
            assert kwargs["use_cache"] and not kwargs["output_attentions"]
            cache = kwargs["past_key_values"]
            cache.key_cache = [Tensor((1, ARCH.kv_heads, 512, ARCH.head_dim), pointer=2*i+1) for i in range(ARCH.layers)]
            cache.value_cache = [Tensor((1, ARCH.kv_heads, 512, ARCH.head_dim), pointer=2*i+2) for i in range(ARCH.layers)]
            class Output: pass
            out = Output()
            out.logits, out.past_key_values = Tensor((1, 1, ARCH.vocab)), cache
            refs["out"], refs["cache"] = weakref.ref(out), weakref.ref(cache)
            return out
    def load(*args, **kwargs):
        events.append("load")
        assert kwargs["revision"] == probe.MODEL_REVISION
        assert kwargs["attn_implementation"] == "sdpa"
        assert not kwargs["trust_remote_code"]
        return Model()
    def randint(low, high, shape, generator):
        events.append("inputs")
        return Tensor(shape, dtype="int64")
    nvml = NS(nvmlInit=lambda: events.append("nvml_init"),
              nvmlShutdown=lambda: events.append("nvml_shutdown"),
              nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
              nvmlDeviceGetUUID=lambda handle: handle,
              nvmlDeviceGetName=lambda handle: b"Tesla T4",
              nvmlSystemGetDriverVersion=lambda: b"test-driver",
              nvmlDeviceGetMemoryInfo=lambda handle: NS(used=150, free=1850, total=2000),
              nvmlDeviceGetComputeRunningProcesses=lambda handle: [], NVMLError=RuntimeError)
    torch = NS(cuda=Cuda(), __version__="2.6.0+cu124", version=NS(cuda="12.4"), float16="float16",
               empty=lambda *a, **k: Tensor((1,)), inference_mode=nullcontext,
               Generator=lambda **k: NS(manual_seed=lambda seed: object()), randint=randint,
               arange=lambda length, **k: Tensor((length,), dtype="int64"))
    rt = NS(torch=torch, nvml=nvml, cache_class=Cache, efficient_context=nullcontext,
            transformers=NS(__version__="4.51.3", AutoModelForCausalLM=NS(from_pretrained=load)))
    return rt, events


class ContractTests(unittest.TestCase):
    def test_diagnostic_grids_and_policies_are_distinct(self):
        for mode, length in (("chunked_dynamic", 32768), ("static_cache", 8192), ("workspace_control", 512)):
            point = request(measurement=mode, length=length)
            self.assertNotEqual(probe.execution_policy(point), probe.POLICY)
            with self.assertRaises(ValueError):
                request(measurement=mode, length=length, attempt_kind="deliberate")
        for mode, length in (("chunked_dynamic",1024),("static_cache",32768),("workspace_control",2048)):
            with self.assertRaises(ValueError): request(measurement=mode,length=length)
        self.assertEqual(probe.execution_policy(request()), probe.POLICY)

    def test_workspace_intervention_is_child_only(self):
        from kv_cache_scaling.processes import worker_env
        with patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", worker_env("default", ROOT))
            self.assertEqual(worker_env("default", ROOT,"workspace_control")["CUBLAS_WORKSPACE_CONFIG"],":0:0")

    def test_diagnostic_scope_cannot_escalate_or_use_deliberate_intent(self):
        self.assertEqual(request(measurement="kernel_trace").length, 512)
        for changes in ({"measurement": "unknown"}, {"measurement": "kernel_trace", "length": 1024},
                        {"measurement": "kernel_trace", "attempt_kind": "deliberate"}):
            with self.assertRaises(ValueError): request(**changes)
    def test_request_and_fingerprints_reject_drift(self):
        self.assertEqual(probe.validate_inputs(request(), ROOT), ARCH)
        for changes in ({"length": True}, {"length": 131073}, {"gpu_uuid": "0"},
                        {"allocator": "math"}, {"repeat": 0}, {"protocol_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): request(**changes)
        with self.assertRaises(ValueError): probe.validate_inputs(request(protocol_sha256="0"*64), ROOT)
        with self.assertRaises(ValueError): probe.ProbeRequest.from_dict({"gpu_measurements_authorized": True})

    def test_allocator_aliases_checked_before_import(self):
        probe.validate_allocator(request(), {})
        probe.validate_allocator(request(allocator="expandable"), {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        for env in ({"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
                    {"PYTORCH_CUDA_ALLOC_CONF": "x", "PYTORCH_ALLOC_CONF": "y"},
                    {"PYTORCH_CUDA_ALLOC_CONF": ""}):
            with self.assertRaises(ValueError): probe.validate_allocator(request(), env)
        with self.assertRaises(ValueError): probe.validate_allocator(request(allocator="expandable"), {})
        with self.assertRaises(ValueError):
            probe.validate_allocator(request(allocator="expandable"), {"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

    def test_result_writer_preserves_failures_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            probe.write_probe_result(path, {"status": "error", "peak_allocated_bytes": None})
            self.assertEqual(json.loads(path.read_text())["status"], "error")
            with self.assertRaises(FileExistsError): probe.write_probe_result(path, {"status": "success"})
            with self.assertRaises(ValueError): probe.write_probe_result(Path(directory) / "nan", {"value": float("nan")})

    def test_cli_validation_is_explicitly_not_authorization(self):
        from kv_cache_scaling.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["validate-probe", "--request", str(ROOT / "configs/probe.example.json"), "--root", str(ROOT)])
        self.assertFalse(json.loads(output.getvalue())["gpu_measurements_authorized"])

    def test_public_gate_never_imports_runtime(self):
        with patch.object(probe, "_import_runtime", side_effect=AssertionError("must not import")):
            with self.assertRaises(probe.PreregistrationRequired): probe.run_probe(request())

    def test_new_colab_stack_is_rejected_before_gpu_imports(self):
        with patch.object(probe, "version", return_value="5.18.0"):
            with self.assertRaises(probe.RuntimeMismatch): probe._import_runtime()


class TelemetryTests(unittest.TestCase):
    def test_storage_deduplicates_views_and_returns_only_metadata(self):
        report = tensor_storage_report([Tensor((2,), size=16), Tensor((3,), size=16)])
        self.assertEqual(report["logical_bytes"], 10)
        self.assertEqual(report["unique_storage_bytes"], 16)
        json.dumps(report)

    def test_uuid_mapping_does_not_assume_nvml_index(self):
        rt, _ = fake_runtime()
        device = NvmlDevice(rt.nvml, UUID.removeprefix("GPU-"))
        self.assertEqual(device.uuid, UUID)
        rt.nvml.nvmlDeviceGetUUID = lambda handle: "GPU-other"
        with self.assertRaises(ValueError): NvmlDevice(rt.nvml, UUID)

    def test_snapshot_records_acquisition_window_and_same_time_slack(self):
        rt, _ = fake_runtime()
        value = snapshot(torch_module=rt.torch)
        self.assertEqual(value["allocator_slack_bytes"], 40)
        self.assertLessEqual(value["monotonic_start_ns"], value["monotonic_end_ns"])
        rt.torch.cuda.memory_reserved = lambda device: 0
        with self.assertRaises(ValueError): snapshot(torch_module=rt.torch)

    def test_sampler_retains_global_sample_max_after_eviction(self):
        values = iter((90, 10, 20))
        sampler = NvmlSampler(NS(memory=lambda: {"used_bytes": next(values)}), max_samples=1)
        sampler._sample(); sampler._sample(); result = sampler.stop()
        self.assertEqual(result["sampled_max_used_bytes"], 90)
        self.assertEqual(result["dropped_samples"], 2)
        self.assertEqual(len(result["samples"]), 1)

    def test_model_inspection_counts_tied_embedding_once_and_checks_dtype(self):
        config = json.loads((ROOT / "configs/models/qwen2.5-1.5b.json").read_text())
        embedding = Tensor((ARCH.vocab, ARCH.hidden), pointer=1)
        rest = Tensor((ARCH.parameter_count() - embedding.numel(),), pointer=2)
        parameters = [embedding, rest]
        model = NS(config=NS(to_dict=lambda: config, _attn_implementation="sdpa", _commit_hash=probe.MODEL_REVISION),
                   parameters=lambda: iter(parameters), buffers=lambda: iter(()),
                   get_input_embeddings=lambda: NS(weight=embedding), get_output_embeddings=lambda: NS(weight=embedding))
        result = probe.inspect_model(model, ARCH, NS(float16="float16"))
        self.assertEqual(result["parameters"]["unique_storage_bytes"], ARCH.weight_bytes())
        self.assertTrue(result["tied_embeddings"])
        rest.dtype = "float32"
        with self.assertRaises(ValueError): probe.inspect_model(model, ARCH, NS(float16="float16"))

    def test_sampler_read_failure_is_not_a_successful_peak(self):
        device = NS(memory=lambda: {"used_bytes": 5})
        sampler = NvmlSampler(device)
        sampler._sample()
        def broken(): raise RuntimeError("driver unavailable")
        device.memory = broken
        result = sampler.stop()
        self.assertEqual(result["sampled_max_used_bytes"], 5)
        self.assertEqual(result["error"]["type"], "RuntimeError")

    def test_cache_validates_shape_dtype_and_storage(self):
        rt, _ = fake_runtime()
        cache = Cache()
        cache.key_cache = [Tensor((1, 2, 512, 128), pointer=2*i+1) for i in range(28)]
        cache.value_cache = [Tensor((1, 2, 512, 128), pointer=2*i+2) for i in range(28)]
        report = probe.inspect_cache(cache, ARCH, 512, rt.torch.float16)
        self.assertEqual(report["unique_storage_bytes"], ARCH.kv_bytes(512))
        json.dumps(report)
        cache.key_cache[0].dtype = "float32"
        with self.assertRaises(ValueError): probe.inspect_cache(cache, ARCH, 512, rt.torch.float16)


class LifecycleTests(unittest.TestCase):
    def run_engine(self, rt):
        with patch.object(probe, "_USED", False), patch.dict(probe.os.environ, {}, clear=True), \
             patch.object(probe, "inspect_model", return_value={"cpu_double": True}):
            return probe._execute_probe(request(), root=ROOT, runtime=rt)

    def test_cold_peak_window_and_release_do_not_retain_outputs_or_cache(self):
        rt, events = fake_runtime()
        result = self.run_engine(rt)
        self.assertEqual(result["status"], "success", result)
        resets = [i for i, event in enumerate(events) if event == "reset"]
        self.assertEqual(len(resets), 2)
        self.assertLess(resets[1], events.index("inputs"))
        self.assertNotIn("empty_cache", events[resets[1]+1:])
        self.assertLess(events.index("forward"), events.index("out-dead"))
        self.assertIn("cache-dead", events)
        self.assertEqual(result["cache"]["logical_bytes"], ARCH.kv_bytes(512))
        self.assertGreaterEqual(result["peak_allocated_bytes"], result["snapshots"]["after_forward"]["allocated_bytes"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(events[-1], "nvml_shutdown")

    def test_kernel_diagnostic_requires_observed_efficient_event(self):
        for names, expected in ((["aten::_scaled_dot_product_efficient_attention"], "success"),
                                (["aten::_scaled_dot_product_attention_math"], "unsupported_backend")):
            rt, _ = fake_runtime()
            class Profile:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def events(self): return [NS(name=n) for n in names]
            rt.torch.profiler = NS(profile=lambda **kwargs: Profile(), ProfilerActivity=NS(CPU="cpu", CUDA="cuda"))
            with patch.object(probe, "_USED", False), patch.dict(probe.os.environ, {}, clear=True), \
                 patch.object(probe, "inspect_model", return_value={"cpu_double": True}):
                result = probe._execute_probe(request(measurement="kernel_trace"), root=ROOT, runtime=rt)
            self.assertEqual(result["status"], expected, result)
            self.assertEqual(result["measurement"], "kernel_trace")
            self.assertEqual(result["kernel_diagnostic"]["event_names"], names)

    def test_oom_and_backend_errors_do_not_continue_lifecycle(self):
        for error, status in ((OOM("failed"), "cuda_oom"), (RuntimeError("No available kernel"), "unsupported_backend"),
                              (ValueError("bad API"), "error"), (TimeoutError("worker deadline"), "timeout")):
            with self.subTest(status=status):
                rt, events = fake_runtime(error)
                result = self.run_engine(rt)
                self.assertEqual(result["status"], status, result)
                self.assertNotIn("retained", result["snapshots"])
                self.assertIsNone(result["peak_allocated_bytes"])
                self.assertIsNotNone(result["failure_counters"])
                self.assertEqual(events.count("forward"), 1)

    def test_identity_drift_aborts_before_model_load(self):
        rt, events = fake_runtime()
        rt.torch.cuda.get_device_properties = lambda device: NS(uuid="GPU-00000000-0000-0000-0000-000000000000")
        result = self.run_engine(rt)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("load", events)

    def test_unavailable_process_attribution_aborts_before_model_load(self):
        rt, events = fake_runtime()
        def unavailable(handle): raise RuntimeError("NVML process list unavailable")
        rt.nvml.nvmlDeviceGetComputeRunningProcesses = unavailable
        result = self.run_engine(rt)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["processes"]["available"])
        self.assertNotIn("load", events)

    def test_unsupported_allocator_warning_stops_before_load(self):
        import warnings
        rt, events = fake_runtime()
        def empty(*args, **kwargs):
            warnings.warn("expandable_segments not supported on this platform")
            return Tensor((1,))
        rt.torch.empty = empty
        result = self.run_engine(rt)
        self.assertEqual(result["status"], "unsupported_allocator")
        self.assertNotIn("load", events)

    def test_second_point_cannot_reuse_worker(self):
        rt, events = fake_runtime()
        with patch.object(probe, "_USED", True):
            result = probe._execute_probe(request(), root=ROOT, runtime=rt)
        self.assertEqual(result["status"], "error")
        self.assertEqual(events, [])

    def test_existing_cuda_context_is_rejected(self):
        rt, events = fake_runtime()
        rt.torch.cuda.initialized = True
        result = self.run_engine(rt)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("load", events)
