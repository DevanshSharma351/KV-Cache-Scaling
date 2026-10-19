"""Cold single-prefill engine, guarded by prospective receipt verification.

Only the receipt-checking coordinator may call _execute_probe, once in a
fresh interpreter. CPU tests inject doubles; they are never research records.
"""

from dataclasses import dataclass
from contextlib import nullcontext
import gc
import hashlib
import inspect
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace
import warnings
from itertools import chain

from .model_math import Architecture, positive_int
from .telemetry import NvmlDevice, NvmlSampler, snapshot, storage_key, tensor_storage_report

MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
CONFIG_SHA256 = "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77"
EXPECTED_RUNTIME = {"python_minor": "3.11", "torch": "2.6.0", "cuda": "12.4", "transformers": "4.51.3"}
POLICY = {"protocol_version": "prefill-v1", "model_id": MODEL_ID,
          "model_revision": MODEL_REVISION, "dtype": "float16", "batch_size": 1,
          "mode": "single_prefill", "attention_implementation": "sdpa",
          "sdpa_backend": "EFFICIENT_ATTENTION", "cache_type": "DynamicCache",
          "logits_to_keep": 1, "seed": 42, "warmup": False, "allocator_backend": "native"}
_USED = False


def execution_policy(request):
    policy = dict(POLICY)
    if request.measurement == "chunked_dynamic":
        policy.update(mode="chunked_prefill", chunk_size=512)
    elif request.measurement == "static_cache":
        policy.update(mode="static_prefill", cache_type="StaticCache", capacity=request.length)
    elif request.measurement == "workspace_control":
        policy.update(diagnostic="workspace_control", cublas_workspace_config=":0:0")
    return policy


class PreregistrationRequired(RuntimeError):
    pass


class UnsupportedBackend(RuntimeError):
    pass


class UnsupportedAllocator(RuntimeError):
    pass


class EnvironmentDrift(RuntimeError):
    pass


class RuntimeMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeRequest:
    probe_id: str
    length: int
    gpu_uuid: str
    protocol_sha256: str
    config_sha256: str
    allocator: str = "default"
    repeat: int = 1
    attempt_kind: str = "safe"
    measurement: str = "primary"

    def __post_init__(self):
        if self.measurement not in ("primary", "kernel_trace", "chunked_dynamic", "static_cache", "workspace_control"):
            raise ValueError("Unknown measurement scope")
        if self.measurement in ("kernel_trace", "workspace_control") and (self.length != 512 or self.attempt_kind != "safe"):
            raise ValueError("Kernel diagnostics are restricted to safe L=512")
        if self.measurement == "chunked_dynamic" and (self.length not in (512,2048,8192,32768) or self.attempt_kind != "safe"):
            raise ValueError("Chunked diagnostic differs from its registered grid")
        if self.measurement == "static_cache" and (self.length not in (512,2048,8192) or self.attempt_kind != "safe"):
            raise ValueError("Static diagnostic differs from its registered grid")
        if not isinstance(self.probe_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.probe_id):
            raise ValueError("Invalid probe_id")
        if positive_int(self.length, "length") > 131072:
            raise ValueError("Length exceeds the primary config stress domain")
        positive_int(self.repeat, "repeat")
        if self.attempt_kind not in ("safe", "deliberate"):
            raise ValueError("Invalid attempt intent; intent never grants execution authorization")
        if self.allocator not in ("default", "expandable"):
            raise ValueError("Unsupported allocator variant")
        if not isinstance(self.gpu_uuid, str) or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", self.gpu_uuid):
            raise ValueError("A full physical GPU UUID is required")
        for value in (self.protocol_sha256, self.config_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Scientific fingerprints must be SHA-256 hex strings")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Request must be a JSON object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("Missing or unknown request fields") from error


def validate_inputs(request, root):
    root = Path(root)
    protocol_bytes = (root / "configs/experiment.json").read_bytes()
    config_bytes = (root / "configs/models/qwen2.5-1.5b.json").read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != request.protocol_sha256:
        raise ValueError("Protocol fingerprint mismatch")
    if hashlib.sha256(config_bytes).hexdigest() != request.config_sha256 or request.config_sha256 != CONFIG_SHA256:
        raise ValueError("Config fingerprint mismatch")
    protocol, config = json.loads(protocol_bytes), json.loads(config_bytes)
    for key, expected in POLICY.items():
        if type(protocol.get(key)) is not type(expected) or protocol.get(key) != expected:
            raise ValueError(f"Unsupported primary policy: {key}")
    return Architecture.from_config(config)


def validate_allocator(request, environ):
    """Called before CUDA/import; never mutate allocator env in this process."""
    names = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    values = [environ.get(name) for name in names]
    if values[0] is not None and values[1] is not None and values[0] != values[1]:
        raise ValueError("Conflicting allocator aliases")
    expected = None if request.allocator == "default" else "expandable_segments:True"
    for value in values:
        if value is not None and value != expected:
            raise ValueError("Allocator env differs from the requested variant")
    if request.allocator == "expandable" and values[0] != expected:
        raise ValueError("Set PYTORCH_CUDA_ALLOC_CONF before starting the pinned 2.6 worker interpreter")
    return dict(zip(names, values))


def inspect_cache(cache, arch, length, fp16):
    if len(cache.key_cache) != arch.layers or len(cache.value_cache) != arch.layers:
        raise ValueError("Wrong cache layer count")
    if int(cache.get_seq_length()) != length:
        raise ValueError("Cache populated length differs from request")
    tensors = []
    expected = (1, arch.kv_heads, length, arch.head_dim)
    for key, value in zip(cache.key_cache, cache.value_cache):
        for tensor in (key, value):
            if tuple(tensor.shape) != expected or tensor.dtype != fp16 or str(tensor.device) != "cuda:0":
                raise ValueError("Wrong cache shape, dtype or device")
            tensors.append(tensor)
    report = tensor_storage_report(tensors)
    report["expected_logical_bytes"] = arch.kv_bytes(length)
    if report["logical_bytes"] != arch.kv_bytes(length):
        raise ValueError("Cache logical byte count differs from analytical formula")
    report["storage_formula_relative_error"] = abs(report["unique_storage_bytes"] / arch.kv_bytes(length) - 1)
    if report["storage_formula_relative_error"] > 0.01:
        raise ValueError("Cache backing storage differs from formula by more than 1%")
    return report


def inspect_model(model, arch, torch):
    if Architecture.from_config(model.config.to_dict()) != arch:
        raise ValueError("Loaded architecture differs from pinned config")
    if model.config._attn_implementation != "sdpa":
        raise ValueError("Loaded model did not select SDPA")
    if getattr(model.config, "_commit_hash", None) != MODEL_REVISION:
        raise ValueError("Loaded model config revision is not the pinned SHA")
    count = 0
    for parameter in model.parameters():
        if parameter.dtype != torch.float16 or str(parameter.device) != "cuda:0":
            raise ValueError("Model must be wholly on cuda:0 in FP16")
        count += int(parameter.numel())
    report = tensor_storage_report(model.parameters())
    if count != arch.parameter_count() or report["unique_storage_bytes"] != arch.weight_bytes():
        raise ValueError("Loaded unique parameters/storage disagree with analytical weights")
    tied = storage_key(model.get_input_embeddings().weight) == storage_key(model.get_output_embeddings().weight)
    if tied != arch.tied_embeddings:
        raise ValueError("Embedding storage tie differs from config")
    return {"unique_parameters": count, "parameters": report, "tied_embeddings": tied,
            "buffers": tensor_storage_report(model.buffers()),
            "all_model_storage": tensor_storage_report(chain(model.parameters(), model.buffers()))}


def _import_runtime():
    # Metadata checks precede heavy imports; the observed new Colab stack is refused.
    observed = {"python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch": version("torch").split("+")[0], "transformers": version("transformers")}
    if any(observed[key] != EXPECTED_RUNTIME[key] for key in observed):
        raise RuntimeMismatch(f"Pinned runtime required: {EXPECTED_RUNTIME}; observed {observed}")
    import torch
    if torch.version.cuda != EXPECTED_RUNTIME["cuda"]:
        raise RuntimeMismatch("Pinned torch CUDA runtime required")
    import transformers
    import pynvml
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    sources = {}
    for name, obj in (("qwen2_class", Qwen2ForCausalLM), ("dynamic_cache", transformers.DynamicCache),
                      ("qwen2_forward", inspect.unwrap(Qwen2ForCausalLM.forward)),
                      ("sdpa_adapter", sdpa_attention_forward)):
        path = Path(inspect.getsourcefile(obj))
        sources[name] = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return SimpleNamespace(torch=torch, transformers=transformers, nvml=pynvml,
                           cache_class=transformers.DynamicCache, source_hashes=sources,
                           static_cache_class=transformers.StaticCache,
                           efficient_context=lambda: sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))


def classify_error(error, stage, torch=None):
    if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
        return "cuda_oom"
    if isinstance(error, UnsupportedBackend):
        return "unsupported_backend"
    if isinstance(error, UnsupportedAllocator):
        return "unsupported_allocator"
    if isinstance(error, EnvironmentDrift):
        return "environment_drift"
    if isinstance(error, RuntimeMismatch):
        return "runtime_mismatch"
    if isinstance(error, TimeoutError):
        return "timeout"
    # Kernel failures remain errors, never OOM inferred from text alone.
    if stage == "forward" and any(marker in str(error).lower() for marker in
                                  ("no available kernel", "no viable backend", "no suitable kernel")):
        return "unsupported_backend"
    return "error"


def reject_unsupported_allocator(caught):
    if any("expandable_segments" in str(item.message).lower() and "not supported" in str(item.message).lower()
           for item in caught):
        raise UnsupportedAllocator("expandable_segments not supported; refusing a native fallback run")


def _execute_probe(request, *, root=".", runtime=None, verified=None):
    """Internal engine for coordinator integration; NOT a preregistration authorization API.

    Caller must persist start/token events and enforce receipts before invoking.
    Call once per fresh process. No retries, no warmup, no diagnostics here.
    """
    global _USED
    result = {"schema_version": "primary-probe-v1", "probe_id": request.probe_id,
              "length": request.length, "repeat": request.repeat, "allocator": request.allocator,
              "execution_policy": execution_policy(request),
              "context_domain": "within_model_card_limit" if request.length <= 32768 else "synthetic_stress_above_model_card_limit",
              "protocol_sha256": request.protocol_sha256, "config_sha256": request.config_sha256,
              "status": "incomplete", "stage": "preflight", "snapshots": {},
              "attempt_kind": request.attempt_kind, "oom_classification": None,
              "measurement": request.measurement,
              "peak_allocated_bytes": None, "peak_reserved_bytes": None,
              "weights": None, "cache": None, "sampling": None, "warnings": [],
              "pid": os.getpid(), "started_unix_ns": time.time_ns(),
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("probe.py", "telemetry.py")}}
    rt, nvml, sampler = None, None, None
    initialized_nvml = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if _USED:
                raise ValueError("Worker process already used; launch a fresh interpreter")
            _USED = True
            arch = validate_inputs(request, root)
            result["allocator_env"] = validate_allocator(request, os.environ)
            expected_workspace = ":0:0" if request.measurement == "workspace_control" else None
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_workspace:
                raise ValueError("cuBLAS workspace environment differs from measurement scope")
            rt = runtime or _import_runtime()
            torch, cuda = rt.torch, rt.torch.cuda
            if cuda.is_initialized():
                raise ValueError("Worker inherited an initialized CUDA context")
            result["runtime"] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                                 "torch": torch.__version__, "cuda": torch.version.cuda,
                                 "transformers": rt.transformers.__version__,
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                                 "source_files": getattr(rt, "source_hashes", {}),
                                 "package_versions": {dist.metadata["Name"]: dist.version for dist in distributions()
                                                      if dist.metadata["Name"]}}
            result["stage"] = "context"
            cuda.init()
            tiny = torch.empty(1, device="cuda:0")
            cuda.synchronize(0)
            del tiny
            cuda.empty_cache()
            if cuda.get_allocator_backend() != "native":
                raise ValueError("Only native allocator is supported")
            result["allocator_backend"] = "native"
            rt.nvml.nvmlInit()
            initialized_nvml = True
            nvml = NvmlDevice(rt.nvml, cuda.get_device_properties(0).uuid)
            if nvml.uuid.lower() != request.gpu_uuid.lower():
                raise ValueError("GPU identity drift")
            result["gpu_uuid"] = nvml.uuid
            result["gpu_identity"] = nvml.identity()
            result["processes"] = nvml.processes()
            if not result["processes"]["available"]:
                raise ValueError("GPU process attribution is unavailable")
            if result["processes"]["other_pids"]:
                raise ValueError("Other GPU processes are present")
            snap = lambda: snapshot(0, torch_module=torch, nvml_device=nvml)
            result["snapshots"]["context_only"] = snap()
            if verified is not None:
                baseline = result["snapshots"]["context_only"]
                if (abs(baseline["cuda_free_bytes"] - verified.usable_bytes) > 64 * 1024**2
                        or baseline["cuda_total_bytes"] != verified.total_bytes
                        or result["gpu_identity"]["driver_version"] != verified.driver_version):
                    raise EnvironmentDrift("Context budget/total/driver differs from frozen baseline")
            reject_unsupported_allocator(caught)
            result["stage"] = "load"
            cuda.reset_peak_memory_stats(0)
            result["active_peak_scope"] = "model_load"
            model = rt.transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.float16, attn_implementation="sdpa")
            model.to("cuda:0")
            model.eval()
            cuda.synchronize(0)
            result["load_peaks"] = {"allocated_bytes": int(cuda.max_memory_allocated(0)),
                                    "reserved_bytes": int(cuda.max_memory_reserved(0))}
            result["weights"] = inspect_model(model, arch, torch)
            gc.collect()
            cuda.empty_cache()
            result["snapshots"]["post_load"] = snap()
            result["post_load_accounting"] = {
                "allocated_minus_analytical_weights_bytes":
                    result["snapshots"]["post_load"]["allocated_bytes"] - arch.weight_bytes(),
                "interpretation": "includes live model buffers and tracked non-parameter storage; not pure activation memory"}
            reject_unsupported_allocator(caught)
            result["stage"] = "inputs"
            cuda.reset_peak_memory_stats(0)
            result["active_peak_scope"] = "prefill_inputs_through_forward"
            sampler = NvmlSampler(nvml)
            sampler.start()
            with torch.inference_mode():
                generator = torch.Generator(device="cpu").manual_seed(42)
                ids = torch.randint(0, arch.vocab, (1, request.length), generator=generator).to("cuda:0")
                positions = torch.arange(request.length, device="cuda:0")
                cache = (rt.static_cache_class(config=model.config,max_batch_size=1,max_cache_len=request.length,
                                              device="cuda:0",dtype=torch.float16)
                         if request.measurement == "static_cache" else rt.cache_class())
                result["input_logical_bytes"] = int(ids.numel() * ids.element_size() + positions.numel() * positions.element_size())
                result["stage"] = "forward"
                profile = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA])
                           if request.measurement == "kernel_trace" else nullcontext())
                with profile as trace, rt.efficient_context():
                    if request.measurement == "chunked_dynamic":
                        for start in range(0,request.length,512):
                            end = min(start+512,request.length)
                            # Explicit absolute-position causal bias for a chunk with a past cache.
                            bias = torch.full((end-start,end),torch.finfo(torch.float16).min,
                                              device="cuda:0",dtype=torch.float16)
                            bias.masked_fill_(positions[:end].unsqueeze(0) <= positions[start:end].unsqueeze(1),0)
                            out = model(input_ids=ids[:,start:end],cache_position=positions[start:end],
                                        attention_mask=bias.unsqueeze(0).unsqueeze(0),past_key_values=cache,
                                        use_cache=True,logits_to_keep=1,output_attentions=False,
                                        output_hidden_states=False,return_dict=True)
                            del bias
                            if end < request.length:
                                del out
                    else:
                        out = model(input_ids=ids, cache_position=positions, past_key_values=cache,
                                    use_cache=True, logits_to_keep=1, output_attentions=False,
                                    output_hidden_states=False, return_dict=True)
                cuda.synchronize(0)
                result["peak_allocated_bytes"] = int(cuda.max_memory_allocated(0))
                result["peak_reserved_bytes"] = int(cuda.max_memory_reserved(0))
                if request.measurement == "kernel_trace":
                    names = sorted({event.name for event in trace.events()})
                    result["kernel_diagnostic"] = {"event_names": names,
                        "interpretation": "profiled diagnostic; exclude these peaks from primary fits"}
                    if not any("_scaled_dot_product_efficient_attention" in name for name in names):
                        raise UnsupportedBackend("Profiler did not observe efficient SDPA")
                result["snapshots"]["after_forward"] = snap()
                result["stage"] = "inspection"
                if tuple(out.logits.shape) != (1, 1, arch.vocab) or out.logits.dtype != torch.float16:
                    raise ValueError("Wrong reduced logits shape/dtype")
                result["logits"] = tensor_storage_report([out.logits])
                if out.past_key_values is not cache:
                    raise ValueError("Forward returned a different cache object")
                result["cache"] = inspect_cache(cache, arch, request.length, torch.float16)
                del out, ids, positions
                result["snapshots"]["retained"] = snap()
                del cache
                result["snapshots"]["released"] = snap()
            if result["peak_allocated_bytes"] < result["snapshots"]["after_forward"]["allocated_bytes"]:
                raise ValueError("Peak does not cover live allocation")
            if result["peak_reserved_bytes"] < result["snapshots"]["after_forward"]["reserved_bytes"]:
                raise ValueError("Reservation peak does not cover the live reservation")
            result["status"], result["stage"] = "success", "complete"
        except Exception as error:
            result["status"] = classify_error(error, result["stage"], rt.torch if rt else None)
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            if result["status"] == "cuda_oom":
                result["oom_classification"] = "deliberate_request" if request.attempt_kind == "deliberate" else "accidental"
            if rt is not None and rt.torch.cuda.is_initialized():
                try:
                    result["failure_counters"] = {
                        "allocated_bytes": int(rt.torch.cuda.memory_allocated(0)),
                        "reserved_bytes": int(rt.torch.cuda.memory_reserved(0))}
                    if result.get("active_peak_scope"):
                        result["failure_counters"].update({
                            "partial_peak_allocated_bytes": int(rt.torch.cuda.max_memory_allocated(0)),
                            "partial_peak_reserved_bytes": int(rt.torch.cuda.max_memory_reserved(0)),
                            "scope": result["active_peak_scope"]})
                except Exception as counter_error:
                    result["failure_counter_error"] = str(counter_error)
        finally:
            if sampler is not None:
                result["sampling"] = sampler.stop()
                result["sampling"]["scope"] = "before inputs through retained/released snapshots; separate from exact forward peak scope"
                if result["sampling"]["error"] and result["status"] == "success":
                    result["status"] = "telemetry_error"
            result["warnings"] = [str(item.message) for item in caught]
            if any("expandable_segments" in message and "not supported" in message.lower()
                   for message in result["warnings"]):
                if result["status"] == "success":
                    result["status"] = "unsupported_allocator"
            if initialized_nvml:
                try:
                    rt.nvml.nvmlShutdown()
                except Exception as error:
                    result["nvml_shutdown_error"] = str(error)
            result["finished_unix_ns"] = time.time_ns()
    return result


def run_probe(request, *, root=".", receipt_path=None, run_id=None):
    """Recheck pushed receipt and durable launch before any GPU import."""
    if receipt_path is None or run_id is None:
        raise PreregistrationRequired("A pushed reviewed receipt and durable coordinator launch are required")
    from .gate import claim_launch, verify_receipt, verify_launch
    verified = verify_receipt(root, receipt_path)
    verified.check_request(request)
    verify_launch(verified, request, run_id)
    claim_launch(verified, request, run_id)
    return _execute_probe(request, root=root, verified=verified)


def write_probe_result(path, result):
    """Exclusive, fsynced worker result; parent must reject incomplete/torn JSON.

    The coordinator owns durable start events, process termination, journal mirroring and
    interpretation of absent output. This writer cannot make timeouts into OOMs.
    """
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
