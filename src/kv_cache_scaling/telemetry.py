"""A2 telemetry. Importing this module never imports torch or initializes CUDA."""

from collections import deque
import os
import threading
import time


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def tensor_storage_report(tensors):
    """Return CPU scalars only; count aliased backing storage once, views in full."""
    seen, rows = set(), []
    logical = unique = 0
    for tensor in tensors:
        key = storage_key(tensor)
        size = int(tensor.numel()) * int(tensor.element_size())
        logical += size
        if key not in seen:
            unique += key[2]
            seen.add(key)
        rows.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                     "device": str(tensor.device), "logical_bytes": size,
                     "storage_bytes": key[2]})
    return {"logical_bytes": logical, "unique_storage_bytes": unique, "tensors": rows}


class NvmlDevice:
    """Resolve physical identity from CUDA's UUID, never from logical index 0."""

    def __init__(self, nvml, cuda_uuid):
        self.api = nvml
        uuid = str(cuda_uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        self.handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        observed = nvml.nvmlDeviceGetUUID(self.handle)
        self.uuid = observed.decode() if isinstance(observed, bytes) else str(observed)
        if self.uuid.lower() != uuid.lower():
            raise ValueError("CUDA/NVML UUID mismatch")

    def identity(self):
        def text(value):
            return value.decode() if isinstance(value, bytes) else str(value)
        return {"uuid": self.uuid, "name": text(self.api.nvmlDeviceGetName(self.handle)),
                "driver_version": text(self.api.nvmlSystemGetDriverVersion())}

    def memory(self):
        info = self.api.nvmlDeviceGetMemoryInfo(self.handle)
        return {"used_bytes": int(info.used), "free_bytes": int(info.free),
                "total_bytes": int(info.total)}

    def processes(self):
        try:
            rows = self.api.nvmlDeviceGetComputeRunningProcesses(self.handle)
            pids = sorted({int(row.pid) for row in rows})
            return {"available": True, "pids": pids,
                    "other_pids": [pid for pid in pids if pid != os.getpid()]}
        except self.api.NVMLError as error:
            return {"available": False, "pids": None, "other_pids": None,
                    "error": str(error)}


def snapshot(device=0, *, torch_module=None, nvml_device=None):
    """Synchronized sequential acquisition, not an instantaneous peak identity."""
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda
    cuda.synchronize(device)
    start = time.monotonic_ns()
    allocated, reserved = int(cuda.memory_allocated(device)), int(cuda.memory_reserved(device))
    free, total = cuda.mem_get_info(device)
    stats = cuda.memory_stats(device)
    driver = nvml_device.memory() if nvml_device is not None else None
    if reserved < allocated:
        raise ValueError("Snapshot reserved bytes are below allocated bytes")
    return {"monotonic_start_ns": start, "monotonic_end_ns": time.monotonic_ns(),
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "allocator_slack_bytes": reserved - allocated,
            "cuda_free_bytes": int(free), "cuda_total_bytes": int(total),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
            "nvml": driver,
            "nvml_minus_reserved_bytes": None if driver is None else driver["used_bytes"] - reserved,
            "driver_residual_attribution": "combined context/external/non-PyTorch; not pure framework overhead"}


class NvmlSampler:
    """Bounded observations; full-run sampled maximum survives ring-buffer eviction."""

    def __init__(self, device, interval_seconds=0.01, max_samples=4096):
        if interval_seconds <= 0 or max_samples < 1:
            raise ValueError("Sampling interval and capacity must be positive")
        self.device, self.interval = device, interval_seconds
        self.samples = deque(maxlen=max_samples)
        self.count, self.maximum, self.error = 0, None, None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        row = {"monotonic_ns": time.monotonic_ns(), **self.device.memory()}
        self.samples.append(row)
        self.count += 1
        self.maximum = max(self.maximum or 0, row["used_bytes"])

    def _loop(self):
        try:
            while not self._stop.wait(self.interval):
                self._sample()
        except Exception as error:
            self.error = {"type": type(error).__name__, "message": str(error)}
            self._stop.set()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Sampler cannot be restarted")
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                self.error = {"type": "SamplerStopTimeout", "message": "NVML reader did not stop"}
        if self.error is None:
            try:
                self._sample()
            except Exception as error:
                self.error = {"type": type(error).__name__, "message": str(error)}
        return {"kind": "sampled_driver_usage_not_exact_peak",
                "requested_interval_seconds": self.interval,
                "sample_count": self.count, "dropped_samples": self.count - len(self.samples),
                "sampled_max_used_bytes": self.maximum, "samples": list(self.samples),
                "error": self.error}
