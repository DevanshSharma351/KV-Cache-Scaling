"""A2 / Person 1: synchronized snapshots and sampled driver telemetry adapter.

Implement allocated/reserved/free, inactive_split_bytes, CUDA/NVML identity,
monotonic timestamps, peak counters and storage-deduplicated tensor bytes.
NVML sampling cannot establish an exact transient driver peak; label it sampled.
"""


def snapshot(device):
    raise NotImplementedError("A2: CUDA + NVML telemetry adapter pending")
