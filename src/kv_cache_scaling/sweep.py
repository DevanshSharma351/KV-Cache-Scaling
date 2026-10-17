"""A3 / Devansh: CPU coordinator; never import torch in this process.

Launch sys.executable -m ...probe in a new subprocess per point/repeat. Supply
allocator env before process start; enforce prereg receipt, lock, fsync journal,
resume fingerprints, safety envelope, timeouts, and one global OOM attempt token.
"""


def run_sweep(experiment, journal):
    raise NotImplementedError("A3: isolated sweep and boundary estimation pending")
