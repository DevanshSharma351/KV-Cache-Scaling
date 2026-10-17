"""A2 / Person 1: primary GPU probe; B4 / Person 2: diagnostic helpers.

See docs/MEASUREMENT_PROTOCOL.md for the shared measurement contract.

Implement locally imported torch/pynvml, version/backend preflight, post-load
baseline, peak reset, single prefill, cache storage inspection, synchronized
snapshots, release snapshot, and structured errors. Never generate fake results.
"""


def run_probe(request):
    raise NotImplementedError("A2: GPU probe pending; preregistration required")
