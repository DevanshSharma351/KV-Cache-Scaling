"""B2 / Teammate: append-only persistence and global deliberate-attempt ledger.

Implement the docs/DATA_CONTRACT.md lifecycle, file lock, flush/fsync, durable
mirroring, torn-line recovery and point-key resume. Use CPU synthetic workers for
fault tests. Do not implement consume-on-success: token consumption precedes launch.
"""


def append_event(path, event):
    raise NotImplementedError("B2: validated durable journal pending")
