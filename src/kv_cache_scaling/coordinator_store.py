"""Shared POSIX persistence: locked runs, durable mirrors and one-use ledger.

Local fsync cannot prove a cloud provider's durability. Actual external storage
must be validated before treating an ephemeral mirror as a durable checkpoint.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID, uuid4

from .gate import digest, json_bytes


def now():
    return datetime.now(timezone.utc).isoformat()


def sync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value) + b"\n")


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another coordinator holds the run/ledger lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class RunStore:
    def __init__(self, local_root, persistent_root, run_id):
        self.run_id = str(UUID(run_id))
        self.local = Path(local_root).resolve() / self.run_id
        self.mirror = Path(persistent_root).resolve() / "runs" / self.run_id
        if self.local == self.mirror or self.local.is_relative_to(self.mirror) or self.mirror.is_relative_to(self.local):
            raise ValueError("Local and mirror trees must be distinct")
        self.events = []
        self._locked = False

    def __enter__(self):
        self._lock = file_lock(self.mirror / ".lock")
        self._lock.__enter__()
        self._locked = True
        try:
            self.local.mkdir(parents=True, exist_ok=True)
            # Restore lost local files; existing immutable artifacts must agree.
            for path in self.mirror.rglob("*"):
                if path.is_file() and path.name != ".lock" and not path.name.startswith(".pending-"):
                    relative = path.relative_to(self.mirror)
                    target = self.local / relative
                    if not target.exists():
                        atomic_bytes(target, path.read_bytes())
                    elif relative != Path("events.jsonl") and target.read_bytes() != path.read_bytes():
                        raise ValueError("Immutable local/mirror artifact diverged: " + str(relative))
            self._recover_events()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self._locked = False
        self._lock.__exit__(*args)

    def _assert_lock(self):
        if not self._locked:
            raise RuntimeError("Acquire RunStore before writing")

    def artifact(self, relative, data):
        self._assert_lock()
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.name == ".lock":
            raise ValueError("Artifact must stay within the run")
        for base in (self.local, self.mirror):
            path = base / relative
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Refusing to overwrite an immutable artifact: " + str(relative))
            if not path.exists():
                atomic_bytes(path, data)
        return {"path": str(relative), "sha256": digest(data), "bytes": len(data)}

    def manifest(self, value):
        return self.artifact("manifest.json", json_bytes(value) + b"\n")

    def _recover_events(self):
        self._assert_lock()
        primary, mirror = self.local / "events.jsonl", self.mirror / "events.jsonl"
        data = primary.read_bytes() if primary.exists() else b""
        durable = mirror.read_bytes() if mirror.exists() else b""
        if durable.startswith(data):
            data = durable
        elif not data.startswith(durable):
            raise ValueError("Local and durable journals diverged")
        tail = b""
        if data and not data.endswith(b"\n"):
            index = data.rfind(b"\n") + 1
            data, tail = data[:index], data[index:]
        rows = []
        for index, line in enumerate(data.splitlines(), 1):
            value = json.loads(line)
            if value.get("sequence") != index or value.get("run_id") != self.run_id:
                raise ValueError("Interior journal corruption or run identity mismatch")
            rows.append(value)
        if tail:
            self.artifact("recovery/torn-" + str(uuid4()) + ".bin", tail)
        atomic_bytes(primary, data)
        atomic_bytes(mirror, data)
        self.events = rows
        if tail:
            self.append("journal_recovered", torn_bytes=len(tail), torn_sha256=digest(tail))

    def append(self, kind, **fields):
        self._assert_lock()
        if any(name in fields for name in ("sequence", "run_id", "event", "timestamp_utc")):
            raise ValueError("Reserved journal field")
        row = {"sequence": len(self.events) + 1, "run_id": self.run_id,
               "event": kind, "timestamp_utc": now(), **fields}
        payload = json_bytes(row) + b"\n"
        with (self.local / "events.jsonl").open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(row)
        # No next launch is allowed until this durable mirror succeeds.
        atomic_bytes(self.mirror / "events.jsonl", (self.local / "events.jsonl").read_bytes())
        return row

    def orphaned(self):
        started = {row["probe_id"]: row for row in self.events if row["event"] == "probe_started"}
        finished = {row["probe_id"] for row in self.events if row["event"] in ("probe_result", "probe_interrupted")}
        return [value for key, value in started.items() if key not in finished]


class AttemptLedger:
    """One project-wide token; experiment/run/allocator changes do not reset it."""
    def __init__(self, persistent_root, experiment_id):
        self.experiment_id = str(UUID(experiment_id))
        self.path = Path(persistent_root) / "global_deliberate_attempt.json"

    def consume(self, run_id, probe_id, point_key):
        with file_lock(self.path.with_suffix(".lock")):
            if self.path.exists():
                raise RuntimeError("Global deliberate attempt is already consumed or blocked")
            row = {"state": "consumed_before_launch", "experiment_id": self.experiment_id, "run_id": run_id, "probe_id": probe_id,
                   "point_key": point_key, "consumed_at_utc": now()}
            atomic_json(self.path, row)
            return row

    def finish(self, probe_id, status):
        with file_lock(self.path.with_suffix(".lock")):
            row = json.loads(self.path.read_text())
            if row.get("probe_id") != probe_id or row["state"] != "consumed_before_launch":
                raise ValueError("Ledger attempt identity/state mismatch")
            row.update(state={"success": "completed_success", "cuda_oom": "completed_oom"}.get(status, "interrupted"),
                       result_status=status, finished_at_utc=now())
            atomic_json(self.path, row)

    def block_accidental(self, run_id, probe_id):
        with file_lock(self.path.with_suffix(".lock")):
            previous = json.loads(self.path.read_text()) if self.path.exists() else None
            atomic_json(self.path, {"state": "blocked_after_accidental_oom", "run_id": run_id,
                                    "probe_id": probe_id, "timestamp_utc": now(), "previous_state": previous})
