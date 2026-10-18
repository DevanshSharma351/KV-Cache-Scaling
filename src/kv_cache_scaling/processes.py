"""Fresh worker process, bounded streaming logs, timeout/interrupt cleanup."""

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def worker_env(allocator, root):
    env = dict(os.environ)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)
    if allocator == "expandable":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    elif allocator != "default":
        raise ValueError("Unknown allocator")
    env["PYTHONPATH"] = str(Path(root).resolve() / "src")
    env["PYTHONUNBUFFERED"] = "1"
    return env


class BoundedStream:
    def __init__(self, pipe, limit):
        self.pipe, self.limit = pipe, limit
        self.data, self.count, self.hash = bytearray(), 0, hashlib.sha256()
        self.error = None

    def drain(self):
        try:
            while True:
                chunk = self.pipe.read(65536)
                if not chunk:
                    break
                self.count += len(chunk)
                self.hash.update(chunk)
                remaining = self.limit - len(self.data)
                if remaining > 0:
                    self.data.extend(chunk[:remaining])
        except Exception as error:
            self.error = str(error)
        finally:
            self.pipe.close()

    def report(self):
        return {"total_bytes": self.count, "retained_bytes": len(self.data),
                "truncated": self.count > len(self.data), "full_stream_sha256": self.hash.hexdigest(),
                "retained_sha256": hashlib.sha256(self.data).hexdigest(), "read_error": self.error}


def stop_group(process):
    # Kill descendants too, including inherited pipe holders after parent exit.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def launch_worker(command, *, env, cwd, timeout, log_limit=1024**2):
    if timeout <= 0 or log_limit < 1:
        raise ValueError("Positive timeout and log bound required")
    start = time.monotonic_ns()
    process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    streams = [BoundedStream(process.stdout, log_limit), BoundedStream(process.stderr, log_limit)]
    threads = [threading.Thread(target=stream.drain, daemon=True) for stream in streams]
    status = "exited"
    try:
        for thread in threads:
            thread.start()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
        stop_group(process)
    except KeyboardInterrupt:
        status = "interrupted"
        stop_group(process)
    except BaseException:
        stop_group(process)
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=2)
        raise
    finally:
        # A normal parent exit may leave descendants holding stdout/stderr.
        stop_group(process)
    for thread in threads:
        thread.join(timeout=2)
        if thread.is_alive():
            status = "log_drain_error"
    if any(stream.error for stream in streams):
        status = "log_drain_error"
    return {"process_status": status, "returncode": process.returncode, "pid": process.pid,
            "elapsed_ns": time.monotonic_ns() - start,
            "stdout": bytes(streams[0].data), "stderr": bytes(streams[1].data),
            "logs": {name: stream.report() for name, stream in zip(("stdout", "stderr"), streams)}}
