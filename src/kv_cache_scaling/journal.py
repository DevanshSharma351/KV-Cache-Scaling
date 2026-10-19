"""Public persistence API backed by the coordinator's tested POSIX store.

A RunStore context owns the lock, immutable artifacts, torn-tail recovery and
fsynced local/mirror writes. Both APIs use the same implementation; there is no
second journal format and no path-only append that could bypass the mirror.
"""
from .coordinator_store import AttemptLedger, RunStore, atomic_bytes, atomic_json, file_lock, now, sync_dir

__all__ = ['AttemptLedger', 'RunStore', 'append_event', 'atomic_bytes', 'atomic_json', 'file_lock', 'now', 'sync_dir']


def append_event(store, event):
    """Append an event through an acquired RunStore; return its canonical row.

    ``event`` contains an ``event`` name and payload fields. Sequence, run ID
    and timestamp are assigned by RunStore. It must mirror successfully before
    the caller can schedule another worker. Failure is propagated unchanged.
    """
    if not isinstance(store, RunStore):
        raise TypeError('Pass an acquired RunStore with distinct local and persistent roots')
    if not isinstance(event, dict):
        raise ValueError('Event must be a JSON object')
    kind = event.get('event')
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError('Event requires a nonempty event name')
    fields = {key: value for key, value in event.items() if key != 'event'}
    return store.append(kind, **fields)
