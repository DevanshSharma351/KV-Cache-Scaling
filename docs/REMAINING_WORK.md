# Measurement limitations and optional future research

**20 October 2026: further GPU work closed at the user's resource limit.**
The current resource-bounded delivery is finalized. No extra probe is scheduled.
All six artifact types, the five phases, PDF and CPU presentation are delivered.
See [final scope](FINAL_SCOPE.md) and [project status](PROJECT_STATUS.md).

The original full experiment still lacks two expandable L=114688 repeats on its
original allocation and a failed hardware upper point. They are **deferred
research**, not completed measurements and not hidden requirements claimed to
pass. Both allocators succeeded at 114688, which remains a successful lower bound.
The 80-worker core and two-worker replication are separate UUID cohorts.

Consequences: exact OOM-threshold error, causal decomposition of an OOM-limit gap
and exact hardware allocator-context gain are not established. Fixed-length
peak discrepancy accounting and direct retained KV verification are available.
The published transfer comparison validates KV blocks, not unrun HF thresholds.

If separately authorized later, new experiments must reconcile actual hardware,
runtime and usable VRAM, prospectively register changes, checkpoint externally,
and preserve the project-wide single-attempt ledger. A new UUID never silently
fills old repeats. No OOM allowance was consumed by this delivery. Do not resume
the stopped bulk cell or weaken the original completion checks to manufacture a
pass. The final-scope verifier makes these limitations explicit.

```bash
python scripts/verify_final_delivery.py
```

CPU reproduction and the clean presentation remain available without a GPU.
Historical predictions, receipts, raw bytes and original audit journals remain
unchanged. The decision to close further measurement is documented, not applied
retroactively to the original preregistration.
