"""Run from repository root; emits a draft only, never writes a preregistration."""

from kv_cache_scaling.cli import main

raise SystemExit(main(["predict", "--usable-vram-gib", "14.5"]))
