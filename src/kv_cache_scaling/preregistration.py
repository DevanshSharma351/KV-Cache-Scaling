"""CPU-only preparation of reviewable prediction drafts, never frozen receipts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .model_math import (
    Architecture, GIB, peak_budget_bytes, peak_tensor_bytes,
    predict_limit, prefill_scenarios, positive_int, vram_gib_to_bytes,
)

STATUS = "DRAFT_THEORY_ONLY_NOT_PREREGISTERED"
REQUIRED_LENGTHS = [512, 1024, 2048, 4096, 8192]
DEFAULT_BUDGETS = ["12", "13", "14", "14.5"]


def _read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_input(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Scientific input paths must be inside the repository root")
    return path


def _validate_protocol(experiment, architecture):
    expected = {
        "protocol_version": "prefill-v1", "model_id": "Qwen/Qwen2.5-1.5B",
        "dtype": "float16", "batch_size": 1, "mode": "single_prefill",
        "attention_implementation": "sdpa", "sdpa_backend": "EFFICIENT_ATTENTION",
        "cache_type": "DynamicCache", "logits_to_keep": 1, "warmup": False,
        "allocator_backend": "native", "allocator_variants": ["default", "expandable"],
        "required_lengths": REQUIRED_LENGTHS, "quality_context_limit": 32768,
        "memory_stress_limit": architecture.max_positions, "repeats": 3,
        "deliberate_oom_attempt_budget": 1,
    }
    for key, value in expected.items():
        if type(experiment.get(key)) is not type(value) or experiment[key] != value:
            raise ValueError(f"Unsupported prefill-v1 protocol value: {key}")
    seed = experiment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("boundary_resolution_tokens", "per_probe_timeout_seconds"):
        positive_int(experiment.get(key), key)
    fraction = experiment.get("safe_budget_fraction")
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1):
        raise ValueError("safe_budget_fraction must be between zero and one")
    lengths = experiment.get("candidate_lengths")
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("candidate_lengths must be a nonempty list")
    for length in REQUIRED_LENGTHS + lengths:
        positive_int(length, "candidate length")
        if length > architecture.max_positions:
            raise ValueError("Candidate exceeds config position ceiling")
    if len(set(REQUIRED_LENGTHS + lengths)) != len(REQUIRED_LENGTHS + lengths):
        raise ValueError("Duplicate primary candidate lengths")
    diagnostics = experiment.get("diagnostics", {})
    expected_diagnostics = {
        "chunked_prefill": {"chunk_size": 512, "lengths": [512, 2048, 8192, 32768]},
        "static_cache": {"lengths": [512, 2048, 8192], "capacity_equals_length": True},
        "eager_attention": False, "full_logits": False,
    }
    if json.dumps(diagnostics, sort_keys=True) != json.dumps(expected_diagnostics, sort_keys=True):
        raise ValueError("Diagnostic grid differs from the reviewed plan")


def build_draft(experiment_path="configs/experiment.json",
                provenance_path="configs/models/provenance.json",
                usable_vram_gib=DEFAULT_BUDGETS, root=None):
    """Return deterministic CPU calculations, with no hardware/review claims.

    Paths are repository-relative (absolute in-repository paths also work).
    A successful draft validates input integrity, not actual GPU compatibility.
    """
    root = Path.cwd().resolve() if root is None else Path(root).resolve()
    experiment_file = _repo_input(root, experiment_path)
    provenance_file = _repo_input(root, provenance_path)
    experiment, provenance = _read_json(experiment_file), _read_json(provenance_file)
    if not isinstance(experiment.get("config_path"), str):
        raise ValueError("config_path must name the pinned config")
    config_file = _repo_input(root, experiment["config_path"])
    config = _read_json(config_file)
    architecture = Architecture.from_config(config)
    revision = experiment.get("model_revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a complete commit SHA")
    if provenance.get("model_id") != experiment.get("model_id"):
        raise ValueError("Model identity differs from provenance")
    if provenance.get("revision") != revision:
        raise ValueError("Model revision differs from provenance")
    if provenance.get("sha256") != _sha(config_file):
        raise ValueError("Pinned config SHA-256 mismatch")
    expected_url = (f"https://huggingface.co/{experiment['model_id']}/resolve/"
                    f"{revision}/config.json")
    if provenance.get("url") != expected_url:
        raise ValueError("Provenance URL differs from model/revision")
    _validate_protocol(experiment, architecture)
    if not isinstance(usable_vram_gib, (tuple, list)):
        raise ValueError("Supply usable-VRAM budgets as a list")
    budgets = [vram_gib_to_bytes(value) for value in usable_vram_gib]
    if not budgets or len(budgets) != len(set(budgets)):
        raise ValueError("Supply a nonempty set of distinct usable-VRAM budgets")
    budgets.sort()
    scenarios = prefill_scenarios(architecture)
    memory_rows = []
    for length in REQUIRED_LENGTHS:
        memory_rows.append({
            "length_tokens": length, "kv_bytes": architecture.kv_bytes(length),
            "naive_kv_bytes": architecture.kv_bytes(length, naive=True),
            "scenarios": [{
                "name": scenario.name,
                "tensor_peak_bytes": peak_tensor_bytes(architecture, length, scenario),
                "budget_bytes": peak_budget_bytes(architecture, length, scenario),
            } for scenario in scenarios],
        })
    limit_rows = []
    for budget in budgets:
        values = []
        for scenario in scenarios:
            uncapped = predict_limit(architecture, budget, scenario)
            values.append({
                "name": scenario.name, "uncapped_memory_limit_tokens": uncapped,
                "config_capped_limit_tokens": min(uncapped, architecture.max_positions),
                "quality_domain_limit_tokens": min(uncapped, experiment["quality_context_limit"]),
                "config_ceiling_reached": uncapped >= architecture.max_positions,
                "at_least_one_token_fits": uncapped >= 1,
            })
        limit_rows.append({"usable_vram_bytes": budget, "scenarios": values})
    inputs = [experiment_file, provenance_file, config_file,
              _repo_input(root, "docs/DERIVATION.md")]
    code_paths = [Path(__file__), Path(__file__).with_name("model_math.py")]
    return {
        "schema_version": 1, "status": STATUS,
        "gpu_measurements_authorized": False,
        "missing_evidence": ["actual_gpu_context_only_inventory", "independent_review",
                             "formal_prediction_push", "remote_receipt"],
        "units": {"memory": "bytes", "length": "tokens", "GiB": GIB, "MiB": 1024**2},
        "usable_vram_definition": "after context/other users, before weights; do not subtract context twice",
        "uncertainty_type": "shape-inspired assumption sensitivity; not statistical confidence or proven bounds",
        "input_sha256": {str(path.relative_to(root)): _sha(path) for path in inputs},
        "generator_sha256": {path.name: _sha(path) for path in code_paths},
        "protocol": experiment,
        "architecture": asdict(architecture),
        "parameter_breakdown": architecture.parameter_breakdown(),
        "unique_parameters": architecture.parameter_count(),
        "weight_bytes_fp16": architecture.weight_bytes(),
        "duplicate_tied_embedding_error_bytes": architecture.vocab * architecture.hidden * 2,
        "kv_bytes_per_token": architecture.kv_bytes_per_token(),
        "naive_kv_bytes_per_token": architecture.kv_bytes_per_token(naive=True),
        "gqa_overestimate_factor": architecture.query_heads // architecture.kv_heads,
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "memory_predictions": memory_rows,
        "vram_sensitivity": limit_rows,
    }


def render_draft(draft):
    """Render self-contained review tables; no invented measurement timestamps."""
    protocol = draft["protocol"]
    lines = [
        "# Phase 1 prediction draft", "", f"**Status: {draft['status']}**", "",
        "CPU calculations only. This package does not authorize GPU measurements.",
        "Actual GPU inventory, independent review, formal freeze and remote receipt are pending.", "",
        "## Operational definition", "",
        f"Model: `{protocol['model_id']}` at `{protocol['model_revision']}`.",
        "One cold FP16 prefill of L tokens, B=1, empty DynamicCache, logits_to_keep=1.",
        "HF SDPA with EFFICIENT_ATTENTION only; no fallback or generation.",
        "Config ceiling 131072; lengths above the card's 32768 are synthetic memory stress.", "",
        "## Derivation", "", "```text",
        "q = query_heads * head_dim; k = kv_heads * head_dim",
        "P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H  (tied embeddings)",
        "W = 2P; KV(L,B) = 2 * B * L * N * kv_heads * head_dim * 2",
        "P_tensor(L,B) = W + B * L * [kv_bytes_per_token_at_B1 + a]",
        "P_budget(L,B) = P_tensor(L,B) + H0",
        "L(V,B) = max(0, floor((V-W-H0) / (B * [kv_bytes_per_token_at_B1+a])))",
        "```", "",
        "N=layers; H=hidden; I=intermediate; V=vocabulary in the parameter equation.",
        "In L(V,B), V is usable VRAM in bytes after context/other users, before weights.",
        "a is the proposed B=1 activation bytes/token; H0 is fixed byte headroom.",
        "The shape envelope adds full final KV to a transient allowance; phase lifetimes may not coincide.",
        "Neither the high scenario nor its L estimate is a guaranteed admission bound.", "",
        "## Config and unique weights", "",
        "| Architecture field | Value |", "| --- | ---: |",
    ]
    lines += [f"| {key} | {value} |" for key, value in draft["architecture"].items()]
    lines += ["", "| Disjoint component | Parameters | FP16 bytes |", "| --- | ---: | ---: |"]
    lines += [f"| {key} | {value} | {value * 2} |"
              for key, value in draft["parameter_breakdown"].items()]
    lines += [f"| **TOTAL** | **{draft['unique_parameters']}** | **{draft['weight_bytes_fp16']}** |", "",
              f"Tied embedding double-count error: {draft['duplicate_tied_embedding_error_bytes']} bytes.",
              f"Correct/naive KV slopes: {draft['kv_bytes_per_token']} / {draft['naive_kv_bytes_per_token']} bytes/token.",
              f"GQA overestimate factor: {draft['gqa_overestimate_factor']}x.", "",
              "## Proposed sensitivity assumptions", "",
              "These scenarios are assumptions, not measured coefficients or confidence intervals.",
              "H0 covers fixed logits/buffers/workspace/allocator uncertainty; context is already deducted.", "",
              "| Scenario | a (bytes/token at B=1) | H0 (bytes) |", "| --- | ---: | ---: |"]
    lines += [f"| {s['name']} | {s['activation_bytes_per_token']} | {s['fixed_headroom_bytes']} |"
              for s in draft["scenarios"]]
    lines += ["", "## Required-length predictions", "",
              "Canonical values below are exact integer bytes; nothing in this table was measured.", "",
              "| L | KV bytes | Naive KV bytes | Scenario | Tensor peak bytes | Budget bytes |",
              "| ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in draft["memory_predictions"]:
        lines += [f"| {row['length_tokens']} | {row['kv_bytes']} | {row['naive_kv_bytes']} | "
                  f"{s['name']} | {s['tensor_peak_bytes']} | {s['budget_bytes']} |"
                  for s in row["scenarios"]]
    lines += ["", "## Usable-VRAM sensitivity", "",
              "Budgets are hypothetical inputs, not observations of an allocated GPU.",
              "A cap is a domain limit, not an experimentally observed failure. Zero means no token fits.", "",
              "| Usable GiB | Scenario | Uncapped L | Config-capped L | Quality-domain L |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for row in draft["vram_sensitivity"]:
        lines += [f"| {row['usable_vram_bytes'] / GIB:g} | {s['name']} | "
                  f"{s['uncapped_memory_limit_tokens']} | {s['config_capped_limit_tokens']} | "
                  f"{s['quality_domain_limit_tokens']} |" for s in row["scenarios"]]
    lines += ["", "## Frozen-policy proposal", "",
              "```json", json.dumps(protocol, indent=2, allow_nan=False), "```", "",
              "## Review and experiment gate", "",
              "Preserve three fresh-process repeats, safe-budget admission, and the global single deliberate-attempt ledger.",
              "Validate cache bytes within 1% and retained slope within 3%; keep peak regression separate.",
              "Require same-metric discrepancy closure within max(1 MiB, 0.1% measured bytes) and >=90% supported attribution",
              "for gaps >=10 MiB; below that, residual <=1 MiB with the ratio labelled ill-conditioned.",
              "Actual GPU/backend/allocator support is still untested; no CUDA/model load occurred during generation.",
              "Obtain context-only inventory and a documented review, then freeze a separately named prediction and push it.",
              "Record its SHA/remote evidence in a subsequent receipt before model-memory measurements.", "",
              "## Input and generator provenance", "",
              "```json", json.dumps({"inputs": draft["input_sha256"],
                                      "generator": draft["generator_sha256"]}, indent=2), "```", ""]
    return "\n".join(lines)


def write_draft(output_directory, draft):
    """Write a new package; never overwrite a prior draft/frozen prediction."""
    output = Path(output_directory)
    if draft.get("status") != STATUS or draft.get("gpu_measurements_authorized") is not False:
        raise ValueError("This writer accepts non-authorizing prediction drafts only")
    prediction = json.dumps(draft, indent=2, allow_nan=False) + "\n"
    markdown = render_draft(draft)
    output.mkdir(parents=True, exist_ok=False)
    (output / "prediction.json").write_text(prediction)
    (output / "prediction.md").write_text(markdown)
    manifest = {
        "status": STATUS,
        "files_sha256": {name: _sha(output / name)
                         for name in ("prediction.json", "prediction.md")},
        "note": "Local integrity manifest; not a remote receipt or review sign-off.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
