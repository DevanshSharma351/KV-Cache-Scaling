"""Config-only transfer calculations; no model weights, torch or GPU imports."""
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .model_math import Architecture, GIB, predict_limit, prefill_scenarios, peak_budget_bytes, vram_gib_to_bytes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGETS = (
    {"gpu": "A100-40GB", "usable_gib_min": "37", "usable_gib_central": "38", "usable_gib_max": "39"},
    {"gpu": "RTX-4090-24GB", "usable_gib_min": "22", "usable_gib_central": "23", "usable_gib_max": "23.5"},
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_configs(provenance_path=None):
    path = Path(provenance_path or ROOT/"configs/transfer/provenance.json")
    provenance = json.loads(path.read_text())
    models = {}
    for entry in provenance["models"]:
        config_path = (ROOT/entry["config_path"]).resolve()
        if not config_path.is_relative_to(ROOT):
            raise ValueError("Transfer config escapes repository")
        data = config_path.read_bytes()
        if sha(data) != entry["config_sha256"]:
            raise ValueError("Transfer config hash mismatch")
        if len(entry["revision"]) != 40 or any(c not in "0123456789abcdef" for c in entry["revision"]):
            raise ValueError("Transfer revision must be pinned")
        models[entry["model_id"]] = {"architecture": Architecture.from_config(json.loads(data)), "provenance": entry}
    return models


def published_comparison(models, evidence):
    if evidence["tokens_per_block"] != 16 or evidence["element_bytes_assumption"] != 2 or evidence["tensor_parallel_size"] != 1:
        raise ValueError("Unsupported external-comparison assumptions")
    rows = []
    for reference in evidence["rows"]:
        base = models[reference["base_model_id"]]["architecture"]
        instruct = models[reference["model_id"]]["architecture"]
        fields = ("layers", "kv_heads", "head_dim")
        if any(getattr(base, key) != getattr(instruct, key) for key in fields):
            raise ValueError("Published Instruct cache architecture differs from base model")
        predicted = base.kv_bytes(16)
        shown = Decimal(reference["reported_new_block_MB"])
        decimal_bytes = shown*1_000_000
        binary_bytes = shown*1024**2
        rows.append({"base_model_id": reference["base_model_id"], "published_model_id": reference["model_id"],
                     "predicted_block_bytes": predicted, "predicted_block_MiB": predicted/1024**2,
                     "published_display_MB": str(shown),
                     "display_error_percent_if_binary": float(abs(Decimal(predicted)-binary_bytes)/binary_bytes*100),
                     "display_error_percent_if_decimal": float(abs(Decimal(predicted)-decimal_bytes)/decimal_bytes*100),
                     "kv_architecture_matches": True, "source_url": evidence["source_url"],
                     "verifies": "KV block byte scaling under a 2-byte dtype assumption only",
                     "does_not_verify": "HF prefill peak, activation envelope, usable VRAM or transferred context threshold"})
    return {"status": "KV_SUBMODEL_COMPARISON_WITH_UNIT_CAVEAT", "source": evidence, "comparisons": rows,
            "full_peak_threshold_externally_validated": False}


def generalize(configs, budgets=DEFAULT_BUDGETS, assumptions=None):
    """Return four theory cells with VRAM and activation assumption sensitivity.

    The callable's input configs use pinned model identities. Activation scenarios
    are recomputed from each target architecture; no fitted T4 activation slope
    is blindly asserted to hold on another GPU/model.
    """
    if assumptions is not None:
        raise ValueError("Custom/calibrated scenarios require a separately reviewed transfer protocol")
    predictions = []
    for model_id in ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"):
        model = configs[model_id]
        arch = model["architecture"]
        scenarios = prefill_scenarios(arch)
        for budget in budgets:
            low, central, high = [vram_gib_to_bytes(budget[k]) for k in
                                 ("usable_gib_min", "usable_gib_central", "usable_gib_max")]
            if not low <= central <= high:
                raise ValueError("Usable VRAM range must be ordered")
            central_raw = predict_limit(arch, central, scenarios[1])
            low_raw = predict_limit(arch, low, scenarios[2])
            high_raw = predict_limit(arch, high, scenarios[0])
            predictions.append({"model_id": model_id, "gpu": budget["gpu"],
                "prediction_kind": "UNRUN_CONFIG_DERIVED_TRANSFER", "weights_loaded": False, "target_gpu_measured": False,
                "model_revision": model["provenance"]["revision"], "config_sha256": model["provenance"]["config_sha256"],
                "unique_parameters": arch.parameter_count(), "weight_bytes_fp16": arch.weight_bytes(),
                "kv_bytes_per_token": arch.kv_bytes_per_token(), "naive_gqa_overestimate_factor": arch.query_heads/arch.kv_heads,
                "usable_vram_bytes": {"min": low, "central": central, "max": high},
                "budget_origin": "Hypothetical sensitivity budgets; nominal GPU labels are not measurements of usable memory",
                "scenarios": [asdict(s) for s in scenarios],
                "parametric_limit": "max(0, floor((U-W-H0)/(B*(kv_B1+a))))",
                "uncapped_limit_tokens": {"min": low_raw, "central": central_raw, "max": high_raw},
                "config_capped_limit_tokens": {"min": min(low_raw,arch.max_positions), "central": min(central_raw,arch.max_positions), "max": min(high_raw,arch.max_positions)},
                "config_ceiling_tokens": arch.max_positions,
                "quality_domain_limit_tokens": min(central_raw,32768,arch.max_positions),
                "config_censored_central": central_raw >= arch.max_positions,
                "central_budget_bytes_at_required_lengths": {str(n):peak_budget_bytes(arch,n,scenarios[1]) for n in (512,1024,2048,4096,8192)},
                "uncertainty": "VRAM plus architecture-shaped activation assumptions, not a confidence interval or validated GPU boundary"})
    return {"schema_version": "transfer-v1", "predictions": predictions,
            "calibration_status": "Only retained KV scaling is analytically established; target peak envelopes remain unvalidated"}


def write_transfer(output, budgets=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite transfer artifacts")
    models = load_configs()
    result = generalize(models, budgets or DEFAULT_BUDGETS)
    evidence_path = ROOT/"configs/transfer/published_verification.json"
    result["published_verification"] = published_comparison(models,json.loads(evidence_path.read_text()))
    result["provenance"] = {"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "generator_sha256":sha(Path(__file__).read_bytes()), "config_inventory_sha256":sha((ROOT/"configs/transfer/provenance.json").read_bytes()),
        "published_evidence_sha256":sha(evidence_path.read_bytes())}
    output.mkdir(parents=True)
    (output/"transfer.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    rows=["# Config-derived transfer predictions", "", "No target checkpoint was loaded and no target GPU was measured.", "",
          "| Model | GPU | Assumed usable GiB (low/central/high) | Raw central L | Config-capped L | Capped assumption range |", "| --- | --- | --- | ---: | ---: | --- |"]
    for p in result["predictions"]:
        u=p["usable_vram_bytes"];c=p["config_capped_limit_tokens"]
        rows.append(f"| {p['model_id']} | {p['gpu']} | {u['min']/GIB:g}/{u['central']/GIB:g}/{u['max']/GIB:g} | {p['uncapped_limit_tokens']['central']} | {c['central']} | {c['min']}–{c['max']} |")
    rows += ["", "Activation/VRAM ranges are assumptions, not statistical confidence intervals. Config censoring is not an observed OOM.", "", "## Published KV-only comparison", "",
             f"Source: [{result['published_verification']['source']['source_title']}]({evidence_path and result['published_verification']['source']['source_url']}).", "",
             "The publication uses MB without defining decimal versus binary units. Both errors are reported; the 2-byte cache dtype is an explicit compatibility assumption.", "",
             "| Base / matching Instruct KV architecture | Predicted 16-token block bytes | Published MB | Error if MiB | Error if decimal MB |", "| --- | ---: | ---: | ---: | ---: |"]
    for p in result["published_verification"]["comparisons"]:
        rows.append(f"| {p['base_model_id']} | {p['predicted_block_bytes']} | {p['published_display_MB']} | {p['display_error_percent_if_binary']:.4f}% | {p['display_error_percent_if_decimal']:.4f}% |")
    rows += ["", "This checks KV-byte scaling only. It does not verify transferred HF single-prefill peak context limits.", ""]
    (output/"transfer.md").write_text("\n".join(rows))
    (output/"manifest.json").write_text(json.dumps({p.name:sha(p.read_bytes()) for p in sorted(output.iterdir())},indent=2)+"\n")
    return result
