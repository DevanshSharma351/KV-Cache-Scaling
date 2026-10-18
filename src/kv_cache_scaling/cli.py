"""Working CPU entry points. GPU execution is deliberately unimplemented."""

import argparse
import json
from pathlib import Path

from .model_math import (Architecture, peak_budget_bytes, peak_tensor_bytes,
                         predict_limit, prefill_scenarios, positive_int, vram_gib_to_bytes)
from .preregistration import DEFAULT_BUDGETS, build_draft, write_draft


def load_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kvcache")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="CPU-only draft sensitivity prediction")
    predict.add_argument("--config", default="configs/models/qwen2.5-1.5b.json")
    predict.add_argument("--usable-vram-gib", type=float, required=True)
    predict.add_argument("--batch-size", type=int, default=1)
    prepare = sub.add_parser("prepare-prereg", help="CPU-only prediction draft; does not authorize probes")
    prepare.add_argument("--experiment", default="configs/experiment.json")
    prepare.add_argument("--provenance", default="configs/models/provenance.json")
    prepare.add_argument("--usable-vram-gib", nargs="+", default=DEFAULT_BUDGETS)
    prepare.add_argument("--output", required=True, help="New directory; existing paths are never overwritten")
    plan = sub.add_parser("plan", help="List candidate points; never start CUDA")
    plan.add_argument("--experiment", default="configs/experiment.json")
    for name in ("probe", "sweep", "analyze", "generalize"):
        sub.add_parser(name, help="Reserved implementation entry point")
    args = parser.parse_args(argv)
    if args.command == "predict":
        try:
            arch = Architecture.from_config(load_json(args.config))
            usable = vram_gib_to_bytes(args.usable_vram_gib)
            batch = positive_int(args.batch_size, "batch_size")
            scenarios = []
            for scenario in prefill_scenarios(arch):
                scenarios.append({
                    "name": scenario.name,
                    "activation_bytes_per_token": scenario.activation_bytes_per_token,
                    "fixed_headroom_bytes": scenario.fixed_headroom_bytes,
                    "uncapped_memory_limit_tokens": predict_limit(arch, usable, scenario, batch_size=batch),
                    "config_capped_memory_limit_tokens": predict_limit(arch, usable, scenario, arch.max_positions, batch),
                    "tensor_peak_bytes_at_required_lengths": {
                        str(length): peak_tensor_bytes(arch, length, scenario, batch)
                        for length in (512, 1024, 2048, 4096, 8192)
                    },
                    "budget_bytes_at_required_lengths": {
                        str(length): peak_budget_bytes(arch, length, scenario, batch)
                        for length in (512, 1024, 2048, 4096, 8192)
                    },
                })
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps({
            "status": "DRAFT_THEORY_ONLY_NOT_PREREGISTERED",
            "usable_vram_bytes": usable,
            "batch_size": batch,
            "parameter_breakdown": arch.parameter_breakdown(),
            "unique_parameters": arch.parameter_count(),
            "weight_bytes_fp16": arch.parameter_count() * 2,
            "kv_bytes_per_token": arch.kv_bytes_per_token(batch),
            "naive_kv_bytes_per_token": arch.kv_bytes_per_token(batch, naive=True),
            "uncertainty_type": "assumption sensitivity, not statistical confidence",
            "scenarios": scenarios,
        }, indent=2, allow_nan=False))
    elif args.command == "prepare-prereg":
        try:
            draft = build_draft(args.experiment, args.provenance, args.usable_vram_gib)
            output = write_draft(args.output, draft)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps({"status": draft["status"], "output_directory": str(output),
                          "gpu_measurements_authorized": False}, indent=2))
    elif args.command == "plan":
        exp = load_json(args.experiment)
        points = [{"length": length, "allocator": allocator, "repeat": repeat,
                   "status": "candidate_requires_safety_gate"}
                  for allocator in exp["allocator_variants"]
                  for length in exp["required_lengths"] + exp["candidate_lengths"]
                  for repeat in range(1, exp["repeats"] + 1)]
        print(json.dumps({"protocol_version": exp["protocol_version"],
                          "executes_gpu_work": False, "points": points}, indent=2))
    else:
        parser.error(f"'{args.command}' is scaffolded only. Follow TASK_ALLOCATION.md; "
                     "complete preregistration before any GPU probe.")
    return 0
