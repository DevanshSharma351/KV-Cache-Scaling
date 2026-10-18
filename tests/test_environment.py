from copy import deepcopy
import json
from pathlib import Path
import unittest

from kv_cache_scaling.environment import context_prediction
from kv_cache_scaling.model_math import Architecture

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads((ROOT / "prereg/environment/t4-context-2026-10-18.user.json").read_text())
        self.arch = Architecture.from_config(json.loads((ROOT / "configs/models/qwen2.5-1.5b.json").read_text()))

    def test_exact_free_budget_is_not_reduced_twice(self):
        result = context_prediction(self.snapshot, self.arch)
        self.assertEqual(result["usable_vram_bytes"], 15527116800)
        central = result["scenario_limits"][1]
        self.assertEqual(central["uncapped_limit_tokens"], 111232)
        self.assertEqual(result["other_gpu_pids_observed"], [])
        self.assertEqual(result["cuda_total_minus_free_bytes"], 109969408)

    def test_runtime_mismatch_is_visible_and_never_authorizes_probes(self):
        result = context_prediction(self.snapshot, self.arch)
        self.assertFalse(result["runtime_matches_proposal"])
        self.assertEqual(len(result["runtime_differences"]), 4)
        self.assertFalse(result["gpu_measurements_authorized"])
        self.assertIn("NOT_PREREGISTERED", result["status"])

    def test_same_stack_still_requires_review_and_receipt(self):
        self.snapshot.update(python="3.11.9", torch="2.6.0+cu124",
                             torch_cuda_runtime="12.4", transformers="4.51.3")
        result = context_prediction(self.snapshot, self.arch)
        self.assertTrue(result["runtime_matches_proposal"])
        self.assertFalse(result["gpu_measurements_authorized"])

    def test_invalid_or_non_context_inventory_is_rejected(self):
        for changes in ({"cuda_free_bytes": 0}, {"cuda_free_bytes": True},
                        {"cuda_total_bytes": 1}, {"allocated_bytes": 1024},
                        {"reserved_bytes": 4096}, {"timestamp_utc": "2026-10-18"},
                        {"gpu_identity_csv": "GPU-bad, Tesla T4, 580.82.07"},
                        {"gpu_processes": ""}, {"current_pid": 999},
                        {"allocator_env": {}}, {"transformers": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                snapshot = deepcopy(self.snapshot)
                snapshot.update(changes)
                context_prediction(snapshot, self.arch)

    def test_other_users_are_reported_instead_of_hidden(self):
        self.snapshot["gpu_processes"] += "\n1111, another-python, 100 MiB"
        result = context_prediction(self.snapshot, self.arch)
        self.assertEqual(result["other_gpu_pids_observed"], [1111])
        self.assertFalse(result["gpu_measurements_authorized"])
