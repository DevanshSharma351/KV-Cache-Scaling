import json
from pathlib import Path
import unittest

from kv_cache_scaling.model_math import Architecture, GIB, predict_limit, prefill_scenarios

ROOT = Path(__file__).resolve().parents[1]


class ModelMathTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/models/qwen2.5-1.5b.json").read_text())
        self.arch = Architecture.from_config(self.config)

    def test_hand_derived_gqa_and_parameter_counts(self):
        self.assertEqual(self.arch.kv_bytes_per_token(), 28672)
        self.assertEqual(self.arch.kv_bytes_per_token(naive=True), 172032)
        self.assertEqual(self.arch.parameter_count(), 1543714304)
        self.assertEqual(self.arch.kv_bytes_per_token() * 8192, 224 * 1024**2)

    def test_untied_embeddings_add_one_matrix(self):
        self.config["tie_word_embeddings"] = False
        untied = Architecture.from_config(self.config)
        self.assertEqual(2 * (untied.parameter_count() - self.arch.parameter_count()), 466747392)

    def test_more_vram_cannot_reduce_limit(self):
        for scenario in prefill_scenarios(self.arch):
            self.assertLessEqual(predict_limit(self.arch, 12 * GIB, scenario),
                                 predict_limit(self.arch, 14.5 * GIB, scenario))
            self.assertEqual(predict_limit(self.arch, GIB, scenario), 0)
            self.assertLessEqual(predict_limit(self.arch, 80 * GIB, scenario, 131072), 131072)

    def test_sensitivity_interval_order(self):
        optimistic, central, conservative = prefill_scenarios(self.arch)
        limits = [predict_limit(self.arch, 14.5 * GIB, s)
                  for s in (conservative, central, optimistic)]
        self.assertEqual(limits, sorted(limits))

    def test_invalid_inputs_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                predict_limit(self.arch, value, prefill_scenarios(self.arch)[0])
        with self.assertRaises(ValueError):
            self.arch.kv_bytes_per_token(batch_size=0)
        self.config["num_key_value_heads"] = 5
        with self.assertRaises(ValueError):
            Architecture.from_config(self.config)


if __name__ == "__main__":
    unittest.main()
