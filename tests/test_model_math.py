import json
from pathlib import Path
import unittest

from kv_cache_scaling.model_math import (
    Architecture, GIB, PeakScenario, peak_budget_bytes, peak_tensor_bytes,
    predict_limit, prefill_scenarios, vram_gib_to_bytes,
)

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

    def test_component_counts_close_without_duplicate_embedding(self):
        parts = self.arch.parameter_breakdown()
        self.assertEqual(parts["input_embedding"], 233373696)
        self.assertEqual(parts["untied_output_embedding"], 0)
        self.assertEqual(parts["attention_qkv_biases"], 57344)
        self.assertEqual(parts["layer_norm_weights"], 86016)
        self.assertEqual(parts["final_norm_weight"], 1536)
        self.assertEqual(sum(parts.values()), 1543714304)
        self.assertEqual(self.arch.weight_bytes(), 3087428608)

    def test_kv_scales_with_batch_length_and_element_size(self):
        self.assertEqual(self.arch.kv_bytes(2048, batch_size=3),
                         3 * self.arch.kv_bytes(2048))
        self.assertEqual(self.arch.kv_bytes(4096), 2 * self.arch.kv_bytes(2048))
        self.assertEqual(self.arch.kv_bytes(2048, element_bytes=4),
                         2 * self.arch.kv_bytes(2048))

    def test_batch_scales_variable_memory_not_weights_or_fixed_allowance(self):
        scenario = prefill_scenarios(self.arch)[1]
        single = peak_tensor_bytes(self.arch, 8192, scenario)
        triple = peak_tensor_bytes(self.arch, 8192, scenario, batch_size=3)
        self.assertEqual(triple - self.arch.weight_bytes(),
                         3 * (single - self.arch.weight_bytes()))
        self.assertEqual(peak_budget_bytes(self.arch, 8192, scenario, 3) - triple,
                         scenario.fixed_headroom_bytes)

    def test_predicted_integer_limit_is_exact_inverse_of_assumed_budget(self):
        for batch in (1, 2, 4):
            for scenario in prefill_scenarios(self.arch):
                for budget in (12 * GIB, 14 * GIB, 80 * GIB):
                    limit = predict_limit(self.arch, budget, scenario, batch_size=batch)
                    self.assertLessEqual(peak_budget_bytes(self.arch, limit, scenario, batch), budget)
                    self.assertGreater(peak_budget_bytes(self.arch, limit + 1, scenario, batch), budget)

    def test_decimal_gib_conversion_and_invalid_values(self):
        self.assertEqual(vram_gib_to_bytes("14.5"), 15569256448)
        self.assertEqual(vram_gib_to_bytes("0.25"), GIB // 4)
        for value in (0, -1, True, "nan", "Infinity", "bad", "0.00000000001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vram_gib_to_bytes(value)

    def test_bad_config_shapes_and_unsupported_execution_rejected(self):
        cases = [None, [], {**self.config, "num_hidden_layers": True},
                 {**self.config, "head_dim": 0}, {**self.config, "head_dim": 64},
                 {**self.config, "use_sliding_window": True},
                 {**self.config, "tie_word_embeddings": "true"},
                 {**self.config, "hidden_act": "relu"},
                 {**self.config, "model_type": "llama"}]
        missing = dict(self.config)
        del missing["vocab_size"]
        cases.append(missing)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                Architecture.from_config(config)

    def test_invalid_scenarios_and_dimensions_rejected(self):
        for arguments in (("", 1, 0), ("bad", 0, 0), ("bad", 1, -1), ("bad", 1, True)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PeakScenario(*arguments)
        scenario = prefill_scenarios(self.arch)[0]
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.arch.kv_bytes(value)
            with self.subTest(batch=value), self.assertRaises(ValueError):
                peak_tensor_bytes(self.arch, 512, scenario, value)


if __name__ == "__main__":
    unittest.main()
