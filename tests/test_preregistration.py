import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from kv_cache_scaling.cli import main
from kv_cache_scaling.preregistration import STATUS, build_draft, render_draft, write_draft

ROOT = Path(__file__).resolve().parents[1]


class PreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        shutil.copytree(ROOT / "configs", self.root / "configs")
        (self.root / "docs").mkdir()
        shutil.copyfile(ROOT / "docs/DERIVATION.md", self.root / "docs/DERIVATION.md")

    def mutate_json(self, relative_path, changes):
        path = self.root / relative_path
        content = json.loads(path.read_text())
        content.update(changes)
        path.write_text(json.dumps(content))

    def test_review_package_contains_required_tables_and_truthful_status(self):
        draft = build_draft(root=self.root)
        self.assertEqual(draft["status"], STATUS)
        self.assertIs(draft["gpu_measurements_authorized"], False)
        self.assertIn("actual_gpu_context_only_inventory", draft["missing_evidence"])
        self.assertEqual(len(draft["memory_predictions"]), 5)
        self.assertEqual(len(draft["vram_sensitivity"]), 4)
        self.assertEqual(draft["weight_bytes_fp16"], 3087428608)
        self.assertEqual(draft["duplicate_tied_embedding_error_bytes"], 466747392)
        self.assertEqual(draft["gqa_overestimate_factor"], 6)
        central = draft["vram_sensitivity"][-1]["scenarios"][1]
        self.assertEqual(central["uncapped_memory_limit_tokens"], 111626)
        self.assertEqual(central["quality_domain_limit_tokens"], 32768)

    def test_config_tampering_fails_even_when_json_is_valid(self):
        self.mutate_json("configs/models/qwen2.5-1.5b.json", {"vocab_size": 151935})
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            build_draft(root=self.root)

    def test_revision_and_identity_mismatches_fail(self):
        for field, value in (("model_revision", "a" * 40), ("model_id", "other/model")):
            with self.subTest(field=field):
                shutil.copyfile(ROOT / "configs/experiment.json", self.root / "configs/experiment.json")
                self.mutate_json("configs/experiment.json", {field: value})
                with self.assertRaisesRegex(ValueError, "differs from provenance"):
                    build_draft(root=self.root)

    def test_unsafe_protocol_changes_fail(self):
        for field, value in (("logits_to_keep", 0), ("batch_size", True),
                             ("cache_type", "StaticCache"), ("sdpa_backend", "MATH"),
                             ("warmup", True), ("required_lengths", [512]),
                             ("safe_budget_fraction", float("nan"))):
            with self.subTest(field=field):
                shutil.copyfile(ROOT / "configs/experiment.json", self.root / "configs/experiment.json")
                self.mutate_json("configs/experiment.json", {field: value})
                with self.assertRaises(ValueError):
                    build_draft(root=self.root)

    def test_candidates_cannot_duplicate_or_exceed_domain(self):
        for values in ([512, 8192], [262144], [True], []):
            with self.subTest(values=values):
                self.mutate_json("configs/experiment.json", {"candidate_lengths": values})
                with self.assertRaises(ValueError):
                    build_draft(root=self.root)

    def test_budget_validation_and_infeasible_output(self):
        for values in ([], ["nan"], ["14.5", "14.50"], "14.5"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_draft(root=self.root, usable_vram_gib=values)
        draft = build_draft(root=self.root, usable_vram_gib=["1"])
        for scenario in draft["vram_sensitivity"][0]["scenarios"]:
            self.assertEqual(scenario["uncapped_memory_limit_tokens"], 0)
            self.assertFalse(scenario["at_least_one_token_fits"])

    def test_input_paths_must_stay_inside_repo(self):
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            build_draft(experiment_path="../outside.json", root=self.root)

    def test_byte_for_byte_regeneration_and_manifest_integrity(self):
        draft = build_draft(root=self.root)
        first = write_draft(self.root / "one", draft)
        second = write_draft(self.root / "two", build_draft(root=self.root))
        manifest = json.loads((first / "manifest.json").read_text())
        for filename in ("prediction.json", "prediction.md", "manifest.json"):
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
        for filename, digest in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((first / filename).read_bytes()).hexdigest(), digest)
        self.assertIn("Actual GPU inventory", render_draft(draft))

    def test_existing_output_is_not_overwritten(self):
        draft = build_draft(root=self.root)
        destination = write_draft(self.root / "existing", draft)
        original = (destination / "prediction.json").read_bytes()
        with self.assertRaises(FileExistsError):
            write_draft(destination, draft)
        self.assertEqual((destination / "prediction.json").read_bytes(), original)

    def test_writer_refuses_authorizing_or_formal_status(self):
        draft = build_draft(root=self.root)
        draft["gpu_measurements_authorized"] = True
        with self.assertRaises(ValueError):
            write_draft(self.root / "forged", draft)
        self.assertFalse((self.root / "forged").exists())

    def test_cli_draft_generation_and_invalid_prediction_input(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["prepare-prereg", "--experiment", str(ROOT / "configs/experiment.json"),
                  "--provenance", str(ROOT / "configs/models/provenance.json"),
                  "--output", str(self.root / "cli")])
        self.assertEqual(json.loads(out.getvalue())["status"], STATUS)
        self.assertTrue((self.root / "cli/prediction.md").is_file())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main(["predict", "--usable-vram-gib", "nan"])
        self.assertEqual(failure.exception.code, 2)

    def test_generation_never_imports_torch(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; from kv_cache_scaling.preregistration import build_draft; "
            "build_draft(); assert 'torch' not in sys.modules; assert 'pynvml' not in sys.modules",
        ], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
