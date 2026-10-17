import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

from kv_cache_scaling.cli import main

ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_config_matches_pinned_source_hash(self):
        provenance = json.loads((ROOT / "configs/models/provenance.json").read_text())
        digest = hashlib.sha256((ROOT / "configs/models/qwen2.5-1.5b.json").read_bytes()).hexdigest()
        self.assertEqual(digest, provenance["sha256"])
        exp = json.loads((ROOT / "configs/experiment.json").read_text())
        self.assertEqual(exp["model_revision"], provenance["revision"])

    def test_gpu_command_fails_closed(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["probe"])
        self.assertEqual(error.exception.code, 2)

    def test_prediction_is_explicitly_draft(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["predict", "--config", str(ROOT / "configs/models/qwen2.5-1.5b.json"),
                  "--usable-vram-gib", "14.5"])
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "DRAFT_THEORY_ONLY_NOT_PREREGISTERED")
        self.assertEqual(len(result["scenarios"]), 3)
