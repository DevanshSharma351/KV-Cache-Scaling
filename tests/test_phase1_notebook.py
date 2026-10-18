import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_p1_phase1_colab.ipynb"


class Phase1NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        for cell in self.notebook["cells"]:
            if isinstance(cell["source"], list):
                cell["source"] = "".join(cell["source"])
        self.cells = {cell["id"]: cell for cell in self.notebook["cells"]}

    def test_cells_are_syntactically_valid_and_no_execution_is_invented(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def helper_namespace(self, directory):
        namespace = {}
        with patch.dict(os.environ, {"KVCACHE_NOTEBOOK_RUNS_ROOT": directory}):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells["audit-setup"]["source"], namespace)
        return namespace

    def test_success_failure_and_repeat_protection_leave_a_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.helper_namespace(directory)
            run = ns["audit_run"]
            with contextlib.redirect_stdout(io.StringIO()):
                run("success", [sys.executable, "-c", "print('recorded output')"])
                with self.assertRaises(RuntimeError):
                    run("failure", [sys.executable, "-c", "import sys; print('failure evidence'); sys.exit(7)"])
                with self.assertRaises(FileExistsError):
                    run("success", [sys.executable, "-c", "print('overwrite')"])
            rows = [json.loads(line) for line in ns["JOURNAL"].read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows],
                             ["command_started", "command_finished", "command_started", "command_finished"])
            self.assertEqual(rows[-1]["returncode"], 7)
            self.assertEqual((ns["RUN_DIR"] / "success.stdout.txt").read_text(), "recorded output\n")
            self.assertIn("failure evidence", (ns["RUN_DIR"] / "failure.stdout.txt").read_text())

    def test_timeout_is_preserved_as_incomplete_instead_of_success(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            ns = self.helper_namespace(directory)
            with patch.object(ns["subprocess"], "run", side_effect=subprocess.TimeoutExpired(["test"], 1)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    ns["audit_run"]("timeout", ["test"], timeout=1)
            rows = [json.loads(line) for line in ns["JOURNAL"].read_text().splitlines()]
            self.assertEqual(rows[-1]["event"], "command_failed_to_complete")
            self.assertTrue((ns["RUN_DIR"] / "timeout.stderr.txt").is_file())

    def test_notebook_prediction_script_reproduces_report_without_cuda(self):
        import subprocess
        tree = ast.parse(self.cells["conditional-prediction"]["source"])
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "prediction_script" for t in node.targets))
        script = ast.literal_eval(assignment.value)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            completed = subprocess.run([
                sys.executable, "-c", script,
                str(ROOT / "prereg/environment/t4-context-2026-10-18.user.json"), str(output),
            ], cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated = json.loads(output.read_text())
            saved = json.loads((ROOT / "prereg/environment/t4-context-2026-10-18.conditional-prediction.json").read_text())
            saved.pop("calculation_execution")
            self.assertEqual(generated, saved)


if __name__ == "__main__":
    unittest.main()
