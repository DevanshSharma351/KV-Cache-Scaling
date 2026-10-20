"""The live demo must be traceable to actual evidence and cannot launch CUDA."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from kv_cache_scaling.presentation import Presentation,SECTIONS,checkpoint_demo

ROOT=Path(__file__).resolve().parents[1]

class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.results=Path(cls.directory.name)
        shutil.copytree(ROOT/'data/processed/p1-2026-10-19',cls.results/'analysis')
        shutil.copytree(ROOT/'data/processed/p1-2026-10-19/transfer',cls.results/'transfer')
        shutil.copytree(ROOT/'artifacts/figures/p1-2026-10-19',cls.results/'analysis/figures')
        shutil.copytree(ROOT/'artifacts/reports/p1-2026-10-19',cls.results/'report')
        cls.presentation=Presentation(ROOT,cls.results)
    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_all_six_deliverables_are_separately_presentable_without_role_labels(self):
        for section in SECTIONS:
            blocks=self.presentation.content(section);self.assertTrue(blocks)
            for kind,value in blocks:
                if kind=='image':self.assertTrue(Path(value).is_file())
                else:self.assertIsNone(re.search(r'Person [12]|\bP[12]\b|55%|45%',value))
        self.assertIn('2026-10-18',self.presentation.commit_timestamp)
        with self.assertRaises(ValueError):self.presentation.content('invented')

    def test_parametric_explorer_keeps_original_prediction_and_batch_scaling(self):
        one=self.presentation.explore();two=self.presentation.explore(batch_size=2)
        self.assertEqual(next(r[1] for r in one if r[0]=='central'),111232)
        self.assertTrue(all(b[1]<=a[1] for a,b in zip(one,two)))
        with self.assertRaises(ValueError):self.presentation.explore(-1)
        self.assertEqual(self.presentation.analysis['hardware_brackets'][0]['smallest_observed_oom_tokens'],None)

    def test_generated_data_tampering_cannot_be_presented(self):
        path=self.results/'analysis/analysis.json';original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaises(ValueError):Presentation(ROOT,self.results)
        finally:path.write_bytes(original)

    def test_live_checkpoint_demo_restores_real_temporary_files(self):
        result=checkpoint_demo()
        self.assertTrue(result['restored_missing_local_artifact'])
        self.assertTrue(result['mirrored_journals_match'])
        self.assertEqual(result['recovered_events'],1)
        subprocess.run([sys.executable,'-c',"import sys;import kv_cache_scaling.presentation;assert 'torch' not in sys.modules and 'pynvml' not in sys.modules"],check=True,cwd=ROOT)

    def test_runtime_free_board_has_all_six_pinned_images_and_original_commitment(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('notebook_builder',ROOT/'scripts/build_presentation_notebook.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        commit='0'*40;notebook=builder.build(commit)
        self.assertEqual(sum(c['cell_type']=='code' for c in notebook['cells']),12)
        board=''.join(notebook['cells'][-1]['source'])
        self.assertIn('Final presentation board',board)
        self.assertEqual(board.count('https://raw.githubusercontent.com/DevanshSharma351/KV-Cache-Scaling/'+commit+'/'),6)
        self.assertIn('2026-10-18T19:54:52+05:30',board)
        self.assertIn('exact-OOM objectives are explicitly unfulfilled',board)

if __name__=='__main__':unittest.main()
