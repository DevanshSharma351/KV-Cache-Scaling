import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('targeted_boundary',Path(__file__).resolve().parents[1]/'scripts/targeted_boundary.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TargetedBoundaryTests(unittest.TestCase):
    def test_requires_same_allocation_success_before_single_attempt(self):
        self.assertEqual(m.choose_step([],None),(114688,'safe'))
        row={'status':'success','request':{'length':114688}}
        self.assertEqual(m.choose_step([row],'a'*64),(131072,'deliberate'))
        for rows,sha in [([], 'a'*64),([row],None),([row],'x'*64),([row,row],'a'*64),([dict(row,status='cuda_oom')],'a'*64)]:
            with self.assertRaises(ValueError):m.choose_step(rows,sha)
