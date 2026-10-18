"""CPU arithmetic fixtures only; synthetic modified records never enter reports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from kv_cache_scaling.analysis import read_capsules, analyze, regression, discrepancy, write_analysis, matched_workspace_controls
from kv_cache_scaling.model_math import Architecture

ROOT=Path(__file__).resolve().parents[1]
CAPSULE=ROOT/'artifacts/acceptance/2026-10-18-t4/gpu-acceptance-complete.zip'


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records,cls.runs,cls.sources=read_capsules([CAPSULE])
        cls.arch=Architecture.from_config(json.loads((ROOT/'configs/models/qwen2.5-1.5b.json').read_text()))
        cls.good=next(x['record'] for x in cls.records if x['record']['status']=='success' and x['record']['measurement']=='primary')

    def test_actual_acceptance_is_not_a_five_length_sweep(self):
        result=analyze(self.records)
        self.assertFalse(result['acceptance']['five_length_paired_primary_complete'])
        self.assertFalse(result['acceptance']['cache_slope_pass'])
        self.assertTrue(result['acceptance']['discrepancy_closure_pass'])
        self.assertFalse(result['acceptance']['discrepancy_coverage_pass'])
        self.assertEqual(len(result['failures']),1)
        self.assertEqual(len(result['groups']),3) # allocator variants plus separate profiler
        for t in result['discrepancy_tables']:
            self.assertEqual(t['closure_error_bytes'],0)
            self.assertGreater(t['unaccounted_absolute_bytes'],1024**2)

    def test_duplicate_capsules_deduplicate_and_divergent_member_rejected(self):
        r,_,_=read_capsules([CAPSULE,CAPSULE]);self.assertEqual(len(r),len(self.records))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(CAPSULE) as src,zipfile.ZipFile(p,'w') as dst:
                for n in src.namelist():dst.writestr(n,b'altered' if n=='audit/context-inventory.json' else src.read(n))
            with self.assertRaises(ValueError):read_capsules([p])

    def test_ols_requires_distinct_lengths_and_has_known_solution(self):
        self.assertEqual(regression([(512,1),(512,2)])['status'],'INSUFFICIENT_LENGTHS')
        r=regression([(n,123+28672*n) for n in (512,1024,2048,4096,8192)])
        self.assertAlmostEqual(r['slope_bytes_per_token'],28672,places=7)
        self.assertAlmostEqual(r['intercept_bytes'],123,places=5)
        self.assertEqual(r['r_squared'],1)

    def test_signed_accounting_closes_without_fake_attribution(self):
        t=discrepancy(self.good,self.arch)
        self.assertEqual(sum(x['signed_bytes'] for x in t['rows']),t['signed_gap_bytes'])
        self.assertFalse(t['coverage_pass'])
        fake=copy.deepcopy(self.good)
        fake['analysis_evidence']={'confirmed_cublas_workspace_bytes':8519680}
        self.assertEqual(discrepancy(fake,self.arch)['unaccounted_absolute_bytes'],t['unaccounted_absolute_bytes'])

    def test_workspace_attribution_requires_three_matching_actual_controls(self):
        # Explicit CPU-only fixtures: never saved as measured evidence.
        controls=[]
        for repeat in (1,2,3):
            c=copy.deepcopy(self.good);c.update(probe_id=f'synthetic-control-{repeat}',repeat=repeat,measurement='workspace_control')
            c['runtime']['cublas_workspace_config']=':0:0'
            c['execution_policy'].update(diagnostic='workspace_control',cublas_workspace_config=':0:0')
            c['snapshots']['released']['allocated_bytes']=c['snapshots']['post_load']['allocated_bytes']
            controls.append(c)
        self.assertFalse(matched_workspace_controls([self.good]+controls[:2],self.arch))
        proof=matched_workspace_controls([self.good]+controls,self.arch)
        self.assertEqual(proof[self.good['probe_id']]['bytes'],8519680)
        longer=copy.deepcopy(self.good);longer.update(probe_id='synthetic-longer',length=1024)
        longer['cache']['logical_bytes']=self.arch.kv_bytes(1024)
        longer['cache']['unique_storage_bytes']=self.arch.kv_bytes(1024)
        inferred=matched_workspace_controls([longer]+controls,self.arch)
        self.assertIn('Source-supported',inferred[longer['probe_id']]['scope'])
        wrong_gpu=copy.deepcopy(longer);wrong_gpu['gpu_identity']['name']='unverified GPU'
        self.assertFalse(matched_workspace_controls([wrong_gpu]+controls,self.arch))
        altered=copy.deepcopy(controls);altered[0]['allocator']='expandable'
        self.assertFalse(matched_workspace_controls([self.good]+altered,self.arch))

    def test_failed_metrics_not_imputed_and_output_never_overwrites(self):
        r=analyze([{'probe_id':'synthetic-failed','status':'timeout','length':512}])
        self.assertEqual(len(r['probes']),0);self.assertEqual(r['failures'][0]['status'],'timeout')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'processed'
            write_analysis([CAPSULE],out)
            self.assertTrue((out/'analysis.json').is_file())
            with self.assertRaises(ValueError):write_analysis([CAPSULE],out)

    def test_invalid_snapshot_rejected_and_groups_do_not_pool_hardware(self):
        bad=copy.deepcopy(self.good);bad['snapshots']['retained']['reserved_bytes']=0
        with self.assertRaises(ValueError):analyze([bad])
        other=copy.deepcopy(self.good);other.update(probe_id='synthetic-other',gpu_uuid='GPU-synthetic-other')
        r=analyze([self.good,other]);self.assertEqual(len(r['groups']),2)

if __name__=='__main__':unittest.main()
