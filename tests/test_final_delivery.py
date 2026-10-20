"""Final signoff must reject a genuine but incomplete saved GPU cohort."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p1_delivery_verifier',ROOT/'scripts/verify_p1_delivery.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)


class FinalDeliveryTests(unittest.TestCase):
    def test_real_partial_checkpoint_cannot_pass_final_signoff(self):
        capsule=ROOT/'artifacts/experiments/2026-10-19-t4-interim/primary-043-recovered.zip'
        with self.assertRaisesRegex(ValueError,'four completed execution scopes'):
            verifier.verify(capsule,ROOT/'absent-intent.json',ROOT/'absent-ledger.json')



import json,tempfile
from kv_cache_scaling.delivery import verify_final_scope

class BoundedDeliveryTests(unittest.TestCase):
    def test_real_evidence_is_closed_but_not_promoted_to_oom_completion(self):
        result=verify_final_scope()
        self.assertTrue(result['bounded_delivery_complete'])
        self.assertFalse(result['all_original_research_objectives_met'])
        self.assertFalse(result['exact_oom_threshold_measured'])
        self.assertFalse(result['additional_gpu_runs_planned'])
        self.assertEqual(result['total_successful_workers'],82)
        self.assertEqual(result['sources']['primary']['workers'],80)
        self.assertEqual(result['sources']['replication']['workers'],2)
        self.assertNotEqual(result['sources']['primary']['gpu_uuid'],result['sources']['replication']['gpu_uuid'])
        self.assertEqual(result['closure']['live_gpu_processes'],[])

    def test_scope_cannot_claim_pooling_or_a_measured_oom(self):
        for field in ('pool_cohorts','exact_oom_threshold_measured','all_original_research_objectives_met'):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'configs').mkdir()
                config=json.loads((ROOT/'configs/delivery.json').read_text());config[field]=True
                (root/'configs/delivery.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):verify_final_scope(root)

if __name__=='__main__':unittest.main()
