import copy
import json
from pathlib import Path
import tempfile
import unittest
from kv_cache_scaling.generalize import load_configs,generalize,published_comparison,write_transfer,ROOT

class TransferTests(unittest.TestCase):
    def setUp(self):
        self.models=load_configs()
        self.evidence=json.loads((ROOT/'configs/transfer/published_verification.json').read_text())
    def test_four_unrun_cells_tying_and_domain_censoring(self):
        p=generalize(self.models)['predictions'];self.assertEqual(len(p),4)
        self.assertEqual({x['model_id'] for x in p},{'Qwen/Qwen2.5-3B','Qwen/Qwen2.5-7B'})
        for x in p:
            self.assertFalse(x['weights_loaded']);self.assertFalse(x['target_gpu_measured'])
            v=x['config_capped_limit_tokens'];self.assertLessEqual(v['min'],v['central']);self.assertLessEqual(v['central'],v['max'])
            self.assertLessEqual(v['max'],x['config_ceiling_tokens'])
            if x['model_id'].endswith('3B'):
                self.assertEqual(x['unique_parameters'],3085938688);self.assertEqual(x['kv_bytes_per_token'],36864)
                self.assertEqual(v['central'],32768)
            else:
                self.assertEqual(x['unique_parameters'],7615616512);self.assertEqual(x['kv_bytes_per_token'],57344)
    def test_published_comparison_checks_compatible_architecture_and_both_units(self):
        c=published_comparison(self.models,self.evidence)
        self.assertFalse(c['full_peak_threshold_externally_validated'])
        self.assertEqual(c['comparisons'][0]['predicted_block_bytes'],589824)
        self.assertLess(c['comparisons'][0]['display_error_percent_if_binary'],1)
        self.assertGreater(c['comparisons'][0]['display_error_percent_if_decimal'],5)
        bad=copy.deepcopy(self.evidence);bad['tokens_per_block']=32
        with self.assertRaises(ValueError):published_comparison(self.models,bad)
    def test_refuses_unreviewed_calibration_and_reversed_budget(self):
        with self.assertRaises(ValueError):generalize(self.models,assumptions={'fitted':True})
        with self.assertRaises(ValueError):generalize(self.models,[{'gpu':'synthetic','usable_gib_min':'30','usable_gib_central':'20','usable_gib_max':'10'}])
    def test_provenance_is_checked_and_artifacts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'transfer';write_transfer(out)
            self.assertTrue((out/'transfer.md').is_file())
            with self.assertRaises(ValueError):write_transfer(out)
            p=json.loads((ROOT/'configs/transfer/provenance.json').read_text());p['models'][0]['config_sha256']='0'*64
            path=Path(d)/'bad.json';path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_configs(path)
if __name__=='__main__':unittest.main()
