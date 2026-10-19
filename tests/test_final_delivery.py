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

if __name__=='__main__':unittest.main()
