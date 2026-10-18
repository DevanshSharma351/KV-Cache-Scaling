"""Temporary CPU dispatch claims only; no real project's allowance is consumed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('external_claim',ROOT/'scripts/claim_external_attempt.py')
claim=importlib.util.module_from_spec(spec);spec.loader.exec_module(claim)


class ExternalClaimTests(unittest.TestCase):
    def test_local_claim_is_consumed_before_dispatch_and_cannot_be_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'synthetic-only'
            result=claim.reserve(ROOT/'prereg/formal-2026-10-18-t4/receipt.json',str(uuid4()),131072,state)
            ledger=(state/'global_deliberate_attempt.json').read_bytes()
            intent=(state/'external_dispatch_intent.json').read_bytes()
            self.assertEqual(json.loads(ledger)['state'],'consumed_before_launch')
            self.assertEqual(json.loads(intent)['state'],'consumed_before_external_dispatch')
            with self.assertRaises((ValueError,RuntimeError)):
                claim.reserve(ROOT/'prereg/formal-2026-10-18-t4/receipt.json',str(uuid4()),131072,state)
            self.assertEqual((state/'global_deliberate_attempt.json').read_bytes(),ledger)
            self.assertEqual((state/'external_dispatch_intent.json').read_bytes(),intent)
            self.assertEqual(result['length'],131072)

    def test_bad_length_does_not_create_a_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'synthetic-only'
            with self.assertRaises(ValueError):
                claim.reserve(ROOT/'prereg/formal-2026-10-18-t4/receipt.json',str(uuid4()),True,state)
            self.assertFalse(state.exists())
