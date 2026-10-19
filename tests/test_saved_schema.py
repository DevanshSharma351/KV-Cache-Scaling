"""Validate real saved records; optional JSON Schema dependency is used in report CI."""
import importlib.util
import json
from pathlib import Path
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(importlib.util.find_spec('jsonschema'),'JSON Schema checks run in the notebook/report environment')
class SavedSchemaTests(unittest.TestCase):
    def validator(self):
        from jsonschema import Draft202012Validator, FormatChecker
        schema=json.loads((ROOT/'docs/measurement.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema,format_checker=FormatChecker())

    def test_all_eighty_actual_workers_and_envelopes_validate(self):
        validator=self.validator();count=0
        with zipfile.ZipFile(ROOT/'artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip') as z:
            for name in z.namelist():
                if name.startswith('state/runs/') and name.endswith('/events.jsonl'):
                    for row in map(json.loads,z.read(name).splitlines()):
                        if row['event']=='probe_result':
                            validator.validate(row);validator.validate(row['metrics']);count+=1
        self.assertEqual(count,80)

    def test_empty_success_and_negative_bytes_are_rejected(self):
        validator=self.validator()
        self.assertFalse(validator.is_valid({'schema_version':'primary-probe-v1','status':'success'}))
        with zipfile.ZipFile(ROOT/'artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip') as z:
            name=next(n for n in z.namelist() if n.startswith('local/') and '/worker/' in n and n.endswith('.json'))
            row=json.loads(z.read(name));row['peak_allocated_bytes']=-1
        self.assertFalse(validator.is_valid(row))

if __name__=='__main__':unittest.main()
