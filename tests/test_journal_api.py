"""Public API integration: temporary CPU-only persistence, never GPU records."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from kv_cache_scaling import journal,coordinator_store


class JournalApiTests(unittest.TestCase):
    def test_public_api_reuses_live_coordinator_store_and_mirrors(self):
        self.assertIs(journal.RunStore,coordinator_store.RunStore)
        self.assertIs(journal.AttemptLedger,coordinator_store.AttemptLedger)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);run_id=str(uuid4())
            with journal.RunStore(base/'local',base/'mirror',run_id) as store:
                row=journal.append_event(store,{'event':'demo_checkpoint','label':'CPU fixture only'})
                self.assertEqual(row['sequence'],1)
                self.assertEqual((store.local/'events.jsonl').read_bytes(),(store.mirror/'events.jsonl').read_bytes())
            with journal.RunStore(base/'local',base/'mirror',run_id) as resumed:
                self.assertEqual(resumed.events,[row])

    def test_public_api_cannot_bypass_lock_mirror_or_reserved_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);store=journal.RunStore(base/'local',base/'mirror',str(uuid4()))
            with self.assertRaises(RuntimeError):journal.append_event(store,{'event':'demo_checkpoint'})
            with store:
                for bad in ({},{'event':''},{'event':'demo_checkpoint','run_id':'replacement'}):
                    with self.assertRaises(ValueError):journal.append_event(store,bad)
                with patch.object(coordinator_store,'atomic_bytes',side_effect=OSError('synthetic mirror outage')):
                    with self.assertRaises(OSError):journal.append_event(store,{'event':'demo_checkpoint'})
            with self.assertRaises(TypeError):journal.append_event(base/'unsafe.jsonl',{'event':'demo_checkpoint'})

if __name__=='__main__':unittest.main()
