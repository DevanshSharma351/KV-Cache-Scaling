"""Scientific completion gates; CPU-only fixtures never become measured data."""
import unittest
from kv_cache_scaling.reporting import evidence_gaps


def complete_fixture():
    rows=[]
    for mode,lengths,allocators in [('chunked_dynamic',[512,2048,8192,32768],['default']),
                                    ('static_cache',[512,2048,8192],['default']),
                                    ('workspace_control',[512],['default','expandable'])]:
        rows += [dict(measurement=mode,length=n,allocator=a,diagnostic_repeats=3)
                 for n in lengths for a in allocators]
    return dict(acceptance=dict(five_length_paired_primary_complete=True,discrepancy_coverage_pass=True),
                allocator_boundary_deltas=[{}],diagnostic_comparisons=rows,
                probes=[dict(measurement='kernel_trace')],
                held_out_validation=[dict(allocator=a,length=114688,repeats=3) for a in ('default','expandable')],
                hardware_brackets=[dict(smallest_observed_oom_tokens=131072,consistent_monotone_bracket=True)])


class ReportEvidenceTests(unittest.TestCase):
    def test_complete_evidence_has_no_missing_experiment(self):
        self.assertEqual(evidence_gaps(complete_fixture()),[])

    def test_one_allocator_workspace_controls_are_insufficient(self):
        a=complete_fixture()
        a['diagnostic_comparisons']=[r for r in a['diagnostic_comparisons'] if r['allocator']!='expandable']
        self.assertIn('workspace_control diagnostic grid incomplete.',evidence_gaps(a))

    def test_held_out_repeats_and_separate_trace_are_required(self):
        a=complete_fixture();a['held_out_validation'][1]['repeats']=2;a['probes']=[]
        self.assertIn('Paired prospective held-out validation at L=114688.',evidence_gaps(a))
        self.assertIn('Separate efficient-SDPA kernel trace.',evidence_gaps(a))

    def test_safe_budget_or_inconsistent_bracket_cannot_substitute_for_oom(self):
        a=complete_fixture();a['allocator_boundary_deltas']=[]
        for bracket in ({'smallest_observed_oom_tokens':None,'consistent_monotone_bracket':True},
                        {'smallest_observed_oom_tokens':65536,'consistent_monotone_bracket':False}):
            a['hardware_brackets']=[bracket]
            gaps=evidence_gaps(a)
            self.assertIn('Paired safe-budget boundary search.',gaps)
            self.assertTrue(any('Actual OOM boundary unobserved' in item for item in gaps))



class ReportAlignmentTests(unittest.TestCase):
    def test_original_commitment_is_read_without_rewriting(self):
        from kv_cache_scaling.reporting import report_context
        context=report_context()
        self.assertEqual(context['commit_timestamp'],'2026-10-18T19:54:52+05:30')
        self.assertEqual([r[0] for r in context['prediction_rows']],[512,1024,2048,4096,8192])
        self.assertEqual(context['prediction_rows'][0],(512,14680064,3142216704,3679087616))
        self.assertEqual(context['prediction_rows'][-1],(8192,234881024,3964038144,4500909056))

    def test_generated_pdf_covers_deliverables_and_preserves_incomplete_status(self):
        import importlib.util
        if not all(importlib.util.find_spec(name) for name in ('reportlab','pypdf','PIL')):
            self.skipTest('PDF dependencies are exercised in the full report environment')
        import json
        from pathlib import Path
        import shutil
        import tempfile
        from pypdf import PdfReader
        from kv_cache_scaling.reporting import build_report,ROOT,DELIVERABLE_MAP,PHASE_FLOW
        with tempfile.TemporaryDirectory() as temporary:
            staging=Path(temporary)/'analysis';staging.mkdir()
            source=ROOT/'data/processed/project-results'
            shutil.copy2(source/'analysis.json',staging/'analysis.json')
            (staging/'figures').symlink_to(ROOT/'artifacts/figures/project-results',target_is_directory=True)
            output=Path(temporary)/'report'
            result=build_report(staging,source/'transfer',output)
            self.assertEqual(result['status'],'FINAL_WITH_MEASUREMENT_LIMITATIONS')
            text=' '.join(' '.join(page.extract_text() for page in PdfReader(output/'report.pdf').pages).split())
            for title,_ in DELIVERABLE_MAP+PHASE_FLOW:
                self.assertIn(title,text)
                self.assertIn(title,(output/'report.md').read_text())
            for evidence in ('2026-10-18T19:54:52+05:30','3679087616','4500909056',
                             'reset_peak_memory_stats()', 'max_memory_allocated()',
                             'FINAL_PRESENTATION.ipynb','Completed targeted OOM boundary',
                             'transferred HF context thresholds are unvalidated',
                             'Byte-level discrepancy example', '-1339293184', '-1350870016'):
                self.assertIn(evidence,text)
            for ownership in ('Person 1','Person 2','55%','45%'):
                self.assertNotIn(ownership,text)
            self.assertEqual(json.loads((output/'provenance.json').read_text())['status'],result['status'])


if __name__=='__main__':unittest.main()
