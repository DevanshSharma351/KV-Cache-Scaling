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

if __name__=='__main__':unittest.main()
