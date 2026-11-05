import inspect,json,tempfile,unittest
from pathlib import Path
from src.dlr_opt_data import build,score

class TestDLROptimization(unittest.TestCase):
    def test_dataset_size_freshness_and_long_design(self):
        with tempfile.TemporaryDirectory() as folder:
            fresh,long=build(folder);self.assertEqual(len(fresh),32);self.assertEqual(set(long),{'long_high','long_low'})
            self.assertTrue(all(len(t['target_words'])==40 for ts in long.values() for t in ts))
            self.assertTrue(all(len(set(t['target_words']))==1 for t in long['long_high']))
            self.assertTrue(all(len(set(t['target_words']))>1 for t in long['long_low']))
            Path(folder,'prior.json').write_text(json.dumps(fresh))
            with self.assertRaises(ValueError):build(folder)
    def test_copy_completion_not_just_prefix(self):
        t={'score_kind':'copy_words','target_words':['echo']*40}
        self.assertFalse(score(t,'echo '*39)['content_correct']);self.assertTrue(score(t,'echo '*40)['content_correct']);self.assertFalse(score(t,'echo '*41)['content_correct'])
    def test_reference_not_in_runtime_gate_interface_source(self):
        source=Path('src/dlr_opt_runtime.py').read_text()
        self.assertNotIn("task['reference']",source);self.assertNotIn('expected_object',source)
        # Per-request counter collection is allowed; there is no scalar gate read in forward.
        f=source[source.index('    def forward('):source.index('    def generate(')]
        self.assertNotIn('.item()',f);self.assertNotIn('mx.eval(',f)
    def test_custom_branch_precedes_weight_reads(self):
        s=Path('src/dlr_opt_kernels.py').read_text();self.assertIn('if (reuse[0])',s);self.assertLess(s.index('if (reuse[0])'),s.index('wg[row * D + col]'))
