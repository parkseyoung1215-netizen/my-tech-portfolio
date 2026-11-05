import inspect,json,tempfile,unittest
from pathlib import Path
from src.dlr_policy import reuse_allowed,estimate
from src.dlr_data import fresh,score

class TestDLR(unittest.TestCase):
    def test_gate_is_input_only_and_age_limited(self):
        self.assertEqual(list(inspect.signature(reuse_allowed).parameters),['distance','age','method'])
        self.assertTrue(reuse_allowed(.3,0,'dlr'));self.assertFalse(reuse_allowed(.3,0,'conservative'))
        for x in (float('nan'),float('inf'),-.1):self.assertFalse(reuse_allowed(x,0,'dlr'))
        self.assertFalse(reuse_allowed(.01,1,'dlr'));self.assertFalse(reuse_allowed(0,0,'baseline'))
    def test_freshness_and_balanced_formats(self):
        with tempfile.TemporaryDirectory() as d:
            tasks=fresh(d);self.assertEqual(len(tasks),32);self.assertEqual(sum(t['score_kind']=='json' for t in tasks),8)
            Path(d,'nested').mkdir();Path(d,'nested','previous.json').write_text(json.dumps(tasks))
            with self.assertRaises(ValueError):fresh(d)
    def test_typed_json_duplicates_and_format_separation(self):
        t={'score_kind':'json','expected_object':{'code':'x','quantity':1}}
        self.assertTrue(score(t,'{"quantity":1,"code":"x"}')['content_correct'])
        self.assertFalse(score(t,'{"quantity":true,"code":"x"}')['content_correct'])
        self.assertFalse(score(t,'{"quantity":1,"quantity":1,"code":"x"}')['content_correct'])
        s=score(t,'Answer: {"quantity":1,"code":"x"}');self.assertTrue(s['content_correct']);self.assertFalse(s['format_ok'])
    def test_unambiguous_short_and_choice_content(self):
        t={'score_kind':'short','aliases':['red']};self.assertTrue(score(t,'The scarf is red.')['content_correct']);self.assertFalse(score(t,'red or blue')['content_correct']);self.assertFalse(score(t,'It is not red.')['content_correct'])
        t={'score_kind':'choice','reference':'B'};self.assertTrue(score(t,'B) The right option.')['content_correct']);self.assertFalse(score(t,'A or B')['content_correct'])
    def test_flops_mlp_reduction_and_nonzero_attention(self):
        c={'hidden_size':8,'intermediate_size':16,'vocab_size':20,'num_key_value_heads':1,'num_attention_heads':2}
        a=estimate(c,[[1,10,28,28]]);b=estimate(c,[[1,10,28,24]])
        self.assertEqual(a-b,4*6*8*16);self.assertGreater(b,0)
