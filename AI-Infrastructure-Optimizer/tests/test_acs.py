import inspect,json,tempfile,unittest
from pathlib import Path
from src.acs_policy import boundary,request_kind,requested_keys,threshold,limit
from src.acs_data import build
from src.acs_scoring import score,json_equal

class TestACS(unittest.TestCase):
    def test_json_boundaries_respect_strings_keys_duplicates(self):
        q='Return only a JSON object with keys code and quantity. Values: code="a", quantity=2.'
        self.assertEqual(requested_keys(q),['code','quantity']);self.assertFalse(boundary(q,'{"code":"a"}'))
        self.assertTrue(boundary(q,'{"code":"}","quantity":2}'));self.assertFalse(boundary(q,'{"code":"a","quantity":2,"quantity":3}'))
        self.assertFalse(boundary(q,'```json\n{"code":"a","quantity":2}\n```'))
    def test_adaptive_abbreviation_and_minimum_are_not_gold(self):
        q='Use exactly one sentence to preserve these facts.'
        self.assertTrue(boundary(q,'Dr.'));self.assertFalse(boundary(q,'Dr.',adaptive=True));self.assertFalse(boundary(q,'It is done.',adaptive=True));self.assertTrue(boundary(q,'The doctor now works at Studio 44.',adaptive=True))
    def test_choices_and_unknown_fallback(self):
        q='Q\nReturn only the option letter.'
        self.assertTrue(boundary(q,'B'));self.assertFalse(boundary(q,'B)'));self.assertFalse(boundary('Write code.','A'))
        self.assertEqual(limit(q,'fixed8'),8);self.assertEqual(limit(q,'baseline'),16);self.assertEqual(limit('Write code.','baseline'),48)
    def test_thresholds_fixed_and_inputs_only(self):
        self.assertEqual(list(inspect.signature(threshold).parameters),['question','input_tokens','method'])
        q='Q\nReturn only the option letter.';self.assertEqual(threshold(q,10,'confidence'),.9);self.assertEqual(threshold(q,10,'adaptive'),.75);self.assertEqual(threshold(q,129,'adaptive'),.8)
    def test_fresh_splits_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=build(folder);self.assertEqual(len(rows),64);self.assertEqual(sum(r['score_kind']=='choice' for r in rows),32)
            Path(folder,'old.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):build(folder)
    def test_content_format_and_incomplete_are_separate(self):
        t={'score_kind':'constraints','question':'Use one sentence. "Dr. Leno is a doctor. Dr. Leno works at Studio 9."','reference':'','required':['Dr. Leno','doctor','Studio 9']}
        s=score(t,'Dr. Leno is a doctor working at Studio 9.');self.assertTrue(s['content_correct']);self.assertTrue(s['format_ok']);self.assertFalse(s['legacy_grade']['correct'])
        bad=score(t,'Dr.');self.assertFalse(bad['content_correct']);self.assertTrue(bad['incomplete_output'])
        two=score(t,'Dr. Leno is a doctor. Dr. Leno works at Studio 9.');self.assertTrue(two['content_correct']);self.assertFalse(two['format_ok'])
        self.assertFalse(json_equal({'x':[True]},{'x':[1]}))
