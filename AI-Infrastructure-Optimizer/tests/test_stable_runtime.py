import json,tempfile,unittest
from pathlib import Path
from src.stable_runtime_data import build
from src.stable_runtime_records import Store
from src.stable_runtime_protocol import option_complete,VARIANTS

class TestStableRuntime(unittest.TestCase):
    def test_checkpoint_resume_partial_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);p=Path(folder,'natural/task');p.mkdir(parents=True);(p/'partial.json').write_text('{')
            self.assertIsNone(store.load('natural/task'));store.save('natural/task',{'value':3});self.assertEqual(Store(folder).load('natural/task')['value'],3)
            with self.assertRaises(FileExistsError):store.save('natural/task',{'value':4})
            self.assertEqual((p/'partial.json').read_text(),'{')
    def test_fresh64_and_no_old_stem(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=build(folder);self.assertEqual(len(rows),64);self.assertEqual(len({r['task_id'] for r in rows}),64)
            Path(folder,'past.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):build(folder)
    def test_output_completion_uses_request_not_reference(self):
        q='Q\nReturn only the option letter.'
        self.assertTrue(option_complete(q,' B '));self.assertFalse(option_complete(q,'B)'));self.assertFalse(option_complete('Write a sentence.','A'));self.assertFalse(option_complete(q,'Answer B'))
    def test_factors_are_complete_and_same_stop_per_policy(self):
        self.assertEqual(len(VARIANTS),9)
        self.assertEqual({r['method'] for r in VARIANTS.values() if r['format_stop']},{'full','stable1','sketch'})
