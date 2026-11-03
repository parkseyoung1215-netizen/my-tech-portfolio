from collections import Counter
import contextlib
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import solver300_dataset as data
from src import solver300_runtime as runtime
from src import solver300_experiment as experiment
from src.solver300_protocol import content_extract


def profile(correct=20):
    return runtime.make_profile([{'inferred_category':c,'model':m,'correct':i<correct}
        for c in experiment.CATEGORIES for m in (runtime.SMALL,runtime.LARGE) for i in range(20)])


def fake_generate(model,question):
    return {'model':model,'answer':'A' if runtime.features(question)['answer_kind']=='choice' else '0',
        'status':'generated','error':'','input_tokens':10,'output_tokens':1,'eval_seconds':.01,'load_seconds':0,
        'elapsed_seconds':.01,'done_reason':'stop'}


class TestSolver300(unittest.TestCase):
    def test_balanced_unique_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            cal,ev=data.build(folder)
        self.assertEqual(len(cal),100)
        self.assertEqual(Counter(t['category'] for t in ev),{c:60 for c in experiment.CATEGORIES})
        self.assertEqual(len({data.task_key(t['question']) for t in ev}),300)
        self.assertFalse({data.task_key(t['question']) for t in cal}&{data.task_key(t['question']) for t in ev})
        self.assertTrue(all('originally owned' not in t['question'] for t in cal))

    def test_duplicate_stem_is_not_new_after_choice_shuffle(self):
        self.assertEqual(data.task_key('What is X?\nA) a\nB) b\nC) c\nD) d'),data.task_key('What is X?\nA) d\nB) a\nC) b\nD) c'))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'old.json').write_text(json.dumps({'question':'Which data structure is last-in, first-out?'}))
            with self.assertRaises(ValueError): data.build(folder)

    def test_all_reference_code_and_arithmetic_labels(self):
        with tempfile.TemporaryDirectory() as folder: _,tasks=data.build(folder)
        checks=experiment.check_references(tasks)
        self.assertEqual(len(checks),60)
        self.assertTrue(all(c['matches'] for c in checks))

    def test_confidence_is_calibrated_and_difficulty_conservative(self):
        q='What is a stack?\nA) a\nB) b\nC) c\nD) d'
        self.assertEqual(runtime.choose(q,'confidence',profile())['model'],runtime.SMALL)
        self.assertEqual(runtime.choose(q,'confidence',profile(19))['model'],runtime.LARGE)
        hard='Reason about P and Q.\nA) a\nB) b\nC) c\nD) d'
        self.assertEqual(runtime.choose(hard,'confidence',profile())['model'],runtime.LARGE)

    def test_four_policies_and_tool_guard(self):
        self.assertEqual(runtime.choose('29 * 47','fixed',{})['model'],runtime.LARGE)
        self.assertEqual(runtime.choose('29 * 47','rule',{})['source'],'tool')
        self.assertEqual(runtime.choose('29 * 47','hybrid',profile())['source'],'tool')
        self.assertEqual(runtime.choose('29 * 47','confidence',profile())['source'],'model')
        self.assertFalse(runtime.features("Read the passage: '2 + 3' is a password.\nA) a\nB) b\nC) c\nD) d")['tool_supported'])

    def test_runtime_accepts_no_labels(self):
        self.assertEqual(list(inspect.signature(runtime.execute).parameters),['question','strategy','profile'])
        self.assertEqual(list(inspect.signature(runtime.generate).parameters),['model','question'])

    def test_tool_does_not_generate(self):
        with patch.object(runtime,'generate') as generate:
            row=runtime.execute('29 * 47','hybrid',profile())
        generate.assert_not_called()
        self.assertEqual(row['prediction'],'1363')
        self.assertEqual(row['attempts'],[])

    def test_format_error_fallback_without_gold(self):
        q='What is X?\nA) a\nB) b\nC) c\nD) d'
        first=fake_generate(runtime.SMALL,q);first['answer']='A or B'
        second=fake_generate(runtime.LARGE,q);second['answer']='C'
        with patch.object(runtime,'generate',side_effect=[first,second]): row=runtime.execute(q,'rule',{})
        self.assertTrue(row['fallback']);self.assertEqual(row['format_errors'],1)
        self.assertEqual(row['prediction'],'C');self.assertEqual(len(row['attempts']),2)

    def test_well_formatted_wrong_content_does_not_trigger_oracle(self):
        q='What is X?\nA) a\nB) b\nC) c\nD) d'
        with patch.object(runtime,'generate',side_effect=lambda m,q:fake_generate(m,q)) as generate:
            row=runtime.execute(q,'rule',{})
        self.assertEqual(row['prediction'],'A');generate.assert_called_once()

    def test_protocol_rejects_ambiguity(self):
        for text in ['A or B','FINAL_ANSWER: A\nFINAL_ANSWER: B','3,000','The answer is B']:
            self.assertIsNone(content_extract(text,'choice')[0])
        self.assertEqual(content_extract('FINAL_ANSWER: B','choice')[0],'B')
        self.assertEqual(content_extract('+0012','integer')[0],'12')

    def test_offline_pipeline_in_temporary_test_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            cal,tasks=data.build(folder)
            tiny=[next(t for t in tasks if t['category']==c) for c in experiment.CATEGORIES]
            snapshot={'time':1,'duration':0,'error':'','processes':{'1':{'cpu_seconds':1,'rss_kib':1024}}}
            with patch.object(experiment,'build',return_value=(cal[:5],tiny)), \
                 patch.object(experiment,'check_references',return_value=[{'matches':True}]), \
                 patch.object(experiment,'check_ollama'),patch.object(experiment,'process_snapshot',return_value=snapshot), \
                 patch.object(runtime,'generate',side_effect=fake_generate), \
                 patch.object(experiment.urllib.request,'urlopen') as urlopen,contextlib.redirect_stdout(io.StringIO()):
                urlopen.return_value.__enter__.return_value=io.StringIO('{"models":[]}')
                original=Path(folder)/'keep.csv';original.write_bytes(b'keep')
                prefix=experiment.run(folder)
                self.assertEqual(original.read_bytes(),b'keep')
                self.assertTrue(Path(f'{prefix}_comparison.csv').is_file())
                self.assertTrue(Path(f'{prefix}_report.md').is_file())


if __name__=='__main__': unittest.main()
