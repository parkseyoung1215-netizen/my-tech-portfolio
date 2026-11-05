import tempfile,json,unittest
from pathlib import Path
from src import alsr_dataset as data,alsr_policy as policy

class TestALSR(unittest.TestCase):
    def test_splits_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            splits=data.build(folder);self.assertEqual({k:len(v) for k,v in splits.items()},{'training':12,'calibration':8,'pilot':32})
            ids=[t['task_id'] for vv in splits.values() for t in vv];self.assertEqual(len(ids),len(set(ids)))
            Path(folder,'past.json').write_text(json.dumps(splits))
            with self.assertRaises(ValueError):data.build(folder)

    def test_skip_pool_and_random_matching(self):
        for n in (24,28):
            pool=policy.pool_for(n);self.assertEqual(len(pool),8);self.assertTrue(all(2<l<n-2 for l in pool));self.assertEqual(len(policy.fixed_layers(n)),4)
            for k in range(5):
                a=policy.random_layers(pool,k,'q',42);self.assertEqual(len(a),k);self.assertEqual(a,policy.random_layers(pool,k,'q',42))

    def test_risk_threshold_budget_and_zero_skip(self):
        pool=list(range(8));risks={l:l*.1 for l in pool}
        self.assertEqual(policy.choose_by_risk(risks,0,pool),[])
        self.assertEqual(policy.choose_by_risk(risks,1,pool),[0,1,2,3])

    def test_flops_are_reduced_only_with_actual_execution_trace(self):
        c={'hidden_size':896,'intermediate_size':4864,'vocab_size':151936,'num_key_value_heads':2,'num_attention_heads':14}
        full=[{'query_tokens':10,'context_tokens':10,'executed':list(range(24))}];skipped=[dict(full[0],executed=list(range(20)))]
        self.assertLess(policy.estimated_matmul_flops(c,skipped),policy.estimated_matmul_flops(c,full))
        self.assertEqual(policy.estimated_matmul_flops(c,full,100)-policy.estimated_matmul_flops(c,full),100)

    def test_content_and_format_separate_without_gold_guided_extraction(self):
        t={'question':'Q\nA) 15\nB) 14\nC) 12\nD) 16','reference':'A','score_kind':'choice'}
        self.assertTrue(data.grade(t,'A) 15')['correct']);self.assertFalse(data.grade(t,'A) 15')['format_ok'])
        self.assertFalse(data.grade(t,'A) 99')['correct'])
        self.assertTrue(data.grade(t,'The correct option is A) 15.')['correct'])
        self.assertFalse(data.grade(t,'The correct option is A) 1.')['correct'])
        self.assertTrue(data.grade(t,'The correct answer is 15.')['correct'])
        t={'score_kind':'json','expected_object':{'x':3}}
        self.assertTrue(data.grade(t,'```json\n{"x":3}\n```')['correct']);self.assertFalse(data.grade(t,'```json\n{"x":3}\n```')['format_ok'])
        self.assertFalse(data.grade(t,'{"x":2,"x":3}')['correct'])

    def test_ridge_prediction_is_clamped_and_no_reference_input(self):
        m={'mean':[0,0],'scale':[1,1],'weights':[-1,0,0]}
        self.assertEqual(policy.predict_ridge([5,6],m),0)

if __name__=='__main__':unittest.main()
