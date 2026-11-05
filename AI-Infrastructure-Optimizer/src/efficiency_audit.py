"""Independently reconstruct measured parsing, unique calls and repeated scores."""
import csv,hashlib,json,sys,statistics
from collections import Counter
from pathlib import Path
if __package__:
    from . import efficiency_runtime as runtime
    from .efficiency_protocol import extract,extract_batch
    from .efficiency_repeat import read_csv,aggregate,summarize,paired,frozen_check
    from .solver300_dataset import task_key
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import efficiency_runtime as runtime
    from efficiency_protocol import extract,extract_batch
    from efficiency_repeat import read_csv,aggregate,summarize,paired,frozen_check
    from solver300_dataset import task_key
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def parse_csv_value(value):
    if value=='True':return True
    if value=='False':return False
    if value=='':return None
    try:return json.loads(value)
    except (ValueError,TypeError):return value

def decode_row(row):
    # Text remains text, notably numeric predictions and expected answers.
    string_fields={'method','question','primary_answer','primary_prediction','primary_source','primary_parse_mode',
        'tool_error','final_answer','final_prediction','final_source','verification_answer','verification_prediction',
        'verification_parse_mode','batch_envelope_error','category','expected','task_format','verification_outcome',
        'block_id','call_id','stage','model','request_sha256','answer','status','error','done_reason','parse_mode',
        'cost_status','resource_error'}
    result={}
    for k,v in row.items():
        if k in string_fields:result[k]=v if v!='' else (None if k.endswith('_prediction') else '')
        else:result[k]=parse_csv_value(v)
    return result

def audit(prefix):
    prefix=Path(prefix).resolve()
    def js(suffix):return json.loads(Path(f'{prefix}_{suffix}.json').read_text())
    manifest,metadata,profile=js('manifest'),js('metadata'),js('profile')
    frozen_check(manifest)
    for name,digest in metadata['frozen_sha256'].items():assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==digest
    rows=[decode_row(r) for r in read_csv(f'{prefix}_evaluation.csv')]
    calls=[decode_row(a) for a in read_csv(f'{prefix}_calls.csv')]
    blocks=[decode_row(b) for b in read_csv(f'{prefix}_blocks.csv')]
    tasks={t['question']:t for t in manifest['tasks']};cal={t['question']:t for t in manifest['calibration']}
    assert len(tasks)==150
    assert not {task_key(q) for q in tasks}&{task_key(q) for q in cal}
    plans={(r['method'],r['question']):r['decision'] for r in js('plans')}
    ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)
    usage=Counter();seen=set();categories=[];failures=[];envelopes=Counter()
    batch_cache={}
    for a in calls:
        payload=a['request_payload'];assert payload['model']==runtime.MODEL
        assert set(payload)=={'model','prompt','stream','options'}
        assert payload['options']=={'temperature':0,'seed':42,'num_predict':8 if a['stage']=='primary' and a['method']=='budget_cap' else (128 if a['stage']=='batch_verify' else 64)}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
        if a['stage']=='batch_verify':
            items=json.loads(payload['prompt'].split('\n',1)[1])
            expected_prompt=('Independently verify each request and candidate below. Return ONLY a JSON object with an "answers" object '
                'mapping every exact id to its correct single option letter or integer string. No explanations.\n'+json.dumps(items,ensure_ascii=False))
            assert payload['prompt']==expected_prompt
            assert [item['id'] for item in items]==[f'Q{i}' for i in range(len(items))]
            decoded=extract_batch(a['answer'],[item['question'] for item in items],[runtime.features(item['question'])['answer_kind'] for item in items])
            batch_cache[a['call_id']]=(items,decoded)
            assert a['format_compliant']==decoded['format_compliant']
            envelopes['fenced' if decoded['envelope_fenced'] else 'plain']+=1
    for r in rows:
        rep,m,q=r['repetition'],r['method'],r['question'];key=(rep,m,q)
        assert q in tasks and key not in seen;seen.add(key)
        d=r['decision'];assert d==plans[m,q]==runtime.preflight(q,m,profile)
        p=extract(q,r['primary_answer'],d['answer_kind'])
        assert p['prediction']==r['primary_prediction'] and p['format_compliant']==r['primary_format_compliant']
        ids=r['call_ids'];aa=[ci[id] for id in ids]
        for a in aa:assert a['method']==m and a['repetition']==rep and a['block_id']==r['block_id'];usage[a['call_id']]+=1
        prim=[a for a in aa if a['stage']=='primary'];vv=[a for a in aa if a['stage']!='primary']
        if r['primary_source']=='model':
            assert len(prim)==1 and prim[0]['answer']==r['primary_answer']
            instruction='Return only the option letter A, B, C or D.' if d['answer_kind']=='choice' else 'Return only the calculated integer.'
            assert prim[0]['request_payload']['prompt']==q+'\n'+instruction+' Do not explain.'
            assert prim[0]['format_compliant']==p['format_compliant']
        else:assert not prim and r['tool_calls']==1
        if vv:
            assert len(vv)==1 and not d['skip_verifier'];a=vv[0]
            if a['stage']=='batch_verify':
                items,decoded=batch_cache[a['call_id']];i=next(i for i,item in enumerate(items) if item['question']==q)
                assert items[i]['candidate']==r['primary_answer'];v=decoded['items'][i]
            else:
                prompt=(q+'\nCandidate response: '+json.dumps(r['primary_answer'],ensure_ascii=False)+
                    '\nIndependently check the candidate and solve the original request. The candidate may be wrong. '+
                    ('Return only the correct option letter A, B, C or D.' if d['answer_kind']=='choice' else 'Return only the correct integer.')+' Do not explain.')
                assert a['request_payload']['prompt']==prompt
                v=extract(q,a['answer'],d['answer_kind']);assert v['format_compliant']==a['format_compliant']
            assert r['verification_prediction']==v['prediction']
            assert r['verification_format_compliant']==v['format_compliant']
            expected_final=v if v['prediction'] is not None else p
        else:assert d['skip_verifier'];expected_final=p
        assert r['final_prediction']==expected_final['prediction']
        assert r['final_format_compliant']==expected_final['format_compliant']
        assert r['expected']==tasks[q]['expected']
        pc=p['prediction'] is not None and p['prediction']==tasks[q]['expected']
        assert r['primary_correct']==pc
        assert r['final_correct']==(expected_final['prediction'] is not None and expected_final['prediction']==tasks[q]['expected'])
        assert r['answered']==(r['final_prediction'] is not None)
        assert r['verification_executed']==bool(vv)
        fc=r['final_correct']
        outcome=('not_executed' if not vv else 'beneficial' if not pc and fc else 'harmful' if pc and not fc
                 else 'redundant_correct_by_outcome' if pc and fc else 'still_wrong_or_unresolved')
        assert r['verification_outcome']==outcome
        if not r['final_correct'] or not r['final_format_compliant'] or r['verification_outcome']=='harmful':failures.append(r)
    assert seen=={(rep,m,q) for rep in range(1,manifest['repeats']+1) for m in runtime.METHODS for q in tasks}
    assert set(usage)==set(ci)
    assert all(n==1 or ci[id]['stage']=='batch_verify' for id,n in usage.items())
    calrows=read_csv(f'{prefix}_calibration.csv')
    for r in calrows:
        q=r['question'];assert r['expected']==cal[q]['expected']
        p=extract(q,r['primary_answer'],runtime.features(q)['answer_kind'])
        v=extract(q,r['verification_answer'],runtime.features(q)['answer_kind'])
        final=v['prediction'] if v['prediction'] is not None else p['prediction']
        assert r['primary_correct']==str(p['prediction']==r['expected'])
        assert r['final_correct']==str(final==r['expected'])
        for k in ('primary_correct','final_correct'):r[k]=r[k]=='True'
    assert runtime.make_profile(calrows)==profile
    reconstructed=[]
    for rep in range(1,manifest['repeats']+1):
        reconstructed.extend(summarize([r for r in rows if r['repetition']==rep],[a for a in calls if a['repetition']==rep],[b for b in blocks if b['repetition']==rep],rep))
    recorded=[decode_row(r) for r in read_csv(f'{prefix}_repeat_summary.csv')]
    assert reconstructed==recorded
    ag=aggregate(reconstructed);pa=paired(reconstructed,rows)
    assert ag==[decode_row(r) for r in read_csv(f'{prefix}_aggregate.csv')]
    assert pa==[decode_row(r) for r in read_csv(f'{prefix}_paired.csv')]
    for rep in range(1,manifest['repeats']+1):
        for m in runtime.METHODS:
            for category in sorted({t['category'] for t in tasks.values()}):
                rr=[r for r in rows if r['repetition']==rep and r['method']==m and r['category']==category]
                categories.append({'repetition':rep,'method':m,'category':category,'requests':len(rr),
                    'accuracy':sum(r['final_correct'] for r in rr)/len(rr),
                    'answer_rate':sum(r['answered'] for r in rr)/len(rr),
                    'format_compliance':sum(r['final_format_compliant'] for r in rr)/len(rr),
                    'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
                    'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr)})
    per_question=[];primary_diagnostics=[]
    for m in runtime.METHODS:
        for q in tasks:
            rr=[r for r in rows if r['method']==m and r['question']==q]
            values=[r['completion_latency_seconds'] for r in rr]
            per_question.append({'method':m,'question':q,'category':tasks[q]['category'],'repeats':len(rr),
                'content_accuracy':sum(r['final_correct'] for r in rr)/len(rr),
                'format_compliance':sum(r['final_format_compliant'] for r in rr)/len(rr),
                'latency_mean_seconds':statistics.mean(values),'latency_min_seconds':min(values),
                'latency_max_seconds':max(values),'latency_sample_std_seconds':statistics.stdev(values),
                'distinct_primary_responses':len({r['primary_answer'] for r in rr}),
                'distinct_final_predictions':len({r['final_prediction'] for r in rr})})
        for rep in range(1,manifest['repeats']+1):
            rr=[r for r in rows if r['method']==m and r['repetition']==rep]
            primary_diagnostics.append({'method':m,'repetition':rep,'requests':len(rr),
                'primary_content_correct':sum(r['primary_correct'] for r in rr),'final_content_correct':sum(r['final_correct'] for r in rr),
                'warning':'Observed intermediate answers, not a separately timed primary-only policy.'})
    result={'all_original_files_unchanged':True,'preserved_files_checked':len(manifest['preserved_sha256']),
        'frozen_manifest_profile_plans_unchanged':True,'reconstructed_all_scores_and_formats':True,
        'reconstructed_summaries_aggregates_pairs':True,'exact_runtime_prompts_contain_only_questions_and_actual_candidates':True,
        'calibration_only_profile_reconstructed':True,'calibration_evaluation_overlap':0,
        'unique_questions':len(tasks),'repeats':manifest['repeats'],'evaluated_rows':len(rows),
        'unique_actual_evaluation_calls':len(calls),'shared_calls_not_duplicated':True,'request_hashes_valid':True,
        'batch_envelopes_actual_calls':dict(envelopes),'api_errors':sum(a['status']!='generated' for a in calls),
        'limits':'Replayed tasks, not independent holdout; gold-independent interpretable content scoring is not arbitrary prose adjudication.'}
    write_records(Path(f'{prefix}_category_summary.csv'),categories)
    write_records(Path(f'{prefix}_failure_cases.csv'),failures)
    write_records(Path(f'{prefix}_question_repeats.csv'),per_question)
    write_records(Path(f'{prefix}_primary_diagnostics.csv'),primary_diagnostics)
    write_json(Path(f'{prefix}_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':audit(sys.argv[1])
