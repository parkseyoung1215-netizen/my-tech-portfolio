"""Read-only audit of measured artifacts; writes only new exclusive audit files."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
if __package__:
    from . import skipwork_runtime as runtime
    from .skipwork_experiment import score,CATEGORIES
    from .solver300_dataset import task_key
    from .solver300_protocol import content_extract
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import skipwork_runtime as runtime
    from skipwork_experiment import score,CATEGORIES
    from solver300_dataset import task_key
    from solver300_protocol import content_extract
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def load_csv(prefix,suffix):
    with Path(f'{prefix}_{suffix}.csv').open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def audit(prefix):
    prefix=Path(prefix).resolve()
    def js(suffix):return json.loads(Path(f'{prefix}_{suffix}.json').read_text())
    manifest,metadata,profile=js('manifest'),js('metadata'),js('profile')
    rows,calls,blocks=load_csv(prefix,'evaluation'),load_csv(prefix,'calls'),load_csv(prefix,'blocks')
    ci={a['call_id']:a for a in calls};bi={b['block_id']:b for b in blocks}
    assert len(ci)==len(calls)
    for name in ('manifest','profile','plans'):
        assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==metadata[f'{name}_sha256']
    oldkeys=set();preserved=manifest['preserved_sha256']
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):oldkeys.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name,digest in preserved.items():
        path=Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, f'Original changed: {name}'
        if path.parent==prefix.parent and path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.parent==prefix.parent and path.suffix=='.csv':
            with path.open(newline='',encoding='utf-8') as f:
                for r in csv.DictReader(f):walk(r)
    tasks={t['question']:t for t in manifest['evaluation']}
    assert len(tasks)==150
    assert len({task_key(q) for q in tasks})==150
    assert not {task_key(q) for q in tasks}&oldkeys
    assert not {task_key(q) for q in tasks}&{task_key(t['question']) for t in manifest['calibration']}
    cal=load_csv(prefix,'calibration')
    for r in cal:
        for k in ('primary_correct','final_correct'):r[k]=r[k]=='True'
    assert runtime.make_profile(cal)==profile
    plans={(r['method'],r['question']):r['decision'] for r in js('plans')}
    seen=set();usage=Counter()
    for r in rows:
        q,m=r['question'],r['method'];assert q in tasks
        assert (m,q) not in seen;seen.add((m,q))
        d=json.loads(r['decision'])
        assert d==plans[(m,q)]==runtime.preflight(q,m,profile)
        assert r['expected']==tasks[q]['expected']
        assert (r['final_correct']=='True')==(bool(r['final_prediction']) and r['final_prediction']==r['expected'])
        assert (r['answered']=='True')==bool(r['final_prediction'])
        assert content_extract(r['final_answer'],d['answer_kind'])[0]==(r['final_prediction'] or None)
        for id in json.loads(r['call_ids']):
            a=ci[id];assert a['method']==m and a['block_id']==r['block_id'];usage[id]+=1
        primary=[ci[id] for id in json.loads(r['call_ids']) if ci[id]['stage']=='primary']
        if r['primary_source']=='tool':assert not primary
        else:assert len(primary)==1
    assert seen=={(m,q) for m in runtime.METHODS for q in tasks}
    for a in calls:
        payload=json.loads(a['request_payload'])
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
        assert set(payload)=={'model','prompt','stream','options'} and payload['model']==runtime.MODEL
        assert usage[a['call_id']]>0
        if a['stage']!='batch_verify':assert usage[a['call_id']]==1
    category=[]
    for m in runtime.METHODS:
        for c in CATEGORIES:
            rr=[r for r in rows if r['method']==m and r['category']==c]
            ids={id for r in rr for id in json.loads(r['call_ids'])};aa=[ci[id] for id in ids]
            bb=[bi[id] for id in {r['block_id'] for r in rr}]
            correct=sum(r['final_correct']=='True' for r in rr);answered=sum(r['answered']=='True' for r in rr)
            category.append({'method':m,'category':c,'requests':len(rr),'correct':correct,'accuracy':correct/len(rr),
                'primary_correct':sum(r['primary_correct']=='True' for r in rr),'answered':answered,'answer_rate':answered/len(rr),
                'model_calls':len(aa),'input_tokens':sum(int(a['input_tokens']) for a in aa) if all(a['input_tokens'] for a in aa) else None,
                'output_tokens':sum(int(a['output_tokens']) for a in aa) if all(a['output_tokens'] for a in aa) else None,
                'processing_seconds':sum(float(b['processing_seconds']) for b in bb),
                'mean_completion_latency_seconds':sum(float(r['completion_latency_seconds']) for r in rr)/len(rr),
                'verification_format_failures':sum(r['verification_format_error']=='True' for r in rr),
                'primary_format_failures':sum(not r['primary_prediction'] for r in rr),
                'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
                'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr)})
    base={r['question']:r for r in rows if r['method']=='baseline'}
    cases=[]
    for r in rows:
        reg=base[r['question']]['final_correct']=='True' and r['final_correct']!='True'
        if r['final_correct']!='True' or r['verification_format_error']=='True' or r['verification_outcome']=='harmful' or reg:
            cases.append({**r,'baseline_correct_to_wrong':reg,
                'error_class':'final_format_unresolved' if not r['final_prediction'] else ('content_error' if r['final_correct']!='True' else 'correct_final_verifier_format_failure'),
                'actual_calls':[ci[id] for id in json.loads(r['call_ids'])]})
    details={}
    for m in runtime.METHODS:
        aa=[a for a in calls if a['method']==m];rr=[r for r in rows if r['method']==m]
        details[m]={'done_reasons':dict(Counter(a['done_reason'] for a in aa)),
            'stage_calls':dict(Counter(a['stage'] for a in aa)),
            'primary_accuracy':sum(r['primary_correct']=='True' for r in rr)/len(rr),
            'verification_outcomes':dict(Counter(r['verification_outcome'] for r in rr)),
            'verifier_parse_failures':sum(r['verification_format_error']=='True' for r in rr),
            'api_errors':sum(a['status']!='generated' for a in aa),
            'skipped_by_reason':dict(Counter(json.loads(r['decision'])['reason'] for r in rr if r['verification_executed']=='False'))}
    calcalls=load_csv(prefix,'calibration_calls')
    audit_result={'preserved_files_checked':len(preserved),'prior_task_keys':len(oldkeys),
        'fresh_questions':len(tasks),'evaluation_rows':len(rows),'unique_actual_calls':len(ci),
        'frozen_policy_and_plan_match':True,'hashes_unchanged':True,'no_old_question_stem_overlap':True,
        'calibration_split_disjoint':True,'all_final_scores_recomputed':True,
        'all_request_hashes_valid':True,'no_duplicate_call_accounting':True,
        'calibration_calls':len(calcalls),'calibration_input_tokens':sum(int(a['input_tokens']) for a in calcalls),
        'calibration_output_tokens':sum(int(a['output_tokens']) for a in calcalls),
        'method_details':details,'limits':'Structural/runtime data-flow audit; not proof of semantic template independence or model pretraining exclusion.'}
    write_records(Path(f'{prefix}_category_summary.csv'),category)
    write_records(Path(f'{prefix}_failure_cases.csv'),cases)
    write_json(Path(f'{prefix}_audit.json'),audit_result)
    print(json.dumps(audit_result,ensure_ascii=False,indent=2))
    return audit_result

if __name__=='__main__':audit(sys.argv[1])
