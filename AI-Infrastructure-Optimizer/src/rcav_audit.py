"""Post-execution reconstruction and failure/forecast diagnostics, never policy tuning."""
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path
if __package__:
    from . import rcav_policy as policy
    from . import rcav_experiment as experiment
    from .rcav_runtime import verifier_prompt
    from .efficiency_protocol import extract
    from .efficiency_audit import decode_row
    from .solver300_dataset import task_key
    from .skipwork_runtime import make_profile,preflight
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import rcav_policy as policy
    import rcav_experiment as experiment
    from rcav_runtime import verifier_prompt
    from efficiency_protocol import extract
    from efficiency_audit import decode_row
    from solver300_dataset import task_key
    from skipwork_runtime import make_profile,preflight
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def audit(prefix):
    prefix=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{prefix}_{s}.json').read_text())
    def read(s):return [decode_row(r) for r in experiment.read_csv(f'{prefix}_{s}.csv')]
    manifest,profiles,meta=js('manifest'),js('profiles'),js('metadata')
    experiment.frozen_check(manifest)
    for s in ('manifest','profiles'):assert hashlib.sha256(Path(f'{prefix}_{s}.json').read_bytes()).hexdigest()==meta[s+'_sha256']
    tasks={t['question']:t for t in manifest['evaluation']};cal={t['question']:t for t in manifest['calibration']}
    keys={task_key(q) for q in tasks};ckeys={task_key(q) for q in cal};assert len(keys)==150 and len(ckeys)==100 and not keys&ckeys
    oldkeys=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):oldkeys.add(task_key(obj['question']))
            for value in obj.values():walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if path.parent!=prefix.parent:continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in experiment.read_csv(path):walk(r)
    assert not (keys|ckeys)&oldkeys
    calibration=read('calibration');rows=read('evaluation');calls=read('calls');blocks=read('blocks')
    for model in manifest['models']:
        rr=[r for r in calibration if r['model']==model]
        assert len(rr)==100
        for r in rr:
            assert r['expected']==cal[r['question']]['expected']
            p=extract(r['question'],r['primary_answer'],r['task_format']);v=extract(r['question'],r['verification_answer'],r['task_format'])
            primary={'answer':r['primary_answer'],**p,'source':'tool' if r['task_format']=='integer' else 'model'}
            assert r['observed']==policy.observable(r['question'],primary)
            assert r['verification_prompt_chars']==len(verifier_prompt(r['question'],r['primary_answer']))
            final=v['prediction'] if v['prediction'] is not None else p['prediction']
            assert r['primary_correct']==(p['prediction'] is not None and p['prediction']==r['expected'])
            assert r['final_correct']==(final is not None and final==r['expected'])
        assert policy.fit(rr)==profiles[model]['rcav']
        assert make_profile(rr)==profiles[model]['selective']
    ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)
    usage=Counter();seen=set()
    for a in calls:
        payload=a['request_payload'];assert set(payload)=={'model','prompt','stream','options'}
        assert payload['model']==a['model'] and payload['options']=={'temperature':0,'seed':42,'num_predict':64}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
    for r in rows:
        model,rep,method,q=r['model'],r['repetition'],r['method'],r['question'];key=model,rep,method,q
        assert key not in seen and q in tasks;seen.add(key)
        f=experiment.runtime.features(q);p=extract(q,r['primary_answer'],f['answer_kind'])
        primary={'answer':r['primary_answer'],**p,'source':r['primary_source']}
        assert r['primary_prediction']==p['prediction'] and r['primary_format_compliant']==p['format_compliant']
        chars=len(verifier_prompt(q,r['primary_answer']));assert r['verification_prompt_chars']==chars
        if method in policy.WEIGHTS:expected_decision=policy.decide(q,primary,chars,profiles[model]['rcav'],policy.WEIGHTS[method])
        else:
            d=preflight(q,method,profiles[model]['selective']);expected_decision={'verify':not d['skip_verifier'],'reason':d['reason']}
        assert r['decision']==expected_decision and r['verification_executed']==expected_decision['verify']
        aa=[ci[id] for id in r['call_ids']]
        for a in aa:assert (a['model'],a['repetition'],a['method'],a['block_id'])==(model,rep,method,r['block_id']);usage[a['call_id']]+=1
        prim=[a for a in aa if a['stage']=='primary'];vv=[a for a in aa if a['stage']=='verify']
        if r['primary_source']=='model':
            instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
            assert len(prim)==1 and prim[0]['answer']==r['primary_answer']
            assert prim[0]['request_payload']['prompt']==q+'\n'+instruction+' Do not explain.'
        else:assert not prim and r['tool_calls']==1
        selected=p
        if r['verification_executed']:
            assert len(vv)==1;a=vv[0];assert a['request_payload']['prompt']==verifier_prompt(q,r['primary_answer'])
            assert a['answer']==r['verification_answer'];v=extract(q,a['answer'],f['answer_kind'])
            assert r['verification_prediction']==v['prediction'] and r['verification_format_compliant']==v['format_compliant']
            actual=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None
            assert r['verification_model_seconds']==actual
            if v['prediction'] is not None:selected=v
        else:assert not vv
        assert r['final_prediction']==selected['prediction'] and r['final_format_compliant']==selected['format_compliant']
        assert r['expected']==tasks[q]['expected']
        scored=experiment.score(r,tasks[q])
        for k in ('final_correct','primary_correct','answered','verification_outcome'):assert r[k]==scored[k]
    assert seen=={(model,rep,m,q) for model in manifest['models'] for rep in range(1,manifest['repeats']+1) for m in policy.METHODS for q in tasks}
    assert set(usage)==set(ci) and all(n==1 for n in usage.values())
    reconstructed=[]
    for model in manifest['models']:
        for rep in range(1,manifest['repeats']+1):
            choose=lambda rr:[r for r in rr if r['model']==model and r['repetition']==rep]
            reconstructed.extend(experiment.summarize(choose(rows),choose(calls),choose(blocks),model,rep))
    assert reconstructed==read('repeat_summary')
    assert experiment.aggregate(reconstructed)==read('aggregate')
    assert experiment.compare(reconstructed,rows)==read('paired')
    assert experiment.pareto(experiment.aggregate(reconstructed))==read('pareto')
    # Posthoc diagnostics only. A skipped verifier has no own execution cost.
    ri={(r['model'],r['repetition'],r['method'],r['question']):r for r in rows}
    residuals=[];riskrows=[];failures=[];category=[];diagnostics=[]
    for r in rows:
        if not r['final_correct'] or r['verification_outcome']=='harmful':failures.append(r)
        if r['method'] not in policy.WEIGHTS:continue
        d=r['decision'];riskrows.append({'model':r['model'],'method':r['method'],'repetition':r['repetition'],
            'question':r['question'],'category':r['category'],'source':r['primary_source'],'estimated_risk':d['risk_estimate'],
            'observed_primary_error':not r['primary_correct'],'brier_squared_error':(d['risk_estimate']-int(not r['primary_correct']))**2,
            'estimated_error_reduction':d['expected_error_reduction'],'observed_error_reduction_if_executed':int(r['final_correct'])-int(r['primary_correct']) if r['verification_executed'] else None})
        base=ri[r['model'],r['repetition'],'baseline',r['question']]
        for kind,actual in [('executed',r['verification_model_seconds']),
            ('paired_baseline_reference',base['verification_model_seconds'] if base['primary_answer']==r['primary_answer'] else None)]:
            if actual is None:continue
            predicted=d['estimated_verification_model_seconds']
            residuals.append({'model':r['model'],'method':r['method'],'repetition':r['repetition'],'question':r['question'],
                'category':r['category'],'reference_kind':kind,'verification_executed':r['verification_executed'],
                'estimated_seconds':predicted,'actual_seconds':actual,'signed_error_seconds':predicted-actual,
                'absolute_error_seconds':abs(predicted-actual),'absolute_percentage_error':abs(predicted-actual)/actual if actual>0 else None,
                'cost_extrapolation':d['cost_extrapolation'],
                'estimated_error_reduction':d['expected_error_reduction'],
                'observed_reference_error_reduction':int(base['final_correct'])-int(base['primary_correct']) if kind=='paired_baseline_reference' else int(r['final_correct'])-int(r['primary_correct']),
                'warning':'Paired reference ran at a different time/cache state; not the exact counterfactual cost.' if kind=='paired_baseline_reference' else ''})
    for model in manifest['models']:
        for method in policy.METHODS:
            for rep in range(1,manifest['repeats']+1):
                for c in sorted({t['category'] for t in tasks.values()}):
                    rr=[r for r in rows if (r['model'],r['method'],r['repetition'],r['category'])==(model,method,rep,c)]
                    category.append({'model':model,'method':method,'repetition':rep,'category':c,'requests':len(rr),
                        'accuracy':sum(r['final_correct'] for r in rr)/len(rr),'answer_rate':sum(r['answered'] for r in rr)/len(rr),
                        'verified_requests':sum(r['verification_executed'] for r in rr),'primary_correct':sum(r['primary_correct'] for r in rr),
                        'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
                        'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr)})
            if method not in policy.WEIGHTS:continue
            risk=[r for r in riskrows if (r['model'],r['method'])==(model,method)]
            for scope in ('all','non_tool'):
                rr=[r for r in risk if scope=='all' or r['source']!='tool']
                diagnostics.append({'model':model,'method':method,'kind':'risk_brier_'+scope,'n':len(rr),
                    'mae_seconds':None,'mean_absolute_percentage_error':None,
                    'brier':statistics.mean(r['brier_squared_error'] for r in rr) if rr else None,
                    'scope':'Repeated150 tasks, not independent samples'})
            for kind in ('executed','paired_baseline_reference'):
                rr=[r for r in residuals if (r['model'],r['method'],r['reference_kind'])==(model,method,kind)]
                mape=[r['absolute_percentage_error'] for r in rr if r['absolute_percentage_error'] is not None]
                diagnostics.append({'model':model,'method':method,'kind':'cost_'+kind,'n':len(rr),
                    'mae_seconds':statistics.mean(r['absolute_error_seconds'] for r in rr) if rr else None,
                    'mean_absolute_percentage_error':statistics.mean(mape) if mape else None,'brier':None,
                    'scope':'Selected verification calls only' if kind=='executed' else 'Matched raw primary candidate, different execution/cache state'})
    result={'preserved_files_unchanged':True,'preserved_files_checked':len(manifest['preserved_sha256']),
        'old_task_keys_checked':len(oldkeys),'new_calibration_questions':len(cal),'new_evaluation_questions':len(tasks),
        'old_or_split_task_overlap':0,'calibration_profiles_reconstructed':True,'frozen_hashes_valid':True,
        'all_decisions_reconstructed_from_preverification_information':True,'all_runtime_prompts_checked':True,
        'scores_formats_and_aggregates_recomputed':True,'evaluation_rows':len(rows),'unique_actual_calls':len(calls),
        'models':manifest['models'],'repeats':manifest['repeats'],'api_errors':sum(a['status']!='generated' for a in calls),
        'limits':'No guarantee of semantic template independence, calibrated probabilities, exact skipped-call counterfactual cost, or FLOPs savings.'}
    for suffix,records in [('category_summary',category),('failure_cases',failures),('risk_diagnostics',riskrows),
        ('cost_residuals',residuals),('forecast_summary',diagnostics)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':audit(sys.argv[1])
