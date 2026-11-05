"""Immutable-input audit and statistical analysis of controlled actual execution."""
from collections import defaultdict,Counter
import hashlib,json,statistics,sys
from pathlib import Path
if __package__:
    from . import controlled_validation as experiment
    from . import controlled_runtime as runtime
    from . import controlled_statistics as stats
    from .efficiency_audit import decode_row
    from .efficiency_protocol import extract
    from .solver300_dataset import task_key
    from .rcav_runtime import verifier_prompt
    from . import rcav_policy as base
    from . import rcav_guard_policy as guard
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import controlled_validation as experiment
    import controlled_runtime as runtime
    import controlled_statistics as stats
    from efficiency_audit import decode_row
    from efficiency_protocol import extract
    from solver300_dataset import task_key
    from rcav_runtime import verifier_prompt
    import rcav_policy as base
    import rcav_guard_policy as guard
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def read_decoded(path):
    rows=[]
    for raw in experiment.read_csv(path):
        row=decode_row(raw)
        for key,value in raw.items():
            if key.endswith('_id') or key.endswith('_hash') or key.endswith('_sha256'):row[key]=value
        rows.append(row)
    return rows

def analyze(prefix):
    prefix=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{prefix}_{s}.json').read_text())
    def read(s):return read_decoded(f'{prefix}_{s}.csv')
    manifest,profiles,cache,meta=js('manifest'),js('profiles'),js('primary_cache'),js('metadata')
    experiment.frozen_check(manifest)
    for suffix in ('manifest','profiles','primary_cache'):assert hashlib.sha256(Path(f'{prefix}_{suffix}.json').read_bytes()).hexdigest()==meta[suffix+'_sha256']
    assert manifest['schedule']==experiment.make_schedule(manifest['datasets'],manifest['models'],manifest['repeats'])
    source=json.loads(Path(f'{experiment.SOURCE}_profiles.json').read_text())
    for model in manifest['models']:
        for name in ('rcav','selective'):assert profiles[model][name]==source[model][name]
    assert profiles[experiment.LARGE]['guard']==json.loads(Path(f'{experiment.GUARD_SOURCE}_guard_profile.json').read_text())
    if experiment.SMALL in manifest['models']:assert profiles[experiment.SMALL]['guard']==guard.fit(read('small_guard_training'))
    tasks={t['task_id']:t for tt in manifest['datasets'].values() for t in tt};cached={(r['model'],r['task_id']):r for r in cache}
    assert len(cached)==len(tasks)*len(manifest['models'])
    freshkeys={task_key(t['question']) for t in manifest['datasets']['fresh300']};oldkeys=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):oldkeys.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if path.parent!=prefix.parent:continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in experiment.read_csv(path):walk(r)
    assert len(freshkeys)==300 and not freshkeys&oldkeys
    primary_calls=read('primary_calls');pci={a['call_id']:a for a in primary_calls};assert len(pci)==len(primary_calls)
    for p in cache:
        assert 'expected' not in p and 'category' not in p
        assert p['cache_entry_sha256']==experiment.digest({k:v for k,v in p.items() if k!='cache_entry_sha256'})
        decoded=extract(p['question'],p['answer'],base.features(p['question'])['answer_kind'])
        assert decoded['prediction']==p['prediction'] and decoded['format_compliant']==p['format_compliant']
        for id in p['primary_call_ids']:assert pci[id]['stage']=='primary' and pci[id]['answer']==p['answer']
    rows,calls,blocks=read('outcomes'),read('calls'),read('blocks');ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)
    assert not set(ci)&set(pci);seen=set();usage=Counter();failures=[]
    for a in calls:
        assert a['stage']=='verify'
        payload=a['request_payload'];assert set(payload)=={'model','prompt','stream','options'}
        assert payload['options']=={'temperature':0,'seed':42,'num_predict':64}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
    for r in rows:
        model,rep,method,id=r['model'],r['repetition'],r['method'],r['task_id'];p=cached[model,id];t=tasks[id]
        key=model,rep,method,id;assert key not in seen;seen.add(key)
        assert r['primary_cache_hash']==p['cache_entry_sha256'] and r['primary_answer']==p['answer'] and r['primary_prediction']==p['prediction']
        decision=runtime.choose(p['question'],p,method,profiles[model]);assert decision==r['decision']
        assert r['verification_executed']==decision['verify'];selected={'prediction':p['prediction'],'format_compliant':p['format_compliant']}
        if decision['verify']:
            assert len(r['call_ids'])==1;a=ci[r['call_ids'][0]];usage[a['call_id']]+=1
            assert (a['model'],a['method'],a['repetition'],a['block_id'])==(model,method,rep,r['block_id'])
            assert a['request_payload']['prompt']==verifier_prompt(p['question'],p['answer'])
            v=extract(p['question'],a['answer'],base.features(p['question'])['answer_kind'])
            assert v['prediction']==r['verification_prediction'] and a['answer']==r['verification_answer']
            if v['prediction'] is not None:selected=v
        else:assert not r['call_ids']
        assert r['final_prediction']==selected['prediction'] and r['final_format_compliant']==selected['format_compliant']
        scored=experiment.grade(r,t)
        for k in ('primary_correct','final_correct','answered','right_to_wrong','wrong_to_right'):assert scored[k]==r[k]
        if not r['final_correct'] or r['right_to_wrong']:failures.append(r)
    assert seen=={(model,rep,method,id) for model in manifest['models'] for rep in range(1,manifest['repeats']+1) for method in runtime.METHODS for id in tasks}
    assert set(usage)==set(ci) and all(n==1 for n in usage.values())
    planned={(p['model'],p['repetition'],p['sequence'],p['position']):p for p in manifest['schedule']}
    for b in blocks:
        p=planned[b['model'],b['repetition'],b['sequence'],b['position']]
        for k in ('method','dataset','block_key','task_ids'):assert b[k]==p[k]
        assert b['rss_sample_count']==len(b['rss_samples'])
        assert b['sampled_max_rss_mib']==(max(v[1] for v in b['rss_samples']) if b['rss_samples'] else None)
    reconstructed=[]
    for model in manifest['models']:
        for rep in range(1,manifest['repeats']+1):
            choose=lambda rr:[r for r in rr if r['model']==model and r['repetition']==rep]
            reconstructed.extend(experiment.summaries(choose(rows),choose(calls),choose(blocks),model,rep))
    assert reconstructed==read('repeat_summary')
    averages=[]
    for model in manifest['models']:
        for ds in manifest['datasets']:
            for method in runtime.METHODS:
                rr=[r for r in reconstructed if (r['model'],r['dataset'],r['method'])==(model,ds,method)]
                out={'model':model,'dataset':ds,'method':method,'repetitions':len(rr)}
                for k in rr[0]:
                    if k in ('model','dataset','method','repetition','cost_status'):continue
                    result=stats.describe([r[k] for r in rr])
                    out.update({k+'_'+s:v for s,v in result.items()})
                out['interval_scope']='Repeat-mean bootstrap estimate conditional on frozen questions/primaries; not population-accuracy CI.'
                averages.append(out)
    ri={(r['model'],r['dataset'],r['repetition'],r['method'],r['task_id']):r for r in rows}
    bi={(b['model'],b['dataset'],b['repetition'],b['method'],b['block_key']):b for b in blocks}
    controls=read('cache_controls');comparisons=[];positions=[]
    for model in manifest['models']:
        for ds,tt in manifest['datasets'].items():
            keys=sorted({b['block_key'] for b in blocks if b['model']==model and b['dataset']==ds})
            for method in runtime.METHODS:
                if method=='selective':continue
                question_pairs=[[int(ri[model,ds,rep,method,t['task_id']]['final_correct'])-int(ri[model,ds,rep,'selective',t['task_id']]['final_correct']) for rep in range(1,manifest['repeats']+1)] for t in tt]
                row={'model':model,'dataset':ds,'method':method,'control':'selective',**stats.accuracy_comparison(question_pairs)}
                for metric,field in [('wall','stage_wall_seconds'),('cpu',None)]:
                    matrix=[]
                    for key in keys:
                        values=[]
                        for rep in range(1,manifest['repeats']+1):
                            a=bi[model,ds,rep,method,key];b=bi[model,ds,rep,'selective',key]
                            if field:delta=a[field]-b[field]
                            else:delta=(a['cpu_seconds']+a['router_cpu_seconds']-b['cpu_seconds']-b['router_cpu_seconds']) if a['cpu_seconds'] is not None and b['cpu_seconds'] is not None else None
                            values.append(delta)
                        matrix.append(values)
                    row.update({metric+'_'+k:v for k,v in stats.paired_cost(matrix).items()})
                ordered=[]
                for pos in range(1,5):
                    pp=[]
                    for b in blocks:
                        if (b['model'],b['dataset'],b['method'],b['position'])!=(model,ds,method,pos):continue
                        c=bi[model,ds,b['repetition'],'selective',b['block_key']]
                        pp.append((b,c))
                    wall=[a['stage_wall_seconds']-b['stage_wall_seconds'] for a,b in pp]
                    cpu=[a['cpu_seconds']+a['router_cpu_seconds']-b['cpu_seconds']-b['router_cpu_seconds'] for a,b in pp if a['cpu_seconds'] is not None and b['cpu_seconds'] is not None]
                    pr={'model':model,'dataset':ds,'method':method,'position':pos,'paired_blocks':len(pp),
                        'wall_delta_mean_per_block':statistics.mean(wall),'cpu_delta_mean_per_block':statistics.mean(cpu) if len(cpu)==len(pp) else None}
                    positions.append(pr);ordered.append(pr)
                row['quality_pass']=row['conservative_exact_lower95']>=-manifest['protocol']['quality_tolerance_absolute']
                cov_target=sum(r['answered'] for r in rows if (r['model'],r['dataset'],r['method'])==(model,ds,method))
                cov_control=sum(r['answered'] for r in rows if (r['model'],r['dataset'],r['method'])==(model,ds,'selective'))
                row['coverage_pass']=cov_target>=cov_control
                row['cpu_pass']=row['cpu_ci95_high'] is not None and row['cpu_ci95_high']<0
                row['time_pass']=row['wall_ci95_high'] is not None and row['wall_ci95_high']<0
                row['position_sign_pass']=all(p['wall_delta_mean_per_block']<=0 and p['cpu_delta_mean_per_block'] is not None and p['cpu_delta_mean_per_block']<=0 for p in ordered)
                row['runner_reset_pass']=all(c['reset_observed'] and not c['error'] for c in controls if c['model']==model)
                row['dataset_success']=all(row[k] for k in ('quality_pass','coverage_pass','cpu_pass','time_pass','position_sign_pass','runner_reset_pass'))
                comparisons.append(row)
    verdict=[]
    for model in manifest['models']:
        for method in ('rcav','improved'):
            rr=[r for r in comparisons if r['model']==model and r['method']==method]
            verdict.append({'model':model,'method':method,'success_both_datasets':all(r['dataset_success'] for r in rr),
                'failed_criteria_by_dataset':{r['dataset']:[k for k in ('quality_pass','coverage_pass','cpu_pass','time_pass','position_sign_pass','runner_reset_pass') if not r[k]] for r in rr},
                'scope':'Prespecified observational criteria; no monetary, energy or GPU-FLOPs claim.'})
    risk=[];risk_summary=[]
    for p in cache:
        model=p['model'];d=base.decide(p['question'],p,len(verifier_prompt(p['question'],p['answer'])),profiles[model]['rcav'],.05)
        actual=p['prediction'] is None or p['prediction']!=tasks[p['task_id']]['expected']
        risk.append({'model':model,'dataset':p['dataset'],'task_id':p['task_id'],'source':p['source'],
            'estimated_error_risk':d['risk_estimate'],'observed_initial_error':actual,'brier':(d['risk_estimate']-int(actual))**2,
            'threshold_05_classification_correct':(d['risk_estimate']>=.5)==actual})
    for model in manifest['models']:
        for ds in manifest['datasets']:
            for scope in ('all','non_tool'):
                rr=[r for r in risk if r['model']==model and r['dataset']==ds and (scope=='all' or r['source']!='tool')]
                ece=0.
                for i in range(10):
                    bb=[r for r in rr if min(int(r['estimated_error_risk']*10),9)==i]
                    if bb:ece+=len(bb)/len(rr)*abs(statistics.mean(r['estimated_error_risk'] for r in bb)-statistics.mean(r['observed_initial_error'] for r in bb))
                risk_summary.append({'model':model,'dataset':ds,'scope':scope,'unique_initial_outputs':len(rr),
                    'brier':statistics.mean(r['brier'] for r in rr),'ece_10_bins':ece,
                    'threshold_05_classification_accuracy':statistics.mean(r['threshold_05_classification_correct'] for r in rr),
                    'native_predictor':'RCAV and improved base component; Baseline/Selective have no native probability predictor'})
    frontier=[];accounting=[];psummaries=read('primary_summary')
    for a in averages:
        for cost in ('measured_cpu_seconds','verification_wall_seconds'):
            other=[b for b in averages if b['model']==a['model'] and b['dataset']==a['dataset'] and b[cost+'_mean'] is not None]
            dominators=[b['method'] for b in other if b['method']!=a['method'] and b['accuracy_mean']>=a['accuracy_mean'] and b['answer_rate_mean']>=a['answer_rate_mean'] and b[cost+'_mean']<=a[cost+'_mean'] and (b['accuracy_mean']>a['accuracy_mean'] or b['answer_rate_mean']>a['answer_rate_mean'] or b[cost+'_mean']<a[cost+'_mean'])] if a[cost+'_mean'] is not None else []
            frontier.append({'model':a['model'],'dataset':a['dataset'],'method':a['method'],'cost_axis':cost,'accuracy':a['accuracy_mean'],
                'measured_mean_cost':a[cost+'_mean'],'dominated_by':dominators,'on_observed_frontier':not dominators,
                'scope':'Four observed points only; ignores inference uncertainty and is not proof of success.'})
        p=next(p for p in psummaries if p['model']==a['model'] and p['dataset']==a['dataset'])
        pcpu=p['cpu_seconds']+p['router_cpu_seconds'] if p['cpu_seconds'] is not None else None
        accounting.append({'model':a['model'],'dataset':a['dataset'],'method':a['method'],'shared_primary_wall_once':p['stage_wall_seconds'],
            'verification_wall_mean':a['verification_wall_seconds_mean'],'accounted_combined_wall':p['stage_wall_seconds']+a['verification_wall_seconds_mean'],
            'shared_primary_cpu_once':pcpu,'accounted_combined_cpu':pcpu+a['measured_cpu_seconds_mean'] if pcpu is not None and a['measured_cpu_seconds_mean'] is not None else None,
            'accounted_model_calls':p['model_calls']+a['verification_calls_mean'],
            'interpretation':'Sum of separately measured components; NOT an independently executed end-to-end pipeline or amortized observed production result.'})
    audit={'preserved_files_unchanged':True,'preserved_files_checked':len(manifest['preserved_sha256']),
        'fresh_questions':300,'old_question_keys_checked':len(oldkeys),'fresh_overlap_old':0,'shared_initial_entries':len(cache),
        'all_methods_used_identical_initial_answers':True,'primary_regeneration_during_verification':0,
        'actual_primary_calls':len(primary_calls),'actual_verification_calls':len(calls),'all_verification_prompts_reconstructed':True,
        'outcome_rows':len(rows),'summaries_reconstructed':True,'schedule_and_positions_verified':True,
        'runner_resets_observed':sum(c['reset_observed'] for c in controls),'runner_reset_attempts':len(controls),
        'verification_api_errors':sum(a['status']!='generated' for a in calls),
        'limits':'Sampled RSS only; intra-repeat cache hits, power, GPU FLOPs and monetary costs unmeasured. Statistical inference conditional on frozen primaries.'}
    for suffix,records in [('aggregate',averages),('comparisons',comparisons),('order_effects',positions),('verdict',verdict),('risk_predictions',risk),('risk_summary',risk_summary),('pareto',frontier),('accounting',accounting),('failure_cases',failures)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_audit.json'),audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print('VERDICT '+json.dumps(verdict,ensure_ascii=False))
    return audit

if __name__=='__main__':analyze(sys.argv[1])
