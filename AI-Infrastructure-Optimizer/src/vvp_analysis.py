"""Posthoc matched-reference effects, immutable-input audit, budget curves and inference."""
import hashlib,json,math,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
if __package__:
    from . import vvp_model as model
    from . import vvp_runtime as runtime
    from . import vvp_experiment as experiment
    from .controlled_analysis import read_decoded
    from . import controlled_statistics as stats
    from .rcav_runtime import verifier_prompt
    from .efficiency_protocol import extract
    from .solver300_dataset import task_key
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import vvp_model as model
    import vvp_runtime as runtime
    import vvp_experiment as experiment
    from controlled_analysis import read_decoded
    import controlled_statistics as stats
    from rcav_runtime import verifier_prompt
    from efficiency_protocol import extract
    from solver300_dataset import task_key
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def auc(scores,labels):
    pos=[s for s,y in zip(scores,labels) if y];neg=[s for s,y in zip(scores,labels) if not y]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None

def analyze(prefix):
    prefix=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{prefix}_{s}.json').read_text())
    def read(s):return read_decoded(f'{prefix}_{s}.csv')
    manifest,predictors,cache,meta=js('manifest'),js('predictors'),js('initial_cache'),js('metadata');experiment.frozen_check(manifest)
    for name in ('manifest','predictors','initial_cache','plans'):assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==meta[name+'_sha256']
    models=manifest['models'];reps=range(1,manifest['repeats']+1);tasks={t['task_id']:t for t in manifest['test']};cached={(p['model'],p['task_id']):p for p in cache}
    assert len(tasks)==300 and len(cached)==300*len(models)
    old_eval=experiment.old_evaluation_keys(experiment.ROOT/'data')
    # Exclude this run's own final test from the historical role check.
    for m in models:assert not {task_key(r['question']) for r in manifest['training_questions'][m]}&old_eval
    training=read('training')
    training_keys={task_key(r['question']) for r in training};testkeys={task_key(t['question']) for t in tasks.values()}
    assert not training_keys&testkeys
    validation_keys={task_key(t['question']) for t in manifest['validation']};prior_stems=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):prior_stems.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if path.parent!=prefix.parent:continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in experiment.read_csv(path):walk(r)
    assert not (testkeys|validation_keys)&prior_stems
    for m in models:assert not {task_key(r['question']) for r in manifest['training_questions'][m]}&validation_keys
    for r in training:
        p=r['primary'];kind='integer' if r['category']=='arithmetic' else 'choice';v=extract(r['question'],r['verification_answer'],kind)
        pc=p['prediction'] is not None and p['prediction']==r['expected'];final=v['prediction'] if v['prediction'] is not None else p['prediction'];fc=final is not None and final==r['expected']
        assert (r['primary_correct'],r['final_correct'],r['label'])==(pc,fc,model.label(pc,fc))
        assert r['features']==model.observed(r['question'],p)
    # Recompute the validation-selected small model from training only.
    for m in models:
        tr=[r for r in training if r['model']==m and r['phase']=='training'];va=[r for r in training if r['model']==m and r['phase']=='validation']
        assert model.train(tr,va)==predictors[m]
    rows,calls,summaries,resources=read('outcomes'),read('calls'),read('repeat_summary'),read('resources')
    ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)
    plans={(s['model'],s['repetition'],s['method'],s['budget_percent']):s['plan'] for s in js('plans')}
    profiles=manifest['existing_profiles'];seen=set();usage=Counter();prediction={}
    for p in cache:
        assert 'expected' not in p and 'category' not in p
        assert p['entry_hash']==experiment.digest({k:v for k,v in p.items() if k!='entry_hash'})
        prediction[p['model'],p['task_id']]=model.predict(p['question'],p,predictors[p['model']])
    for s in manifest['schedule']:
        m,rep,method,b=s['model'],s['repetition'],s['method'],s['budget_percent'];ps=[p for p in cache if p['model']==m]
        rebuilt=runtime.allocate(ps,method,b,profiles[m],predictors[m],manifest['arrivals'][m+':'+str(rep)])
        assert rebuilt==plans[m,rep,method,b]
        assert len(rebuilt['selected'])<=rebuilt['call_cap'] and rebuilt['selected_estimated_cpu']<=rebuilt['predicted_cpu_quota']+1e-12
        if method=='full' and b==100:assert len(rebuilt['selected'])==300
    for a in calls:
        assert a['stage']=='verify';p=cached[a['model'],a['task_id']]
        assert set(a['request_payload'])=={'model','prompt','stream','options'} and a['request_payload']['model']==a['model']
        assert a['request_payload']['options']=={'temperature':0,'seed':42,'num_predict':64}
        assert a['request_payload']['prompt']==verifier_prompt(p['question'],p['answer'])
        assert hashlib.sha256(json.dumps(a['request_payload']).encode()).hexdigest()==a['request_sha256']
    for r in rows:
        m,rep,method,b,id=r['model'],r['repetition'],r['method'],r['budget_percent'],r['task_id'];key=m,rep,method,b,id
        assert key not in seen;seen.add(key);p=cached[m,id];t=tasks[id]
        assert r['primary_answer']==p['answer'] and r['primary_prediction']==p['prediction']
        selected=id in plans[m,rep,method,b]['selected'];assert selected==r['verification_executed'];pred=p['prediction'];fmt=p['format_compliant']
        if selected:
            a=ci[r['call_id']];usage[a['call_id']]+=1;v=extract(p['question'],a['answer'],t['task_format'])
            assert (a['method'],a['budget_percent'],a['repetition'])==(method,b,rep)
            assert r['verification_prediction']==v['prediction'] and r['verification_answer']==a['answer']
            if v['prediction'] is not None:pred,fmt=v['prediction'],v['format_compliant']
        else:assert not r['call_id']
        assert r['final_prediction']==pred and r['final_format_compliant']==fmt
        pc=p['prediction'] is not None and p['prediction']==t['expected'];fc=pred is not None and pred==t['expected']
        assert (r['primary_correct'],r['final_correct'],r['wrong_to_right'],r['right_to_wrong'])==(pc,fc,not pc and fc,pc and not fc)
    assert len(seen)==300*len(models)*manifest['repeats']*len(runtime.METHODS)*len(runtime.BUDGETS)
    assert set(usage)==set(ci) and all(n==1 for n in usage.values())
    ri={(r['model'],r['repetition'],r['method'],r['budget_percent'],r['task_id']):r for r in rows};grouped=defaultdict(list)
    for r in rows:grouped[r['model'],r['repetition'],r['method'],r['budget_percent']].append(r)
    calls_grouped=defaultdict(list)
    for a in calls:calls_grouped[a['model'],a['repetition'],a['method'],a['budget_percent']].append(a)
    resource_index={(r['model'],r['repetition'],r['method'],r['budget_percent']):r for r in resources}
    effects=[];augmented=[]
    for s in summaries:
        key=s['model'],s['repetition'],s['method'],s['budget_percent'];rr=grouped[key];missed=0;same=0;changed_same=0;fixed_unresolved=0;variable=0
        result={'plan':plans[key],'planning_wall_seconds':s['planning_wall_seconds'],'planning_cpu_seconds':s['planning_cpu_seconds']}
        assert experiment.summarize(rr,calls_grouped[key],resource_index[key],result,s['method'],s['budget_percent'],s['repetition'],s['model'])==s
        for r in rr:
            reference=ri[r['model'],r['repetition'],'full',100,r['task_id']]
            if not r['verification_executed'] and not r['primary_correct'] and reference['final_correct']:missed+=1
            if r['verification_executed'] and not r['wrong_to_right'] and not r['right_to_wrong']:same+=1;changed_same+=r['primary_prediction']!=r['final_prediction']
            fixed_unresolved+=r['wrong_to_right'] and r['primary_prediction'] is None
            variable+=r['verification_executed'] and r['verification_prediction']!=reference['verification_prediction']
        assert s['correct']==sum(r['final_correct'] for r in rr) and s['verification_calls']==sum(r['verification_executed'] for r in rr)
        augmented.append({**s,'missed_repairs_full_reference':missed,'verified_unchanged_correctness':same,
            'changed_answer_unchanged_correctness':changed_same,'unresolved_output_recoveries':fixed_unresolved,
            'parsed_wrong_repairs':s['repairs']-fixed_unresolved,'different_verifier_predictions_vs_full_reference':variable})
    averages=[]
    for m in models:
        for method in runtime.METHODS:
            for b in runtime.BUDGETS:
                rr=[s for s in augmented if (s['model'],s['method'],s['budget_percent'])==(m,method,b)]
                out={'model':m,'method':method,'budget_percent':b,'repeats':len(rr)}
                for k in rr[0]:
                    if k in ('model','method','budget_percent','repetition','cost_status'):continue
                    out.update({k+'_'+name:value for name,value in stats.describe([r[k] for r in rr]).items()})
                averages.append(out)
    si={(s['model'],s['repetition'],s['method'],s['budget_percent']):s for s in augmented};comparisons=[];cohorts=[];verdict=[]
    for m in models:
        gates=[]
        for control,budget in [('selective',100),('none',0),('selective',25)]:
            pairs=[[int(ri[m,rep,'vvp',25,id]['final_correct'])-int(ri[m,rep,control,budget,id]['final_correct']) for rep in reps] for id in tasks]
            acc=stats.accuracy_comparison(pairs)
            cpu=[si[m,rep,'vvp',25]['measured_cpu_seconds']-si[m,rep,control,budget]['measured_cpu_seconds'] if si[m,rep,'vvp',25]['measured_cpu_seconds'] is not None and si[m,rep,control,budget]['measured_cpu_seconds'] is not None else None for rep in reps]
            cost=stats.describe(cpu);row={'model':m,'comparison':'vvp25_vs_'+control+str(budget),**acc,
                'cpu_delta_mean':cost['mean'],'cpu_delta_ci95_low':cost['ci95_low'],'cpu_delta_ci95_high':cost['ci95_high']}
            comparisons.append(row)
            if control=='selective' and budget==100:gates.append(('A_accuracy_not_lower',acc['conservative_exact_lower95']>=0));gates.append(('B_conditional_accounted_cpu_lower',cost['ci95_high'] is not None and cost['ci95_high']<0))
            if control=='none':gates.append(('D_benefit_over_no_verification',acc['conservative_exact_lower95']>0))
            if (control,budget) in [('selective',100),('none',0)]:
                for cohort in ('A','B'):
                    ids=[id for id,t in tasks.items() if t['cohort']==cohort]
                    delta=statistics.mean(int(ri[m,rep,'vvp',25,id]['final_correct'])-int(ri[m,rep,control,budget,id]['final_correct']) for rep in reps for id in ids)
                    cohorts.append({'model':m,'cohort':cohort,'control':control,'control_budget':budget,'accuracy_delta':delta,'unique_questions':len(ids)})
        c=[r for r in cohorts if r['model']==m];gates.append(('C_two_cohorts_direction',all(r['accuracy_delta']>=0 if r['control']=='selective' else r['accuracy_delta']>0 for r in c)))
        verdict.append({'model':m,'main_budget':25,'success':all(v for k,v in gates),'criteria':dict(gates),
            'limits':'Conditional measured-component CPU accounting, not independent production end-to-end proof; training acquisition overhead separate.'})
    native=[];forecasts=[];categories=[];budget_actual=[]
    for m in models:
        for rep in reps:
            full=[ri[m,rep,'full',100,id] for id in tasks]
            for scope in ('all','non_tool'):
                subset=[r for r in full if scope=='all' or r['primary_source']!='tool']
                pr=[prediction[m,r['task_id']] for r in subset];ys=[model.label(r['primary_correct'],r['final_correct']) for r in subset]
                for label,name in [(0,'repair'),(1,'damage')]:
                    scores=[p['p_'+name] for p in pr];labels=[y==label for y in ys]
                    native.append({'model':m,'repetition':rep,'event':name,'scope':scope,'unique_questions':len(subset),'brier':statistics.mean((s-y)**2 for s,y in zip(scores,labels)),
                        'roc_auc':auc(scores,labels),'predicted_rate':statistics.mean(scores),'observed_full_reference_rate':statistics.mean(labels)})
        for id,t in tasks.items():
            p=prediction[m,id];gain=statistics.mean(int(ri[m,rep,'full',100,id]['final_correct'])-int(ri[m,rep,'full',100,id]['primary_correct']) for rep in reps)
            forecasts.append({'model':m,'task_id':id,'category':t['category'],**p,'observed_mean_full_reference_gain':gain})
        for method in runtime.METHODS:
            for b in runtime.BUDGETS:
                for c in ('arithmetic','knowledge','logic','coding','comprehension'):
                    rr=[r for rep in reps for r in grouped[m,rep,method,b] if r['category']==c]
                    categories.append({'model':m,'method':method,'budget_percent':b,'category':c,'unique_questions':60,
                        'accuracy':statistics.mean(r['final_correct'] for r in rr),'repairs_per_repeat':sum(r['wrong_to_right'] for r in rr)/len(list(reps)),
                        'damage_per_repeat':sum(r['right_to_wrong'] for r in rr)/len(list(reps))})
    for s in augmented:budget_actual.append({'model':s['model'],'repetition':s['repetition'],'method':s['method'],'budget_percent':s['budget_percent'],
        'call_cap':int(300*s['budget_percent']/100),'actual_calls':s['verification_calls'],'predicted_cpu_quota':s['predicted_cpu_quota'],
        'actual_total_cpu':s['measured_cpu_seconds'],'cpu_quota_ratio':s['measured_cpu_seconds']/s['predicted_cpu_quota'] if s['predicted_cpu_quota']>0 and s['measured_cpu_seconds'] is not None else None,
        'scope':'Quota is a forecast, not a hard actual-CPU guarantee; includes measured planning cost.'})
    frontier=[];accounting=[];initial_cost=read('primary_cost')
    for a in averages:
        others=[s for s in averages if s['model']==a['model'] and s['measured_cpu_seconds_mean'] is not None]
        dominated=[s['method']+':'+str(s['budget_percent']) for s in others if (s['method'],s['budget_percent'])!=(a['method'],a['budget_percent']) and s['accuracy_mean']>=a['accuracy_mean'] and s['measured_cpu_seconds_mean']<=a['measured_cpu_seconds_mean'] and (s['accuracy_mean']>a['accuracy_mean'] or s['measured_cpu_seconds_mean']<a['measured_cpu_seconds_mean'])]
        frontier.append({'model':a['model'],'method':a['method'],'budget_percent':a['budget_percent'],'accuracy':a['accuracy_mean'],
            'measured_cpu_mean':a['measured_cpu_seconds_mean'],'on_observed_frontier':not dominated,'dominated_by':dominated,
            'scope':'Observed grid points; no best-budget success selection or significance claim.'})
        p=next(p for p in initial_cost if p['model']==a['model']);pcpu=p['cpu_seconds']+p['router_cpu_seconds'] if p['cpu_seconds'] is not None else None
        accounting.append({'model':a['model'],'method':a['method'],'budget_percent':a['budget_percent'],'shared_initial_cpu_once':pcpu,
            'actual_planning_verification_cpu_mean':a['measured_cpu_seconds_mean'],
            'accounted_full_cpu':pcpu+a['measured_cpu_seconds_mean'] if pcpu is not None and a['measured_cpu_seconds_mean'] is not None else None,
            'accounted_full_wall':p['stage_wall_seconds']+a['verification_wall_seconds_mean'],
            'accounted_model_calls':p['model_calls']+a['verification_calls_mean'],
            'scope':'Separately measured generation plus actual planning/verification; not an independently executed production pipeline. Training overhead not amortized.'})
    variations=[];hashes=defaultdict(list)
    for a in calls:hashes[a['request_sha256']].append(a)
    for h,aa in hashes.items():
        predictions=[extract(tasks[a['task_id']]['question'],a['answer'],tasks[a['task_id']]['task_format'])['prediction'] for a in aa]
        variations.append({'model':aa[0]['model'],'task_id':aa[0]['task_id'],'request_sha256':h,'actual_calls':len(aa),
            'distinct_raw_responses':len({a['answer'] for a in aa}),'distinct_predictions':len(set(predictions))})
    failures=[r for r in rows if r['method']=='vvp' and (not r['final_correct'] or r['right_to_wrong'])]
    audit={'preserved_files_unchanged':True,'preserved_files_checked':len(manifest['preserved_sha256']),
        'past_stems_checked':len(prior_stems),'fresh_validation_and_test_overlap_past':0,
        'training_past_evaluation_overlap':0,'training_validation_test_disjoint':True,'all_predictors_reconstructed_from_training_validation':True,
        'same_initial_answers':True,'all_allocations_frozen_and_reconstructed':True,'all_verifier_prompts_reconstructed':True,
        'unique_final_test_questions':300,'outcome_rows':len(rows),'actual_verifier_calls':len(calls),
        'api_errors':sum(a['status']!='generated' for a in calls),'variable_verifier_requests':sum(r['distinct_predictions']>1 for r in variations),
        'limits':'Matched Full100 reference is separate-time/cache, not an exact same-execution counterfactual. Repeated outputs are not independent questions.'}
    for suffix,records in [('effects_repeat_summary',augmented),('aggregate',averages),('comparisons',comparisons),('cohorts',cohorts),('verdict',verdict),
        ('predictive_metrics',native),('forecasts',forecasts),('category_summary',categories),('budget_actual',budget_actual),('pareto',frontier),
        ('accounting',accounting),('verifier_variation',variations),('failure_cases',failures)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_audit.json'),audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print('VERDICT '+json.dumps(verdict,ensure_ascii=False));return audit

if __name__=='__main__':analyze(sys.argv[1])
