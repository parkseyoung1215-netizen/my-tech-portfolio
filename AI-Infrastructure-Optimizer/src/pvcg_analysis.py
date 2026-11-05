"""Audit frozen pre-verification decisions and compare measured accuracy/CPU."""
import hashlib,json,statistics,sys
from collections import defaultdict,Counter
from pathlib import Path
from . import pvcg_experiment as exp
from . import pvcg_policy as policy
from .controlled_analysis import read_decoded
from .controlled_statistics import describe,accuracy_comparison
from .efficiency_repeat import frozen_check
from .rcav_runtime import verifier_prompt
from .solver300_dataset import task_key
from .tool_experiment import write_records
from .adaptive_experiment import write_json

def analyze(prefix,output_dir=None):
    p=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{p}_{s}.json').read_text())
    def read(s):return read_decoded(f'{p}_{s}.csv')
    manifest,profiles,cache,plans,meta=js('manifest'),js('profiles'),js('initial_cache'),js('plans'),js('metadata');frozen_check(manifest)
    for name,h in js('frozen').items():assert hashlib.sha256(Path(f'{p}_{name}.json').read_bytes()).hexdigest()==h
    models=manifest['models'];reps=list(range(1,meta['repeats']+1));tasks={t['task_id']:t for t in manifest['test']};cached={(r['model'],r['task_id']):r for r in cache}
    assert len(tasks)==300 and len(cached)==600
    trainkeys={task_key(t['question']) for rr in manifest['training'].values() for t in rr};vakeys={task_key(t['question']) for t in manifest['validation']};testkeys={task_key(t['question']) for t in tasks.values()}
    assert not trainkeys&vakeys and not trainkeys&testkeys and not vakeys&testkeys and not trainkeys&exp.old_evaluation_keys(exp.ROOT/'data')
    past=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):past.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if not path.is_relative_to(p.parent):continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in read_decoded(path):walk(r)
    assert not (vakeys|testkeys)&past
    calibration=read('calibration')
    for m in models:
        va=[r for r in calibration if r['model']==m]
        for r in manifest['training'][m]+va:
            reconstructed=exp.record(r['question'],r['expected'],r['category'],r['primary'],r['verification_answer'],r['verifier_cpu_seconds'])
            assert all(r[k]==v for k,v in reconstructed.items())
        assert policy.train(manifest['training'][m],va,manifest['old_profiles'][m])==profiles[m]
    for c in cache:
        assert not policy.FORBIDDEN&c.keys() and 'category' not in c
        assert c['entry_hash']==exp.digest({k:v for k,v in c.items() if k!='entry_hash'})
        assert all(c[k]==v for k,v in exp.parse(c['question'],c['answer']).items())
        for method in exp.METHODS:assert plans[c['model']][method][c['task_id']]==exp.decision(c['question'],c,method,profiles[c['model']],manifest['old_profiles'][c['model']])
        assert not plans[c['model']]['pvcg'][c['task_id']]['verify'] or plans[c['model']]['selective'][c['task_id']]['verify']
    initial_calls=read('primary_calls');prompts={}
    for c in cache:
        if c['source']!='model':continue
        kind=policy.features(c['question'])['answer_kind'];instruction='Return only the option letter A, B, C or D.' if kind=='choice' else 'Return only the calculated integer.'
        prompts[c['model'],c['question']+'\n'+instruction+' Do not explain.']=c
    assert len(prompts)==len(initial_calls)==meta['actual_primary_calls']
    for a in initial_calls:
        payload=a['request_payload'];c=prompts.pop((a['model'],payload['prompt']));assert c['answer']==a['answer'] and set(payload)=={'model','prompt','stream','options'}
        assert payload['options']=={'temperature':0,'seed':42,'num_predict':64} and hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
    assert not prompts
    rows,calls,summaries,resources,initial=read('outcomes'),read('calls'),read('repeat_summary'),read('resources'),read('primary_cost')
    ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)==meta['actual_verification_calls'];ri={};usage=Counter();group=defaultdict(list);cg=defaultdict(list)
    for a in calls:
        c=cached[a['model'],a['task_id']];payload={'model':a['model'],'prompt':verifier_prompt(c['question'],c['answer']),'stream':False,'options':{'temperature':0,'seed':42,'num_predict':64}}
        assert a['stage']=='verify' and a['request_payload']==payload and hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256'];cg[a['model'],a['repetition'],a['method']].append(a)
    for r in rows:
        m,rep,method,id=r['model'],r['repetition'],r['method'],r['task_id'];key=m,rep,method,id;assert key not in ri;ri[key]=r;group[m,rep,method].append(r)
        c=cached[m,id];d=plans[m][method][id];assert r['decision']==d and r['decision_phase']=='before_verification' and r['verification_executed']==d['verify']
        assert r['primary_answer']==c['answer'] and r['primary_prediction']==c['prediction'];pred=c['prediction'];answer=c['answer'];fmt=c['format_compliant']
        if d['verify']:
            a=ci[r['call_id']];usage[a['call_id']]+=1;assert (a['model'],a['repetition'],a['method'],a['task_id'])==(m,rep,method,id)
            v=exp.parse(c['question'],a['answer']);assert r['verification_answer']==a['answer'] and r['verification_prediction']==v['prediction']
            if v['prediction'] is not None:pred,answer,fmt=v['prediction'],a['answer'],v['format_compliant']
        else:assert not r['call_id']
        t=tasks[id];pc=c['prediction'] is not None and c['prediction']==t['expected'];fc=pred is not None and pred==t['expected']
        assert (r['final_prediction'],r['final_answer'],r['final_format_compliant'])==(pred,answer,fmt)
        assert (r['primary_correct'],r['final_correct'],r['expected'],r['transition'])==(pc,fc,t['expected'],('C' if pc else 'W')+('C' if fc else 'W'))
    assert set(ri)=={(m,rep,method,id) for m in models for rep in reps for method in exp.METHODS for id in tasks}
    assert set(usage)==set(ci) and all(n==1 for n in usage.values())
    resource={(r['model'],r['repetition'],r['method']):r for r in resources};augmented=[]
    for s in summaries:
        key=s['model'],s['repetition'],s['method'];result={k:s[k] for k in ('gate_cpu_seconds','gate_wall_seconds')};ic=next(r for r in initial if r['model']==s['model'])
        assert exp.summarize(group[key],cg[key],resource[key],result,*key,ic)==s
        rr=group[key];missed=sum(not r['verification_executed'] and not r['primary_correct'] and ri[s['model'],s['repetition'],'full',r['task_id']]['final_correct'] for r in rr)
        prevented=sum(not r['verification_executed'] and r['primary_correct'] and not ri[s['model'],s['repetition'],'full',r['task_id']]['final_correct'] for r in rr)
        unchanged=sum(r['verification_executed'] and r['transition'] in ('CC','WW') for r in rr)
        forecast=sum(r['decision'].get('estimated_cpu_seconds',0) for r in rr if r['verification_executed']) if s['method']=='pvcg' else None
        augmented.append(dict(s,missed_repairs_full_reference=missed,prevented_harm_full_reference=prevented,verified_without_correctness_change=unchanged,
                             estimated_selected_cpu=forecast,scope='Full reference is same frozen initial, separate-time/cache verifier execution, not exact simultaneous counterfactual.'))
    aggregate=[]
    for m in models:
        for method in exp.METHODS:
            rr=[r for r in augmented if r['model']==m and r['method']==method];a={'model':m,'method':method,'repeats':len(rr)}
            keys={k for r in augmented for k,v in r.items() if (v is None or isinstance(v,(int,float))) and k!='repetition'}
            for k in sorted(keys):a.update({k+'_'+n:v for n,v in describe([r.get(k) for r in rr]).items()})
            aggregate.append(a)
    si={(s['model'],s['repetition'],s['method']):s for s in summaries};comparisons=[];cohorts=[];verdict=[]
    for m in models:
        quality=None;cost=None;benefit=None
        for control in ('selective','none','full'):
            pairs=[[int(ri[m,rep,'pvcg',id]['final_correct'])-int(ri[m,rep,control,id]['final_correct']) for rep in reps] for id in tasks];a=accuracy_comparison(pairs)
            cpu=describe([si[m,rep,'pvcg']['stage_cpu_seconds']-si[m,rep,control]['stage_cpu_seconds'] if si[m,rep,'pvcg']['stage_cpu_seconds'] is not None and si[m,rep,control]['stage_cpu_seconds'] is not None else None for rep in reps])
            wall=describe([si[m,rep,'pvcg']['stage_wall_seconds']-si[m,rep,control]['stage_wall_seconds'] for rep in reps])
            comparisons.append({'model':m,'control':control,**a,'cpu_delta_mean':cpu['mean'],'cpu_delta_low':cpu['ci95_low'],'cpu_delta_high':cpu['ci95_high'],
              'wall_delta_mean':wall['mean'],'wall_delta_low':wall['ci95_low'],'wall_delta_high':wall['ci95_high']})
            if control=='selective':quality,cost=a,cpu
            if control=='none':benefit=a
            for cohort in ('A','B'):
                ids=[id for id,t in tasks.items() if t['cohort']==cohort];delta=statistics.mean(int(ri[m,rep,'pvcg',id]['final_correct'])-int(ri[m,rep,control,id]['final_correct']) for rep in reps for id in ids)
                cohorts.append({'model':m,'control':control,'cohort':cohort,'unique_questions':len(ids),'accuracy_delta':delta})
        tokens=all(si[m,rep,'pvcg'][k] is not None and si[m,rep,'selective'][k] is not None and si[m,rep,'pvcg'][k]<=si[m,rep,'selective'][k] for rep in reps for k in ('input_tokens','output_tokens'))
        gates={'accuracy_noninferior_to_selective':quality['conservative_exact_lower95']>=0,
               'strict_benefit_over_none':benefit['conservative_exact_lower95']>0,'measured_cpu_lower_than_selective':cost['ci95_high'] is not None and cost['ci95_high']<0,
               'input_output_tokens_not_increased':tokens,'both_cohorts_benefit_over_none':all(r['accuracy_delta']>0 for r in cohorts if r['model']==m and r['control']=='none'),
               'not_skip_all':sum(si[m,rep,'pvcg']['verification_calls'] for rep in reps)>0}
        verdict.append({'model':m,'success':all(gates.values()),'criteria':gates})
    frontier=[]
    for a in aggregate:
        competitors=[b for b in aggregate if b['model']==a['model'] and b['method']!=a['method'] and b['stage_cpu_seconds_mean'] is not None]
        dominated=[b['method'] for b in competitors if a['stage_cpu_seconds_mean'] is not None and b['accuracy_mean']>=a['accuracy_mean'] and b['stage_cpu_seconds_mean']<=a['stage_cpu_seconds_mean'] and (b['accuracy_mean']>a['accuracy_mean'] or b['stage_cpu_seconds_mean']<a['stage_cpu_seconds_mean'])]
        frontier.append({'model':a['model'],'method':a['method'],'accuracy':a['accuracy_mean'],'stage_cpu_mean':a['stage_cpu_seconds_mean'],'dominated_by':dominated,'on_observed_frontier':not dominated,'scope':'Observed points only, not significance guarantee; initial shared cost shift does not alter dominance.'})
    categories=[]
    for m in models:
        for method in exp.METHODS:
            for cat in ('ALL','MODEL_ONLY','arithmetic','context_lookup','logic','coding','comprehension'):
                rr=[r for r in rows if r['model']==m and r['method']==method and (cat=='ALL' or cat=='MODEL_ONLY' and r['primary_source']=='model' or r['category']==cat)];n=len(rr)
                categories.append({'model':m,'method':method,'category':cat,'unique_questions':n/len(reps),'initial_accuracy':statistics.mean(r['primary_correct'] for r in rr),'accuracy':statistics.mean(r['final_correct'] for r in rr),
                   'CW_per_repeat':sum(r['transition']=='CW' for r in rr)/len(reps),'WC_per_repeat':sum(r['transition']=='WC' for r in rr)/len(reps),'verification_calls_per_repeat':sum(r['verification_executed'] for r in rr)/len(reps)})
    variation=[];hashes=defaultdict(list)
    for a in calls:hashes[a['request_sha256']].append(a)
    for h,aa in hashes.items():
        q=tasks[aa[0]['task_id']]['question'];variation.append({'model':aa[0]['model'],'task_id':aa[0]['task_id'],'request_sha256':h,'actual_calls':len(aa),'distinct_raw':len({a['answer'] for a in aa}),'distinct_predictions':len({exp.parse(q,a['answer'])['prediction'] for a in aa})})
    out=Path(output_dir).resolve() if output_dir else p.parent/p.name;out.mkdir(exist_ok=False)
    reason_counts=Counter((c['model'],plans[c['model']]['pvcg'][c['task_id']]['reason']) for c in cache)
    reasons=[{'model':m,'reason':reason,'unique_requests':n,'scope':'Frozen initial-only decision; no outcome used.'} for (m,reason),n in reason_counts.items()]
    traincost=read('training_cost');costanalysis=[]
    for a in aggregate:
        t=next(t for t in traincost if t['model']==a['model']);costanalysis.append(dict(a,pvcg_fit_cpu=t['fit_cpu_seconds'] if a['method']=='pvcg' else None,
          pvcg_calibration_cpu=t['collection_cpu_seconds'] if a['method']=='pvcg' else None,pvcg_calibration_calls=t['new_calibration_calls'] if a['method']=='pvcg' else None,
          scope='Actual stage and measured initial component accounting; acquisition costs separate, historical cost not free.',GPU_FLOPs='unmeasured',power='unmeasured',monetary_cost='unmeasured'))
    for name,rr in [('PVCG_RESULTS.csv',aggregate),('PVCG_REPEAT_METRICS.csv',augmented),('PVCG_COST_ANALYSIS.csv',costanalysis),('PVCG_COMPARISONS.csv',comparisons),('PVCG_COHORTS.csv',cohorts),
                    ('PVCG_VERDICT.csv',verdict),('PVCG_PARETO.csv',frontier),('PVCG_CATEGORY_RESULTS.csv',categories),('PVCG_CANDIDATE_VARIATION.csv',variation),('PVCG_GATE_REASONS.csv',reasons)]:write_records(out/name,rr)
    audit={'preserved_files_unchanged':len(manifest['preserved_sha256']),'frozen_inputs_unchanged':True,'past_stems_checked':len(past),'new_validation_test_past_overlap':0,
      'train_validation_test_disjoint':True,'no_previous_evaluation_training':True,'profiles_refitted_only_from_train_calibration':True,'all_initial_prompts_reconstructed':True,
      'all_gate_decisions_frozen_before_test_verification':True,'all_gate_decisions_reconstructed':True,'PVCG_subset_of_Selective':True,'all_prompts_grades_calls_verified':True,
      'actual_verification_calls':len(calls),'outcome_rows':len(rows),'api_errors':sum(a['status']!='generated' for a in calls),'varying_verifier_requests':sum(r['distinct_predictions']>1 for r in variation),
      'limits':'Stem disjointness not semantic independence; repeated fixed initial outputs not independent questions.'}
    write_json(out/'PVCG_AUDIT.json',audit);print(json.dumps(audit,indent=2));print('VERDICT '+json.dumps(verdict));return out

if __name__=='__main__':analyze(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
