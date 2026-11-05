"""Post-experiment integrity audit, transitions, uncertainty and oracle upper bound."""
import hashlib,json,statistics,sys
from collections import defaultdict,Counter
from pathlib import Path
from . import vrg_experiment as exp
from . import vrg_policy as policy
from .controlled_analysis import read_decoded
from . import controlled_statistics as stats
from .efficiency_repeat import frozen_check
from .rcav_runtime import verifier_prompt
from .solver300_dataset import task_key,prior_keys
from .tool_experiment import write_records
from .adaptive_experiment import write_json

def analyze(prefix,output_dir=None):
    p=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{p}_{s}.json').read_text())
    def read(s):return read_decoded(f'{p}_{s}.csv')
    manifest,profiles,cache,meta=js('manifest'),js('profiles'),js('initial_cache'),js('metadata');frozen_check(manifest)
    for name,h in js('frozen').items():assert hashlib.sha256(Path(f'{p}_{name}.json').read_bytes()).hexdigest()==h
    tasks={t['task_id']:t for t in manifest['test']};cached={(r['model'],r['task_id']):r for r in cache};models=manifest['models'];reps=list(range(1,meta['repeats']+1))
    assert len(tasks)==300 and len(cached)==600 and set(manifest['models'])=={exp.LARGE,exp.SMALL}
    trainkeys={task_key(r['question']) for rr in manifest['training'].values() for r in rr};vakeys={task_key(t['question']) for t in manifest['validation']};testkeys={task_key(t['question']) for t in tasks.values()}
    assert not trainkeys&vakeys and not trainkeys&testkeys and not vakeys&testkeys
    assert not trainkeys&exp.old_evaluation_keys(exp.ROOT/'data')
    past=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):past.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if path.parent!=p.parent:continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in exp.read_decoded(path):walk(r)
    assert not (vakeys|testkeys)&past
    calibration=read('calibration')
    for m in models:
        for r in manifest['training'][m]+[r for r in calibration if r['model']==m]:
            reconstructed=exp.training_record(r['question'],r['expected'],r['category'],r['primary'],r['revision']['answer'])
            assert all(r[k]==v for k,v in reconstructed.items())
        assert policy.train(manifest['training'][m],[r for r in calibration if r['model']==m])==profiles[m]
    for r in cache:
        assert not {'expected','correct','label','category'}&r.keys()
        assert r['entry_hash']==exp.digest({k:v for k,v in r.items() if k!='entry_hash'})
        parsed=policy.parse(r['question'],r['answer']);assert all(r[k]==v for k,v in parsed.items())
    rows,calls,summaries,resources,primary=read('outcomes'),read('calls'),read('repeat_summary'),read('resources'),read('primary_cost')
    initial_calls=read('primary_calls');expected_prompts={}
    for initial in cache:
        if initial['source']!='model':continue
        f=exp.policy.features(initial['question']);instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
        expected_prompts[initial['model'],initial['question']+'\n'+instruction+' Do not explain.']=initial
    assert len(initial_calls)==len(expected_prompts)==meta['primary_calls']
    for a in initial_calls:
        payload=a['request_payload'];initial=expected_prompts.pop((a['model'],payload['prompt']))
        assert a['answer']==initial['answer'] and payload=={'model':a['model'],'prompt':payload['prompt'],'stream':False,'options':{'temperature':0,'seed':42,'num_predict':64}}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
    assert not expected_prompts
    ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)==meta['verification_calls'];ri={};usage=Counter();bygroup=defaultdict(list);bycalls=defaultdict(list)
    for a in calls:
        q=cached[a['model'],a['task_id']];payload=a['request_payload'];assert payload=={'model':a['model'],'prompt':verifier_prompt(q['question'],q['answer']),'stream':False,'options':{'temperature':0,'seed':42,'num_predict':64}}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256'];bycalls[a['model'],a['repetition'],a['method']].append(a)
    for r in rows:
        m,rep,method,id=r['model'],r['repetition'],r['method'],r['task_id'];key=(m,rep,method,id);assert key not in ri;ri[key]=r;bygroup[m,rep,method].append(r)
        initial=cached[m,id];assert r['primary_answer']==initial['answer'] and r['primary_prediction']==initial['prediction']
        verify=method!='none'
        if method=='selective':verify=exp.controls.choose(initial['question'],initial,'selective',manifest['old_profiles'][m])['verify']
        assert verify==r['verification_executed'];accept=False;v=None
        if verify:
            a=ci[r['call_id']];usage[a['call_id']]+=1;assert (a['model'],a['repetition'],a['method'],a['task_id'])==(m,rep,method,id)
            v=dict(policy.parse(initial['question'],a['answer']),answer=a['answer']);assert v['prediction']==r['verification_prediction']
            if method=='always':accept=True
            elif method=='selective':accept=v['prediction'] is not None
            elif method=='heuristic':accept=policy.heuristic(initial['question'],initial,v)[0]
            else:
                accept,scores=policy.learned(initial['question'],initial,v,profiles[m]);assert scores==r['guard_scores']
        else:assert not r['call_id']
        selected=v if accept else initial;assert accept==r['accepted_revision']
        assert selected['prediction']==r['final_prediction'] and selected['answer']==r['final_answer'] and selected['format_compliant']==r['final_format_compliant']
        t=tasks[id];pc=initial['prediction'] is not None and initial['prediction']==t['expected'];fc=selected['prediction'] is not None and selected['prediction']==t['expected']
        assert r['expected']==t['expected'] and r['primary_correct']==pc and r['final_correct']==fc and r['transition']==('C' if pc else 'W')+('C' if fc else 'W')
    assert set(ri)=={(m,rep,method,id) for m in models for rep in reps for method in exp.METHODS for id in tasks}
    assert len(ri)==300*len(models)*len(reps)*len(exp.METHODS) and set(usage)==set(ci) and all(n==1 for n in usage.values())
    resource={(r['model'],r['repetition'],r['method']):r for r in resources}
    for s in summaries:
        key=s['model'],s['repetition'],s['method'];result={k:s[k] for k in ('guard_cpu_seconds','guard_wall_seconds','gate_cpu_seconds')};initial=next(c for c in primary if c['model']==s['model'])
        assert exp.summarize(bygroup[key],bycalls[key],resource[key],result,*key,initial)==s
    # Oracle references the same Always candidate; it is never called by execute().
    oracle_rows=[];oracle_summary=[]
    for m in models:
        for rep in reps:
            rr=[]
            for id in tasks:
                r=ri[m,rep,'always',id];t=tasks[id];use=r['verification_prediction'] is not None and r['verification_prediction']==t['expected']
                pred=r['verification_prediction'] if use else r['primary_prediction'];pc=r['primary_correct'];fc=pred is not None and pred==t['expected']
                raw=r['verification_answer'] if use else r['primary_answer']
                rr.append(dict(r,method='oracle',accepted_revision=use,final_answer=raw,final_prediction=pred,final_format_compliant=policy.parse(t['question'],raw)['format_compliant'],final_correct=fc,transition=('C' if pc else 'W')+('C' if fc else 'W'),oracle=True,
                   scope='Gold-informed upper bound using Always candidate. Not executable; no measured oracle cost.'))
            g={x:sum(r['transition']==x for r in rr) for x in ('CC','CW','WC','WW')};n=len(rr)
            oracle_summary.append({'model':m,'repetition':rep,'method':'oracle','requests':n,**g,'initial_accuracy':(g['CC']+g['CW'])/n,'accuracy':(g['CC']+g['WC'])/n,
               'CW_rate_all':g['CW']/n,'WC_rate_all':g['WC']/n,'CW_rate_initial_correct':0.,'WC_rate_initial_wrong':g['WC']/(g['WC']+g['WW']) if g['WC']+g['WW'] else None,
               'cost_scope':'Theoretical quality bound; cost unmeasured and not zero.'})
            oracle_rows.extend(rr)
    all_summaries=summaries+oracle_summary;all_rows=rows+oracle_rows;aggregate=[]
    numeric=[k for k,v in summaries[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool) and k not in ('repetition',)]
    for m in models:
        for method in (*exp.METHODS,'oracle'):
            ss=[s for s in all_summaries if s['model']==m and s['method']==method];o={'model':m,'method':method,'repeats':len(ss),'oracle':method=='oracle'}
            for k in numeric:o.update({k+'_'+name:v for name,v in stats.describe([s.get(k) for s in ss]).items()})
            aggregate.append(o)
    comparisons=[];cohorts=[];hypotheses=[]
    si={(r['model'],r['repetition'],r['method']):r for r in summaries}
    for m in models:
        for guard in ('heuristic','learned'):
            quality=None;cost=None;harm=None
            for control in ('selective','always','none'):
                pairs=[[int(ri[m,rep,guard,id]['final_correct'])-int(ri[m,rep,control,id]['final_correct']) for rep in reps] for id in tasks]
                aa=stats.accuracy_comparison(pairs);cpu=stats.describe([si[m,rep,guard]['stage_cpu_seconds']-si[m,rep,control]['stage_cpu_seconds'] if si[m,rep,guard]['stage_cpu_seconds'] is not None and si[m,rep,control]['stage_cpu_seconds'] is not None else None for rep in reps])
                harm_pairs=[[int(ri[m,rep,control,id]['transition']=='CW')-int(ri[m,rep,guard,id]['transition']=='CW') for rep in reps] for id in tasks];hh=stats.accuracy_comparison(harm_pairs)
                comparisons.append({'model':m,'guard':guard,'control':control,**aa,'cpu_delta_mean':cpu['mean'],'cpu_delta_low':cpu['ci95_low'],'cpu_delta_high':cpu['ci95_high'],
                  'harm_reduction':hh['accuracy_delta'],'harm_reduction_low':hh['paired_bootstrap_low'],'harm_reduction_high':hh['paired_bootstrap_high'],'harm_conservative_lower':hh['conservative_exact_lower95']})
                if control=='selective':quality=aa;cost=cpu
                if control=='always':harm=hh
                for cohort in ('A','B'):
                    ids=[id for id,t in tasks.items() if t['cohort']==cohort];delta=statistics.mean(int(ri[m,rep,guard,id]['final_correct'])-int(ri[m,rep,control,id]['final_correct']) for rep in reps for id in ids)
                    cohorts.append({'model':m,'guard':guard,'control':control,'cohort':cohort,'unique_questions':len(ids),'accuracy_delta':delta})
            repairs=sum(si[m,rep,guard]['WC'] for rep in reps);fullrepairs=sum(si[m,rep,'always']['WC'] for rep in reps);retention=repairs/fullrepairs if fullrepairs else None
            directional=all(c['accuracy_delta']>0 for c in cohorts if c['model']==m and c['guard']==guard and c['control']=='selective')
            gates={'H1_harm_reduction':harm['conservative_exact_lower95']>0,'H2_repair_retention80':retention is not None and retention>=.8,
              'H3_accuracy_strict':quality['paired_bootstrap_low']>0 and quality['conservative_exact_lower95']>0,
              'H3_CPU_not_increased':cost['ci95_high'] is not None and cost['ci95_high']<=0,
              'H4_net_post_verification_savings':False,'H5_cohorts_direction':directional}
            hypotheses.append({'model':m,'guard':guard,'repair_retention':retention,'success':all(gates.values()),'criteria':gates,
              'H4_scope':'No verifier calls avoided by a post-verification gate. Marginal guard CPU is positive; no measured downstream savings.'})
    transitions=[]
    for m in models:
        for method in (*exp.METHODS,'oracle'):
            for cat in ('ALL','MODEL_ONLY','arithmetic','knowledge','logic','coding','comprehension'):
                rr=[r for r in all_rows if r['model']==m and r['method']==method and (cat=='ALL' or cat=='MODEL_ONLY' and r['primary_source']=='model' or r['category']==cat)];g={x:sum(r['transition']==x for r in rr) for x in ('CC','CW','WC','WW')};n=len(rr)
                transitions.append({'model':m,'method':method,'category':cat,'unique_questions':n/len(reps),'repeated_rows':n,**g,
                   **{x+'_per_repeat':g[x]/len(reps) for x in g},**{x+'_rate_all':g[x]/n for x in g},
                   'CW_rate_initial_correct':g['CW']/(g['CC']+g['CW']) if g['CC']+g['CW'] else None,
                   'WC_rate_initial_wrong':g['WC']/(g['WC']+g['WW']) if g['WC']+g['WW'] else None})
    traincost=read('training_cost');cost_analysis=[]
    for a in aggregate:
        t=next(t for t in traincost if t['model']==a['model']);cost_analysis.append(dict(a,
           learned_fit_cpu_seconds=t['fit_cpu_seconds'] if a['method']=='learned' else None,
           learned_new_calibration_cpu_seconds=t['new_collection_cpu_seconds'] if a['method']=='learned' else None,
           learned_new_calibration_calls=t['new_collection_calls'] if a['method']=='learned' else None,
           scope='Actual stage measurements plus once-measured shared initial component accounting. Oracle costs unknown; historical collection sunk, not free.',
           GPU_FLOPs='unmeasured',power='unmeasured',monetary_cost='unmeasured'))
    variation=[];hashes=defaultdict(list)
    for a in calls:hashes[a['request_sha256']].append(a)
    for h,aa in hashes.items():
        question=tasks[aa[0]['task_id']]['question'];pred={policy.parse(question,a['answer'])['prediction'] for a in aa}
        variation.append({'request_sha256':h,'model':aa[0]['model'],'task_id':aa[0]['task_id'],'actual_calls':len(aa),'distinct_raw':len({a['answer'] for a in aa}),'distinct_predictions':len(pred)})
    paired=[]
    for m in models:
        for rep in reps:
            for id in tasks:
                r=ri[m,rep,'always',id];initial=cached[m,id];v=dict(policy.parse(initial['question'],r['verification_answer']),answer=r['verification_answer'])
                for guard in ('heuristic','learned'):
                    accept=policy.heuristic(initial['question'],initial,v)[0] if guard=='heuristic' else policy.learned(initial['question'],initial,v,profiles[m])[0]
                    pred=v['prediction'] if accept else initial['prediction'];correct=pred is not None and pred==tasks[id]['expected']
                    paired.append({'model':m,'repetition':rep,'task_id':id,'guard':guard,'accepted':accept,'always_transition':r['transition'],
                       'guard_correct':correct,'avoided_damage':r['transition']=='CW' and not accept,'lost_repair':r['transition']=='WC' and not accept,
                       'scope':'Offline identical-candidate guard diagnostic only; no measured cost or actual extra call assigned.'})
    out=Path(output_dir).resolve() if output_dir is not None else p.parent/p.name;out.mkdir(exist_ok=False)
    frontier=[]
    for a in aggregate:
        if a['oracle']:continue
        other=[b for b in aggregate if b['model']==a['model'] and not b['oracle'] and b['method']!=a['method'] and b['stage_cpu_seconds_mean'] is not None]
        dominated=[b['method'] for b in other if a['stage_cpu_seconds_mean'] is not None and b['accuracy_mean']>=a['accuracy_mean'] and b['stage_cpu_seconds_mean']<=a['stage_cpu_seconds_mean'] and (b['accuracy_mean']>a['accuracy_mean'] or b['stage_cpu_seconds_mean']<a['stage_cpu_seconds_mean'])]
        frontier.append({'model':a['model'],'method':a['method'],'accuracy':a['accuracy_mean'],'stage_cpu_mean':a['stage_cpu_seconds_mean'],'dominated_by':dominated,'on_observed_frontier':not dominated,'scope':'Observed points only; not a statistical dominance guarantee. Oracle excluded.'})
    for path,rr in [('VRG_RESULTS.csv',aggregate),('VRG_ERROR_TRANSITIONS.csv',transitions),('VRG_COST_ANALYSIS.csv',cost_analysis),('VRG_COMPARISONS.csv',comparisons),
       ('VRG_HYPOTHESES.csv',hypotheses),('VRG_COHORTS.csv',cohorts),('VRG_ORACLE.csv',oracle_rows),('VRG_CANDIDATE_VARIATION.csv',variation),('VRG_IDENTICAL_CANDIDATE_DIAGNOSTIC.csv',paired),('VRG_PARETO.csv',frontier)]:write_records(out/path,rr)
    audit={'preserved_files_unchanged':len(manifest['preserved_sha256']),'frozen_inputs_verified':True,'previous_question_stems':len(past),
      'fresh_calibration_test_past_overlap':0,'train_validation_test_disjoint':True,'previous_evaluation_not_used_for_training':True,
      'profiles_refitted_from_training_validation_only':True,'same_initial_outputs':True,'all_acceptance_decisions_reconstructed':True,
      'all_prompts_and_grades_reconstructed':True,'actual_verification_calls':len(calls),'actual_rows':len(rows),'oracle_rows_analysis_only':len(oracle_rows),
      'api_errors':sum(a['status']!='generated' for a in calls),'varying_candidate_requests':sum(r['distinct_predictions']>1 for r in variation),
      'scope':'Repeated frozen primaries are not independent questions. Semantic/template overlap remains.'}
    write_json(out/'VRG_AUDIT.json',audit);print(json.dumps(audit,indent=2));print('HYPOTHESES '+json.dumps(hypotheses));return out

if __name__=='__main__':analyze(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
