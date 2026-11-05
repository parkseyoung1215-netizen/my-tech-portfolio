"""Read-only reconstruction plus prespecified paired uncertainty analysis."""
import hashlib,json,math,random,statistics,sys
from pathlib import Path
from collections import Counter,defaultdict
if __package__:
    from . import rcav_guard_experiment as experiment
    from . import rcav_guard_policy as guard
    from . import rcav_policy as base
    from .rcav_runtime import verifier_prompt
    from .skipwork_runtime import preflight
    from .efficiency_protocol import extract
    from .efficiency_audit import decode_row
    from .solver300_dataset import task_key
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import rcav_guard_experiment as experiment
    import rcav_guard_policy as guard
    import rcav_policy as base
    from rcav_runtime import verifier_prompt
    from skipwork_runtime import preflight
    from efficiency_protocol import extract
    from efficiency_audit import decode_row
    from solver300_dataset import task_key
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)

def paired_interval(strata,draws=4000,seed=812067):
    """Stratified question cluster bootstrap; input averages keep all repeats together."""
    rng=random.Random(seed);values=[];n=sum(len(v) for v in strata.values())
    for _ in range(draws):values.append(sum(sum(rng.choice(v) for i in range(len(v))) for v in strata.values())/n)
    return percentile(values,.025),percentile(values,.975)

def exact_mcnemar(losses,gains):
    n=losses+gains
    return min(1.,2*sum(math.comb(n,i) for i in range(min(losses,gains)+1))/2**n) if n else 1.

def analyze(prefix):
    prefix=Path(prefix).resolve()
    def js(s):return json.loads(Path(f'{prefix}_{s}.json').read_text())
    def read(s):return [decode_row(r) for r in experiment.read_csv(f'{prefix}_{s}.csv')]
    manifest,profile,meta=js('manifest'),js('guard_profile'),js('metadata');experiment.frozen_check(manifest)
    for name,key in [('manifest','manifest_sha256'),('guard_profile','guard_profile_sha256')]:assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==meta[key]
    assert manifest['base_profiles']==json.loads(Path(f'{experiment.SOURCE}_profiles.json').read_text())[experiment.LARGE]
    pool=read('calibration_pool');assert len(pool)==200 and guard.fit(pool)==profile
    tasks={name:{t['question']:t for t in tt} for name,tt in manifest['datasets'].items()}
    pkeys={task_key(r['question']) for r in pool};fkeys={task_key(q) for q in tasks['fresh150']};oldkeys=set()
    assert len(pkeys)==200 and not pkeys&fkeys and not pkeys&{task_key(q) for q in tasks['replay150']}
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
    assert not fkeys&oldkeys
    newcalkeys={task_key(t['question']) for t in manifest['new_calibration']}
    assert not newcalkeys&oldkeys
    for r in pool:
        rebuilt=experiment.pool_record(r['question'],r['expected'],r['category'],r['primary_answer'],r['verification_answer'],r['verification_model_seconds'],r['calibration_source'])
        assert rebuilt==r
    rows,calls,blocks=read('evaluation'),read('calls'),read('blocks');ci={a['call_id']:a for a in calls};assert len(ci)==len(calls)
    seen=set();usage=Counter();failures=[]
    for a in calls:
        payload=a['request_payload'];assert set(payload)=={'model','prompt','stream','options'}
        assert payload['model']==experiment.LARGE and payload['options']=={'temperature':0,'seed':42,'num_predict':64}
        assert hashlib.sha256(json.dumps(payload).encode()).hexdigest()==a['request_sha256']
    for r in rows:
        ds,rep,m,q=r['dataset'],r['repetition'],r['method'],r['question'];key=ds,rep,m,q
        assert key not in seen and q in tasks[ds];seen.add(key)
        t=tasks[ds][q];f=base.features(q);p=extract(q,r['primary_answer'],f['answer_kind'])
        primary={'answer':r['primary_answer'],**p,'source':r['primary_source']};chars=len(verifier_prompt(q,r['primary_answer']))
        assert r['primary_prediction']==p['prediction'] and r['primary_format_compliant']==p['format_compliant']
        if m=='guarded':decision=guard.decide(q,primary,chars,manifest['base_profiles']['rcav'],profile)
        elif m=='rcav':decision=base.decide(q,primary,chars,manifest['base_profiles']['rcav'],.05)
        else:
            d=preflight(q,'selective',manifest['base_profiles']['selective']);decision={'verify':not d['skip_verifier'],'reason':d['reason']}
        assert r['decision']==decision and r['verification_executed']==decision['verify']
        aa=[ci[id] for id in r['call_ids']]
        for a in aa:assert (a['dataset'],a['repetition'],a['method'],a['block_id'])==(ds,rep,m,r['block_id']);usage[a['call_id']]+=1
        pp=[a for a in aa if a['stage']=='primary'];vv=[a for a in aa if a['stage']=='verify']
        if r['primary_source']=='model':
            assert len(pp)==1 and pp[0]['answer']==r['primary_answer']
            instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
            assert pp[0]['request_payload']['prompt']==q+'\n'+instruction+' Do not explain.'
        else:assert not pp and r['tool_calls']==1
        selected=p
        if decision['verify']:
            assert len(vv)==1;a=vv[0];assert a['request_payload']['prompt']==verifier_prompt(q,r['primary_answer'])
            assert a['answer']==r['verification_answer'];v=extract(q,a['answer'],f['answer_kind'])
            assert r['verification_prediction']==v['prediction']
            if v['prediction'] is not None:selected=v
        else:assert not vv
        assert r['final_prediction']==selected['prediction'] and r['final_format_compliant']==selected['format_compliant']
        assert r['expected']==t['expected'] and r['category']==t['category'] and r['regime']==t['regime']
        scored=experiment.score(r,t)
        for k in ('primary_correct','final_correct','answered','verification_outcome'):assert r[k]==scored[k]
        if not r['final_correct'] or r['verification_outcome']=='harmful':failures.append(r)
    assert seen=={(ds,rep,m,q) for ds in tasks for rep in range(1,manifest['repeats']+1) for m in experiment.runtime.METHODS for q in tasks[ds]}
    assert set(usage)==set(ci) and all(n==1 for n in usage.values())
    reconstructed=[]
    for ds in tasks:
        for rep in range(1,manifest['repeats']+1):
            choose=lambda rr:[r for r in rr if r['dataset']==ds and r['repetition']==rep]
            reconstructed.extend(experiment.summarize(choose(rows),choose(calls),choose(blocks),ds,rep))
    assert reconstructed==read('repeat_summary') and experiment.aggregate(reconstructed)==read('aggregate')
    historical=experiment.historical_loss();lostset={r['question'] for r in historical};checks=read('loss_rechecks');checkcalls=read('loss_recheck_calls');cci={a['call_id']:a for a in checkcalls}
    assert len(checks)==11*manifest['repeats'] and not set(cci)&set(ci)
    for r in checks:
        assert r['question'] in lostset;a=cci[r['call_id']]
        assert a['request_payload']['prompt']==verifier_prompt(r['question'],r['fixed_candidate'])
        v=extract(r['question'],a['answer'],'choice');assert r['prediction']==v['prediction']
        assert r['correct']==(v['prediction'] is not None and v['prediction']==r['expected'])
    ri={(r['dataset'],r['repetition'],r['method'],r['question']):r for r in rows}
    paired=[];uncertainty=[];recovery=[];category=[]
    for ds,tt in tasks.items():
        for control in ('rcav','selective'):
            strata=defaultdict(list)
            for q,t in tt.items():
                diffs=[int(ri[ds,rep,'guarded',q]['final_correct'])-int(ri[ds,rep,control,q]['final_correct']) for rep in range(1,manifest['repeats']+1)]
                strata[t['category']].append(statistics.mean(diffs))
            low,high=paired_interval(strata)
            uncertainty.append({'dataset':ds,'comparison':'guarded_vs_'+control,'unique_questions':len(tt),'repeats':manifest['repeats'],
                'observed_accuracy_delta':statistics.mean(v for values in strata.values() for v in values),
                'estimated_ci95_lower':low,'estimated_ci95_upper':high,'bootstrap_draws':4000,
                'ci_excludes_zero':low>0 or high<0,'uncertainty_type':'Statistical estimate: category-stratified paired question-cluster percentile bootstrap; repeated outcomes kept together.',
                'limits':'Fixed training profile and synthetic task sample; excludes training, hardware and template-family uncertainty.'})
            for rep in range(1,manifest['repeats']+1):
                rr=[r for r in rows if (r['dataset'],r['repetition'],r['method'])==(ds,rep,'guarded')]
                losses=sum(ri[ds,rep,control,r['question']]['final_correct'] and not r['final_correct'] for r in rr)
                gains=sum(not ri[ds,rep,control,r['question']]['final_correct'] and r['final_correct'] for r in rr)
                paired.append({'dataset':ds,'repetition':rep,'control':control,'correct_to_wrong':losses,'wrong_to_correct':gains,
                    'accuracy_delta':(gains-losses)/len(rr),'same_raw_primary':sum(r['primary_answer']==ri[ds,rep,control,r['question']]['primary_answer'] for r in rr),
                    'estimated_exact_mcnemar_p':exact_mcnemar(losses,gains),'p_scope':'Per repetition; do not pool repeated identical questions as independent pairs'})
        for method in experiment.runtime.METHODS:
            for subset in ('all','arithmetic','knowledge','logic','coding','comprehension','near','shift','fact_holdout'):
                rr=[r for r in rows if r['dataset']==ds and r['method']==method and (subset=='all' or r['category']==subset or r['regime']==subset)]
                if not rr:continue
                category.append({'dataset':ds,'method':method,'subset':subset,'repeated_rows':len(rr),'unique_questions':len({r['question'] for r in rr}),
                    'accuracy':sum(r['final_correct'] for r in rr)/len(rr),'answer_rate':sum(r['answered'] for r in rr)/len(rr),
                    'verification_rate':sum(r['verification_executed'] for r in rr)/len(rr),
                    'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr),
                    'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr)})
    for loss in historical:
        rr=[r for r in rows if r['dataset']=='replay150' and r['question']==loss['question']]
        recovery.append({'question':loss['question'],'category':loss['category'],'family':loss['family'],
            'guarded_correct_repeats':sum(r['method']=='guarded' and r['final_correct'] for r in rr),
            'rcav_correct_repeats':sum(r['method']=='rcav' and r['final_correct'] for r in rr),
            'guard_added_repeats':sum(r['method']=='guarded' and r['decision']['additional_verification'] for r in rr),
            'fixed_candidate_recheck_correct':sum(r['question']==loss['question'] and r['correct'] for r in checks)})
    audit={'preserved_files_unchanged':True,'preserved_files_checked':len(manifest['preserved_sha256']),
        'original_rcav_selective_profiles_unchanged':True,'guard_fit_reconstructed_from_separate200':True,
        'new_training_questions':len(newcalkeys),'fresh_holdout_questions':len(fkeys),'old_question_keys_checked':len(oldkeys),
        'fresh_or_new_training_overlap_old':0,'guard_training_overlap_either_evaluation':0,
        'all_decisions_and_prompts_reconstructed_without_evaluation_gold':True,'scores_aggregates_recomputed':True,
        'evaluation_rows':len(rows),'actual_evaluation_calls':len(calls),'diagnostic_verifier_calls':len(checkcalls),
        'diagnostic_recheck_correct':sum(r['correct'] for r in checks),'api_errors':sum(a['status']!='generated' for a in calls),
        'limits':'Development-informed feature selection; fresh template-semantic independence and statistical guarantees are not established.'}
    for suffix,records in [('paired',paired),('uncertainty',uncertainty),('category_summary',category),('historical11_recovery',recovery),('failure_cases',failures)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_audit.json'),audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print('UNCERTAINTY '+json.dumps(uncertainty,ensure_ascii=False))
    return audit

if __name__=='__main__':analyze(sys.argv[1])
