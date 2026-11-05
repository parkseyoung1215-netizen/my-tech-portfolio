"""Real VRG controlled trial. Run: python -B -u -m src.vrg_experiment."""
import hashlib,json,random,time,platform
from pathlib import Path
from uuid import uuid4
from . import vrg_policy as policy
from .vrg_dataset import ROOT,SEED,build
from .rcav_runtime import LocalClient
from .rcav_experiment import LARGE,SMALL
from .controlled_validation import measured,digest
from . import controlled_runtime as controls
from .efficiency_repeat import total,frozen_check
from .controlled_analysis import read_decoded
from .solver300_dataset import task_key
from .solver300_experiment import check_references
from .vvp_experiment import old_evaluation_keys
from .tool_experiment import write_records
from .adaptive_experiment import write_json
from .routing_runner import check_ollama
SOURCE=ROOT/'data/vvp_2a90be1867974c2cbc6862856d207d44'
CONTROLLED=ROOT/'data/controlled_1ee08ac18eaa428f9644ef66daa2fcaa'
METHODS=('none','always','selective','heuristic','learned')
REPEATS=5
PROTOCOL={'seed':SEED,'main_comparisons':'Both guards vs Selective and Always; No Verification reference. No posthoc best model/method choice.',
 'quality':'paired unique-question bootstrap95 accuracy lower >0 vs Selective, and conservative exact lower >0; both models, both fixed cohorts directional positive',
 'damage':'H1 paired harm reduction vs Always; H2 retain at least80% of Always repairs (predefined descriptive floor)',
 'cost':'No increase in measured component CPU vs Selective (paired repeat upper95<=0), including guard CPU; zero added AI predictor calls.',
 'H4':'Post-verification guard cannot recover spent verification computation. Guard overhead measured; no assumed future savings.',
 'success':'All quality, damage, repair-retention, cost conditions for BOTH models, no leakage. Oracle excluded.',
 'parser':'Frozen original parser plus previous VVP explicit-content audit; same interpretation in all methods. Unresolved remains incorrect. Format scored independently.',
 'always':'Always return verifier raw response including unresolved output. Selective retains its existing invalid-revision fallback.',
 'oracle':'Posthoc only from Always actual candidates, never executable routing, no real cost assigned.',
 'total':'Measured initial generation once + actually executed per-method stage. Component accounting, not independent end-to-end repeated generation.',
 'budgets':'No pre-verification gate for VRG: full verifier calls, then guard. No hidden call saving.'}

def parsed_primary(client,q):
    p=client.primary(q);p.update(policy.parse(q,p['answer']));return p

def training_record(q,expected,category,p,answer):
    primary={k:p[k] for k in ('answer','prediction','format_compliant','source')};v=dict(policy.parse(q,answer),answer=answer)
    pc=primary['prediction'] is not None and primary['prediction']==expected;vc=v['prediction'] is not None and v['prediction']==expected
    return {'question':q,'expected':expected,'category':category,'primary':primary,'revision':v,'primary_correct':pc,'revision_correct':vc,'label':0 if not pc and vc else 1 if pc and not vc else 2}

def historical():
    rows=read_decoded(f'{SOURCE}_outcomes.csv');groups={}
    for r in rows:
        if (r['method'],r['budget_percent']) not in [('none',0),('full',100),('selective',100),('rcav',100),('improved',100),('vvp',25)]:continue
        for grading in ('original','explicit_content'):
            pc,fc=r['primary_correct'],r['final_correct']
            if grading=='explicit_content':
                p=policy.parse(r['question'],r['primary_answer'])['prediction'];f=policy.parse(r['question'],r['final_answer'])['prediction'];pc=p is not None and p==r['expected'];fc=f is not None and f==r['expected']
            for cat in ('ALL',r['category']):
                key=(r['model'],r['method'],r['budget_percent'],grading,cat);g=groups.setdefault(key,{'CC':0,'CW':0,'WC':0,'WW':0});g[('C' if pc else 'W')+('C' if fc else 'W')]+=1
    out=[]
    for key,g in groups.items():
        n=sum(g.values());out.append(dict(zip(('model','method','budget_percent','grading','category'),key),**g,repeated_rows=n,
          CW_rate_all=g['CW']/n,WC_rate_all=g['WC']/n,CW_rate_initial_correct=g['CW']/(g['CC']+g['CW']) if g['CC']+g['CW'] else None,
          WC_rate_initial_wrong=g['WC']/(g['WC']+g['WW']) if g['WC']+g['WW'] else None,scope='Five repeated frozen-primary trials, not independent questions; diagnosis only, never training.'))
    return out

def execute(primaries,method,client,old_profile,profile,arrival):
    if method not in METHODS:raise ValueError('Executable methods exclude oracle')
    lookup={p['task_id']:p for p in primaries};rows=[];calls=[];guard_cpu=0.;guard_wall=0.;gate_cpu=0.
    for id in arrival:
        p=lookup[id];q=p['question'];start=time.perf_counter();cpu=time.process_time()
        verify=method!='none'
        if method=='selective':verify=controls.choose(q,p,'selective',old_profile)['verify']
        gate_cpu+=time.process_time()-cpu
        a=None;v=None;accept=False;reason='no_verification';scores={}
        if verify:
            a,_=client.verify(q,p['answer']);calls.append(dict(a,task_id=id));v=dict(policy.parse(q,a['answer']),answer=a['answer'])
            before,cpu=time.perf_counter(),time.process_time()
            if method=='always':accept,reason=True,'always_return_revision'
            elif method=='selective':accept=v['prediction'] is not None;reason='legacy_resolved_fallback'
            elif method=='heuristic':accept,reason=policy.heuristic(q,p,v)
            else:accept,scores=policy.learned(q,p,v,profile);reason=scores['reason']
            guard_cpu+=time.process_time()-cpu;guard_wall+=time.perf_counter()-before
        chosen=v if accept else p
        rows.append({'task_id':id,'question':q,'primary_answer':p['answer'],'primary_prediction':p['prediction'],'primary_format_compliant':p['format_compliant'],
          'primary_source':p['source'],'verification_executed':verify,'verification_answer':v['answer'] if v else '',
          'verification_prediction':v['prediction'] if v else None,'call_id':a['call_id'] if a else '',
          'accepted_revision':accept,'accept_reason':reason,'guard_scores':scores,'final_answer':chosen['answer'],
          'final_prediction':chosen['prediction'],'final_format_compliant':chosen['format_compliant'],'service_seconds':time.perf_counter()-start})
    return {'rows':rows,'calls':calls,'guard_cpu_seconds':guard_cpu,'guard_wall_seconds':guard_wall,'gate_cpu_seconds':gate_cpu}

def summarize(rows,calls,res,result,m,rep,method,initial):
    n=len(rows);g={s:sum(r['transition']==s for r in rows) for s in ('CC','CW','WC','WW')};cpu=res['cpu_seconds']+res['router_cpu_seconds'] if res['cpu_seconds'] is not None else None
    initial_cpu=initial['cpu_seconds']+initial['router_cpu_seconds'] if initial['cpu_seconds'] is not None else None
    return {'model':m,'repetition':rep,'method':method,'requests':n,**g,'initial_accuracy':(g['CC']+g['CW'])/n,'accuracy':(g['CC']+g['WC'])/n,
       'CW_rate_all':g['CW']/n,'WC_rate_all':g['WC']/n,'CW_rate_initial_correct':g['CW']/(g['CC']+g['CW']) if g['CC']+g['CW'] else None,
       'WC_rate_initial_wrong':g['WC']/(g['WC']+g['WW']) if g['WC']+g['WW'] else None,
       'answer_rate':sum(r['final_prediction'] is not None for r in rows)/n,'format_compliance':sum(r['final_format_compliant'] for r in rows)/n,
       'verification_calls':len(calls),'accounted_total_model_calls':initial['model_calls']+len(calls),
       'verification_input_tokens':total(calls,'input_tokens') if calls else 0,'verification_output_tokens':total(calls,'output_tokens') if calls else 0,
       'accounted_total_input_tokens':initial['input_tokens']+(total(calls,'input_tokens') if calls else 0),
       'accounted_total_output_tokens':initial['output_tokens']+(total(calls,'output_tokens') if calls else 0),
       'stage_wall_seconds':res['stage_wall_seconds'],'stage_cpu_seconds':cpu,'ollama_cpu_seconds':res['cpu_seconds'],'python_cpu_seconds':res['router_cpu_seconds'],
       'accounted_total_wall_seconds':initial['stage_wall_seconds']+res['stage_wall_seconds'],
       'accounted_total_cpu_seconds':initial_cpu+cpu if initial_cpu is not None and cpu is not None else None,
       'guard_cpu_seconds':result['guard_cpu_seconds'],'guard_wall_seconds':result['guard_wall_seconds'],'gate_cpu_seconds':result['gate_cpu_seconds'],
       'sampled_max_rss_mib':res['sampled_max_rss_mib'],'api_errors':sum(a['status']!='generated' for a in calls),
       'power_gpu_monetary_cost':'unmeasured','accepted_changes':sum(r['accepted_revision'] and r['primary_prediction']!=r['verification_prediction'] for r in rows)}

def run(data_dir=ROOT/'data',repeats=REPEATS):
    if repeats<5:raise ValueError('At least five repeats')
    data_dir=Path(data_dir);validation,test=build(data_dir);references=check_references(validation+test)
    models=[LARGE,SMALL];check_ollama(models);tags=controls.http_json('/api/tags')
    training={m:[] for m in models};past_eval=old_evaluation_keys(ROOT/'data')
    for r in read_decoded(f'{SOURCE}_training.csv'):
        if r['phase']!='training':continue
        if task_key(r['question']) in past_eval:raise ValueError('Historical evaluation reused for VRG training')
        p=r['primary'];p.update(policy.parse(r['question'],p['answer']))
        training[r['model']].append(training_record(r['question'],r['expected'],r['category'],p,r['verification_answer']))
    assert all(len(rr)==200 for rr in training.values())
    preserved={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (ROOT/'src',ROOT/'tests',ROOT/'data',ROOT) for p in folder.iterdir() if p.is_file()}
    for t in test:t['task_id']=hashlib.sha256(t['question'].encode()).hexdigest()
    schedule=[];arrivals={};model_orders={}
    base=list(METHODS);random.Random(SEED).shuffle(base)
    for rep in range(1,repeats+1):
        order=models[:] if rep%2 else models[::-1];model_orders[str(rep)]=order
        for m in order:
            ids=[t['task_id'] for t in test];random.Random(SEED+rep*100+models.index(m)).shuffle(ids);arrivals[m+':'+str(rep)]=ids
            shifted=base[(rep-1)%5:]+base[:(rep-1)%5]
            if m==SMALL:shifted=list(reversed(shifted))
            for pos,method in enumerate(shifted,1):schedule.append({'model':m,'repetition':rep,'position':pos,'method':method})
    old=json.loads(Path(f'{CONTROLLED}_profiles.json').read_text());run_id=uuid4().hex;prefix=data_dir/f'vrg_{run_id}'
    manifest={'run_id':run_id,'protocol':PROTOCOL,'seed':SEED,'models':models,'repeats':repeats,'training':training,'validation':validation,'test':test,
      'old_profiles':old,'schedule':schedule,'arrivals':arrivals,'model_orders':model_orders,'preserved_sha256':preserved,'tags':tags,'python':platform.python_version(),'platform':platform.platform()}
    write_json(Path(f'{prefix}_manifest.json'),manifest);write_records(Path(f'{prefix}_historical_transitions.csv'),historical())
    profiles={};calibration=[];cal_calls=[];traincost=[];warmups=[];resetlog=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','vrg_calibration_warmup'));rr=[];aa=[]
        def collect():
            for t in validation:
                p=parsed_primary(client,t['question']);aa.extend(p['calls']);a,_=client.verify(t['question'],p['answer']);aa.append(a)
                if a['status']!='generated':raise RuntimeError('Calibration API failure')
                rr.append(dict(training_record(t['question'],t['expected'],t['category'],p,a['answer']),model=m))
        _,res=measured(collect);calibration.extend(rr);cal_calls.extend(dict(a,model=m) for a in aa)
        start,cpu=time.perf_counter(),time.process_time();profiles[m]=policy.train(training[m],rr);fitwall=time.perf_counter()-start;fitcpu=time.process_time()-cpu
        traincost.append({'model':m,'historical_training_rows':len(training[m]),'new_calibration_rows':len(rr),'new_collection_calls':len(aa),
          'new_collection_input_tokens':total(aa,'input_tokens'),'new_collection_output_tokens':total(aa,'output_tokens'),
          'new_collection_wall_seconds':res['stage_wall_seconds'],'new_collection_cpu_seconds':res['cpu_seconds']+res['router_cpu_seconds'] if res['cpu_seconds'] is not None else None,
          'fit_cpu_seconds':fitcpu,'fit_wall_seconds':fitwall,'historical_collection_cost':'sunk measurements in VVP training artifacts; not re-executed or counted as free',
          'resource':res});print('VRG calibrated '+m,flush=True)
    write_json(Path(f'{prefix}_profiles.json'),profiles);write_records(Path(f'{prefix}_calibration.csv'),calibration);write_records(Path(f'{prefix}_calibration_calls.csv'),cal_calls);write_records(Path(f'{prefix}_training_cost.csv'),traincost)
    cache=[];primary_calls=[];initialcost=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','vrg_primary_warmup'));rr=[];aa=[]
        def generate():
            for t in test:
                p=parsed_primary(client,t['question']);aa.extend(p['calls']);e={k:p[k] for k in ('answer','prediction','format_compliant','parse_mode','source')}
                e.update(model=m,task_id=t['task_id'],question=t['question']);e['entry_hash']=digest(e);rr.append(e)
        _,res=measured(generate);cache.extend(rr);primary_calls.extend(aa);initialcost.append(dict(res,model=m,model_calls=len(aa),input_tokens=total(aa,'input_tokens'),output_tokens=total(aa,'output_tokens')))
        print('VRG frozen shared primaries '+m,flush=True)
    write_json(Path(f'{prefix}_initial_cache.json'),cache);write_records(Path(f'{prefix}_primary_calls.csv'),primary_calls);write_records(Path(f'{prefix}_primary_cost.csv'),initialcost)
    frozen={name:hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest() for name in ('manifest','profiles','initial_cache')}
    write_json(Path(f'{prefix}_frozen.json'),frozen);tasks={t['task_id']:t for t in test};allrows=[];allcalls=[];summaries=[];resources=[];start=time.perf_counter()
    for rep in range(1,repeats+1):
        for m in model_orders[str(rep)]:
            client=LocalClient(m);reset,warm=controls.reset_and_warm(client,models);resetlog.append(dict(reset,repetition=rep,model=m));warmups.append(warm)
            rows=[];calls=[];ss=[];rs=[];ps=[p for p in cache if p['model']==m];ic=next(c for c in initialcost if c['model']==m)
            for s in [s for s in schedule if s['model']==m and s['repetition']==rep]:
                method=s['method'];result,res=measured(lambda:execute(ps,method,client,old[m],profiles[m],arrivals[m+':'+str(rep)]))
                graded=[]
                for r in result['rows']:
                    t=tasks[r['task_id']];pc=r['primary_prediction'] is not None and r['primary_prediction']==t['expected'];fc=r['final_prediction'] is not None and r['final_prediction']==t['expected']
                    graded.append(dict(r,**s,expected=t['expected'],category=t['category'],cohort=t['cohort'],primary_correct=pc,final_correct=fc,transition=('C' if pc else 'W')+('C' if fc else 'W')))
                cc=[dict(a,**s) for a in result['calls']];summary=summarize(graded,cc,res,result,m,rep,method,ic)
                rows.extend(graded);calls.extend(cc);ss.append(summary);rs.append(dict(res,**s))
                print(f'VRG actual {m} repeat{rep}/{repeats} {method}: {(summary["CC"]+summary["WC"])}/300 CW{summary["CW"]} WC{summary["WC"]} calls{len(cc)}',flush=True)
            for name,rr in [('outcomes',rows),('calls',calls),('summary',ss),('resources',rs)]:write_records(Path(f'{prefix}_m{models.index(m)}_r{rep}_{name}.csv'),rr)
            allrows.extend(rows);allcalls.extend(calls);summaries.extend(ss);resources.extend(rs)
    elapsed=time.perf_counter()-start;frozen_check(manifest)
    for name,h in frozen.items():assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==h
    for name,rr in [('outcomes',allrows),('calls',allcalls),('repeat_summary',summaries),('resources',resources),('warmup_calls',warmups),('cache_controls',resetlog),('reference_checks',references)]:write_records(Path(f'{prefix}_{name}.csv'),rr)
    write_json(Path(f'{prefix}_metadata.json'),{'run_id':run_id,'models':models,'repeats':repeats,'real_inference':True,'verification_calls':len(allcalls),'primary_calls':len(primary_calls),'calibration_calls':len(cal_calls),'verification_loop_seconds':elapsed,'raw_rows':len(allrows),'preserved_files':len(preserved),'power_gpu_money':'unmeasured'})
    print('VRG PREFIX: '+str(prefix),flush=True);return prefix

if __name__=='__main__':run()
