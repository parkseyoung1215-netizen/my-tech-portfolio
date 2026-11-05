"""Actual pre-verification gate trial; gold exists only in collection/grading."""
import hashlib,json,random,time,platform
from pathlib import Path
from uuid import uuid4
from .pvcg_dataset import ROOT,SEED,build
from . import pvcg_policy as policy
from .vrg_policy import parse
from .vrg_experiment import parsed_primary,SOURCE,CONTROLLED
from .rcav_runtime import LocalClient
from .rcav_experiment import LARGE,SMALL
from .controlled_validation import measured,digest
from . import controlled_runtime as controls
from .controlled_analysis import read_decoded
from .efficiency_repeat import frozen_check,total
from .solver300_experiment import check_references
from .solver300_dataset import task_key
from .vvp_experiment import old_evaluation_keys
from .tool_experiment import write_records
from .adaptive_experiment import write_json
from .routing_runner import check_ollama
METHODS=('none','full','selective','pvcg')
PROTOCOL={'seed':SEED,'repeats':5,'quality_tolerance':0,
 'success':'Both models: conservative question-paired lower95 accuracy delta vs Selective >=0 and vs None >0; paired CPU upper95 vs Selective <0; tokens not increased; positive accuracy direction vs None on both prespecified150 cohorts; actually verify some requests.',
 'gate':'Subset of existing Selective eligible requests; group posterior net gain minus uncertainty penalty > CPU penalty*historical median CPU. Support>=10, backoff to category.',
 'calibration':'Grid uncertainty0/.5/1, CPU penalty0/.05/.2. Max calibration correctness, tie lower actually measured verifier CPU sum, then conservative parameters. No test tuning.',
 'adoption':'Same in Full/Selective/PVCG: adopt parsed verifier answer, else retain initial. No post-verification guard.',
 'cost_scope':'Measured initial once + actually executed gate/verification stage component accounting; not independently repeated end-to-end generation. Train collection/fit separate.',
 'novelty':'Empirical support-aware filter of an existing gate, not a claim of a new foundational algorithm.',
 'inference':'Bootstrap4000; question clusters, five frozen-primary repeats not independent questions; zero-tolerance success is conservative.'}

def record(q,expected,cat,p,revision,cpu):
    primary={k:p[k] for k in ('answer','prediction','format_compliant','source')};v=parse(q,revision);pred=v['prediction'] if v['prediction'] is not None else primary['prediction']
    pc=primary['prediction'] is not None and primary['prediction']==expected;fc=pred is not None and pred==expected
    return {'question':q,'expected':expected,'category':cat,'primary':primary,'verification_answer':revision,'primary_correct':pc,'final_correct':fc,'gain':int(fc)-int(pc),'verifier_cpu_seconds':cpu}

def decision(q,p,method,profile,old):
    policy.public(q,p)
    if method=='none':return {'verify':False,'reason':'no_verification'}
    if method=='full':return {'verify':True,'reason':'full_verification'}
    if method=='selective':return controls.choose(q,p,'selective',old)
    if method=='pvcg':return policy.decide(q,p,profile,old)
    raise ValueError('Unknown method')

def execute(primaries,method,client,profile,old,arrival,frozen_plan):
    lookup={p['task_id']:p for p in primaries};rows=[];calls=[];gatecpu=0.;gatewall=0.
    for id in arrival:
        p=lookup[id];q=p['question'];start,cpu=time.perf_counter(),time.process_time();d=decision(q,p,method,profile,old)
        gatecpu+=time.process_time()-cpu;gatewall+=time.perf_counter()-start
        if d!=frozen_plan[id]:raise RuntimeError('Decision changed after test verification began')
        # d is complete before any verifier response for this request exists.
        a=None;v=None
        if d['verify']:
            a,_=client.verify(q,p['answer']);calls.append(dict(a,task_id=id));v=dict(parse(q,a['answer']),answer=a['answer'])
        selected=v if v and v['prediction'] is not None else p
        rows.append({'task_id':id,'question':q,'primary_answer':p['answer'],'primary_prediction':p['prediction'],'primary_source':p['source'],
          'primary_format_compliant':p['format_compliant'],'decision':d,'decision_phase':'before_verification','verification_executed':d['verify'],
          'verification_answer':a['answer'] if a else '','verification_prediction':v['prediction'] if v else None,'call_id':a['call_id'] if a else '',
          'final_answer':selected['answer'],'final_prediction':selected['prediction'],'final_format_compliant':selected['format_compliant']})
    return {'rows':rows,'calls':calls,'gate_cpu_seconds':gatecpu,'gate_wall_seconds':gatewall}

def summarize(rows,calls,res,result,m,rep,method,initial):
    n=len(rows);g={s:sum(r['transition']==s for r in rows) for s in ('CC','CW','WC','WW')};cpu=res['cpu_seconds']+res['router_cpu_seconds'] if res['cpu_seconds'] is not None else None
    icpu=initial['cpu_seconds']+initial['router_cpu_seconds'] if initial['cpu_seconds'] is not None else None
    it=total(calls,'input_tokens') if calls else 0;ot=total(calls,'output_tokens') if calls else 0
    return {'model':m,'repetition':rep,'method':method,'requests':n,**g,'initial_accuracy':(g['CC']+g['CW'])/n,'accuracy':(g['CC']+g['WC'])/n,
       'answer_rate':sum(r['final_prediction'] is not None for r in rows)/n,'format_compliance':sum(r['final_format_compliant'] for r in rows)/n,
       'verification_calls':len(calls),'input_tokens':it,'output_tokens':ot,'stage_cpu_seconds':cpu,'ollama_cpu_seconds':res['cpu_seconds'],'python_cpu_seconds':res['router_cpu_seconds'],
       'stage_wall_seconds':res['stage_wall_seconds'],'gate_cpu_seconds':result['gate_cpu_seconds'],'gate_wall_seconds':result['gate_wall_seconds'],
       'accounted_total_model_calls':initial['model_calls']+len(calls),'accounted_total_cpu_seconds':icpu+cpu if icpu is not None and cpu is not None else None,
       'accounted_total_wall_seconds':initial['stage_wall_seconds']+res['stage_wall_seconds'],
       'accounted_total_input_tokens':initial['input_tokens']+it if it is not None else None,'accounted_total_output_tokens':initial['output_tokens']+ot if ot is not None else None,
       'sampled_max_rss_mib':res['sampled_max_rss_mib'],'api_errors':sum(a['status']!='generated' for a in calls),'unmeasured':'GPU FLOPs, power, monetary cost'}

def run(data_dir=ROOT/'data',repeats=5):
    if repeats<5:raise ValueError('At least five repeats')
    data_dir=Path(data_dir);validation,test=build(data_dir);refs=check_references(validation+test);models=[LARGE,SMALL];check_ollama(models)
    old=json.loads(Path(f'{CONTROLLED}_profiles.json').read_text());training={m:[] for m in models};seen=old_evaluation_keys(ROOT/'data')
    for r in read_decoded(f'{SOURCE}_training.csv'):
        if r['phase']!='training':continue
        if task_key(r['question']) in seen:raise ValueError('Old evaluation reused for training')
        p=r['primary'];p.update(parse(r['question'],p['answer']));training[r['model']].append(record(r['question'],r['expected'],r['category'],p,r['verification_answer'],r['cpu_seconds']))
    assert all(len(rr)==200 for rr in training.values())
    paths=[p for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()]+[p for p in ROOT.iterdir() if p.is_file()]
    preserved={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for t in test:t['task_id']=hashlib.sha256(t['question'].encode()).hexdigest()
    schedule=[];arrivals={};orders={};base=list(METHODS);random.Random(SEED).shuffle(base)
    for rep in range(1,repeats+1):
        order=models[:] if rep%2 else models[::-1];orders[str(rep)]=order
        for m in order:
            ids=[t['task_id'] for t in test];random.Random(SEED+rep*100+models.index(m)).shuffle(ids);arrivals[m+':'+str(rep)]=ids
            shift=(rep-1)%4;ms=base[shift:]+base[:shift]
            if m==SMALL:ms=list(reversed(ms))
            for pos,method in enumerate(ms,1):schedule.append(dict(model=m,repetition=rep,method=method,position=pos))
    prefix=data_dir/('pvcg_'+uuid4().hex);manifest={'seed':SEED,'protocol':PROTOCOL,'models':models,'repeats':repeats,'training':training,'validation':validation,'test':test,
      'old_profiles':old,'schedule':schedule,'arrivals':arrivals,'model_orders':orders,'preserved_sha256':preserved,'tags':controls.http_json('/api/tags'),'platform':platform.platform(),'python':platform.python_version()}
    write_json(Path(f'{prefix}_manifest.json'),manifest);profiles={};cal=[];cal_calls=[];traincost=[];warmups=[];resets=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','pvcg_calibration_warmup'));rr=[];aa=[]
        def collect():
            for t in validation:
                p=parsed_primary(client,t['question']);aa.extend(p['calls']);(a,_),vcost=measured(lambda:client.verify(t['question'],p['answer']));aa.append(a)
                if a['status']!='generated':raise RuntimeError('Calibration verifier failed')
                cpu=vcost['cpu_seconds']+vcost['router_cpu_seconds'] if vcost['cpu_seconds'] is not None else None
                rr.append(dict(record(t['question'],t['expected'],t['category'],p,a['answer'],cpu),model=m,verifier_resource=vcost))
        _,res=measured(collect);start,cpu=time.perf_counter(),time.process_time();profiles[m]=policy.train(training[m],rr,old[m]);fitcpu=time.process_time()-cpu;fitwall=time.perf_counter()-start
        cal.extend(rr);cal_calls.extend(aa);traincost.append(dict(model=m,new_calibration_calls=len(aa),input_tokens=total(aa,'input_tokens'),output_tokens=total(aa,'output_tokens'),fit_cpu_seconds=fitcpu,fit_wall_seconds=fitwall,
          collection_cpu_seconds=res['cpu_seconds']+res['router_cpu_seconds'] if res['cpu_seconds'] is not None else None,collection_wall_seconds=res['stage_wall_seconds'],resource=res,historical_training_rows=200,historical_cost='Reused historical training measurements, not free or re-executed.'))
        print('PVCG calibrated '+m,flush=True)
    for name,value in [('profiles',profiles)]:write_json(Path(f'{prefix}_{name}.json'),value)
    for name,rr in [('calibration',cal),('calibration_calls',cal_calls),('training_cost',traincost)]:write_records(Path(f'{prefix}_{name}.csv'),rr)
    cache=[];primary_calls=[];initial=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','pvcg_initial_warmup'));entries=[];aa=[]
        def generate():
            for t in test:
                p=parsed_primary(client,t['question']);aa.extend(p['calls']);r={k:p[k] for k in ('answer','prediction','format_compliant','source','parse_mode')};r.update(model=m,task_id=t['task_id'],question=t['question']);r['entry_hash']=digest(r);entries.append(r)
        _,res=measured(generate);cache.extend(entries);primary_calls.extend(aa);initial.append(dict(res,model=m,model_calls=len(aa),input_tokens=total(aa,'input_tokens'),output_tokens=total(aa,'output_tokens')))
        print('PVCG shared initial '+m,flush=True)
    write_json(Path(f'{prefix}_initial_cache.json'),cache);write_records(Path(f'{prefix}_primary_calls.csv'),primary_calls);write_records(Path(f'{prefix}_primary_cost.csv'),initial)
    plans={m:{method:{p['task_id']:decision(p['question'],p,method,profiles[m],old[m]) for p in cache if p['model']==m} for method in METHODS} for m in models}
    write_json(Path(f'{prefix}_plans.json'),plans);frozen={s:hashlib.sha256(Path(f'{prefix}_{s}.json').read_bytes()).hexdigest() for s in ('manifest','profiles','initial_cache','plans')};write_json(Path(f'{prefix}_frozen.json'),frozen)
    tasks={t['task_id']:t for t in test};allrows=[];allcalls=[];summaries=[];resources=[];start=time.perf_counter()
    for rep in range(1,repeats+1):
        for m in orders[str(rep)]:
            client=LocalClient(m);reset,warm=controls.reset_and_warm(client,models);resets.append(dict(reset,model=m,repetition=rep));warmups.append(warm);rr=[];aa=[];ss=[];rs=[];ps=[p for p in cache if p['model']==m];ic=next(r for r in initial if r['model']==m)
            for s in [s for s in schedule if s['model']==m and s['repetition']==rep]:
                method=s['method'];result,res=measured(lambda:execute(ps,method,client,profiles[m],old[m],arrivals[m+':'+str(rep)],plans[m][method]));graded=[]
                for r in result['rows']:
                    t=tasks[r['task_id']];pc=r['primary_prediction'] is not None and r['primary_prediction']==t['expected'];fc=r['final_prediction'] is not None and r['final_prediction']==t['expected']
                    graded.append(dict(r,**s,expected=t['expected'],category=t['category'],cohort=t['cohort'],primary_correct=pc,final_correct=fc,transition=('C' if pc else 'W')+('C' if fc else 'W')))
                calls=[dict(a,**s) for a in result['calls']];summary=summarize(graded,calls,res,result,m,rep,method,ic);rr.extend(graded);aa.extend(calls);ss.append(summary);rs.append(dict(res,**s))
                print(f'PVCG actual {m} repeat{rep}/{repeats} {method}: {summary["CC"]+summary["WC"]}/300 calls{len(calls)} CW{summary["CW"]} WC{summary["WC"]}',flush=True)
            for name,values in [('outcomes',rr),('calls',aa),('summary',ss),('resources',rs)]:write_records(Path(f'{prefix}_m{models.index(m)}_r{rep}_{name}.csv'),values)
            allrows.extend(rr);allcalls.extend(aa);summaries.extend(ss);resources.extend(rs)
    elapsed=time.perf_counter()-start;frozen_check(manifest)
    for name,h in frozen.items():assert hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()==h
    for name,rr in [('outcomes',allrows),('calls',allcalls),('repeat_summary',summaries),('resources',resources),('warmup_calls',warmups),('cache_controls',resets),('reference_checks',refs)]:write_records(Path(f'{prefix}_{name}.csv'),rr)
    write_json(Path(f'{prefix}_metadata.json'),{'prefix':prefix.name,'models':models,'repeats':repeats,'actual_verification_calls':len(allcalls),'actual_primary_calls':len(primary_calls),'new_calibration_calls':len(cal_calls),'outcome_rows':len(allrows),'verification_loop_seconds':elapsed,'real_inference':True,'preserved_files':len(preserved),'GPU_FLOPs_power_money':'unmeasured'})
    print('PVCG PREFIX '+str(prefix),flush=True);return prefix

if __name__=='__main__':run()
