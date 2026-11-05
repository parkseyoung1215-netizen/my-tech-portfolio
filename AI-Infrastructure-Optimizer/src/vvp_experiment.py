"""Actual six-method/six-budget/five-repeat local evaluation with frozen primaries."""
import hashlib,json,random,time
from pathlib import Path
from uuid import uuid4
if __package__:
    from .vvp_dataset import build,ROOT,SEED
    from . import vvp_model as model
    from . import vvp_runtime as runtime
    from .controlled_validation import measured,digest
    from . import controlled_runtime as controls
    from .rcav_runtime import LocalClient
    from .rcav_experiment import LARGE,SMALL
    from .controlled_validation import SOURCE
    from .efficiency_repeat import total,read_csv,frozen_check
    from .solver300_dataset import task_key
    from .solver300_experiment import check_references
    from .routing_runner import check_ollama
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    from vvp_dataset import build,ROOT,SEED
    import vvp_model as model
    import vvp_runtime as runtime
    from controlled_validation import measured,digest
    import controlled_runtime as controls
    from rcav_runtime import LocalClient
    from rcav_experiment import LARGE,SMALL
    from controlled_validation import SOURCE
    from efficiency_repeat import total,read_csv,frozen_check
    from solver300_dataset import task_key
    from solver300_experiment import check_references
    from routing_runner import check_ollama
    from tool_experiment import write_records
    from adaptive_experiment import write_json
CONTROLLED=ROOT/'data/controlled_1ee08ac18eaa428f9644ef66daa2fcaa'
TRAIN={LARGE:ROOT/'data/rcav_guard_b2e19463ef974c54a65b0770f2cf475d_calibration_pool.csv',SMALL:Path(f'{CONTROLLED}_small_guard_training.csv')}
REPEATS=5
PROTOCOL={'main_budget_percent':25,'accuracy_tolerance':0,'bootstrap_draws':4000,
    'success':'At25%, conservative accuracy lower95 vs Selective100 >=0, gain lower95 vs None0 >0, CPU upper95 vs Selective100 <0; both models and both fixed test cohorts must reproduce direction.',
    'budget':'Call cap floor(N*pct) AND common predicted CPU quota pct*sum(cost predictions). Actual CPU is measured, not hard-capped.',
    'secondary_budget_grid':'0,10,25,50,75,100; do not choose a main winner from evaluation curves',
    'total_cost_scope':'Initial generation measured once + decision/verification actual measurements. Sum is accounting, not independent production end-to-end execution.'}

def old_evaluation_keys(data_dir):
    keys=set()
    for path in Path(data_dir).glob('*evaluation.csv'):
        for r in read_csv(path):
            if r.get('question'):keys.add(task_key(r['question']))
    # Controlled evaluation calls its raw outputs outcomes.csv.
    for path in Path(data_dir).glob('*outcomes.csv'):
        for r in read_csv(path):
            if r.get('question'):keys.add(task_key(r['question']))
    return keys

def historical(data_dir):
    rows=read_csv(f'{CONTROLLED}_outcomes.csv');index={(r['model'],r['dataset'],r['repetition'],r['task_id'],r['method']):r for r in rows};out=[]
    for md in {(r['model'],r['dataset'],r['method']) for r in rows}:
        rr=[r for r in rows if (r['model'],r['dataset'],r['method'])==md];missed=0;same=0;changed_same=0
        for r in rr:
            b=index[r['model'],r['dataset'],r['repetition'],r['task_id'],'baseline']
            if r['verification_executed']=='False' and r['primary_correct']=='False' and b['final_correct']=='True':missed+=1
            if r['verification_executed']=='True' and r['right_to_wrong']=='False' and r['wrong_to_right']=='False':
                same+=1;changed_same+=r['primary_prediction']!=r['final_prediction']
        out.append({'model':md[0],'dataset':md[1],'method':md[2],'repeated_rows':len(rr),
            'repairs':sum(r['wrong_to_right']=='True' for r in rr),'damage':sum(r['right_to_wrong']=='True' for r in rr),
            'verified_same_correctness':same,'changed_answer_same_correctness':changed_same,'missed_repairs_paired_full_reference':missed,
            'scope':'Historical diagnosis only. Same frozen primary; separate-time verifier outcomes, not training data.'})
    return out

def collect(tasks,client,phase):
    rows=[];calls=[]
    for i,t in enumerate(tasks):
        p=client.primary(t['question']);calls.extend(p['calls'])
        pair,resource=measured(lambda:client.verify(t['question'],p['answer']));a,v=pair;calls.append(a)
        if a['status']!='generated':raise RuntimeError('Training verifier failed; no synthetic effects')
        final=v['prediction'] if v['prediction'] is not None else p['prediction'];pc=p['prediction'] is not None and p['prediction']==t['expected'];fc=final is not None and final==t['expected']
        cpu=resource['cpu_seconds']+resource['router_cpu_seconds'] if resource['cpu_seconds'] is not None else None
        rows.append({'question':t['question'],'expected':t['expected'],'category':t['category'],'phase':phase,
            'primary':{k:p[k] for k in ('answer','prediction','format_compliant','source')},'verification_answer':a['answer'],
            'primary_correct':pc,'final_correct':fc,'label':model.label(pc,fc),'features':model.observed(t['question'],p),
            'cpu_seconds':cpu,'verifier_resource':resource,'verification_call_id':a['call_id']})
        if (i+1)%50==0:print(f'VVP {client.model} {phase} {i+1}/{len(tasks)}',flush=True)
    return rows,calls

def summarize(rows,calls,resource,result,method,budget,rep,model_name):
    n=len(rows);cpu=resource['cpu_seconds']+resource['router_cpu_seconds'] if resource['cpu_seconds'] is not None else None
    return {'model':model_name,'repetition':rep,'method':method,'budget_percent':budget,'requests':n,
        'initial_correct':sum(r['primary_correct'] for r in rows),'correct':sum(r['final_correct'] for r in rows),
        'accuracy':sum(r['final_correct'] for r in rows)/n,'answer_rate':sum(r['final_prediction'] is not None for r in rows)/n,
        'format_compliance':sum(r['final_format_compliant'] for r in rows)/n,'repairs':sum(r['wrong_to_right'] for r in rows),
        'damage':sum(r['right_to_wrong'] for r in rows),'verification_calls':len(calls),
        'input_tokens':total(calls,'input_tokens') if calls else 0,'output_tokens':total(calls,'output_tokens') if calls else 0,
        'verification_wall_seconds':resource['stage_wall_seconds'],'measured_cpu_seconds':cpu,
        'ollama_cpu_seconds':resource['cpu_seconds'],'router_cpu_seconds':resource['router_cpu_seconds'],
        'planning_wall_seconds':result['planning_wall_seconds'],'planning_cpu_seconds':result['planning_cpu_seconds'],
        'sampled_max_rss_mib':resource['sampled_max_rss_mib'],'predicted_cpu_quota':result['plan']['predicted_cpu_quota'],
        'selected_estimated_cpu':result['plan']['selected_estimated_cpu'],'api_errors':sum(a['status']!='generated' for a in calls),
        'monetary_cost':None,'cost_status':'unmeasured'}

def run(data_dir=ROOT/'data',repeats=REPEATS):
    if repeats<5:raise ValueError('Minimum5 repeats')
    data_dir=Path(data_dir);validation,test=build(data_dir);references=check_references(validation+test)
    check_ollama([LARGE]);tags=controls.http_json('/api/tags');names={r.get('name',r.get('model')) for r in tags['models']};models=[LARGE]+([SMALL] if SMALL in names else [])
    profiles=json.loads(Path(f'{CONTROLLED}_profiles.json').read_text());seen_eval=old_evaluation_keys(ROOT/'data');training={}
    for m in models:
        training[m]=read_csv(TRAIN[m])
        if any(task_key(r['question']) in seen_eval for r in training[m]):raise RuntimeError('Past evaluation used as VVP training')
    preserved={}
    for folder in (ROOT/'src',ROOT/'tests',data_dir):
        for path in folder.iterdir():
            if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.iterdir():
        if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for t in test:t['task_id']=hashlib.sha256(t['question'].encode()).hexdigest()
    run_id=uuid4().hex;prefix=data_dir/f'vvp_{run_id}';conditions=[(m,b) for m in runtime.METHODS for b in runtime.BUDGETS];schedule=[];arrivals={}
    for mi,m in enumerate(models):
        for rep in range(1,repeats+1):
            arrival=[t['task_id'] for t in test];random.Random(SEED+mi*100+rep).shuffle(arrival);arrivals[m+':'+str(rep)]=arrival
            order=conditions[:];random.Random(SEED+mi*1000+rep).shuffle(order)
            for pos,(method,budget) in enumerate(order):schedule.append({'model':m,'repetition':rep,'position':pos+1,'method':method,'budget_percent':budget})
    manifest={'run_id':run_id,'protocol':PROTOCOL,'validation':validation,'test':test,'training_questions':training,'models':models,
        'repeats':repeats,'schedule':schedule,'arrivals':arrivals,'existing_profiles':profiles,'tags':tags,
        'preserved_sha256':preserved,'training_past_evaluation_overlap':0,'regularization_grid':model.REGULARIZATIONS,'temperatures':model.TEMPERATURES}
    write_json(Path(f'{prefix}_manifest.json'),manifest);mh=hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_historical_diagnosis.csv'),historical(ROOT/'data'))
    predictors={};training_rows=[];collection_calls=[];training_resources=[];warmups=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','training_warmup'))
        start=time.perf_counter();cpu=time.process_time();tr,tc=collect(training[m],client,'training');va,vc=collect(validation,client,'validation')
        collection_wall=time.perf_counter()-start;collection_cpu=time.process_time()-cpu
        start=time.perf_counter();cpu=time.process_time();predictors[m]=model.train(tr,va)
        fitwall=time.perf_counter()-start;fitcpu=time.process_time()-cpu
        training_rows.extend({**r,'model':m} for r in tr+va);collection_calls.extend(tc+vc)
        training_resources.append({'model':m,'collection_wall_seconds':collection_wall,'collection_client_cpu_seconds':collection_cpu,
            'model_calls':len(tc+vc),'fit_wall_seconds':fitwall,'fit_cpu_seconds':fitcpu,
            'input_tokens':total(tc+vc,'input_tokens'),'output_tokens':total(tc+vc,'output_tokens')})
    write_json(Path(f'{prefix}_predictors.json'),predictors);ph=hashlib.sha256(Path(f'{prefix}_predictors.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_training.csv'),training_rows);write_records(Path(f'{prefix}_training_calls.csv'),collection_calls);write_records(Path(f'{prefix}_training_resources.csv'),training_resources)
    cache=[];primary_calls=[];primary_cost=[]
    for m in models:
        client=LocalClient(m);warmups.append(client.send('Return only A.','initial_warmup'));entries=[];aa=[]
        def initial():
            for i,t in enumerate(test):
                p=client.primary(t['question']);aa.extend(p['calls']);r={k:p[k] for k in ('answer','prediction','format_compliant','source')}
                r.update(model=m,task_id=t['task_id'],question=t['question']);r['entry_hash']=digest(r);entries.append(r)
                if (i+1)%100==0:print(f'VVP shared initial {m} {i+1}/300',flush=True)
        _,res=measured(initial);cache.extend(entries);primary_calls.extend(aa);primary_cost.append({'model':m,'model_calls':len(aa),'input_tokens':total(aa,'input_tokens'),'output_tokens':total(aa,'output_tokens'),**res})
    write_json(Path(f'{prefix}_initial_cache.json'),cache);ch=hashlib.sha256(Path(f'{prefix}_initial_cache.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_primary_calls.csv'),primary_calls);write_records(Path(f'{prefix}_primary_cost.csv'),primary_cost)
    cached={m:[p for p in cache if p['model']==m] for m in models};plans=[]
    for s in schedule:
        plan=runtime.allocate(cached[s['model']],s['method'],s['budget_percent'],profiles[s['model']],predictors[s['model']],arrivals[s['model']+':'+str(s['repetition'])])
        plans.append({**s,'plan':plan})
    write_json(Path(f'{prefix}_plans.json'),plans);planhash=hashlib.sha256(Path(f'{prefix}_plans.json').read_bytes()).hexdigest()
    frozen_plans={(p['model'],p['repetition'],p['method'],p['budget_percent']):p['plan'] for p in plans};tasks={t['task_id']:t for t in test}
    print('Predictors, primaries, budgets and allocations frozen before all test verification',flush=True)
    allrows=[];allcalls=[];summaries=[];resources=[];resetlog=[];start=time.perf_counter()
    for mi,m in enumerate(models):
        client=LocalClient(m)
        for rep in range(1,repeats+1):
            reset,warm=controls.reset_and_warm(client,models);resetlog.append({'model':m,'repetition':rep,**reset});warmups.append(warm)
            rr=[];aa=[];ss=[];rs=[];order=[s for s in schedule if s['model']==m and s['repetition']==rep]
            for s in order:
                method,budget=s['method'],s['budget_percent'];arrival=arrivals[m+':'+str(rep)]
                result,res=measured(lambda:runtime.execute(cached[m],method,budget,client,profiles[m],predictors[m],arrival))
                if result['plan']!=frozen_plans[m,rep,method,budget]:raise RuntimeError('Allocation changed after outcomes')
                rows=[]
                for r in result['rows']:
                    t=tasks[r['task_id']];pc=r['primary_prediction'] is not None and r['primary_prediction']==t['expected'];fc=r['final_prediction'] is not None and r['final_prediction']==t['expected']
                    rows.append({**r,'model':m,'repetition':rep,'method':method,'budget_percent':budget,'position':s['position'],
                        'category':t['category'],'cohort':t['cohort'],'expected':t['expected'],'primary_correct':pc,'final_correct':fc,'wrong_to_right':not pc and fc,'right_to_wrong':pc and not fc})
                calls=[{**a,'repetition':rep,'method':method,'budget_percent':budget,'position':s['position']} for a in result['calls']]
                summary=summarize(rows,calls,res,result,method,budget,rep,m);ss.append(summary);rr.extend(rows);aa.extend(calls);rs.append({'model':m,'repetition':rep,'method':method,'budget_percent':budget,'position':s['position'],**res})
                print(f'VVP actual {m} repeat{rep}/{repeats} {method} budget{budget}: {summary["correct"]}/300 calls{len(calls)}',flush=True)
            for suffix,records in [('outcomes',rr),('calls',aa),('summary',ss),('resources',rs)]:write_records(Path(f'{prefix}_m{mi}_r{rep}_{suffix}.csv'),records)
            allrows.extend(rr);allcalls.extend(aa);summaries.extend(ss);resources.extend(rs)
    elapsed=time.perf_counter()-start;frozen_check(manifest)
    for name,h in [('manifest',mh),('predictors',ph),('initial_cache',ch),('plans',planhash)]:
        if hashlib.sha256(Path(f'{prefix}_{name}.json').read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen VVP input changed')
    if len({a['call_id'] for a in allcalls})!=len(allcalls):raise RuntimeError('Duplicate actual inference calls')
    meta={'run_id':run_id,'prefix_name':prefix.name,'real_inference':True,'verification_loop_seconds':elapsed,'models':models,'repeats':repeats,
        'actual_verification_calls':len(allcalls),'actual_primary_calls':len(primary_calls),'training_collection_calls':len(collection_calls),
        'outcome_rows':len(allrows),'manifest_sha256':mh,'predictors_sha256':ph,'initial_cache_sha256':ch,'plans_sha256':planhash,
        'preserved_files_checked':len(preserved),'power_gpu_money':'unmeasured'}
    for suffix,records in [('outcomes',allrows),('calls',allcalls),('repeat_summary',summaries),('resources',resources),('cache_controls',resetlog),('warmup_calls',warmups),('reference_checks',references)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_metadata.json'),meta);print('VVP PREFIX: '+str(prefix),flush=True);return prefix

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as exc:raise SystemExit(f'VVP stopped: {exc}')
