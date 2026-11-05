"""Shared immutable primaries, five balanced repeats, verification-only measurements."""
import hashlib,itertools,json,random,statistics,time,platform
from pathlib import Path
from uuid import uuid4
if __package__:
    from .controlled_dataset import build,ROOT,SEED
    from . import controlled_runtime as runtime
    from .rcav_runtime import LocalClient
    from .rcav_experiment import LARGE,SMALL
    from .rcav_guard_experiment import SOURCE,pool_record
    from . import rcav_guard_policy as guard
    from .efficiency_repeat import total,read_csv,frozen_check
    from .solver300_experiment import check_references
    from .verify_routing200 import process_snapshot,resource_delta
    from .routing_runner import check_ollama
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    from controlled_dataset import build,ROOT,SEED
    import controlled_runtime as runtime
    from rcav_runtime import LocalClient
    from rcav_experiment import LARGE,SMALL
    from rcav_guard_experiment import SOURCE,pool_record
    import rcav_guard_policy as guard
    from efficiency_repeat import total,read_csv,frozen_check
    from solver300_experiment import check_references
    from verify_routing200 import process_snapshot,resource_delta
    from routing_runner import check_ollama
    from tool_experiment import write_records
    from adaptive_experiment import write_json
GUARD_SOURCE=ROOT/'data/rcav_guard_b2e19463ef974c54a65b0770f2cf475d'
REPEATS=5
PROTOCOL={'quality_tolerance_absolute':.01,'confidence_level':.95,'bootstrap_draws':4000,
    'quality_rule':'conservative exact paired lower bound from all-repeat gains and any-repeat losses >= -.01; coverage not lower',
    'cpu_rule':'paired block-cluster bootstrap95 upper CPU delta vs selective <0',
    'time_rule':'paired block-cluster bootstrap95 upper wall delta vs selective <0',
    'order_rule':'CPU and wall differences <=0 in each of four target positions',
    'success_rule':'all rules and observed runner reset for both replay150 and fresh300; no threshold tuning'}

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def measured(work):
    before=process_snapshot();sampler=runtime.MemorySampler(before);sampler.start()
    start,cpu=time.perf_counter(),time.process_time()
    try:result=work()
    finally:
        wall=time.perf_counter()-start;pcpu=time.process_time()-cpu;after=process_snapshot();memory=sampler.stop(after)
    return result,{'before':before,'after':after,'stage_wall_seconds':wall,'router_cpu_seconds':pcpu,
        **resource_delta(before,after),**memory}

def make_schedule(datasets,models,repeats):
    schedule=[]
    for mi,model in enumerate(models):
        rng=random.Random(SEED+mi);blocks=[]
        for ds,tasks in datasets.items():
            ids=[t['task_id'] for t in tasks];rng.shuffle(ids)
            for j in range(0,len(ids),10):
                methods=list(runtime.METHODS);rng.shuffle(methods)
                blocks.append({'dataset':ds,'block_key':f'{ds}:B{j//10:03d}','task_ids':ids[j:j+10],'base_order':methods})
        for rep in range(1,repeats+1):
            order=blocks[:];random.Random(SEED+mi*100+rep).shuffle(order)
            for sequence,b in enumerate(order):
                shift=(rep-1)%4;methods=b['base_order'][shift:]+b['base_order'][:shift]
                for pos,method in enumerate(methods,1):schedule.append({'model':model,'repetition':rep,'sequence':sequence,
                    'dataset':b['dataset'],'block_key':b['block_key'],'task_ids':b['task_ids'],'position':pos,'method':method})
    return schedule

def grade(row,task):
    pc=row['primary_prediction'] is not None and row['primary_prediction']==task['expected']
    fc=row['final_prediction'] is not None and row['final_prediction']==task['expected']
    return {**row,'category':task['category'],'expected':task['expected'],'primary_correct':pc,'final_correct':fc,
        'answered':row['final_prediction'] is not None,'right_to_wrong':pc and not fc,'wrong_to_right':not pc and fc}

def summaries(rows,calls,blocks,model,rep):
    out=[]
    for ds in dict.fromkeys(r['dataset'] for r in rows):
        for method in runtime.METHODS:
            rr=[r for r in rows if r['dataset']==ds and r['method']==method];aa=[a for a in calls if a['dataset']==ds and a['method']==method];bb=[b for b in blocks if b['dataset']==ds and b['method']==method]
            n=len(rr);cpu=total(bb,'cpu_seconds');pcpu=total(bb,'router_cpu_seconds');rss=[b['sampled_max_rss_mib'] for b in bb if b['sampled_max_rss_mib'] is not None]
            pre=total(aa,'prefill_seconds') if aa else 0.;gen=total(aa,'eval_seconds') if aa else 0.
            out.append({'model':model,'dataset':ds,'repetition':rep,'method':method,'requests':n,
                'correct':sum(r['final_correct'] for r in rr),'accuracy':sum(r['final_correct'] for r in rr)/n,
                'answer_rate':sum(r['answered'] for r in rr)/n,'format_compliance':sum(r['final_format_compliant'] for r in rr)/n,
                'verification_calls':len(aa),'verification_input_tokens':total(aa,'input_tokens') if aa else 0,
                'verification_output_tokens':total(aa,'output_tokens') if aa else 0,
                'verification_wall_seconds':total(bb,'stage_wall_seconds'),'ollama_cpu_seconds':cpu,'router_cpu_seconds':pcpu,
                'measured_cpu_seconds':cpu+pcpu if cpu is not None and pcpu is not None else None,
                'model_prefill_seconds':pre,'model_generation_seconds':gen,
                'model_processing_seconds':pre+gen if pre is not None and gen is not None else None,
                'server_total_seconds':total(aa,'total_model_seconds') if aa else 0.,
                'sampled_max_rss_mib':max(rss) if rss else None,'rss_samples':sum(b['rss_sample_count'] for b in bb),
                'cpu_valid_blocks':sum(b['cpu_seconds'] is not None for b in bb),'blocks':len(bb),
                'right_to_wrong':sum(r['right_to_wrong'] for r in rr),'wrong_to_right':sum(r['wrong_to_right'] for r in rr),
                'api_errors':sum(a['status']!='generated' for a in aa),'monetary_cost':None,'cost_status':'unmeasured'})
    return out

def run(data_dir=ROOT/'data',repeats=REPEATS):
    if repeats<5:raise ValueError('Minimum five repeats')
    data_dir=Path(data_dir);fresh=build(data_dir)
    old=json.loads(Path(f'{SOURCE}_manifest.json').read_text())['evaluation'];datasets={'replay150':old,'fresh300':fresh}
    for ds,tt in datasets.items():
        for t in tt:t['task_id']=hashlib.sha256(t['question'].encode()).hexdigest()
    refs=check_references(old+fresh);check_ollama([LARGE]);tags=runtime.http_json('/api/tags')
    installed={r.get('name',r.get('model')) for r in tags['models']};models=[LARGE]+([SMALL] if SMALL in installed else [])
    source_profiles=json.loads(Path(f'{SOURCE}_profiles.json').read_text());profiles={}
    for model in models:profiles[model]={'rcav':source_profiles[model]['rcav'],'selective':source_profiles[model]['selective']}
    profiles[LARGE]['guard']=json.loads(Path(f'{GUARD_SOURCE}_guard_profile.json').read_text())
    preserved={}
    for folder in (ROOT/'src',ROOT/'tests',data_dir):
        for path in folder.iterdir():
            if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.iterdir():
        if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    run_id=uuid4().hex;prefix=data_dir/f'controlled_{run_id}';schedule=make_schedule(datasets,models,repeats)
    manifest={'run_id':run_id,'datasets':datasets,'models':models,'repeats':repeats,'protocol':PROTOCOL,
        'schedule':schedule,'preserved_sha256':preserved,'tags':tags,'platform':platform.platform(),
        'source_rcav':str(SOURCE),'source_improved':str(GUARD_SOURCE),'model_seed':42,'temperature':0,'num_predict':64}
    write_json(Path(f'{prefix}_manifest.json'),manifest);mh=hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()
    calibration=[];cal_calls=[];warmups=[]
    if SMALL in models:
        client=LocalClient(SMALL);warmups.append(client.send('Return only A.','calibration_warmup'))
        pool=[]
        for r in read_csv(f'{SOURCE}_calibration.csv'):
            if r['model']==SMALL:pool.append(pool_record(r['question'],r['expected'],r['category'],r['primary_answer'],r['verification_answer'],float(r['verification_model_seconds']),'historical_small100'))
        newcal=json.loads(Path(f'{GUARD_SOURCE}_manifest.json').read_text())['new_calibration']
        def train():
            for i,t in enumerate(newcal):
                p=client.primary(t['question']);a,v=client.verify(t['question'],p['answer']);cal_calls.extend(p['calls']);cal_calls.append(a)
                seconds=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None
                pool.append(pool_record(t['question'],t['expected'],t['category'],p['answer'],a['answer'],seconds,'fixed_guard_training100_small'))
                if (i+1)%25==0:print(f'Existing small-model guard calibration {i+1}/100',flush=True)
        _,cr=measured(train);calibration.append({'model':SMALL,**cr});profiles[SMALL]['guard']=guard.fit(pool)
        write_records(Path(f'{prefix}_small_guard_training.csv'),pool)
    write_json(Path(f'{prefix}_profiles.json'),profiles);ph=hashlib.sha256(Path(f'{prefix}_profiles.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_calibration_calls.csv'),cal_calls,fields=['call_id','model','stage','answer'])
    write_json(Path(f'{prefix}_calibration_resources.json'),calibration)
    # All primary generation completes, and is frozen, before verification starts.
    cache=[];primary_calls=[];primary_summaries=[]
    for model in models:
        client=LocalClient(model);warmups.append(client.send('Return only A.','primary_warmup'))
        for ds,tasks in datasets.items():
            entries=[];aa=[]
            def generate():
                for i,t in enumerate(tasks):
                    p=client.primary(t['question']);aa.extend(p['calls'])
                    record={'model':model,'dataset':ds,'task_id':t['task_id'],'question':t['question'],
                        'answer':p['answer'],'prediction':p['prediction'],'format_compliant':p['format_compliant'],
                        'source':p['source'],'parse_mode':p['parse_mode'],'primary_call_ids':[a['call_id'] for a in p['calls']]}
                    record['cache_entry_sha256']=digest(record);entries.append(record)
                    if (i+1)%75==0:print(f'Primary once {model} {ds} {i+1}/{len(tasks)}',flush=True)
            _,resource=measured(generate);cache.extend(entries);primary_calls.extend(aa)
            primary_summaries.append({'model':model,'dataset':ds,'requests':len(entries),'model_calls':len(aa),
                'tool_calls':sum(r['source']=='tool' for r in entries),'input_tokens':total(aa,'input_tokens'),
                'output_tokens':total(aa,'output_tokens'),'prefill_seconds':total(aa,'prefill_seconds'),'generation_seconds':total(aa,'eval_seconds'),**resource})
    write_json(Path(f'{prefix}_primary_cache.json'),cache);ch=hashlib.sha256(Path(f'{prefix}_primary_cache.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_primary_calls.csv'),primary_calls);write_records(Path(f'{prefix}_primary_summary.csv'),primary_summaries)
    cached={(r['model'],r['task_id']):r for r in cache};task_index={t['task_id']:t for tt in datasets.values() for t in tt}
    print('All shared primaries and policy profiles frozen before verification',flush=True)
    rows=[];calls=[];blocks=[];controls=[];all_summaries=[];start=time.perf_counter()
    for mi,model in enumerate(models):
        client=LocalClient(model)
        for rep in range(1,repeats+1):
            control,warm=runtime.reset_and_warm(client,models);controls.append({'model':model,'repetition':rep,**control});warmups.append(warm)
            rr=[];aa=[];bb=[];planned=[s for s in schedule if s['model']==model and s['repetition']==rep]
            for si,s in enumerate(planned):
                ps=[cached[model,id] for id in s['task_ids']]
                block,resource=measured(lambda:runtime.execute_block(ps,s['method'],client,profiles[model]))
                bid=f'M{mi}:R{rep}:{s["block_key"]}:{s["method"]}'
                for r in block['rows']:rr.append({**grade(r,task_index[r['task_id']]),'dataset':s['dataset'],'repetition':rep,
                    'position':s['position'],'sequence':s['sequence'],'block_key':s['block_key'],'block_id':bid})
                aa.extend({**a,'dataset':s['dataset'],'repetition':rep,'method':s['method'],'position':s['position'],'block_id':bid} for a in block['calls'])
                bb.append({'model':model,'dataset':s['dataset'],'repetition':rep,'method':s['method'],'position':s['position'],
                    'sequence':s['sequence'],'block_key':s['block_key'],'block_id':bid,'task_ids':s['task_ids'],**resource})
                if (si+1)%40==0:print(f'Controlled actual {model} repeat{rep}/{repeats}: {(si+1)//4}/{len(planned)//4} blocks',flush=True)
            ss=summaries(rr,aa,bb,model,rep);all_summaries.extend(ss)
            for suffix,records in [('outcomes',rr),('calls',aa),('blocks',bb),('summary',ss)]:write_records(Path(f'{prefix}_m{mi}_r{rep}_{suffix}.csv'),records)
            rows.extend(rr);calls.extend(aa);blocks.extend(bb);print('REPEAT '+json.dumps(ss),flush=True)
    elapsed=time.perf_counter()-start;frozen_check(manifest)
    for suffix,expected in [('manifest',mh),('profiles',ph),('primary_cache',ch)]:
        if hashlib.sha256(Path(f'{prefix}_{suffix}.json').read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen input changed')
    if len({a['call_id'] for a in calls})!=len(calls):raise RuntimeError('Duplicate actual verifier calls')
    meta={'run_id':run_id,'prefix_name':prefix.name,'real_inference':True,'evaluation_loop_seconds':elapsed,
        'outcome_rows':len(rows),'actual_verifier_calls':len(calls),'shared_primary_model_calls':len(primary_calls),
        'manifest_sha256':mh,'profiles_sha256':ph,'primary_cache_sha256':ch,'preserved_files_checked':len(preserved),
        'all_primaries_shared_and_not_regenerated':True,'power_gpu_flops_money_status':'unmeasured'}
    for suffix,records in [('outcomes',rows),('calls',calls),('blocks',blocks),('repeat_summary',all_summaries),('cache_controls',controls),('warmup_calls',warmups),('reference_checks',refs)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_metadata.json'),meta);print('CONTROLLED PREFIX: '+str(prefix),flush=True)
    return prefix

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as exc:raise SystemExit(f'Controlled validation stopped: {exc}')
