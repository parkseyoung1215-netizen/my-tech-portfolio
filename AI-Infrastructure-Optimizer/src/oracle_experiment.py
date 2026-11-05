"""Append-only offline privileged builder and actual five-repeat hardware experiment."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import fcntl,hashlib,importlib.metadata,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from .dlr_experiment import runtime as old_runtime,ROOT,sha
from .oracle_runtime import OracleRuntime
from .oracle_policy import RATES,SEED,quota,random_plan,drift_plan,validate,choose_candidate
from .oracle_data import fresh,score
from .stable_runtime_records import Store
from .adaptive_experiment import write_json
from .alsr_io import write_union
from .efficiency_repeat import frozen_check
PRESERVE=ROOT/'data/oracle_preservation_fd7a17a855d64265ac85b54aefca1e66.json'
OLD=ROOT/'data/dlr_opt_2c8a76633b5e450f88242020fce6a899/MANIFEST.json'
CORE=('oracle_runtime.py','oracle_experiment.py','oracle_policy.py','oracle_data.py')
def runtime():return OracleRuntime(old_runtime())
def conditions():return [('baseline',None),('existing',None)]+[(method,rate) for method in ('ideal','random','overhead') for rate in RATES]
def key(j):return '/'.join(str(j[k]) for k in ('phase','dataset','repetition','task_id','method','rate'))

def smoke():
    p=ROOT/'data'/('oracle_smoke_'+uuid4().hex);p.mkdir();r=runtime();q='Return only a JSON object with keys x and y. Values: x=3, y=7.';records=[]
    base=r.generate(q,'baseline',max_tokens=12);zero=r.compile_plan(random_plan(12,0,SEED));same=r.generate(q,'ideal',zero,max_tokens=12);assert same['token_ids']==base['token_ids']
    trace,_=r.generate(q,'baseline',max_tokens=12,capture=True,stop_eos=False);assert trace['token_ids'][:len(base['token_ids'])]==base['token_ids']
    for rate in (0,30,50):
        plan=random_plan(12,rate,SEED);routes=r.compile_plan(plan)
        for method in ('ideal','random','overhead'):
            rec=r.generate(q,method,routes,max_tokens=12,fixed_tokens=base['token_ids']);assert sum(rec['reused_mlp_calls'])==sum(len(s) for s in plan[:len(base['token_ids'])]);records.append(dict(method=method,rate=rate,result=rec))
    # The actual MLP branch raises on a fresh step but is bypassed on a reused step.
    from mlx_lm.models.cache import make_prompt_cache
    ids=r.old.prompt_ids(q);cache=make_prompt_cache(r.model);anchors={};ia={};previous=[False]*28;stats=dict(records[0]['result']);stats.update(attention_calls=[0]*28,mlp_calls=[0]*28,reused_mlp_calls=[0]*28,work_shapes=[])
    first=r.compile_plan([[],[0],[]]);logits,_=r.forward(mx.array([ids]),cache,'ideal',first[0],anchors,ia,previous,0,stats,False,False);mx.eval(logits,anchors);r.trap_mlp=0
    try:
        logits,_=r.forward(mx.array([[42]]),cache,'ideal',first[1],anchors,ia,previous,1,stats,False,False);mx.eval(logits);raised=False
        try:r.forward(mx.array([[42]]),cache,'ideal',first[2],anchors,ia,previous,2,stats,False,False)
        except AssertionError:raised=True
        assert raised
    finally:r.trap_mlp=None
    write_json(p/'SMOKE.json',{'baseline':base,'zero_table':same,'records':records,'native_mlp_trap_bypass_passed':True,'positive_control_raised':True});print('ORACLE SMOKE '+str(p),flush=True)

def create():
    old=json.loads(OLD.read_text());short,long=fresh(ROOT/'data');ts=old['datasets']['replay32'];existing=[dict(t,token_cap=32) for t in ts if t['score_kind']=='choice'][:6]+[dict(t,token_cap=32) for t in ts if t['score_kind']=='json'][:2];datasets={'replay8':existing,'fresh8':short,'long4':long};assert sum(map(len,datasets.values()))==20
    schedule=[]
    for phase in ('natural','fixed'):
        for rep in range(1,6):
            pairs=[(ds,t) for ds,rows in datasets.items() for t in rows];random.Random(SEED+rep+(100 if phase=='fixed' else 0)).shuffle(pairs)
            for seq,(ds,t) in enumerate(pairs):
                cc=conditions();random.Random(SEED+rep*1000+seq).shuffle(cc)
                for pos,(method,rate) in enumerate(cc):schedule.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=t['task_id'],method=method,rate=rate,sequence=seq,position=pos))
    p=ROOT/'data'/('oracle_'+uuid4().hex);p.mkdir();m=dict(json.loads(PRESERVE.read_text()),datasets=datasets,schedule=schedule,seed=SEED,repeats=5,model=old['model'],core_sha256={str(ROOT/'src'/n):sha(ROOT/'src'/n) for n in CORE},legacy_sha256=old['legacy_source_sha256'],design_sha256=sha(ROOT/'ORACLE_DESIGN.md'),packages={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','transformers')},oracle_uses_evaluation_references=True,oracle_uses_future_baseline_activations=True,oracle_global_optimality_proven=False,new_exact_stems=True,semantic_template_independence=False)
    write_json(p/'MANIFEST.json',m);write_json(Path(str(p)+'_manifest.json'),{'datasets':datasets,'manifest':str(p/'MANIFEST.json')});return p

def build_task(r,p,task):
    folder=p/'tables'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'TABLES.json').exists():return json.loads((folder/'TABLES.json').read_text())
    start,cpu=time.perf_counter(),time.process_time();trace,activations=r.generate(task['question'],'baseline',max_tokens=task['token_cap'],capture=True,stop_eos=False)
    baseline=[]
    for token in trace['token_ids']:
        baseline.append(token)
        if token in r.old.eos:break
    trace_record=dict(trace,activation_shape=list(activations.shape),activation_sha256=hashlib.sha256(activations.tobytes()).hexdigest(),baseline_token_ids=baseline,baseline_answer=r.tok.decode(baseline,skip_special_tokens=True),future_after_eos_captured=True)
    write_json(folder/('TRACE_'+uuid4().hex+'.json'),trace_record);selected={};randoms={};rows=[]
    salt=int(task['task_id'][:10],16)
    for rate in RATES:
        randoms[str(rate)]=random_plan(task['token_cap'],rate,SEED+salt+rate+999)
        if rate==0:selected[str(rate)]=random_plan(task['token_cap'],0,SEED);continue
        low,risks=drift_plan(activations,rate);plans=[low]+[random_plan(task['token_cap'],rate,SEED+salt+rate+j*101) for j in (1,2)];candidates=[]
        for index,plan in enumerate(plans):
            validate(plan,rate);result=r.generate(task['question'],'ideal',r.compile_plan(plan),max_tokens=task['token_cap']);candidate=dict(index=index,grade=score(task,result['answer']),baseline_token_match=result['token_ids']==baseline,length_difference=abs(len(result['token_ids'])-len(baseline)),result=result,plan=plan);candidates.append(candidate)
            write_json(folder/(f'CANDIDATE_r{rate}_i{index}_'+uuid4().hex+'.json'),candidate);rows.append(dict(rate=rate,index=index,**{k:v for k,v in candidate.items() if k not in ('plan','index')}))
        winner=choose_candidate(candidates);selected[str(rate)]=plans[winner]
    table={'oracle':selected,'random':randoms,'baseline_token_ids':baseline,'builder_wall_seconds':time.perf_counter()-start,'builder_cpu_seconds':time.process_time()-cpu,'trace_model_calls':1,'candidate_model_calls':len(rows),'selected_rule':'ground-truth content, format, exact baseline tokens, length, index','uses_labels_and_future':True}
    write_union(folder/('CANDIDATES_'+uuid4().hex+'.csv'),rows);write_json(folder/'TABLES.json',table);return table

def run(folder=None):
    p=Path(folder).resolve() if folder else create();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for k in ('core_sha256','legacy_sha256'):
        for name,h in m[k].items():assert sha(name)==h
    assert sha(ROOT/'ORACLE_DESIGN.md')==m['design_sha256']
    assert m['packages']=={n:importlib.metadata.version(n) for n in m['packages']}
    lockpath=p/'RUN.lock';lockpath.touch(exist_ok=True);lock=lockpath.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);store=Store(p/'checkpoints');sid=uuid4().hex;start=time.perf_counter();new=0;print('ORACLE OUTPUT '+str(p),flush=True)
    if any(store.load(key(j)) is None for j in m['schedule']):
        begin=time.perf_counter();r=runtime();write_json(p/(sid+'_LOAD.json'),{'wall_seconds':time.perf_counter()-begin,'device':mx.device_info(),'dtype':str(r.model.model.embed_tokens.weight.dtype)})
        tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};tables={};routes={};buildstart=time.perf_counter()
        for count,(id,t) in enumerate(tasks.items(),1):
            table=build_task(r,p,t);tables[id]=table
            for method in ('ideal','random','overhead'):
                for rate in RATES:routes[id,method,rate]=r.compile_plan(table['random' if method=='random' else 'oracle'][str(rate)])
            print(f'ORACLE tables ready {count}/20; construction/resume elapsed {time.perf_counter()-buildstart:.1f}s',flush=True)
        write_json(p/(sid+'_BUILD.json'),{'elapsed_seconds':time.perf_counter()-buildstart,'sum_recorded_table_builder_seconds':sum(t['builder_wall_seconds'] for t in tables.values()),'privileged_labels_and_future':True,'table_sha256':{id:sha(p/'tables'/id/'TABLES.json') for id in tasks}})
        q='Return only a JSON object with keys x and y. Values: x=3, y=7.';warm=[]
        for method in ('baseline','existing','ideal','random','overhead'):
            warm.append(dict(method=method,result=r.generate(q,method,None if method in ('baseline','existing') else r.compile_plan(random_plan(32,50,SEED)),max_tokens=32)))
        write_json(p/(sid+'_WARMUP.json'),{'excluded_from_primary':True,'records':warm})
        mainstart=time.perf_counter()
        for j in m['schedule']:
            if store.load(key(j)) is not None:continue
            t=tasks[j['task_id']];fixed=None
            if j['phase']=='fixed':fixed=store.load(key(dict(j,phase='natural',repetition=1,method='baseline',rate=None)))['token_ids']
            result=r.generate(t['question'],j['method'],routes.get((j['task_id'],j['method'],j['rate'])),max_tokens=t['token_cap'],fixed_tokens=fixed)
            rec=dict(result,**j,question=t['question'],category=t['category'],score_kind=t['score_kind'],grade=score(t,result['answer']),session=sid,monotonic_end=time.perf_counter());store.save(key(j),rec);new+=1
            if new%100==0:print(f'ORACLE saved {new}; {j["phase"]} rep{j["repetition"]}; primary elapsed {time.perf_counter()-mainstart:.1f}s',flush=True)
    frozen_check(m);rows=store.rows();assert len(rows)==len(m['schedule'])==4000
    out=p/('EXPORT_'+uuid4().hex);out.mkdir();write_union(out/'ORACLE_RAW.csv',rows);write_json(p/(sid+'_COMPLETE.json'),{'new_requests':new,'real_primary_requests':len(rows),'elapsed_seconds':time.perf_counter()-start,'export':str(out)});print('ORACLE COMPLETE '+str(p),flush=True)

if __name__=='__main__':
    if sys.argv[1:2]==['--smoke']:smoke()
    else:run(sys.argv[1] if len(sys.argv)>1 else None)
