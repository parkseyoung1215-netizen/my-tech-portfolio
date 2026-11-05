"""Append-only local DLR pilot and randomized, resumable experiment."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import datetime,fcntl,hashlib,importlib.metadata,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from .dlr_runtime import DLRRuntime
from .dlr_policy import METHODS,SEED,THRESHOLDS,CANDIDATES
from .dlr_data import fresh,score,recorded_keys
from .stable_runtime_records import Store
from .adaptive_experiment import write_json
from .alsr_io import write_union
from .efficiency_repeat import frozen_check
ROOT=Path(__file__).resolve().parents[1]
SETUP=ROOT/'data/alsr_setup_1a6f378749694d5d9fd1f51937e952c0.json'
ACS=ROOT/'data/acs_16a7001279c947c0acb730672deb1a97/MANIFEST.json'
PACKAGES=('mlx','mlx-lm','numpy','transformers')
DEV=[f'Internal DLR development card {67600+i}: Return only a JSON object with keys name and count. Values: name="unit{i}", count={i+10}.' for i in range(4)]+[f'DLR development passage {67604+i}: Mira{i} has a gray cap and works at Pier {i+14}. Restate both facts in one sentence.' for i in range(4)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def runtime():
    setup=json.loads(SETUP.read_text());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));info=setup['hf_models'][1]
    for name,meta in info['files'].items():
        assert sha(Path(info['path'])/name)==meta['sha256'],'Pinned model file changed: '+name
    model,tok,config=load(info['path'],return_config=True);model.eval();return DLRRuntime(model,tok,config)

def native_check(r):
    q='Return only the letter A.';ids=r.prompt_ids(q);x=mx.array([ids]);cache=make_prompt_cache(r.model)
    hidden=r.model.model(x,cache=cache)[:,-1:,:]
    native=r.model.model.embed_tokens.as_linear(hidden) if r.config.get('tie_word_embeddings',True) else r.model.lm_head(hidden)
    native=native[:,-1,:];mx.eval(native)
    stats={'attention_calls':[0]*28,'mlp_calls':[0]*28,'reused_mlp_calls':[0]*28,'work_shapes':[],'input_tokens':len(ids)}
    logits=r.forward(x,make_prompt_cache(r.model),'baseline',{},0,stats)[0,-1];mx.eval(logits)
    diff=float(mx.max(mx.abs(native[0].astype(mx.float32)-logits.astype(mx.float32))).item());assert diff==0
    a=r.generate('Return only a JSON object with keys x and y. Values: x=2, y=3.','dlr',max_tokens=12,force_reuse=True)
    assert sum(a['reused_mlp_calls'])>0
    # Independently trap the actual MLP function on a cached step: a reuse must bypass it.
    state={};base={'attention_calls':[0]*28,'mlp_calls':[0]*28,'reused_mlp_calls':[0]*28,'work_shapes':[],'input_tokens':len(ids),'events':[],'gate_wall_seconds':0.,'gate_cpu_seconds':0.,'gate_elements':0,'pre_gate_sync_wall_seconds':0.}
    cc=make_prompt_cache(r.model);z=r.forward(x,cc,'dlr',state,0,base);mx.eval(z,state);r.trap_mlp=8
    try:
        z=r.forward(mx.array([[42]]),cc,'dlr',state,1,base,True);mx.eval(z,state)
        assert base['reused_mlp_calls'][8]==1
        caught=False
        try:r.forward(mx.array([[42]]),cc,'dlr',state,2,base,True)
        except AssertionError:caught=True
        assert caught
    finally:r.trap_mlp=None
    return {'native_logits_max_abs_difference':diff,'forced_reuse':a,'reuse_bypassed_trapped_mlp':True,'fresh_mlp_positive_control_raised':True,'scope':'neutral software controls, excluded from performance'}

def pilot():
    out=ROOT/'data'/('dlr_pilot_'+uuid4().hex);out.mkdir();write_json(out/'PROTOCOL.json',{'thresholds':THRESHOLDS,'candidates':CANDIDATES,'questions':DEV,'design_sha256':sha(ROOT/'DLR_DESIGN.md'),'utc':datetime.datetime.now(datetime.UTC).isoformat()});r=runtime();write_json(out/'NATIVE_AUDIT.json',native_check(r));rows=[]
    for i,q in enumerate(DEV):
        for method in METHODS:
            rec=dict(r.generate(q,method),method=method,question=q,index=i);write_json(out/(f'{i}_{method}.json'),rec);rows.append(rec)
    write_union(out/'PILOT.csv',rows);write_json(out/'COMPLETE.json',{'real_requests':len(rows),'natural_wall_seconds':sum(x['wall_seconds'] for x in rows),'rules_changed':False});print('DLR PILOT COMPLETE '+str(out),flush=True)

def create():
    old=json.loads(ACS.read_text())['datasets']['replay64'];datasets={'replay32':[t for t in old if t['score_kind']=='choice'][:24]+[t for t in old if t['score_kind']=='json'][:8],'fresh32':fresh(ROOT/'data')}
    assert all(len(x)==32 for x in datasets.values());past=recorded_keys(ROOT/'data')
    from .solver300_dataset import task_key
    assert not {task_key(q) for q in DEV}&{task_key(t['question']) for ts in datasets.values() for t in ts}
    fixed={ds:[t['task_id'] for t in ts if t['score_kind']=='json'][:4]+[t['task_id'] for t in ts if t['score_kind']=='choice'][:4] for ds,ts in datasets.items()}
    schedule=[]
    for phase in ('natural','fixed'):
        for rep in (1,2,3):
            jobs=[(ds,t) for ds,ts in datasets.items() for t in ts if phase=='natural' or t['task_id'] in fixed[ds]];random.Random(SEED+rep+(100 if phase=='fixed' else 0)).shuffle(jobs)
            for seq,(ds,t) in enumerate(jobs):
                methods=list(METHODS);random.Random(SEED+rep*1000+seq).shuffle(methods)
                for pos,method in enumerate(methods):schedule.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=t['task_id'],method=method,sequence=seq,position=pos))
    preserved={str(p.resolve()):sha(p) for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()};preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()})
    out=ROOT/'data'/('dlr_'+uuid4().hex);out.mkdir();m={'seed':SEED,'datasets':datasets,'fixed_ids':fixed,'schedule':schedule,'thresholds':THRESHOLDS,'candidates':CANDIDATES,'token_cap':32,'repeats':3,'model':json.loads(SETUP.read_text())['hf_models'][1],'preserved_sha256':preserved,'source_sha256':{str(ROOT/'src'/n):sha(ROOT/'src'/n) for n in ('dlr_runtime.py','dlr_experiment.py','dlr_policy.py','dlr_data.py')},'design_sha256':sha(ROOT/'DLR_DESIGN.md'),'packages':{n:importlib.metadata.version(n) for n in PACKAGES},'prior_stem_count':len(past),'unseen_exact_stems':True,'semantic_template_independence':False}
    write_json(out/'MANIFEST.json',m);write_json(Path(str(out)+'_manifest.json'),{'datasets':datasets,'study_manifest':str(out/'MANIFEST.json')});return out

def key(j):return '/'.join(str(j[k]) for k in ('phase','dataset','repetition','task_id','method'))
def run(folder=None):
    out=Path(folder).resolve() if folder else create();m=json.loads((out/'MANIFEST.json').read_text());frozen_check(m)
    for p,h in m['source_sha256'].items():assert sha(p)==h
    assert sha(ROOT/'DLR_DESIGN.md')==m['design_sha256'] and m['packages']=={n:importlib.metadata.version(n) for n in PACKAGES}
    lockpath=out/'RUN.lock';lockpath.touch(exist_ok=True);lock=lockpath.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=Store(out/'checkpoints');session=uuid4().hex;start=time.perf_counter();new=0;print('DLR OUTPUT '+str(out),flush=True)
    if any(store.load(key(j)) is None for j in m['schedule']):
        loadstart=time.perf_counter();r=runtime();write_json(out/(session+'_LOAD.json'),{'wall_seconds':time.perf_counter()-loadstart,'device':mx.device_info(),'dtype':str(r.model.model.embed_tokens.weight.dtype),'warmup':'native checks plus one neutral generation per method; caches reset per request'})
        write_json(out/(session+'_NATIVE_AUDIT.json'),native_check(r))
        warmups=[dict(method=method,result=r.generate('Return only a JSON object with keys x and y. Values: x=5, y=7.',method)) for method in METHODS]
        write_json(out/(session+'_WARMUPS.json'),{'excluded_from_primary':True,'real_requests':len(warmups),'records':warmups})
        tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts}
        for j in m['schedule']:
            if store.load(key(j)) is not None:continue
            task=tasks[j['task_id']];fixed=None
            if j['phase']=='fixed':
                source=store.load(key(dict(j,phase='natural',repetition=1,method='baseline')));assert source;fixed=source['token_ids']
            result=r.generate(task['question'],j['method'],fixed_tokens=fixed)
            rec=dict(result,**j,session=session,question=task['question'],category=task['category'],score_kind=task['score_kind'],grade=score(task,result['answer']),monotonic_end=time.perf_counter())
            store.save(key(j),rec);new+=1
            if new%64==0:print(f'DLR saved {new} requests; {j["phase"]} repetition {j["repetition"]}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    frozen_check(m);rows=store.rows();assert len(rows)==len(m['schedule'])==960
    exp=out/('EXPORT_'+uuid4().hex);exp.mkdir();write_union(exp/'DLR_RAW.csv',rows);write_json(out/('COMPLETE_'+session+'.json'),{'real_requests':len(rows),'new_requests':new,'elapsed_seconds':time.perf_counter()-start,'export':str(exp),'model_load_and_warmup_in_primary_per_request_metrics':False});print('DLR COMPLETE '+str(out),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--pilot':pilot()
    else:run(sys.argv[1] if len(sys.argv)>1 else None)
