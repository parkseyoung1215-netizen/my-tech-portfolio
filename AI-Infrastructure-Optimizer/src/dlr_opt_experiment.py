"""Real local DLR optimization pilot and append-only three-repeat comparison."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import datetime,fcntl,hashlib,importlib.metadata,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from .dlr_experiment import runtime as load_old,sha,ROOT
from .dlr_opt_runtime import OptimizationRuntime,METHODS
from .dlr_opt_data import build,score,SEED
from .stable_runtime_records import Store
from .efficiency_repeat import frozen_check
from .adaptive_experiment import write_json
from .alsr_io import write_union
PRESERVED=ROOT/'data/dlr_opt_preservation_b276114c0aa942c08c9d5d6e77517c8c.json'
OLD=ROOT/'data/dlr_5a4656b8d37f48dd8a55a0e82b647b37/MANIFEST.json'
SOURCES=('dlr_opt_runtime.py','dlr_opt_experiment.py','dlr_opt_kernels.py','dlr_opt_data.py')
DEV=['Return only a JSON object with keys x and y. Values: x=6, y=8.','Repeat the word hello exactly eight times, separated by single spaces.']
def runtime():
    r=load_old();return OptimizationRuntime(r.model,r.tok,r.config)

def audits(r):
    l=r.model.layers[8];d=r.config['hidden_size'];x=mx.random.normal((1,1,d)).astype(mx.bfloat16);cached=mx.full_like(x,3)
    native=l.mlp(x);fresh,count=r.kernel(l.mlp,x,cached,mx.array(False));reuse,skip=r.kernel(l.mlp,x,cached,mx.array(True));mx.eval(native,fresh,count,reuse,skip)
    diff=float(mx.max(mx.abs(native.astype(mx.float32)-fresh.astype(mx.float32))).item());rel=float(mx.sqrt(mx.sum((native.astype(mx.float32)-fresh.astype(mx.float32))**2)/(mx.sum(native.astype(mx.float32)**2)+1e-8)).item())
    assert int(count.item())==3 and int(skip.item())==0 and bool(mx.all(reuse==cached).item())
    # Native-MLP poison is not called by device decode; GPU branch counter is authoritative.
    controls=[]
    for q in DEV:
        base=r.generate(q,'baseline',max_tokens=16);fresh=r.generate(q,'device',max_tokens=16,force_fresh=True);forced=r.generate(q,'device',max_tokens=16,force_reuse=True)
        assert sum(forced['reused_mlp_calls'])>0 and fresh['gate_host_scalar_reads']==0
        controls.append(dict(question=q,baseline=base,device_fresh=fresh,forced_reuse=forced,fresh_tokens_match=base['token_ids']==fresh['token_ids']))
    return dict(native_mlp_max_abs_difference=diff,native_mlp_relative_l2=rel,conditional_gpu_reuse_exact_cached_output=True,gpu_mac_counter_fresh=3,gpu_mac_counter_reuse=0,has_mx_cond=hasattr(mx,'cond'),custom_kernels=True,controls=controls,excluded_from_primary=True)

def pilot():
    out=ROOT/'data'/('dlr_opt_pilot_'+uuid4().hex);out.mkdir();write_json(out/'PROTOCOL.json',{'development_prompts':DEV,'seed':SEED,'source_sha256':{n:sha(ROOT/'src'/n) for n in SOURCES},'design_sha256':sha(ROOT/'DLR_OPTIMIZATION_DESIGN.md')});mx.random.seed(SEED);r=runtime();write_json(out/'AUDITS.json',audits(r));rows=[]
    for i,q in enumerate(DEV):
        for method in METHODS:
            row=dict(r.generate(q,method,max_tokens=16),method=method,index=i,question=q);write_json(out/f'{i}_{method}.json',row);rows.append(row)
    write_union(out/'PILOT.csv',rows);write_json(out/'COMPLETE.json',{'requests':8,'device_under_20s':all(r['wall_seconds']<20 for r in rows if r['method']=='device')});print('DLR OPT PILOT '+str(out),flush=True)

def create(pilot_path):
    pilot=json.loads((Path(pilot_path)/'COMPLETE.json').read_text());assert pilot['device_under_20s'],'Device pilot too slow; report subset feasibility before full run'
    prior=json.loads(OLD.read_text());fresh,long=build(ROOT/'data');datasets={'replay32':[dict(t,token_cap=32) for t in prior['datasets']['replay32']],'fresh32':fresh,**long};schedule=[]
    for phase in ('natural','fixed'):
        for rep in (1,2,3):
            pairs=[(ds,t) for ds,ts in datasets.items() for t in ts];random.Random(SEED+rep+(100 if phase=='fixed' else 0)).shuffle(pairs)
            for seq,(ds,t) in enumerate(pairs):
                methods=list(METHODS);random.Random(SEED+seq+1000*rep).shuffle(methods)
                for pos,method in enumerate(methods):schedule.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=t['task_id'],method=method,sequence=seq,position=pos))
    out=ROOT/'data'/('dlr_opt_'+uuid4().hex);out.mkdir();m=dict(json.loads(PRESERVED.read_text()),seed=SEED,datasets=datasets,schedule=schedule,pilot_path=str(Path(pilot_path).resolve()),model=prior['model'],design_sha256=sha(ROOT/'DLR_OPTIMIZATION_DESIGN.md'),source_sha256={str(ROOT/'src'/n):sha(ROOT/'src'/n) for n in SOURCES},legacy_source_sha256=prior['source_sha256'],packages={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','transformers')},repeats=3,methods=METHODS,freshness='No exact recorded stems; synthetic template overlap remains; no evaluation tuning')
    write_json(out/'MANIFEST.json',m);write_json(Path(str(out)+'_manifest.json'),{'datasets':datasets,'manifest':str(out/'MANIFEST.json')});return out

def key(j):return '/'.join(str(j[k]) for k in ('phase','dataset','repetition','task_id','method'))
def run(folder=None,pilot=None):
    out=Path(folder).resolve() if folder else create(pilot);m=json.loads((out/'MANIFEST.json').read_text());frozen_check(m)
    for field in ('source_sha256','legacy_source_sha256'):
        for p,h in m[field].items():assert sha(p)==h
    assert sha(ROOT/'DLR_OPTIMIZATION_DESIGN.md')==m['design_sha256']
    lockpath=out/'RUN.lock';lockpath.touch(exist_ok=True);lock=lockpath.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);store=Store(out/'checkpoints');sid=uuid4().hex;start=time.perf_counter();new=0;print('DLR OPT OUTPUT '+str(out),flush=True)
    if any(store.load(key(j)) is None for j in m['schedule']):
        mx.random.seed(SEED);st=time.perf_counter();r=runtime();write_json(out/(sid+'_LOAD.json'),{'wall_seconds':time.perf_counter()-st,'device':mx.device_info(),'packages':m['packages']});write_json(out/(sid+'_AUDITS.json'),audits(r))
        warm=[dict(method=method,result=r.generate(DEV[0],method,max_tokens=16)) for method in METHODS];write_json(out/(sid+'_WARMUP.json'),{'excluded_from_primary':True,'records':warm})
        tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts}
        for j in m['schedule']:
            if store.load(key(j)) is not None:continue
            task=tasks[j['task_id']];fixed=None
            if j['phase']=='fixed':fixed=store.load(key(dict(j,phase='natural',repetition=1,method='baseline')))['token_ids']
            result=r.generate(task['question'],j['method'],max_tokens=task['token_cap'],fixed_tokens=fixed);rec=dict(result,**j,question=task['question'],category=task['category'],score_kind=task['score_kind'],grade=score(task,result['answer']),session=sid,monotonic_end=time.perf_counter());store.save(key(j),rec);new+=1
            if new%72==0:print(f'DLR OPT saved {new}; {j["phase"]} rep{j["repetition"]}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    frozen_check(m);rows=store.rows();assert len(rows)==len(m['schedule'])==1728
    export=out/('EXPORT_'+uuid4().hex);export.mkdir();write_union(export/'DLR_OPTIMIZATION_RAW.csv',rows);write_json(out/(sid+'_COMPLETE.json'),{'real_primary_requests':len(rows),'new_requests':new,'elapsed_seconds':time.perf_counter()-start,'export':str(export)});print('DLR OPT COMPLETE '+str(out),flush=True)

if __name__=='__main__':
    if sys.argv[1:2]==['--pilot']:pilot()
    elif sys.argv[1:2]==['--new']:run(pilot=sys.argv[2])
    else:run(sys.argv[1])
