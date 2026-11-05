"""Resumable real MLX experiment. Only new files; no paid API or policy fitting."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import datetime,fcntl,importlib.metadata,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from mlx_lm import load
from .stable_runtime_data import build,SEED
from .stable_runtime_protocol import VARIANTS
from .stable_runtime_records import Store
from .stable_runtime_engine import Runtime
from .alsr_impact_scoring import grade
from .alsr_experiment import sha,rss,unload_ollama
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .efficiency_repeat import frozen_check
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/alsr_impact_e8eea7cd19db4d8796ea54cfe6f57b2c/CANONICAL_aca75ae9b3014233ba48c949c0186bb4'
SETUP=ROOT/'data/alsr_setup_1a6f378749694d5d9fd1f51937e952c0.json'
SKETCH=ROOT/'data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_m1_profile.json'
PACKAGES=('mlx','mlx-lm','numpy','transformers')

def key(job):return '/'.join(str(job[k]) for k in ('phase','dataset','repetition','task_id','variant'))
def create():
    old=json.loads((SOURCE/'MANIFEST.json').read_text());datasets={'replay64':old['splits']['evaluation'],'fresh64':build(ROOT/'data')};schedule=[];fixedids={};profileids={}
    for name,tasks in datasets.items():
        choose=lambda kind,n:[t['task_id'] for t in tasks if t['score_kind']==kind][:n]
        fixedids[name]=choose('choice',12)+choose('json',2)+choose('constraints',2);profileids[name]=choose('choice',2)+choose('json',1)+choose('constraints',1)
    for rep in range(1,4):
        questions=[(name,t) for name,ts in datasets.items() for t in ts];random.Random(SEED+rep).shuffle(questions)
        for seq,(name,t) in enumerate(questions):
            variants=list(VARIANTS);random.Random(SEED+rep*1000+seq).shuffle(variants)
            for position,variant in enumerate(variants):
                j=dict(dataset=name,repetition=rep,task_id=t['task_id'],variant=variant,sequence=seq,position=position,phase='natural');schedule.append(j)
                if t['task_id'] in fixedids[name] and not VARIANTS[variant]['format_stop']:schedule.append(dict(j,phase='fixed'))
    for rep in range(1,4):
        jobs=[dict(phase='profile',dataset=name,repetition=rep,task_id=id,variant=v,sequence=-1,position=-1) for name,ids in profileids.items() for id in ids for v,c in VARIANTS.items() if not c['format_stop']];random.Random(SEED+9000+rep).shuffle(jobs);schedule.extend(jobs)
    preserved={str(p.resolve()):sha(p) for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()};preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()})
    out=ROOT/'data'/('stable_runtime_'+uuid4().hex);out.mkdir();m={'seed':SEED,'datasets':datasets,'fixed_ids':fixedids,'profile_ids':profileids,'schedule':schedule,'source':str(SOURCE),'source_manifest_sha256':sha(SOURCE/'MANIFEST.json'),'setup_sha256':sha(SETUP),'sketch_sha256':sha(SKETCH),'local_profile_sha256':sha(SOURCE/'PROFILE.json'),'packages':{n:importlib.metadata.version(n) for n in PACKAGES},'preserved_sha256':preserved,'runtime_source_sha256':{str(ROOT/'src'/n):sha(ROOT/'src'/n) for n in ('stable_runtime_engine.py','stable_runtime_experiment.py','stable_runtime_protocol.py','stable_runtime_records.py','stable_runtime_data.py')},'design_sha256':sha(ROOT/'STABLE_RUNTIME_DESIGN.md'),'model':json.loads(SETUP.read_text())['hf_models'][1],'variants':VARIANTS,'repeats':3,'freshness':'fresh64 exact stem independent; synthetic semantic similarity possible; no training/tuning in this study'}
    write_json(out/'MANIFEST.json',m);write_json(Path(str(out)+'_manifest.json'),{'datasets':datasets,'study_manifest':str(out/'MANIFEST.json')});return out

def run(folder=None):
    out=Path(folder).resolve() if folder else create();m=json.loads((out/'MANIFEST.json').read_text());frozen_check(m)
    for p,h in m['runtime_source_sha256'].items():assert sha(p)==h,'Runtime changed; create a new experiment, do not mix resumed results'
    assert {n:importlib.metadata.version(n) for n in PACKAGES}==m['packages'];assert sha(SETUP)==m['setup_sha256'] and sha(SKETCH)==m['sketch_sha256'] and sha(SOURCE/'PROFILE.json')==m['local_profile_sha256']
    lockpath=out/'RUN.lock'
    if not lockpath.exists():lockpath.touch(exist_ok=False)
    lock=lockpath.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);store=Store(out/'checkpoints');session=uuid4().hex;sessions=out/'sessions';sessions.mkdir(exist_ok=True);start=time.perf_counter();write_json(sessions/(session+'.start.json'),{'utc':datetime.datetime.now(datetime.UTC).isoformat(),'resume':folder is not None,'manifest_sha256':sha(out/'MANIFEST.json')});print('STABLE RUNTIME OUTPUT '+str(out),flush=True)
    tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};prepjobs=[dict(phase='prepare',dataset=name,repetition=0,task_id=id,variant='full') for name,ids in m['fixed_ids'].items() for id in ids];missing=[j for j in prepjobs+m['schedule'] if store.load(key(j)) is None]
    if missing or store.load('smoke') is None:
        setup=json.loads(SETUP.read_text());write_json(sessions/(session+'.unload.json'),{'events':unload_ollama()});mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));mx.clear_cache();st,sc=time.perf_counter(),time.process_time();model,tok,config=load(m['model']['path'],return_config=True);model.eval();runtime=Runtime(model,tok,config,json.loads(SKETCH.read_text()),json.loads((SOURCE/'PROFILE.json').read_text()))
        write_json(sessions/(session+'.load.json'),{'wall_seconds':time.perf_counter()-st,'cpu_seconds':time.process_time()-sc,'mlx_active_bytes':mx.get_active_memory(),'device':mx.device_info(),'rss_bytes':rss(),'dtype':str(model.model.embed_tokens.weight.dtype)})
        if store.load('smoke') is None:
            checks={**runtime.engine.check_native(),**runtime.engine.check_skip_trap()};parity=[]
            training=json.loads((ROOT/'data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_manifest.json').read_text())['splits']['training'][:2]
            for q in ['Return only the letter A.']+[t['question'] for t in training]:
                for method in ('full','stable1','sketch'):
                    a=runtime.generate(q,method,'legacy',max_tokens=8,audit=True);b=runtime.generate(q,method,'lean',max_tokens=8,audit=True)
                    assert a['token_ids']==b['token_ids'] and a['skip_layers']==b['skip_layers'] and a['first_logits_sha256']==b['first_logits_sha256'],'Lean math/selection parity failed before main evaluation'
                    parity.append(dict(method=method,question=q,legacy=a,lean=b))
            positive=False;runtime.trap_index=24
            try:
                forced=runtime.generate('Return only A.','stable1','lean',max_tokens=4,audit=True)
                try:runtime.generate('Return only A.','full','lean',max_tokens=4,audit=True)
                except AssertionError:positive=True
                assert positive and forced['layer_calls'][24]==0
            finally:runtime.trap_index=None
            store.save('smoke',dict(checks=checks,parity=parity,native_skip_trap_positive_control=positive,forced_static=forced,session=session))
            print('STABLE parity/logit/native attention+MLP trap passed',flush=True)
        warm=[]
        for v,c in VARIANTS.items():warm.append(dict(runtime.generate('Return only the option letter.',c['method'],c['backend'],max_tokens=4,format_stop=c['format_stop']),variant=v))
        write_union(sessions/(session+'.warmups.csv'),warm)
        for j in prepjobs:
            if store.load(key(j)) is not None:continue
            r=runtime.generate(tasks[j['task_id']]['question'],'full','legacy',max_tokens=16);eos=min(runtime.eos);store.save(key(j),dict(r,**j,session=session,fixed_sequence=(r['token_ids']+[eos]*16)[:16],scope='Benchmark preparation only, excluded from service costs'))
        done=0
        for j in m['schedule']:
            if store.load(key(j)) is not None:continue
            c=VARIANTS[j['variant']];task=tasks[j['task_id']];fixed=None
            if j['phase'] in ('fixed','profile'):
                prepkey='/'.join(map(str,('prepare',j['dataset'],0,j['task_id'],'full')));p=store.load(prepkey)
                assert p is not None;fixed=p['fixed_sequence']
            r=runtime.generate(task['question'],c['method'],c['backend'],max_tokens=16 if fixed is not None or task['score_kind']=='choice' else 48,fixed_tokens=fixed,format_stop=c['format_stop'],audit=j['phase']=='profile',detailed=j['phase']=='profile')
            store.save(key(j),dict(r,**j,session=session,question=task['question'],reference=task['reference'],category=task['category'],score_kind=task['score_kind'],grade=grade(task,r['answer']) if j['phase']=='natural' else None));done+=1
            if done%96==0:print(f'STABLE saved {done} new requests; {j["phase"]} repetition{j["repetition"]} sequence{j["sequence"]}',flush=True)
    frozen_check(m)
    for p,h in m['runtime_source_sha256'].items():assert sha(p)==h
    for j in prepjobs+m['schedule']:assert store.load(key(j)) is not None
    rows=store.rows();export=out/('EXPORT_'+uuid4().hex);export.mkdir()
    for phase in ('natural','fixed','profile','prepare'):write_union(export/(phase.upper()+'.csv'),[r for r in rows if r.get('phase')==phase])
    write_json(sessions/(session+'.end.json'),{'elapsed_seconds':time.perf_counter()-start,'utc':datetime.datetime.now(datetime.UTC).isoformat(),'new_complete_records':len(rows),'real_inference':True,'export':str(export)});write_json(out/('COMPLETE_'+session+'.json'),{'session':session,'real_inference':True,'export':str(export),'completed_rows':len(rows),'GPU_elapsed_power_money':'unmeasured','stages_not_continuous_wall_when_resumed':True});print('STABLE COMPLETE '+str(out),flush=True)

def smoke():
    setup=json.loads(SETUP.read_text());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));model,tok,config=load(setup['hf_models'][1]['path'],return_config=True);model.eval();runtime=Runtime(model,tok,config,json.loads(SKETCH.read_text()),json.loads((SOURCE/'PROFILE.json').read_text()));records=[]
    for method in ('full','stable1','sketch'):
        a=runtime.generate('Return only the letter A.',method,'legacy',max_tokens=8,audit=True);b=runtime.generate('Return only the letter A.',method,'lean',max_tokens=8,audit=True)
        records.append(dict(method=method,legacy=a,lean=b));assert a['token_ids']==b['token_ids'] and a['skip_layers']==b['skip_layers'] and a['first_logits_sha256']==b['first_logits_sha256']
    out=ROOT/'data'/('stable_runtime_smoke_'+uuid4().hex+'.json');write_json(out,{'actual_models':True,'records':records,'scope':'Neutral software parity only; not performance evaluation.'});print('STABLE NEUTRAL SMOKE PASS '+str(out),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--smoke':smoke()
    else:run(sys.argv[1] if len(sys.argv)>1 else None)
