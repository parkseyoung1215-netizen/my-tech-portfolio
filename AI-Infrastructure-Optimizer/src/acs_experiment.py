"""Resumable ACS runs; thresholds frozen before eval, existing data never overwritten."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import datetime,fcntl,importlib.metadata,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from mlx_lm import load
from .acs_runtime import ACSRuntime
from .acs_policy import METHODS
from .acs_data import build,SEED
from .acs_scoring import score
from .stable_runtime_records import Store
from .stable_runtime_experiment import ROOT,SETUP,SKETCH,SOURCE,sha,rss,unload_ollama
from .adaptive_experiment import write_json
from .alsr_io import write_union
from .efficiency_repeat import frozen_check
PACKAGES=('mlx','mlx-lm','numpy','transformers')
PREVIOUS=ROOT/'data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628/MANIFEST.json'

def jobkey(j):return '/'.join(str(j[k]) for k in ('dataset','repetition','task_id','method'))
def create():
    old=json.loads(PREVIOUS.read_text())['datasets']['replay64'];datasets={'replay64':[dict(t,difficulty='unrated') for t in old],'fresh64':build(ROOT/'data')};schedule=[]
    for rep in (1,2,3):
        order=[(ds,t) for ds,ts in datasets.items() for t in ts];random.Random(SEED+rep).shuffle(order)
        for seq,(ds,t) in enumerate(order):
            methods=list(METHODS);random.Random(SEED+rep*1000+seq).shuffle(methods)
            for pos,method in enumerate(methods):schedule.append(dict(dataset=ds,repetition=rep,task_id=t['task_id'],method=method,sequence=seq,position=pos))
    preserved={str(p.resolve()):sha(p) for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()};preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()})
    out=ROOT/'data'/('acs_'+uuid4().hex);out.mkdir();m={'seed':SEED,'datasets':datasets,'schedule':schedule,'methods':METHODS,'repeats':3,'model':json.loads(SETUP.read_text())['hf_models'][1],'preserved_sha256':preserved,'runtime_source_sha256':{str(ROOT/'src'/name):sha(ROOT/'src'/name) for name in ('acs_runtime.py','acs_policy.py','acs_data.py','acs_scoring.py','acs_experiment.py')},'packages':{n:importlib.metadata.version(n) for n in PACKAGES},'design_sha256':sha(ROOT/'ACS_DESIGN.md'),'setup_sha256':sha(SETUP),'sketch_sha256':sha(SKETCH),'local_profile_sha256':sha(SOURCE/'PROFILE.json'),'output_condition':'Same prompts, greedy; boundary is stopping only, no logit masking or rewriting; no eval tuning'}
    write_json(out/'MANIFEST.json',m);write_json(Path(str(out)+'_manifest.json'),{'datasets':datasets,'manifest':str(out/'MANIFEST.json')});return out

def runtime():
    setup=json.loads(SETUP.read_text());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));model,tok,config=load(setup['hf_models'][1]['path'],return_config=True);model.eval();return ACSRuntime(model,tok,config,json.loads(SKETCH.read_text()),json.loads((SOURCE/'PROFILE.json').read_text()))

def smoke():
    r=runtime();records=[]
    for q in ('Return only the letter A.','What is 2 + 2?\nA) 3\nB) 4\nC) 5\nD) 6\nReturn only the option letter.'):
        old=r.generate(q,'full','lean',max_tokens=16,audit=True);new=r.generate_acs(q,'baseline',audit=True);assert old['token_ids']==new['token_ids'];records.append(dict(question=q,legacy=old,acs=new))
    for method in METHODS:records.append(dict(method=method,result=r.generate_acs('Return only a JSON object with keys x and y. Values: x=2, y=3.',method,audit=True)))
    out=ROOT/'data'/('acs_smoke_'+uuid4().hex+'.json');write_json(out,{'real_model':True,'records':records,'scope':'Neutral software validation only'});print('ACS SMOKE PASS '+str(out),flush=True)

def run(folder=None):
    out=Path(folder).resolve() if folder else create();m=json.loads((out/'MANIFEST.json').read_text());frozen_check(m)
    for p,h in m['runtime_source_sha256'].items():assert sha(p)==h
    assert {n:importlib.metadata.version(n) for n in PACKAGES}==m['packages'];assert sha(SETUP)==m['setup_sha256'] and sha(SKETCH)==m['sketch_sha256'] and sha(SOURCE/'PROFILE.json')==m['local_profile_sha256']
    lockpath=out/'RUN.lock'
    if not lockpath.exists():lockpath.touch(exist_ok=False)
    lock=lockpath.open('rb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);store=Store(out/'checkpoints');sid=uuid4().hex;sessions=out/'sessions';sessions.mkdir(exist_ok=True);start=time.perf_counter();write_json(sessions/(sid+'.start.json'),{'utc':datetime.datetime.now(datetime.UTC).isoformat(),'resume':folder is not None});print('ACS OUTPUT '+str(out),flush=True);tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};new_count=0
    if any(store.load(jobkey(j)) is None for j in m['schedule']):
        write_json(sessions/(sid+'.unload.json'),{'events':unload_ollama()});begin,cpu=time.perf_counter(),time.process_time();r=runtime();write_json(sessions/(sid+'.load.json'),{'wall_seconds':time.perf_counter()-begin,'cpu_seconds':time.process_time()-cpu,'device':mx.device_info(),'dtype':str(r.model.model.embed_tokens.weight.dtype),'rss_bytes':rss()})
        audits=[]
        for method in METHODS:audits.append(dict(method=method,result=r.generate_acs('Return only a JSON object with keys x and y. Values: x=2, y=3.',method,audit=True)))
        write_json(sessions/(sid+'.native_audits.json'),{'real_model':True,'scope':'Neutral prompt; actual attention/MLP counts checked; excluded from primary performance','records':audits})
        for j in m['schedule']:
            if store.load(jobkey(j)) is not None:continue
            task=tasks[j['task_id']];result=r.generate_acs(task['question'],j['method']);record=dict(result,**j,session=sid,question=task['question'],reference=task['reference'],category=task['category'],difficulty=task['difficulty'],score_kind=task['score_kind'],reference_tokens=len(r.tok.encode(task['reference'],add_special_tokens=False)),score=score(task,result['answer']));store.save(jobkey(j),record);new_count+=1
            if new_count%96==0:print(f'ACS saved {new_count} new requests; repetition{j["repetition"]} sequence{j["sequence"]}',flush=True)
    frozen_check(m);rows=store.rows();assert len(rows)==128*len(METHODS)*3;export=out/('EXPORT_'+uuid4().hex);export.mkdir();write_union(export/'ACS_RAW.csv',rows);write_json(sessions/(sid+'.end.json'),{'elapsed_seconds':time.perf_counter()-start,'new_requests':new_count,'complete_requests':len(rows),'export':str(export)});write_json(out/('COMPLETE_'+sid+'.json'),{'real_inference':True,'complete_requests':len(rows),'new_requests':new_count,'export':str(export),'GPU_elapsed_power_money':'unmeasured'});print('ACS COMPLETE '+str(out),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--smoke':smoke()
    else:run(sys.argv[1] if len(sys.argv)>1 else None)
