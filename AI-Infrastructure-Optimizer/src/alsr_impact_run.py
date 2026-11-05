"""Real local layer-impact experiment; immutable artifacts and phase-level checkpoints."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import gc,hashlib,json,random,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
import numpy as np
from mlx_lm import load
from .alsr_engine import kl_effect
from .alsr_experiment import sha,rss,unload_ollama
from .alsr_dataset import grade
from .alsr_policy import fit_ridge,predict_ridge
from .alsr_impact_engine import ImpactEngine,ablate_task
from .alsr_impact_policy import METHODS,select_threshold
from .alsr_impact_data import build,SEED
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .efficiency_repeat import frozen_check
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_m1_profile.json'
SETUP=ROOT/'data/alsr_setup_1a6f378749694d5d9fd1f51937e952c0.json'

class Journal:
    def __init__(self,path):self.file=path.open('x');self.rows=[]
    def add(self,row):
        self.rows.append(row);self.file.write(json.dumps(row,ensure_ascii=False)+'\n');self.file.flush()
    def close(self,path):self.file.close();write_union(path,self.rows)

def train_profile(rows,shortlist):
    train=[r for r in rows if r['split']=='training'];ridge={};means={}
    for l in range(28):
        rr=[r for r in train if r['layer']==l];ridge[str(l)]=fit_ridge([r['features'] for r in rr],[r['kl']+r['top1_changed'] for r in rr]);rr0=[r for r in rr if r['position']==0];means[str(l)]=float(np.mean([r['kl']+r['top1_changed'] for r in rr0]))
    candidate=min(range(2,27),key=lambda l:(means[str(l)],l));val=[dict(r,prediction=predict_ridge(r['features'],ridge[str(candidate)])) for r in rows if r['split']=='calibration' and r['position']==0 and r['layer']==candidate]
    threshold,trials=select_threshold(val)
    return {'candidate':candidate,'ridge':ridge,'train_mean_risk':means,'threshold':threshold,'trials':trials,'shortlist':shortlist,'train_ids':sorted({r['task_id'] for r in train}),'calibration_ids':sorted({r['task_id'] for r in val}),'feature_scope':'current block INPUT only; preceding block change; no current/future block output, full logits, references'}

def run():
    jobstart=time.perf_counter();splits=build(ROOT/'data');setup=json.loads(SETUP.read_text());info=setup['hf_models'][1]
    preserved={str(p.resolve()):sha(p) for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()}
    preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()})
    out=ROOT/'data'/('alsr_impact_'+uuid4().hex);out.mkdir();tasks={t['task_id']:t for t in splits['evaluation']}
    fixed_ids=[t['task_id'] for t in splits['evaluation'] if t['score_kind']=='choice'][:12]+[t['task_id'] for t in splits['evaluation'] if t['score_kind']=='json'][:2]+[t['task_id'] for t in splits['evaluation'] if t['score_kind']=='constraints'][:2]
    schedule=[]
    for rep in range(1,4):
        ids=list(tasks);random.Random(SEED+rep).shuffle(ids)
        for seq,id in enumerate(ids):
            methods=list(METHODS);random.Random(SEED+rep*1000+seq).shuffle(methods)
            for pos,method in enumerate(methods):schedule.append(dict(repetition=rep,sequence=seq,position=pos,method=method,task_id=id))
    manifest={'seed':SEED,'splits':splits,'schedule':schedule,'fixed_ids':fixed_ids,'model':info,'setup_sha256':sha(SETUP),'old_profile_sha256':sha(OLD),'preserved_sha256':preserved,'design_sha256':sha(ROOT/'ALSR_IMPACT_DESIGN.md'),'repeats':3,'training_count':32,'calibration_count':16,'evaluation_count':64}
    write_json(out/'MANIFEST.json',manifest);write_json(Path(str(out)+'_manifest.json'),manifest);print('IMPACT OUTPUT '+str(out),flush=True)
    write_json(out/'OLLAMA_UNLOAD.json',unload_ollama());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));gc.collect();mx.clear_cache()
    start,cpu=time.perf_counter(),time.process_time();model,tok,config=load(info['path'],return_config=True);model.eval();engine=ImpactEngine(model,tok,config);engine.prepare_profile(json.loads(OLD.read_text()))
    write_json(out/'LOAD.json',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'dtype':str(model.model.embed_tokens.weight.dtype),'mlx_active_bytes':mx.get_active_memory(),'device':mx.device_info(),'rss_bytes':rss()})
    write_json(out/'SMOKE.json',{**engine.check_native(),**engine.check_skip_trap()});print('IMPACT native parity/trap passed',flush=True)
    # Small shortlist is selected from training-only full logits, never from evaluation.
    shortlist=[];prep=Journal(out/'PREPARATION.jsonl');phase=time.perf_counter()
    for task in splits['training']:
        engine.reset();start,cpu=time.perf_counter(),time.process_time();logits,_=engine.forward(mx.array([engine.prompt_ids(task['question'])]));arr=np.array(logits[0,-1].astype(mx.float32));ids=np.argpartition(arr,-2)[-2:];shortlist.extend(map(int,ids));prep.add(dict(task_id=task['task_id'],purpose='training shortlist only',wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
    shortlist=list(dict.fromkeys(shortlist))[:32]
    for c in 'ABCD':
        token=tok.encode(c,add_special_tokens=False)[0]
        if token not in shortlist:shortlist.append(token)
    engine.prepare_local({'shortlist':shortlist});write_json(out/'SHORTLIST.json',{'ids':shortlist,'source':'training only'})
    ablations=Journal(out/'LAYER_ABLATIONS.jsonl');contexts=Journal(out/'DIAGNOSTIC_CONTEXTS.jsonl');costs=[]
    for split in ('training','calibration'):
        start,cpu=time.perf_counter(),time.process_time()
        for j,t in enumerate(splits[split]):
            rr,cc=ablate_task(engine,t,split,SEED+j);[ablations.add(r) for r in rr];[contexts.add(r) for r in cc]
            if (j+1)%4==0:print(f'IMPACT {split} ablation {j+1}/{len(splits[split])}',flush=True)
        costs.append(dict(phase=split+'_diagnostic',wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,scope='Offline causal measurement including feature collection and effect calculations'))
    profile=train_profile(ablations.rows,shortlist);write_json(out/'PROFILE.json',profile);profilehash=sha(out/'PROFILE.json');engine.prepare_local(profile)
    print('IMPACT frozen candidate/threshold '+str(profile['candidate'])+'/'+str(profile['threshold']),flush=True)
    # Freeze predictors before final holdout outputs are observed.
    start,cpu=time.perf_counter(),time.process_time()
    for j,t in enumerate(splits['evaluation']):
        rr,cc=ablate_task(engine,t,'evaluation',SEED+1000+j);[ablations.add(r) for r in rr];[contexts.add(r) for r in cc]
        if (j+1)%4==0:print(f'IMPACT evaluation ablation {j+1}/64',flush=True)
    costs.append(dict(phase='evaluation_diagnostic',wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,scope='Offline causal evaluation, not live inference or threshold fitting'))
    ablations.close(out/'LAYER_ABLATIONS.csv');contexts.close(out/'DIAGNOSTIC_CONTEXTS.csv')
    fixed_sequences={};eos=list(getattr(tok,'eos_token_ids',[]) or [tok.eos_token_id])[0]
    for id in fixed_ids:
        r=engine.generate(tasks[id]['question'],'baseline',SEED,max_tokens=16);fixed_sequences[id]=(r['token_ids']+[eos]*16)[:16];prep.add(dict(r,task_id=id,purpose='fixed-work baseline token sequence preparation only'))
    prep.close(out/'PREPARATION.csv');write_json(out/'FIXED_SEQUENCES.json',fixed_sequences)
    warm=[]
    for method in METHODS:warm.append(dict(engine.generate('Return only A.',method,SEED,max_tokens=4),method=method))
    write_union(out/'WARMUPS.csv',warm)
    gen=Journal(out/'GENERATION.jsonl');fixed=Journal(out/'FIXED_WORK.jsonl');quality=Journal(out/'QUALITY.jsonl')
    for s in schedule:
        t=tasks[s['task_id']];method=s['method'];before=rss();r=engine.generate(t['question'],method,SEED+s['repetition'],max_tokens=16 if t['score_kind']=='choice' else 48)
        gen.add(dict(r,**s,question=t['question'],reference=t['reference'],category=t['category'],score_kind=t['score_kind'],grade=grade(t,r['answer']),rss_before_bytes=before,rss_after_bytes=rss()))
        if s['task_id'] in fixed_sequences:
            f=engine.generate(t['question'],method,SEED+s['repetition'],max_tokens=16,fixed_tokens=fixed_sequences[s['task_id']]);assert f['skip_layers']==r['skip_layers'];fixed.add(dict(f,**s))
        if s['repetition']==1:quality.add(dict(engine.nll(t['question'],t['reference'],r['skip_layers']),method=method,task_id=s['task_id']))
        if s['position']==len(METHODS)-1 and (s['sequence']+1)%8==0:print(f'IMPACT generation repeat{s["repetition"]}/3 question{s["sequence"]+1}/64',flush=True)
    gen.close(out/'GENERATION.csv');fixed.close(out/'FIXED_WORK.csv');quality.close(out/'QUALITY.csv');write_union(out/'OFFLINE_COSTS.csv',costs)
    frozen_check(manifest);assert sha(out/'PROFILE.json')==profilehash and sha(SETUP)==manifest['setup_sha256'] and sha(OLD)==manifest['old_profile_sha256']
    write_json(out/'METADATA.json',{'real_inference':True,'all_stages_wall_seconds':time.perf_counter()-jobstart,'profile_sha256':profilehash,'preserved_files':len(preserved),'generation_rows':len(gen.rows),'ablation_rows':len(ablations.rows),'fixed_rows':len(fixed.rows),'quality_rows':len(quality.rows),'GPU_elapsed_power_money':'unmeasured','FLOPs':'Analytical matmul estimate, not hardware counter','gate_timer':'Feature and prediction timer after materializing pre-block hidden state; additional synchronization effect included in whole request wall; old Sketch timer also contains preceding queued GPU work.','reference_evaluation':'once per method/question, excluded from live latency, never gate input'})
    print('IMPACT COMPLETE '+str(out),flush=True)

if __name__=='__main__':run()
