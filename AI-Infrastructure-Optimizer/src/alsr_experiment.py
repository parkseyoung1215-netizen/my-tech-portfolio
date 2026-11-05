"""Run the actual small MLX pilot. No paid inference API, no fake performance data."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import gc,hashlib,json,random,resource,subprocess,sys,time,urllib.request
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from mlx_lm import load
from . import alsr_policy as policy
from .alsr_dataset import ROOT,SEED,build,grade
from .alsr_engine import Engine,calibrate
from .tool_experiment import write_records
from .adaptive_experiment import write_json
from .efficiency_repeat import frozen_check
REPEATS=3
PROTOCOL={'seed':SEED,'phase':'small_pilot','training_prompts':12,'calibration_prompts':8,'pilot_prompts':32,'repeats':REPEATS,
 'generation':'greedy, EOS respected, max16 MCQ/max48 open; same tokenizer/chat template and original HF BF16 per model',
 'quality_target':'<=1pp loss vs full in correctness and format; mean reference NLL delta<=.05 nats/token',
 'speed_target':'Natural wall AND same16-token workload upper95 paired ratio<=.90, with actual skipped blocks and lower analytic FLOPs',
 'uncertainty':'32 prompts insufficient to establish1pp population noninferiority; positive pilot means promising, not confirmed success',
 'cache':'One prompt-only mask after first2 prefill layers; fixed for all decode, skipped caches offset0, active caches aligned',
 'random':'Fixed random4 and separately adaptive-count-matched random; matched controls pay identical adaptive probe computation',
 'novelty':'Two project-local hypotheses, global novelty unconfirmed. BI static is ShortGPT-inspired partial baseline; no LayerSkip training reproduction',
 'costs':'Real CPU/synchronized wall/MLX peak memory. GPU elapsed, power and money unmeasured. Dense matmul FLOPs analytic, not hardware counter.',
 'fixed_work':'8 prespecified pilot inputs, baseline-derived token sequence padded with EOS to16; gate cannot see future tokens; not free-running quality',
 'quality_evaluation':'Reference NLL once/method/question with generated prompt-only mask frozen; evaluation time separately logged, no policy using reference tokens'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1<<20):h.update(chunk)
    return h.hexdigest()
def rss():
    try:return int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())]).strip())*1024
    except Exception:return None

def unload_ollama():
    records=[]
    for name in ('qwen2.5:0.5b','qwen2.5:1.5b'):
        try:
            payload={'model':name,'keep_alive':0};req=urllib.request.Request('http://localhost:11434/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30) as r:answer=json.load(r)
            records.append({'request':payload,'response':answer})
        except Exception as exc:records.append({'model':name,'error':str(exc)})
    return records

def run(setup_path):
    setup_path=Path(setup_path).resolve();setup=json.loads(setup_path.read_text());splits=build(ROOT/'data');pilot=splits['pilot']
    # Ensure fixed-work subset includes both structured and prose outputs.
    fixed_ids=[t['task_id'] for t in pilot if t['score_kind']=='choice'][:4]+[t['task_id'] for t in pilot if t['score_kind']=='json'][:2]+[t['task_id'] for t in pilot if t['score_kind']=='constraints'][:2]
    preserved={str(p.resolve()):sha(p) for folder in (ROOT/'src',ROOT/'tests',ROOT/'data') for p in folder.rglob('*') if p.is_file()}
    preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()});prefix=ROOT/'data'/('alsr_'+uuid4().hex)
    schedule=[]
    for mi,info in enumerate(setup['hf_models']):
        for rep in range(1,REPEATS+1):
            ids=[t['task_id'] for t in pilot];random.Random(SEED+rep+mi*100).shuffle(ids)
            for seq,id in enumerate(ids):
                methods=list(policy.METHODS);random.Random(SEED+mi*10000+rep*1000+seq).shuffle(methods)
                for pos,method in enumerate(methods,1):schedule.append({'model':info['repo'],'repetition':rep,'task_id':id,'sequence':seq,'position':pos,'method':method})
    manifest={'protocol':PROTOCOL,'setup_path':str(setup_path),'setup_sha256':sha(setup_path),'splits':splits,'schedule':schedule,'fixed_work_ids':fixed_ids,'preserved_sha256':preserved,'models':setup['hf_models']}
    write_json(Path(f'{prefix}_manifest.json'),manifest);write_json(Path(f'{prefix}_ollama_unload.json'),unload_ollama())
    limit=int(min(setup['mlx_device']['max_recommended_working_set_size']*.8,9*(1<<30)));mx.set_memory_limit(limit);mx.set_cache_limit(128*(1<<20))
    allrows=[];allfixed=[];quality=[];profiles={};cal_rows=[];cal_cost=[];loads=[];smokes=[];prep=[];warmups=[];trial_start=time.perf_counter();tasks={t['task_id']:t for t in pilot}
    for mi,info in enumerate(setup['hf_models']):
        gc.collect();mx.clear_cache();mx.synchronize();start,cpu=time.perf_counter(),time.process_time();model,tok,config=load(info['path'],return_config=True);model.eval();engine=Engine(model,tok,config)
        loads.append({'model':info['repo'],'load_wall_seconds':time.perf_counter()-start,'load_cpu_seconds':time.process_time()-cpu,'active_bytes':mx.get_active_memory(),'dtype':str(model.model.embed_tokens.weight.dtype),'config':config})
        smokes.append({'model':info['repo'],**engine.check_native(),**engine.check_skip_trap()});print('ALSR native parity '+info['repo'],flush=True)
        profile,trace,cost=calibrate(engine,splits['training'],splits['calibration']);profiles[info['repo']]=profile;cal_rows.extend(dict(r,model=info['repo']) for r in trace);cal_cost.append(dict(cost,model=info['repo']))
        write_json(Path(f'{prefix}_m{mi}_profile.json'),profile);profile_hash=sha(Path(f'{prefix}_m{mi}_profile.json'))
        print('ALSR calibrated '+info['repo']+' '+json.dumps(profile['thresholds']),flush=True)
        fixed_sequences={};eos=list(getattr(tok,'eos_token_ids',[]) or [tok.eos_token_id])[0]
        for id in fixed_ids:
            r=engine.generate(tasks[id]['question'],'baseline',SEED,max_tokens=16);fixed_sequences[id]=(r['token_ids']+[eos]*16)[:16];prep.append(dict(r,model=info['repo'],task_id=id,scope='Fixed-work benchmark sequence preparation, not gate training; cost separately recorded.'))
        write_json(Path(f'{prefix}_m{mi}_fixed_sequences.json'),fixed_sequences)
        # Neutral warmup for each method: exclude its input from all learning/evaluation.
        for method in policy.METHODS:warmups.append(dict(engine.generate('Return only A.',method,SEED,max_tokens=4),model=info['repo'],method=method))
        rows=[];fixed=[];qq=[]
        for rep in range(1,REPEATS+1):
            for s in [r for r in schedule if r['model']==info['repo'] and r['repetition']==rep]:
                task=tasks[s['task_id']];method=s['method'];cap=16 if task['score_kind']=='choice' else 48;before=rss();r=engine.generate(task['question'],method,SEED+rep,max_tokens=cap);after=rss()
                row=dict(r,**s,question=task['question'],category=task['category'],reference=task['reference'],score_kind=task['score_kind'],grade=grade(task,r['answer']),rss_before_bytes=before,rss_after_bytes=after)
                rows.append(row)
                if task['task_id'] in fixed_sequences:
                    f=engine.generate(task['question'],method,SEED+rep,max_tokens=16,fixed_tokens=fixed_sequences[task['task_id']]);assert f['skip_layers']==r['skip_layers']
                    fixed.append(dict(f,**s,scope='Same prompt + same16 input/output token workload; not natural-output quality.'))
                if rep==1:
                    nll=engine.nll(task['question'],task['reference'],r['skip_layers']);assert nll['quality_eval_gate_calls']==0
                    qq.append(dict(nll,model=info['repo'],task_id=task['task_id'],method=method,skip_layers=r['skip_layers']))
            write_records(Path(f'{prefix}_m{mi}_r{rep}_generation.csv'),[r for r in rows if r['repetition']==rep]);write_records(Path(f'{prefix}_m{mi}_r{rep}_fixed_work.csv'),[r for r in fixed if r['repetition']==rep])
            print('ALSR pilot '+info['repo']+f' repeat{rep}/{REPEATS} complete',flush=True)
        assert sha(Path(f'{prefix}_m{mi}_profile.json'))==profile_hash
        allrows.extend(rows);allfixed.extend(fixed);quality.extend(qq);del engine,model,tok;gc.collect();mx.clear_cache();mx.synchronize()
    frozen_check(manifest);assert sha(setup_path)==manifest['setup_sha256']
    for tag,old in setup['ollama'].items():
        assert sha(old['model_path'])==next(l['digest'].split(':')[1] for l in old['manifest']['layers'] if l['mediaType'].endswith('.model'))
    for name,rr in [('generation',allrows),('fixed_work',allfixed),('quality',quality),('calibration',cal_rows),('calibration_cost',cal_cost),('loads',loads),('native_parity',smokes),('fixed_preparation',prep),('warmups',warmups)]:write_records(Path(f'{prefix}_{name}.csv'),rr)
    write_json(Path(f'{prefix}_profiles.json'),profiles);write_json(Path(f'{prefix}_metadata.json'),{'prefix':prefix.name,'models':[i['repo'] for i in setup['hf_models']],'repeats':REPEATS,'pilot_questions':32,
      'generation_rows':len(allrows),'fixed_work_rows':len(allfixed),'quality_rows':len(quality),'total_trial_wall_seconds':time.perf_counter()-trial_start,'real_inference':True,'memory_limit_bytes':limit,
      'preserved_files':len(preserved),'profile_hashes':{str(mi):sha(Path(f'{prefix}_m{mi}_profile.json')) for mi in range(len(setup['hf_models']))},'GPU_elapsed_power_monetary_cost':'unmeasured'})
    print('ALSR PREFIX '+str(prefix),flush=True);return prefix

if __name__=='__main__':run(sys.argv[1])
