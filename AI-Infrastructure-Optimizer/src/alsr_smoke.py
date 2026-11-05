"""Actual 0.5B native parity, skipped-block trap and KV alignment smoke test."""
import json,hashlib,time,sys,os
from pathlib import Path
from uuid import uuid4
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from .alsr_engine import Engine,calibrate
ROOT=Path(__file__).resolve().parents[1]

def run(model_path,mini_calibration=False):
    start=time.perf_counter();model,tok,config=load(model_path,return_config=True);model.eval();engine=Engine(model,tok,config);native=engine.check_native()
    q='Return only the correct option. What is 9 + 6?\nA) 15\nB) 14\nC) 12\nD) 16'
    baseline=engine.generate(q,'baseline',1,max_tokens=8);static=engine.generate(q,'static',1,max_tokens=8)
    # If any supposedly skipped block is invoked, the real GPU smoke test fails.
    index=engine.pool[1];saved=engine.layers[index]
    class Trap:
        def __call__(self,*a,**k):raise AssertionError('Skipped block was actually invoked')
    engine.layers[index]=Trap();engine.reset();engine.forward(mx.array([engine.prompt_ids(q)]),skip=[index]);engine.layers[index]=saved
    mx.synchronize();record={'model_path':model_path,'config':config,'dtype':str(model.model.embed_tokens.weight.dtype),'native_parity':native,
      'baseline':baseline,'static':static,'exception_trap_skipped_layer':index,'trap_passed':True,'seconds':time.perf_counter()-start,
      'weights_modified':False,'gpu_device':mx.device_info()}
    eos=list(getattr(tok,'eos_token_ids',[]) or [tok.eos_token_id])[0];tokens=(baseline['token_ids']+[eos]*8)[:8]
    record['fixed_work_smoke']=engine.generate(q,'static',1,max_tokens=8,fixed_tokens=tokens)
    record['reference_nll_smoke']=engine.nll(q,'A',static['skip_layers'])
    record['source_sha256']={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ('alsr_engine.py','alsr_smoke.py','alsr_policy.py')}
    if mini_calibration:
        train=[{'question':f'Return only the integer: {a} + 2.','task_id':f'diagnostic_train{a}'} for a in (15,21)]
        val=[{'question':f'Return only the integer: {a} - 3.','task_id':f'diagnostic_cal{a}'} for a in (26,31)]
        profile,trace,cost=calibrate(engine,train,val);record.update(mini_calibration=profile,mini_calibration_cost=cost,mini_calibration_trace=trace,diagnostic_questions=train+val)
        record['adaptive_smoke']={method:engine.generate(q,method,1,max_tokens=8) for method in ('adaptive_sketch','random_match_sketch','adaptive_leverage','random_match_leverage')}
    path=ROOT/'data'/('alsr_smoke_'+uuid4().hex+'.json')
    with path.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print('ALSR SMOKE '+str(path));print('native',native,'baseline',repr(baseline['answer']),'static',repr(static['answer']),'dtype',record['dtype']);return path

if __name__=='__main__':run(sys.argv[1],'--mini-calibration' in sys.argv)
