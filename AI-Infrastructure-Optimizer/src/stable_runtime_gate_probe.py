"""Isolate the frozen Sketch gate on already-materialized training hidden states."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import json,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from .alsr_engine import Engine
from .adaptive_experiment import write_json
from .stable_runtime_experiment import SETUP,SKETCH,ROOT,sha

def run(folder):
    out=Path(folder).resolve();setup=json.loads(SETUP.read_text());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));model,tok,config=load(setup['hf_models'][1]['path'],return_config=True);model.eval();engine=Engine(model,tok,config);engine.prepare_profile(json.loads(SKETCH.read_text()));training=json.loads((ROOT/'data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_manifest.json').read_text())['splits']['training'][:4];records=[];preparations=[]
    for task in training:
        ids=engine.prompt_ids(task['question']);cache=make_prompt_cache(model);mx.synchronize();start,cpu=time.perf_counter(),time.process_time();h=model.model.embed_tokens(mx.array([ids]));mask=create_attention_mask(h,cache[0])
        for i in range(2):h=engine.layers[i](h,mask,cache[i])
        mx.eval(h);mx.synchronize();preparations.append({'task_id':task['task_id'],'first2_layers_wall_seconds':time.perf_counter()-start,'first2_layers_cpu_seconds':time.process_time()-cpu})
        engine.choose(h,len(ids),'adaptive_sketch',task['question'],0)  # explicit warmup excluded below
        for repetition in range(1,4):
            for iteration in range(5):
                mx.synchronize();start,cpu=time.perf_counter(),time.process_time();chosen,decision=engine.choose(h,len(ids),'adaptive_sketch',task['question'],0);mx.synchronize();records.append({'task_id':task['task_id'],'repetition':repetition,'iteration':iteration,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'skip_layers':chosen,'decision':decision})
    target=out/('ISOLATED_GATE_'+uuid4().hex+'.json');write_json(target,{'real_model_hidden_states':True,'training_only':True,'profile_sha256':sha(SKETCH),'records':records,'preparation':preparations,'scope':'Warm microbenchmark on reused, fully materialized first2-layer states; no full service answers. Do not subtract this from production gate timer as exact GPU attribution.','GPU_elapsed_power_money':'unmeasured'});print('ISOLATED GATE SAVED '+str(target),flush=True)

if __name__=='__main__':run(sys.argv[1])
