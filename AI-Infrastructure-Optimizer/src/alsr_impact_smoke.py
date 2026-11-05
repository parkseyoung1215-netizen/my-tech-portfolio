"""Actual neutral-input positive/negative controls for the new live gate branch."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import copy,json,sys,time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from .alsr_impact_engine import ImpactEngine
from .alsr_impact_run import OLD,SETUP,sha
from .adaptive_experiment import write_json

def run(folder):
    start=time.perf_counter();out=Path(folder).resolve();m=json.loads((out/'MANIFEST.json').read_text());p=json.loads((out/'PROFILE.json').read_text());before=sha(out/'PROFILE.json');setup=json.loads(SETUP.read_text());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20))
    model,tok,config=load(m['model']['path'],return_config=True);model.eval();engine=ImpactEngine(model,tok,config);engine.prepare_profile(json.loads(OLD.read_text()))
    forced=copy.deepcopy(p);forced['threshold']=1e9;engine.prepare_local(forced);index=p['candidate'];saved=engine.layers[index]
    class Trap:
        def __call__(self,*a,**kw):raise AssertionError('Positive control: candidate block invoked')
    engine.layers[index]=Trap();positive=False
    try:
        yes=engine.generate('Return only the letter A.','prelayer_gate',1,max_tokens=4)
        assert yes['skip_layers']==[index] and yes['layer_calls'][index]==yes['attention_calls'][index]==yes['mlp_calls'][index]==0
        try:engine.generate('Return only the letter A.','baseline',1,max_tokens=4)
        except AssertionError:positive=True
        assert positive
    finally:engine.layers[index]=saved
    engine.prepare_local(p);no=engine.generate('Return only the letter A.','prelayer_gate',1,max_tokens=4);base=engine.generate('Return only the letter A.','baseline',1,max_tokens=4)
    assert no['skip_layers']==[] and no['token_ids']==base['token_ids'] and sha(out/'PROFILE.json')==before
    write_json(out/'GATE_BRANCH_SMOKE.json',{'real_model':True,'scope':'Neutral prompt only; forced threshold is a software branch test, not calibrated/evaluation performance. No policy file changed.','candidate':index,'forced_skip_trap_passed':True,'non_skipped_positive_control_raised':positive,'forced_request':yes,'normal_request':no,'normal_equals_baseline_tokens':True,'profile_sha256_unchanged':before,'test_including_load_wall_seconds':time.perf_counter()-start})
    print('Actual gate branch positive/negative controls passed: '+str(out/'GATE_BRANCH_SMOKE.json'),flush=True)

if __name__=='__main__':run(sys.argv[1])
