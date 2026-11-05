"""Extra real-GPU software controls, excluded from all research timings/scores."""
import json,sys
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
import mlx.nn as nn
from .dlr_opt_experiment import runtime,key
from .adaptive_experiment import write_json
from .stable_runtime_records import Store
from .dlr_opt_data import score
from .alsr_io import write_union

class Tap(nn.Module):
    def __init__(self,inner):super().__init__();self.inner=inner;self.calls=0
    def __call__(self,x):self.calls+=1;return self.inner(x)

def run(folder):
    r=runtime();indices=(8,12,16,24);original={i:r.model.layers[i].mlp for i in indices};taps={i:Tap(v) for i,v in original.items()}
    try:
        for i,t in taps.items():r.model.layers[i].mlp=t
        result=r.generate('Return one word.','lightweight',fixed_tokens=[42,42,42,42])
        observed={i:t.calls for i,t in taps.items()}
        assert all(v==3 for v in observed.values()) and all(result['reused_mlp_calls'][i]==1 for i in indices)
    finally:
        for i,v in original.items():r.model.layers[i].mlp=v
    # Poisoned tiny arrays exercise exactly the same conditional kernel, not model performance.
    class Fake:pass
    mlp=Fake()
    for attr,shape in [('gate_proj',(64,32)),('up_proj',(64,32)),('down_proj',(32,64))]:
        layer=Fake();layer.weight=mx.full(shape,float('nan'),mx.bfloat16);setattr(mlp,attr,layer)
    x=mx.ones((1,1,32),mx.bfloat16);cached=mx.full_like(x,7);skip,cs=r.kernel(mlp,x,cached,mx.array(True));fresh,cf=r.kernel(mlp,x,cached,mx.array(False));mx.eval(skip,cs,fresh,cf)
    assert bool(mx.all(skip==cached).item()) and bool(mx.all(mx.isnan(fresh)).item()) and int(cs.item())==0 and int(cf.item())==3
    out=Path(folder)/('SOFTWARE_CONTROLS_'+uuid4().hex+'.json');write_json(out,{'scope':'Actual GPU software controls ONLY; no evaluation score or latency claim','lightweight_actual_mlp_tap_counts':observed,'lightweight_four_forward_cached_reuses':result['reused_mlp_calls'],'poisoned_weight_reuse_exact_cached_output':True,'poisoned_weight_fresh_nan_positive_control':True,'gpu_projection_branches_reuse':int(cs.item()),'gpu_projection_branches_fresh':int(cf.item())});print('DLR OPT CONTROLS '+str(out),flush=True)
    # Explicitly separate exploratory fresh-kernel fidelity ablation, not policy tuning.
    root=Path(folder);m=json.loads((root/'MANIFEST.json').read_text());store=Store(root/'checkpoints');rows=[]
    selected=[('fresh32',t) for t in m['datasets']['fresh32'][:4]]+[(ds,t) for ds in ('long_high','long_low') for t in m['datasets'][ds][:2]]
    for rep in (1,2,3):
        for ds,t in selected:
            rec=r.generate(t['question'],'device',max_tokens=t['token_cap'],force_fresh=True);base=store.load(key(dict(phase='natural',dataset=ds,repetition=1,task_id=t['task_id'],method='baseline')))
            assert sum(rec['reused_mlp_calls'])==0
            rows.append(dict(rec,dataset=ds,task_id=t['task_id'],repetition=rep,baseline_answer=base['answer'],baseline_token_ids=base['token_ids'],tokens_match_baseline=rec['token_ids']==base['token_ids'],grade=score(t,rec['answer']),scope='Exploratory post-primary fresh-kernel fidelity only; timing not randomized with primary and excluded from primary comparisons'))
    path=root/('FRESH_KERNEL_ABLATION_'+uuid4().hex+'.csv');write_union(path,rows);print('DLR OPT FRESH KERNEL ABLATION '+str(path),flush=True)

if __name__=='__main__':run(sys.argv[1])
