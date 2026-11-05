"""Intrusive phase timing on real equal-stream model execution; not primary latency."""
import json,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
import mlx.nn as nn
from .oracle_experiment import runtime
from .stable_runtime_records import Store
from .adaptive_experiment import write_json
from .alsr_io import write_union

class Timed(nn.Module):
    def __init__(self,inner,owner,kind):super().__init__();self.inner=inner;self.owner=owner;self.kind=kind
    def __call__(self,*args,**kw):
        begin=time.perf_counter();mx.eval([v for v in args if isinstance(v,mx.array)]);self.owner['pre_component_wait_wall_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter();out=self.inner(*args,**kw);mx.eval(out);self.owner[self.kind]+=time.perf_counter()-begin;return out
def run(folder):
    p=Path(folder);m=json.loads((p/'MANIFEST.json').read_text());store=Store(p/'checkpoints');r=runtime();rows=[]
    for ds,task in [('replay8',m['datasets']['replay8'][0]),('long4',m['datasets']['long4'][0])]:
        fixed=next(x['token_ids'] for x in store.rows() if x['dataset']==ds and x['task_id']==task['task_id'] and x['phase']=='natural' and x['repetition']==1 and x['method']=='baseline');table=json.loads((p/'tables'/task['task_id']/'TABLES.json').read_text())
        for method,rate in [('baseline',None),('ideal',0),('ideal',50),('overhead',50),('existing',None)]:
            if method=='existing':
                timing={'profile_attention_sync_wall_seconds':0.,'profile_mlp_sync_wall_seconds':0.,'pre_component_wait_wall_seconds':0.};original=[(l,l.self_attn,l.mlp) for l in r.model.layers]
                try:
                    for l,a,b in original:l.self_attn=Timed(a,timing,'profile_attention_sync_wall_seconds');l.mlp=Timed(b,timing,'profile_mlp_sync_wall_seconds')
                    r.generate(task['question'],method,fixed_tokens=fixed)
                    for k in timing:timing[k]=0.
                    result=r.generate(task['question'],method,fixed_tokens=fixed);result.update(timing,profile_intrusive=True)
                finally:
                    for l,a,b in original:l.self_attn=a;l.mlp=b
            else:
                routes=None if method=='baseline' else r.compile_plan(table['oracle'][str(rate)])
                r.generate(task['question'],method,routes,max_tokens=task['token_cap'],fixed_tokens=fixed,profile=True)
                result=r.generate(task['question'],method,routes,max_tokens=task['token_cap'],fixed_tokens=fixed,profile=True)
            rows.append(dict(result,dataset=ds,task_id=task['task_id'],method=method,rate=rate,attention_scope='inner attention only; norm/add outside' if method=='existing' else 'input norm + attention + residual add',scope='Intrusive phase-synchronized wall, NOT actual GPU timestamps; excluded from all primary comparisons'))
    out=p/('PROFILE_'+uuid4().hex);out.mkdir();write_union(out/'ORACLE_INTRUSIVE_PROFILE.csv',rows);write_json(out/'SCOPE.json',{'actual_model_execution':True,'warmed_per_condition':True,'warmup_requests':10,'measured_profile_requests':10,'extra_sync_changes_overlap':True,'gpu_kernel_counts_measured':False,'xctrace_available':False,'host_cache_dispatch_is_not_gpu_copy_time':True});print('ORACLE PROFILE '+str(out),flush=True)

if __name__=='__main__':run(sys.argv[1])
