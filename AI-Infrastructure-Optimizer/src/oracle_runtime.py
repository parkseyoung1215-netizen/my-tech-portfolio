"""Native MLP table bypass; privileged table construction is separate from inference."""
import resource,time
import numpy as np
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from .dlr_policy import CANDIDATES,estimate

class OracleRuntime:
    def __init__(self,old):
        self.old=old;self.model,self.tok,self.config=old.model,old.tok,old.config;self.n=28;self.trap_mlp=None
    def compile_plan(self,plan):
        return tuple(tuple((i,l,i in selected) for i,l in enumerate(self.model.layers)) for selected in map(set,plan))
    def forward(self,inputs,cache,method,route,anchors,input_anchors,previous,step,stats,capture,profile):
        h=self.model.model.embed_tokens(inputs);mask=create_attention_mask(h,cache[0]);mlp=0;captured=[]
        for i,l,reuse in route:
            stats['attention_calls'][i]+=1
            if profile:mx.eval(h)
            begin=time.perf_counter() if profile else 0.
            h=h+l.self_attn(l.input_layernorm(h),mask,cache[i])
            if profile:mx.eval(h);stats['profile_attention_sync_wall_seconds']+=time.perf_counter()-begin
            z=None
            if method=='overhead' and i in CANDIDATES:
                z=l.post_attention_layernorm(h)
                if step and not previous[i]:
                    begin=time.perf_counter();mx.eval(z);stats['pre_gate_sync_wall_seconds']+=time.perf_counter()-begin
                    begin,cpu=time.perf_counter(),time.process_time();x=z.astype(mx.float32);old=input_anchors[i].astype(mx.float32);distance=mx.sqrt(mx.sum((x-old)**2)/(mx.sum(x*x)+1e-8));mx.eval(distance);unused=float(distance.item())
                    stats['gate_wall_seconds']+=time.perf_counter()-begin;stats['gate_cpu_seconds']+=time.process_time()-cpu;stats['gate_queries']+=1
            if reuse:
                out=anchors[i];stats['reused_mlp_calls'][i]+=1
            else:
                if self.trap_mlp==i:raise AssertionError('Real MLP positive control')
                if z is None:z=l.post_attention_layernorm(h)
                if profile:mx.eval(z)
                begin=time.perf_counter() if profile else 0.;out=l.mlp(z)
                if profile:mx.eval(out);stats['profile_mlp_sync_wall_seconds']+=time.perf_counter()-begin
                stats['mlp_calls'][i]+=1;mlp+=1
                if method in ('ideal','random','overhead'):
                    begin=time.perf_counter();anchors[i]=mx.contiguous(out[:,-1:,:])
                    if method=='overhead' and i in CANDIDATES:input_anchors[i]=mx.contiguous(z[:,-1:,:])
                    stats['cache_python_wall_seconds']+=time.perf_counter()-begin
                    if profile:
                        begin=time.perf_counter();mx.eval(anchors[i]);stats['profile_cache_sync_wall_seconds']+=time.perf_counter()-begin
            previous[i]=reuse
            if capture:captured.append(out[0,-1,:])
            h=h+out
        h=self.model.model.norm(h)[:,-1:,:];logits=self.model.model.embed_tokens.as_linear(h) if self.config.get('tie_word_embeddings',True) else self.model.lm_head(h)
        stats['work_shapes'].append([inputs.shape[1],stats['input_tokens']+step,28,mlp]);return logits,mx.stack(captured) if capture else None

    def generate(self,question,method,routes=None,max_tokens=32,fixed_tokens=None,capture=False,stop_eos=True,profile=False):
        assert method in ('baseline','existing','ideal','random','overhead')
        if method in ('baseline','existing') and not capture and not profile:
            r=self.old.generate(question,'baseline' if method=='baseline' else 'dlr',max_tokens,fixed_tokens)
            r.update(gate_queries=len(r['events']),cache_python_wall_seconds=None,route_lookup_wall_seconds=None,profile_attention_sync_wall_seconds=None,profile_mlp_sync_wall_seconds=None,profile_cache_sync_wall_seconds=None,primary_native=True,profile_intrusive=False,predicted_token_ids=None,gpu_kernel_count=None)
            r['gate_arithmetic_estimate']=r['gate_queries']*(5*self.config['hidden_size']+4);return r
        assert method!='existing','Existing detailed phases require external taps'
        if routes is None:routes=self.compile_plan([[] for _ in range(max_tokens)])
        mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time();ids=self.old.prompt_ids(question);cache=make_prompt_cache(self.model);inputs=mx.array([ids]);anchors={};input_anchors={};previous=[False]*28;tokens=[];predicted=[];captures=[]
        stats={'attention_calls':[0]*28,'mlp_calls':[0]*28,'reused_mlp_calls':[0]*28,'work_shapes':[],'input_tokens':len(ids),'gate_wall_seconds':0.,'gate_cpu_seconds':0.,'gate_queries':0,'pre_gate_sync_wall_seconds':0.,'cache_python_wall_seconds':0.,'route_lookup_wall_seconds':0.,'profile_attention_sync_wall_seconds':0.,'profile_mlp_sync_wall_seconds':0.,'profile_cache_sync_wall_seconds':0.}
        forced=mx.array([fixed_tokens]) if fixed_tokens is not None else None;limit=len(fixed_tokens) if fixed_tokens is not None else max_tokens;assert len(routes)>=limit;reason='fixed_stream' if fixed_tokens is not None else 'max_tokens';modelwall=modelcpu=0.
        for step in range(limit):
            st,sc=time.perf_counter(),time.process_time();begin=time.perf_counter();route=routes[step];stats['route_lookup_wall_seconds']+=time.perf_counter()-begin
            logits,activation=self.forward(inputs,cache,method,route,anchors,input_anchors,previous,step,stats,capture,profile)
            next_gpu=mx.argmax(logits[0,-1]).reshape(1,1);mx.eval(next_gpu,anchors,input_anchors,activation);next_id=int(next_gpu.item());modelwall+=time.perf_counter()-st;modelcpu+=time.process_time()-sc
            predicted.append(next_id);token=int(fixed_tokens[step]) if fixed_tokens is not None else next_id;tokens.append(token)
            if capture:captures.append(activation)
            if fixed_tokens is None and stop_eos and token in self.old.eos:reason='eos';break
            inputs=forced[:,step:step+1] if forced is not None else next_gpu
        answer=self.tok.decode(tokens,skip_special_tokens=True);captured=None
        if capture:
            captured=np.array(mx.stack(captures).astype(mx.float32))
        mx.synchronize();wall=time.perf_counter()-start;hostcpu=time.process_time()-cpu;n=len(tokens);offsets=[c.offset for c in cache];assert offsets==[len(ids)+n-1]*28
        assert all(stats['mlp_calls'][i]+stats['reused_mlp_calls'][i]==n for i in range(28))
        r=dict(stats,answer=answer,token_ids=tokens,predicted_token_ids=predicted,output_tokens=n,completion_reason=reason,kv_offsets=offsets,wall_seconds=wall,cpu_seconds=hostcpu,model_wall_seconds=modelwall,model_cpu_seconds=modelcpu,model_calls=1,analytical_matmul_flops=estimate(self.config,stats['work_shapes']),gate_arithmetic_estimate=stats['gate_queries']*(5*self.config['hidden_size']+4),cache_logical_bytes=sum(a.nbytes for a in anchors.values())+sum(a.nbytes for a in input_anchors.values()),mlx_peak_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory(),process_lifetime_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,primary_native=True,profile_intrusive=profile,gpu_kernel_count=None,gpu_time_flops_power_money=None,fixed_work=fixed_tokens is not None)
        return (r,captured) if capture else r
