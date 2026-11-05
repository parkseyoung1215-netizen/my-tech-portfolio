"""Unmodified old paths plus actual device-conditional and cheap-token reuse."""
import resource,time
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from .dlr_runtime import DLRRuntime
from .dlr_policy import CANDIDATES,estimate
from .dlr_opt_kernels import ConditionalMLP
METHODS=('baseline','old_dlr','device','lightweight')

class OptimizationRuntime:
    def __init__(self,model,tok,config):
        self.old=DLRRuntime(model,tok,config);self.model,self.tok,self.config=model,tok,config;self.kernel=ConditionalMLP();self.n=28

    def forward(self,inputs,cache,method,state,step,input_id,stats,force_fresh=False,force_reuse=False):
        h=self.model.model.embed_tokens(inputs);mask=create_attention_mask(h,cache[0]);light_reuse=False
        if method=='lightweight' and step:
            begin,cpu=time.perf_counter(),time.process_time();light_reuse=state['age']==0 and input_id==state['anchor_id'];stats['id_comparisons']+=1
            stats['gate_wall_seconds']+=time.perf_counter()-begin;stats['gate_cpu_seconds']+=time.process_time()-cpu
        att=mlp=0
        for i,l in enumerate(self.model.layers):
            stats['attention_calls'][i]+=1;att+=1;h=h+l.self_attn(l.input_layernorm(h),mask,cache[i]);z=l.post_attention_layernorm(h)
            if i not in CANDIDATES:
                out=l.mlp(z);stats['mlp_calls'][i]+=1;mlp+=1
            elif method=='lightweight':
                if light_reuse:out=state[i];stats['reused_mlp_calls'][i]+=1
                else:out=l.mlp(z);state[i]=mx.contiguous(out[:,-1:,:]);stats['mlp_calls'][i]+=1;mlp+=1
            elif step==0:
                out=l.mlp(z);stats['mlp_calls'][i]+=1;mlp+=1
                state[i]={'input':mx.contiguous(z[:,-1:,:]),'output':mx.contiguous(out[:,-1:,:]),'age':mx.array(0,mx.int32)}
            else:
                begin,cpu=time.perf_counter(),time.process_time();x=z.astype(mx.float32);old=state[i]['input'].astype(mx.float32)
                distance=mx.sqrt(mx.sum((x-old)**2)/(mx.sum(x*x)+1e-8));reuse=(distance<=.5)&(state[i]['age']==0)
                if force_fresh:reuse=mx.array(False)
                if force_reuse:reuse=mx.array(step%2==1)
                stats['gate_graph_host_wall_seconds']+=time.perf_counter()-begin;stats['gate_graph_host_cpu_seconds']+=time.process_time()-cpu
                stats['gate_elements']+=self.config['hidden_size'];stats['gate_queries']+=1
                out,count=self.kernel(l.mlp,z,state[i]['output'],reuse)
                state[i]={'input':mx.where(reuse,state[i]['input'],z),'output':out,'age':mx.where(reuse,state[i]['age']+1,mx.array(0,mx.int32))}
                stats['device_trace'].append((step,i,reuse,count,distance))
                # Counts/work for device decode are reconstructed from GPU branch counters.
            h=h+out
        if method=='lightweight':
            if light_reuse:state['age']+=1
            else:state['age']=0;state['anchor_id']=input_id
        h=self.model.model.norm(h)[:,-1:,:];logits=self.model.model.embed_tokens.as_linear(h) if self.config.get('tie_word_embeddings',True) else self.model.lm_head(h)
        stats['work_shapes'].append([inputs.shape[1],stats['input_tokens']+step,att,mlp]);return logits

    def generate(self,question,method,max_tokens=32,fixed_tokens=None,force_fresh=False,force_reuse=False):
        assert method in METHODS
        if method in ('baseline','old_dlr'):
            r=self.old.generate(question,'baseline' if method=='baseline' else 'dlr',max_tokens,fixed_tokens)
            r.update(gate_queries=len(r['events']),gate_host_scalar_reads=len(r['events']),explicit_pre_gate_evals=len(r['events']),id_comparisons=0,gate_graph_host_wall_seconds=0.,gate_graph_host_cpu_seconds=0.,device_gate_gpu_seconds=None,device_counter_read_batches=0,device_custom_projection_calls=0,predicted_token_ids=None)
            r['gate_arithmetic_estimate']=r['gate_queries']*(5*self.config['hidden_size']+4)
            r['gate_selection_element_count']=0;r['limited_net_arithmetic_proxy']=r['analytical_matmul_flops']+r['gate_arithmetic_estimate'];return r
        mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time();ids=self.old.prompt_ids(question);cache=make_prompt_cache(self.model);inputs=mx.array([ids]);state={};tokens=[];predictions=[]
        stats={'attention_calls':[0]*28,'mlp_calls':[0]*28,'reused_mlp_calls':[0]*28,'work_shapes':[],'device_trace':[],'gate_queries':0,'gate_wall_seconds':0.,'gate_cpu_seconds':0.,'gate_graph_host_wall_seconds':0.,'gate_graph_host_cpu_seconds':0.,'id_comparisons':0,'gate_elements':0,'input_tokens':len(ids)}
        forced=mx.array([fixed_tokens]) if fixed_tokens is not None else None;limit=len(fixed_tokens) if fixed_tokens is not None else max_tokens;reason='fixed_stream' if fixed_tokens is not None else 'max_tokens';model_wall=model_cpu=0.
        for step in range(limit):
            st,sc=time.perf_counter(),time.process_time();input_id=ids[-1] if step==0 else tokens[-1]
            logits=self.forward(inputs,cache,method,state,step,input_id,stats,force_fresh,force_reuse);next_gpu=mx.argmax(logits[0,-1]).reshape(1,1);mx.eval(next_gpu,state)
            next_id=int(next_gpu.item());predictions.append(next_id);model_wall+=time.perf_counter()-st;model_cpu+=time.process_time()-sc
            token=int(fixed_tokens[step]) if fixed_tokens is not None else next_id;tokens.append(token)
            if fixed_tokens is None and token in self.old.eos:reason='eos';break
            inputs=forced[:,step:step+1] if forced is not None else next_gpu
        trace=stats.pop('device_trace');events=[];count_batches=0
        if trace:
            meta=mx.stack([mx.stack([flag.astype(mx.float32),count[0].astype(mx.float32),dist]) for _,_,flag,count,dist in trace]);mx.eval(meta);values=meta.tolist();count_batches=1
            for (step,i,_,_,_),values in zip(trace,values):
                reused=bool(values[0]);calls=int(values[1]);assert calls==(0 if reused else 3)
                stats['mlp_calls'][i]+=not reused;stats['reused_mlp_calls'][i]+=reused;stats['work_shapes'][step][3]+=not reused
                events.append(dict(step=step,layer=i,reused=reused,gpu_projection_mac_branches=calls,distance=values[2]))
        answer=self.tok.decode(tokens,skip_special_tokens=True);mx.synchronize();wall=time.perf_counter()-start;cpu_time=time.process_time()-cpu;n=len(tokens);offsets=[c.offset for c in cache]
        assert offsets==[len(ids)+n-1]*28
        assert all(stats['mlp_calls'][i]+stats['reused_mlp_calls'][i]==n for i in range(28))
        flops=estimate(self.config,stats['work_shapes']);gate_ops=stats['gate_queries']*(5*self.config['hidden_size']+4)
        return dict(stats,answer=answer,token_ids=tokens,predicted_token_ids=predictions,output_tokens=n,completion_reason=reason,kv_offsets=offsets,wall_seconds=wall,cpu_seconds=cpu_time,model_wall_seconds=model_wall,model_cpu_seconds=model_cpu,model_calls=1,analytical_matmul_flops=flops,gate_arithmetic_estimate=gate_ops,gate_selection_element_count=stats['gate_queries']*(2*self.config['hidden_size']+2),limited_net_arithmetic_proxy=flops+gate_ops,events=events,gate_host_scalar_reads=0,explicit_pre_gate_evals=0,device_gate_gpu_seconds=None,device_counter_read_batches=count_batches,device_custom_projection_calls=len(trace)*3,pre_gate_sync_wall_seconds=0.,mlx_peak_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory(),process_lifetime_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,fixed_work=fixed_tokens is not None,gpu_time_flops_power_money=None,gate_included_in_wall=True)
