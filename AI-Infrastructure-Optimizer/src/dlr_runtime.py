"""Real Qwen MLP temporal reuse; all attention KV updates remain fresh."""
import resource,time
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from .dlr_policy import CANDIDATES,METHODS,reuse_allowed,estimate

class DLRRuntime:
    def __init__(self,model,tok,config):
        self.model,self.tok,self.config=model,tok,config;self.layers=model.layers;self.n=len(self.layers)
        assert self.n==28 and config['model_type']=='qwen2'
        self.eos=set(getattr(tok,'eos_token_ids',[]) or [tok.eos_token_id])
        self.trap_mlp=None

    def prompt_ids(self,question):
        return self.tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':question}],tokenize=True,add_generation_prompt=True)

    def forward(self,inputs,cache,method,state,step,stats,force_reuse=False):
        h=self.model.model.embed_tokens(inputs);mask=create_attention_mask(h,cache[0]);att=mlp=0
        for i,l in enumerate(self.layers):
            if method=='static' and i==24:continue
            stats['attention_calls'][i]+=1;att+=1
            h=h+l.self_attn(l.input_layernorm(h),mask,cache[i]);z=l.post_attention_layernorm(h);reuse=False
            if method in ('dlr','conservative') and i in CANDIDATES:
                if step>0 and state[i]['age']==0:
                    # Synchronize preceding work BEFORE timing gate; wall includes this wait.
                    begin=time.perf_counter();mx.eval(z);stats['pre_gate_sync_wall_seconds']+=time.perf_counter()-begin
                    begin,cpu=time.perf_counter(),time.process_time();x=z.astype(mx.float32);old=state[i]['input'].astype(mx.float32)
                    distance=mx.sqrt(mx.sum((x-old)**2)/(mx.sum(x*x)+1e-8));mx.eval(distance);value=float(distance.item())
                    reuse=force_reuse or reuse_allowed(value,state[i]['age'],method)
                    stats['gate_wall_seconds']+=time.perf_counter()-begin;stats['gate_cpu_seconds']+=time.process_time()-cpu
                    stats['gate_elements']+=self.config['hidden_size'];stats['events'].append(dict(step=step,layer=i,distance=value,reused=reuse,anchor_step=state[i]['step']))
                if reuse:
                    out=state[i]['output'];state[i]['age']+=1;stats['reused_mlp_calls'][i]+=1
                else:
                    if self.trap_mlp==i:raise AssertionError('MLP positive control')
                    out=l.mlp(z);stats['mlp_calls'][i]+=1;mlp+=1
                    # Last-token logical slice; allocator peak also captures retained backing buffers.
                    state[i]={'input':mx.contiguous(z[:,-1:,:]),'output':mx.contiguous(out[:,-1:,:]),'age':0,'step':step}
            else:
                if self.trap_mlp==i:raise AssertionError('MLP positive control')
                out=l.mlp(z);stats['mlp_calls'][i]+=1;mlp+=1
            h=h+out
        h=self.model.model.norm(h)[:,-1:,:];logits=self.model.model.embed_tokens.as_linear(h) if self.config.get('tie_word_embeddings',True) else self.model.lm_head(h)
        stats['work_shapes'].append([inputs.shape[1],stats['input_tokens']+step,att,mlp]);return logits

    def generate(self,question,method,max_tokens=32,fixed_tokens=None,force_reuse=False):
        assert method in METHODS
        mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time()
        ids=self.prompt_ids(question);cache=make_prompt_cache(self.model);inputs=mx.array([ids]);state={};tokens=[]
        stats={'attention_calls':[0]*self.n,'mlp_calls':[0]*self.n,'reused_mlp_calls':[0]*self.n,'work_shapes':[],'events':[],'gate_wall_seconds':0.,'gate_cpu_seconds':0.,'gate_elements':0,'pre_gate_sync_wall_seconds':0.,'input_tokens':len(ids)}
        model_wall=model_cpu=0.;forced=mx.array([fixed_tokens]) if fixed_tokens is not None else None
        limit=len(fixed_tokens) if fixed_tokens is not None else max_tokens;reason='fixed_stream' if fixed_tokens is not None else 'max_tokens'
        for j in range(limit):
            st,sc=time.perf_counter(),time.process_time();logits=self.forward(inputs,cache,method,state,j,stats,force_reuse);next_gpu=mx.argmax(logits[0,-1]).reshape(1,1);mx.eval(next_gpu,state);token=int(next_gpu.item())
            model_wall+=time.perf_counter()-st;model_cpu+=time.process_time()-sc
            if fixed_tokens is not None:token=int(fixed_tokens[j])
            tokens.append(token)
            if fixed_tokens is None and token in self.eos:reason='eos';break
            inputs=forced[:,j:j+1] if forced is not None else next_gpu
        answer=self.tok.decode(tokens,skip_special_tokens=True);mx.synchronize();wall=time.perf_counter()-start;hostcpu=time.process_time()-cpu
        offsets=[c.offset for c in cache];n=len(tokens)
        assert offsets==[0 if method=='static' and i==24 else len(ids)+n-1 for i in range(self.n)]
        assert all(stats['mlp_calls'][i]+stats['reused_mlp_calls'][i]==stats['attention_calls'][i] for i in range(self.n))
        cached_bytes=sum(v['input'].nbytes+v['output'].nbytes for v in state.values())
        return dict(stats,answer=answer,token_ids=tokens,output_tokens=n,completion_reason=reason,kv_offsets=offsets,wall_seconds=wall,cpu_seconds=hostcpu,model_wall_seconds=model_wall,model_cpu_seconds=model_cpu,model_calls=1,analytical_matmul_flops=estimate(self.config,stats['work_shapes']),cached_activation_payload_bytes=cached_bytes,mlx_peak_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory(),process_lifetime_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,extra_gate_model_calls=0,gpu_time_flops_power_money=None,fixed_work=fixed_tokens is not None)
