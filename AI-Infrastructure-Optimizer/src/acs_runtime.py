"""Full 28-layer inference; actual logits probabilities only when a gate requests them."""
import math,resource,time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from .stable_runtime_engine import Runtime
from .acs_policy import METHODS,boundary,request_kind,threshold,limit
from .alsr_policy import estimated_matmul_flops

class ACSRuntime(Runtime):
    def generate_acs(self,question,method,audit=False):
        assert method in METHODS
        mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time();self.engine.reset();self.calls=[0]*self.n;self.tap_attention=[0]*self.n;self.tap_mlp=[0]*self.n;self.work=[];self.dispatch_wall=0.;self.skip=[];self.decision=None;self.active=self.routes[audit,'full']
        ids=self.engine.prompt_ids(question);cache=make_prompt_cache(self.model);inputs=mx.array([ids]);tokens=[];events=[];modelwall=decisionwall=probwall=0.;probe_calls=0;firstwall=None;reason='max_tokens';cap=limit(question,method);kind=request_kind(question);lastprob=None
        for j in range(cap):
            step=time.perf_counter();logits=self._lean(inputs,cache,'full',j==0,question,audit,False);next_gpu=mx.argmax(logits[0,-1]).reshape(1,1);mx.eval(next_gpu);token=int(next_gpu.item());elapsed=time.perf_counter()-step;modelwall+=elapsed
            if firstwall is None:firstwall=elapsed
            tokens.append(token)
            if token in self.eos:reason='eos';break
            if method in ('boundary','confidence','adaptive'):
                begin=time.perf_counter();text=self.tok.decode(tokens,skip_special_tokens=True);candidate=boundary(question,text,adaptive=method=='adaptive');event={'position':j+1,'candidate':candidate,'kind':kind,'selected_probability':None,'eos_competitor_mass':None,'threshold':None,'stop':False}
                if candidate:
                    yes=method=='boundary'
                    if method in ('confidence','adaptive'):
                        pb=time.perf_counter();v=logits[0,-1].astype(mx.float32);lse=mx.logsumexp(v);selected=mx.exp(v[token]-lse);eos=mx.sum(mx.exp(v[mx.array(sorted(self.eos))]-lse));mx.eval(selected,eos);p=float(selected.item());ep=float(eos.item());probe_calls+=1;probwall+=time.perf_counter()-pb
                        cutoff=threshold(question,len(ids),method);yes=p>=cutoff;event.update(selected_probability=p,eos_competitor_mass=ep,threshold=cutoff);lastprob=p
                    event['stop']=yes
                    if yes:reason='answer_boundary' if method=='boundary' else 'confidence_boundary' if method=='confidence' else 'adaptive_boundary'
                events.append(event);decisionwall+=time.perf_counter()-begin
                if event['stop']:break
            inputs=next_gpu
        answer=self.tok.decode(tokens,skip_special_tokens=True);raw=self.tok.decode(tokens);mx.synchronize();offsets=[c.offset for c in cache];n=len(tokens);assert self.calls==[n]*self.n and offsets==[len(ids)+n-1]*self.n
        if audit:assert self.tap_attention==self.tap_mlp==self.calls
        wall=time.perf_counter()-start;hostcpu=time.process_time()-cpu;work=self.work[:];fs=[dict(query_tokens=q,context_tokens=s,executed=list(range(count)),head_positions=1) for q,s,count in work]
        return {'answer':answer,'raw_answer':raw,'token_ids':tokens,'input_tokens':len(ids),'output_tokens':n,'visible_output_tokens':n-int(tokens[-1] in self.eos),'model_calls':1,'forward_calls':len(work),'layer_calls':self.calls[:],'native_attention_calls':self.tap_attention[:] if audit else None,'native_mlp_calls':self.tap_mlp[:] if audit else None,'kv_offsets':offsets,'work_shapes':work,'completion_reason':reason,'token_cap':cap,'request_kind':kind,'stop_events':events,'probability_query_calls':probe_calls,'probability_elements_scanned':probe_calls*self.config['vocab_size'],'last_candidate_probability':lastprob,'wall_seconds':wall,'cpu_seconds':hostcpu,'model_step_wall_seconds':modelwall,'first_step_wall_seconds':firstwall,'decode_steps_wall_seconds':modelwall-firstwall,'decision_wall_seconds':decisionwall,'probability_wall_seconds':probwall,'end_to_end_tokens_per_second':n/wall,'decode_tokens_per_second':(n-1)/(modelwall-firstwall) if n>1 else None,'mlx_peak_bytes':mx.get_peak_memory(),'mlx_active_bytes':mx.get_active_memory(),'process_lifetime_rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'analytical_matmul_flops':estimated_matmul_flops(self.config,fs),'GPU_elapsed_power_money':'unmeasured','native_audit':audit}
