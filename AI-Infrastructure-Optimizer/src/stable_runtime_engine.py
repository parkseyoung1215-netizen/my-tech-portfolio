"""Same Qwen math and frozen policies; independently measured lean host execution."""
import hashlib,resource,time
import numpy as np
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from .alsr_impact_engine import ImpactEngine
from .alsr_policy import estimated_matmul_flops
from .stable_runtime_protocol import option_complete

class Tap:
    def __init__(self,owner,index,kind,inner):self.owner,self.index,self.kind,self.inner=owner,index,kind,inner
    def __call__(self,*args,**kw):
        getattr(self.owner,self.kind)[self.index]+=1
        if self.owner.trap_index==self.index:raise AssertionError('Native attention/MLP positive control')
        return self.inner(*args,**kw)

class Runtime:
    def __init__(self,model,tok,config,sketch,local):
        self.engine=ImpactEngine(model,tok,config);self.engine.prepare_profile(sketch);self.engine.prepare_local(local);self.model,self.tok,self.config=model,tok,config;self.n=len(model.layers);self.candidate=24
        assert local['candidate']==24
        self.native=tuple((i,l.self_attn.inner,l.mlp.inner,l.input_layernorm,l.post_attention_layernorm) for i,l in enumerate(model.layers))
        self.taps=tuple((i,Tap(self,i,'tap_attention',a),Tap(self,i,'tap_mlp',b),n1,n2) for i,a,b,n1,n2 in self.native);self.trap_index=None
        self.routes={(audit,method):tuple(r for r in (self.taps if audit else self.native) if method!='stable1' or r[0]!=24) for audit in (False,True) for method in ('full','stable1')}
        self.eos=set(getattr(tok,'eos_token_ids',[]) or [tok.eos_token_id])

    def _blocks(self,h,mask,cache,route,detailed):
        for i,attn,mlp,n1,n2 in route:
            begin=time.perf_counter() if detailed else 0.
            self.calls[i]+=1
            r=attn(n1(h),mask,cache[i]);h=h+r;h=h+mlp(n2(h))
            if detailed:self.dispatch_wall+=time.perf_counter()-begin
        return h

    def _lean(self,ids,cache,method,first,question,audit,detailed):
        context=ids.shape[1]+cache[0].offset;h=self.model.model.embed_tokens(ids);mask=create_attention_mask(h,cache[0])
        if first and method=='sketch':
            native=self.taps if audit else self.native;h=self._blocks(h,mask,cache,native[:2],detailed)
            self.skip,decision=self.engine.choose(h,ids.shape[1],'adaptive_sketch',question,0);self.decision=decision
            selected=set(self.skip);self.active=tuple(r for r in native if r[0] not in selected)
            h=self._blocks(h,mask,cache,tuple(r for r in self.active if r[0]>=2),detailed)
        else:h=self._blocks(h,mask,cache,self.active,detailed)
        logits=self.engine.head(self.model.model.norm(h)[:,-1:,:])
        self.work.append((ids.shape[1],context,len(self.active)));return logits

    def generate(self,question,method,backend,max_tokens=16,fixed_tokens=None,format_stop=False,audit=False,detailed=False):
        assert method in ('full','stable1','sketch') and backend in ('legacy','lean')
        mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time();t=time.perf_counter()
        self.engine.reset();self.calls=[0]*self.n;self.tap_attention=[0]*self.n;self.tap_mlp=[0]*self.n;self.work=[];self.dispatch_wall=0.;self.skip=[24] if method=='stable1' else [];self.decision=None
        self.active=self.routes[audit,method] if method!='sketch' else ();resetwall=time.perf_counter()-t
        t=time.perf_counter();ids=self.engine.prompt_ids(question);tokenize=time.perf_counter()-t
        t=time.perf_counter();cache=make_prompt_cache(self.model);inputs=mx.array([ids]);forced=mx.array([fixed_tokens]) if fixed_tokens is not None else None;cachewall=time.perf_counter()-t
        generated=[];stepwall=stepcpu=decodewall=stopwall=argmaxwait=0.;host_tensors=1+int(forced is not None);reason='max_tokens';can_stop=format_stop and question.rstrip().endswith('Return only the option letter.');first_hash=None
        for j in range(max_tokens):
            st,sc=time.perf_counter(),time.process_time()
            if backend=='legacy':
                logits,_=self.engine.forward(inputs,cache,self.skip,'adaptive_sketch' if method=='sketch' and j==0 else None,question,0)
                if self.engine.plan:self.skip=self.engine.plan['skip'];self.decision=self.engine.plan['decision']
            else:logits=self._lean(inputs,cache,method,j==0,question,audit,detailed)
            wait=time.perf_counter();next_gpu=mx.argmax(logits[0,-1]).reshape(1,1);mx.eval(next_gpu);next_id=int(next_gpu.item());argmaxwait+=time.perf_counter()-wait
            stepwall+=time.perf_counter()-st;stepcpu+=time.process_time()-sc
            if audit and j==0:first_hash=hashlib.sha256(np.array(logits[0,-1].astype(mx.float32)).tobytes()).hexdigest()
            token=int(fixed_tokens[j]) if fixed_tokens is not None else next_id;generated.append(token)
            if fixed_tokens is None and token in self.eos:reason='eos';break
            if fixed_tokens is None and can_stop:
                st=time.perf_counter();complete=option_complete(question,self.tok.decode(generated,skip_special_tokens=True));stopwall+=time.perf_counter()-st
                if complete:reason='request_option_complete';break
            if j+1<max_tokens:
                if backend=='legacy':inputs=mx.array([[token]]);host_tensors+=1
                else:inputs=forced[:,j:j+1] if forced is not None else next_gpu
        t=time.perf_counter();raw=self.tok.decode(generated);answer=self.tok.decode(generated,skip_special_tokens=True);decodewall=time.perf_counter()-t
        t=time.perf_counter();mx.synchronize();offsets=[c.offset for c in cache];expected=len(ids)+len(generated)-1;assert offsets==[0 if i in self.skip else expected for i in range(self.n)]
        calls=self.engine.counts[:] if backend=='legacy' else self.calls[:];expectedcalls=[0 if i in self.skip else len(generated) for i in range(self.n)];assert calls==expectedcalls
        if backend=='legacy':assert calls==self.engine.attention_counts==self.engine.mlp_counts
        elif audit:assert calls==self.tap_attention==self.tap_mlp
        checkwall=time.perf_counter()-t;wall=time.perf_counter()-start;hostcpu=time.process_time()-cpu
        if backend=='legacy':work=[(r['query_tokens'],r['context_tokens'],len(r['executed'])) for r in self.engine.forwards]
        else:work=self.work[:]
        fs=[{'query_tokens':q,'context_tokens':s,'executed':list(range(n)),'head_positions':1} for q,s,n in work]
        gate=self.engine.gate_wall
        return {'answer':answer,'raw_answer':raw,'token_ids':generated,'first_logits_sha256':first_hash,'input_tokens':len(ids),'output_tokens':len(generated),'completion_reason':reason,'skip_layers':self.skip[:],'gate_decision':self.decision,'layer_calls':calls,'observed_attention_calls':self.engine.attention_counts[:] if backend=='legacy' else self.tap_attention[:] if audit else None,'observed_mlp_calls':self.engine.mlp_counts[:] if backend=='legacy' else self.tap_mlp[:] if audit else None,'kv_offsets':offsets,'work_shapes':work,'analytical_matmul_flops':estimated_matmul_flops(self.config,fs,self.engine.gate_flops),'gate_flops_estimate':self.engine.gate_flops,'wall_seconds':wall,'cpu_seconds':hostcpu,'model_step_wall_seconds':stepwall,'model_step_cpu_seconds':stepcpu,'decision_wall_inclusive_seconds':gate,'decision_cpu_seconds':self.engine.gate_cpu,'model_step_except_gate_wall_seconds':max(0,stepwall-gate),'tokenize_wall_seconds':tokenize,'cache_setup_wall_seconds':cachewall,'reset_wall_seconds':resetwall,'decode_wall_seconds':decodewall,'format_check_wall_seconds':stopwall,'integrity_check_wall_seconds':checkwall,'host_bookkeeping_wall_seconds':wall-stepwall-tokenize-cachewall-resetwall-decodewall-stopwall-checkwall,'argmax_wait_wall_seconds':argmaxwait,'detailed_native_dispatch_wall_seconds':self.dispatch_wall if detailed and backend=='lean' else None,'host_created_input_tensors':host_tensors,'scalar_token_reads':len(generated),'mlx_peak_bytes':mx.get_peak_memory(),'mlx_active_bytes':mx.get_active_memory(),'process_lifetime_rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu_seconds':None,'GPU_power_money':'unmeasured','fixed_work':fixed_tokens is not None,'native_audit_enabled':audit,'detailed_instrumentation':detailed}
