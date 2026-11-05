"""Instrumentation added alongside the frozen ALSR engine, never replacing it."""
import hashlib,math,time
import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask
from .alsr_engine import Engine
from .alsr_policy import random_layers
from .alsr_impact_policy import accept,distribution_effect

class ImpactEngine(Engine):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.local=None;self._live=False

    def prepare_local(self,profile):
        self.local=profile;w=self.model.model.embed_tokens.weight if self.config.get('tie_word_embeddings',True) else self.model.lm_head.weight
        self.local_weight=w[mx.array(profile['shortlist'])].astype(mx.float32);mx.eval(self.local_weight)

    def features(self,h,previous,prompt_len):
        x=h[0,-1,:].astype(mx.float32);prev=x if previous is None else previous.astype(mx.float32)
        sketch=x@self.projection;rms=mx.sqrt(mx.mean(x*x));relative=mx.sqrt(mx.sum((x-prev)**2)/(mx.sum(x*x)+1e-8));cos=mx.sum(x*prev)/(mx.sqrt(mx.sum(x*x)*mx.sum(prev*prev))+1e-8)
        z=self.model.model.norm(x[None,None,:])[0,0];scores=z@self.local_weight.T;logp=scores-mx.logsumexp(scores);entropy=-mx.sum(mx.exp(logp)*logp)
        mx.eval(sketch,rms,relative,cos,scores,entropy)
        ss=sorted(scores.tolist(),reverse=True);margin=ss[0]-ss[1]
        return sketch.tolist()+[math.log1p(prompt_len),float(rms.item()),float(relative.item()),float(1-cos.item()),margin,float(entropy.item()),margin/(float(rms.item())+1e-6)]

    def diagnostic(self,ids):
        """Full forward with input-only features, used ONLY for offline causal labels."""
        self.reset();h=self.model.model.embed_tokens(ids);mask=create_attention_mask(h);previous=None;features={}
        for i,layer in enumerate(self.layers):
            features[str(i)]=self.features(h,previous,ids.shape[1]);previous=h[0,-1,:]
            self.counts[i]+=1;h=layer(h,mask,None)
        logits=self.head(self.model.model.norm(h)[:,-1:,:]);mx.eval(logits)
        assert self.counts==self.attention_counts==self.mlp_counts==[1]*self.n
        return np.array(logits[0,-1].astype(mx.float32)),features

    def initial_skip(self,method,prompt,seed):
        if method=='stable1':return [self.local['candidate']]
        if method=='random1':return random_layers(range(2,self.n-1),1,prompt,seed)
        if method=='prelayer_gate':return []
        return super().initial_skip(method,prompt,seed)

    def generate(self,question,method,seed,**kw):
        self._live=method=='prelayer_gate'
        try:return super().generate(question,method,seed,**kw)
        finally:self._live=False

    def forward(self,ids,cache=None,skip=(),adaptive=None,prompt='',seed=0,capture=False,head_last=1):
        if not self._live or self.plan is not None:
            return super().forward(ids,cache,skip,adaptive,prompt,seed,capture,head_last)
        h=self.model.model.embed_tokens(ids);context=ids.shape[1]+(cache[0].offset if cache else 0);mask=create_attention_mask(h,cache[0] if cache else None);selected=set(skip);executed=[];previous=None
        for i,layer in enumerate(self.layers):
            if i==self.local['candidate']:
                mx.eval(h,previous)  # exclude already-scheduled preceding block work from probe timer
                start,cpu=time.perf_counter(),time.process_time();f=self.features(h,previous,ids.shape[1]);yes,risk=accept(f,self.local)
                selected={i} if yes else set();self.plan={'skip':sorted(selected),'decision':{'features':f,'risk':risk,'threshold':self.local['threshold'],'candidate':i,'stage':'before_candidate_block_prefill_only'}}
                self.gate_wall+=time.perf_counter()-start;self.gate_cpu+=time.process_time()-cpu
                self.gate_flops+=2*self.config['hidden_size']*(8+len(self.local['shortlist']))+2*len(f)
            if i in selected:continue
            previous=h[0,-1,:];self.counts[i]+=1;h=layer(h,mask,cache[i] if cache else None);executed.append(i)
        logits=self.head(self.model.model.norm(h)[:,-head_last:,:]);mx.eval(logits)
        self.forwards.append({'query_tokens':ids.shape[1],'context_tokens':context,'executed':executed,'skipped':sorted(selected),'head_positions':logits.shape[1]})
        return logits,{}

def top_distribution(logits,k=10):
    ids=np.argpartition(logits,-k)[-k:];ids=ids[np.argsort(logits[ids])[::-1]];logp=logits.astype(float)-np.logaddexp.reduce(logits.astype(float))
    return {'sha256_f32':hashlib.sha256(logits.astype(np.float32).tobytes()).hexdigest(),'top_ids':ids.tolist(),'top_probabilities':np.exp(logp[ids]).tolist()}

def ablate_task(engine,task,split,seed):
    rows=[];contexts=[];ids=engine.prompt_ids(task['question']);eos=set(getattr(engine.tokenizer,'eos_token_ids',[]) or [engine.tokenizer.eos_token_id]);prefix=[]
    for position in range(2):
        start,cpu=time.perf_counter(),time.process_time();mx.synchronize();base,features=engine.diagnostic(mx.array([ids+prefix]));mx.synchronize()
        basewall=time.perf_counter()-start;basecpu=time.process_time()-cpu;top=top_distribution(base);first=int(np.argmax(base))
        engine.reset();plain,_=engine.forward(mx.array([ids+prefix]));plain=np.array(plain[0,-1].astype(mx.float32));difference=float(np.max(np.abs(base-plain)))
        assert difference==0,'Offline input probes changed baseline logits'
        contexts.append(dict(task_id=task['task_id'],split=split,position=position,prefix_tokens=prefix[:],input_tokens=len(ids)+len(prefix),baseline_distribution=top,baseline_top1=first,diagnostic_full_wall_seconds=basewall,diagnostic_full_cpu_seconds=basecpu,probe_reference_max_abs_difference=difference,scope='Offline instrumentation; includes probes for all layers; not service latency.'))
        order=list(range(engine.n));import random;random.Random(seed+position).shuffle(order)
        for seq,layer in enumerate(order):
            engine.reset();mx.synchronize();start,cpu=time.perf_counter(),time.process_time();logits,_=engine.forward(mx.array([ids+prefix]),skip=[layer]);mx.synchronize();wall=time.perf_counter()-start;hostcpu=time.process_time()-cpu
            arr=np.array(logits[0,-1].astype(mx.float32));effect=distribution_effect(base,arr)
            assert engine.counts==engine.attention_counts==engine.mlp_counts==[int(i!=layer) for i in range(engine.n)]
            rows.append(dict(task_id=task['task_id'],split=split,category=task['category'],position=position,layer=layer,sequence=seq,features=features[str(layer)],**effect,skip_distribution=top_distribution(arr),ablation_wall_seconds=wall,ablation_cpu_seconds=hostcpu,layer_calls=engine.counts[:],forwards=engine.forwards[:]))
        if first in eos:break
        prefix.append(first)
    return rows,contexts
