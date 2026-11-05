"""Qwen2 MLX execution with real branch-before-block skips and aligned KV caches.

Uses installed MLX-LM Qwen blocks (Apple MIT code), not a replacement architecture.
"""
import math,random,time,hashlib,resource,subprocess,os,weakref
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from . import alsr_policy as policy

class CountedModule(nn.Module):
    def __init__(self,inner,owner,index,kind):
        super().__init__();self.inner=inner;self._owner=weakref.ref(owner);self._index=index;self._kind=kind
    def __call__(self,*args,**kwargs):
        owner=self._owner();getattr(owner,self._kind)[self._index]+=1
        return self.inner(*args,**kwargs)

class Engine:
    def __init__(self,model,tokenizer,config):
        self.model,self.tokenizer,self.config=model,tokenizer,config;self.layers=model.model.layers;self.n=len(self.layers);self.pool=policy.pool_for(self.n);self.profile=None
        self.counts=[0]*self.n;self.attention_counts=[0]*self.n;self.mlp_counts=[0]*self.n;self.forwards=[];self.gate_wall=0.;self.gate_cpu=0.;self.plan=None;self.gate_flops=0
        self.projection=mx.array(np.random.default_rng(policy.SEED).choice([-1.,1.],size=(config['hidden_size'],8))/math.sqrt(config['hidden_size']),dtype=mx.float32)
        for i,layer in enumerate(self.layers):
            layer.self_attn=CountedModule(layer.self_attn,self,i,'attention_counts');layer.mlp=CountedModule(layer.mlp,self,i,'mlp_counts')

    def reset(self):
        self.counts=[0]*self.n;self.attention_counts=[0]*self.n;self.mlp_counts=[0]*self.n;self.forwards=[];self.gate_wall=0.;self.gate_cpu=0.;self.plan=None;self.gate_flops=0

    def head(self,h):
        return self.model.model.embed_tokens.as_linear(h) if self.config.get('tie_word_embeddings',True) else self.model.lm_head(h)

    def prepare_profile(self,profile):
        self.profile=profile;weight=self.model.model.embed_tokens.weight if self.config.get('tie_word_embeddings',True) else self.model.lm_head.weight
        self.probe_weight=weight[mx.array(profile['shortlist'])].astype(mx.float32);mx.eval(self.probe_weight,self.projection)

    def probe(self,h,prompt_len):
        last=h[0,-1,:].astype(mx.float32);sketch=last@self.projection;rms=mx.sqrt(mx.mean(last**2));norm=self.model.model.norm(h[:,-1:,:].astype(mx.float32)).astype(mx.float32)[0,0];logits=norm@self.probe_weight.T
        mx.eval(h,sketch,rms,logits);scores=logits.tolist();order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        margin=max(.05,scores[order[0]]-scores[order[1]])
        return {'features':sketch.tolist()+[math.log1p(prompt_len),float(rms.item()),margin],'pair':order[:2],'margin':margin,'rms':float(rms.item())}

    def choose(self,h,prompt_len,method,prompt,seed):
        start,cpu=time.perf_counter(),time.process_time();probe=self.probe(h,prompt_len)
        family='sketch' if method.endswith('sketch') else 'leverage'
        if family=='sketch':risks={int(l):policy.predict_ridge(probe['features'],r) for l,r in self.profile['ridge'].items()}
        else:
            a,b=probe['pair'];risks={int(l):abs(r['sensitivity'][a]-r['sensitivity'][b])/probe['margin']*(probe['rms']/self.profile['train_probe_rms'])*r['scale'] for l,r in self.profile['leverage'].items()}
        chosen=policy.choose_by_risk(risks,self.profile['thresholds'][family],self.pool)
        if method.startswith('random_match'):chosen=policy.random_layers(self.pool,len(chosen),prompt,seed)
        self.gate_cpu+=time.process_time()-cpu;self.gate_wall+=time.perf_counter()-start
        self.gate_flops+=2*self.config['hidden_size']*(8+len(self.profile['shortlist']))+2*len(self.pool)*12
        return chosen,{'family':family,'risks':risks,'probe_pair':probe['pair'],'probe_margin':probe['margin'],'threshold':self.profile['thresholds'][family]}

    def forward(self,ids,cache=None,skip=(),adaptive=None,prompt='',seed=0,capture=False,head_last=1):
        h=self.model.model.embed_tokens(ids);context=ids.shape[1]+(cache[0].offset if cache else 0);mask=create_attention_mask(h,cache[0] if cache else None);selected=set(skip);captured={};executed=[];decision=None
        for i,layer in enumerate(self.layers):
            if i in selected:continue  # actual block/attention/MLP are never invoked
            before=h;self.counts[i]+=1;h=layer(h,mask,cache[i] if cache else None);executed.append(i)
            if capture and i in self.pool:
                x,y=before.astype(mx.float32),h.astype(mx.float32);cos=mx.sum(x*y,axis=-1)/(mx.sqrt(mx.sum(x*x,axis=-1)*mx.sum(y*y,axis=-1))+1e-8)
                captured[str(i)]={'residual':(y-x)[0,-1,:],'influence':mx.mean(1-cos),'rms':mx.sqrt(mx.mean(y[0,-1,:]**2))}
            if i==1:
                if capture:captured['early']=h[0,-1,:].astype(mx.float32)
                if adaptive:
                    chosen,decision=self.choose(h,ids.shape[1],adaptive,prompt,seed);selected=set(chosen);self.plan={'skip':chosen,'decision':decision}
        hidden=self.model.model.norm(h);hidden=hidden[:,-head_last:,:] if head_last else hidden;logits=self.head(hidden)
        self.forwards.append({'query_tokens':ids.shape[1],'context_tokens':context,'executed':executed,'skipped':sorted(selected),'head_positions':logits.shape[1]})
        if capture:mx.eval(logits,captured)
        else:mx.eval(logits)
        return logits,captured

    def prompt_ids(self,question):
        return self.tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':question}],tokenize=True,add_generation_prompt=True)

    def initial_skip(self,method,prompt,seed):
        if method=='baseline':return []
        if method=='static':return policy.fixed_layers(self.n)
        if method=='random':return policy.random_layers(self.pool,policy.MAX_SKIP,prompt,seed)
        if method=='bi_static':return self.profile['bi_static']
        if method in ('adaptive_sketch','adaptive_leverage','random_match_sketch','random_match_leverage'):return []
        raise ValueError('Unknown ALSR method')

    def generate(self,question,method,seed,max_tokens=16,fixed_tokens=None):
        self.reset();mx.synchronize();mx.reset_peak_memory();start,cpu=time.perf_counter(),time.process_time();ids=self.prompt_ids(question);cache=make_prompt_cache(self.model)
        skip=self.initial_skip(method,question,seed);adaptive=method if 'sketch' in method or 'leverage' in method else None
        logits,_=self.forward(mx.array([ids]),cache,skip,adaptive,question,seed);skip=self.plan['skip'] if self.plan else skip
        generated=[];eos=set(getattr(self.tokenizer,'eos_token_ids',[]) or [self.tokenizer.eos_token_id]);next_id=int(mx.argmax(logits[0,-1]).item())
        for j in range(max_tokens):
            token=int(fixed_tokens[j]) if fixed_tokens is not None else next_id;generated.append(token)
            if fixed_tokens is None and token in eos:break
            if j+1<max_tokens:
                logits,_=self.forward(mx.array([[token]]),cache,skip);next_id=int(mx.argmax(logits[0,-1]).item())
        raw_text=self.tokenizer.decode(generated);text=self.tokenizer.decode(generated,skip_special_tokens=True);mx.synchronize();wall=time.perf_counter()-start;host_cpu=time.process_time()-cpu
        offsets=[c.offset for c in cache];expected=len(ids)+len(generated)-1
        assert all(v==(0 if i in skip else expected) for i,v in enumerate(offsets))
        assert all(c==(0 if i in skip else len(self.forwards)) for i,c in enumerate(self.counts))
        assert self.counts==self.attention_counts==self.mlp_counts
        return {'answer':text,'raw_answer':raw_text,'token_ids':generated,'input_tokens':len(ids),'output_tokens':len(generated),'wall_seconds':wall,'cpu_seconds':host_cpu,
          'gpu_seconds':None,'gpu_time_status':'unmeasured; synchronized wall includes GPU and host',
          'mlx_peak_bytes':mx.get_peak_memory(),'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory(),
          'process_lifetime_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'gate_wall_seconds':self.gate_wall,'gate_cpu_seconds':self.gate_cpu,'skip_layers':skip,'layer_calls':self.counts[:],
          'attention_calls':self.attention_counts[:],'mlp_calls':self.mlp_counts[:],'kv_offsets':offsets,'forwards':self.forwards[:],
          'executed_layer_token_evaluations':sum(f['query_tokens']*len(f['executed']) for f in self.forwards),
          'skipped_layer_token_evaluations':sum(f['query_tokens']*len(f['skipped']) for f in self.forwards),
          'analytical_matmul_flops':policy.estimated_matmul_flops(self.config,self.forwards,self.gate_flops),'gate_flops_estimate':self.gate_flops,
          'gate_decision':self.plan['decision'] if self.plan else None,'fixed_work':fixed_tokens is not None}

    def nll(self,question,reference,skip):
        self.reset();ids=self.prompt_ids(question);ref=self.tokenizer.encode(reference,add_special_tokens=False)
        start,cpu=time.perf_counter(),time.process_time();logits,_=self.forward(mx.array([ids+ref[:-1]]),skip=skip,head_last=len(ref));values=logits[0].astype(mx.float32);logp=values-mx.logsumexp(values,axis=-1,keepdims=True)
        losses=-mx.take_along_axis(logp,mx.array(ref)[:,None],axis=-1).squeeze(-1);mx.eval(losses);mx.synchronize()
        return {'reference_nll':float(mx.mean(losses).item()),'reference_tokens':len(ref),'quality_eval_wall_seconds':time.perf_counter()-start,'quality_eval_cpu_seconds':time.process_time()-cpu,
                'quality_eval_gate_calls':int(self.gate_wall>0),'quality_eval_skip_layers':list(skip),
                'scope':'Teacher-forced reference evaluation only, prompt-derived mask frozen; excluded from service latency.'}

    def check_native(self):
        ids=mx.array([self.prompt_ids('Return only the letter A.')]);native=self.model(ids);mx.eval(native);self.reset();manual,_=self.forward(ids,head_last=0)
        difference=float(mx.max(mx.abs(native.astype(mx.float32)-manual.astype(mx.float32))).item())
        if difference>1e-5:raise RuntimeError('Manual baseline differs from installed native Qwen forward')
        self.reset();last,_=self.forward(ids,head_last=1)
        return {'native_full_logits_max_abs_difference':difference,'native_full_top1_equal':int(mx.argmax(native[0,-1]).item())==int(mx.argmax(manual[0,-1]).item()),
                'last_head_max_abs_difference':float(mx.max(mx.abs(native[0,-1].astype(mx.float32)-last[0,-1].astype(mx.float32))).item()),
                'last_head_top1_equal':int(mx.argmax(native[0,-1]).item())==int(mx.argmax(last[0,-1]).item())}

    def check_skip_trap(self):
        index=self.pool[0];saved=self.layers[index];ids=mx.array([self.prompt_ids('Return only A.')])
        class Trap:
            def __call__(self,*a,**k):raise AssertionError('Trap confirms actual block invocation')
        self.layers[index]=Trap();positive=False
        try:
            self.reset();self.forward(ids,skip=[index]);self.reset()
            try:self.forward(ids)
            except AssertionError:positive=True
            if not positive:raise RuntimeError('Skip trap positive control did not execute')
        finally:self.layers[index]=saved
        return {'skipped_block_trap_passed':True,'non_skipped_positive_control_raised':positive,'trap_layer':index}

def kl_effect(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);la=a-np.logaddexp.reduce(a);lb=b-np.logaddexp.reduce(b)
    kl=float(np.sum(np.exp(la)*(la-lb)));flip=int(np.argmax(a)!=np.argmax(b))
    return {'kl':max(0,kl),'top1_changed':flip,'risk':max(0,kl)+flip}

def calibrate(engine,training,validation):
    start,cpu=time.perf_counter(),time.process_time();model=engine.model;records=[];captures=[];full_logits=[];top_ids=[];prefill_passes=0
    for task in training:
        ids=mx.array([engine.prompt_ids(task['question'])]);engine.reset();logits,capture=engine.forward(ids,capture=True);prefill_passes+=1;arr=logits[0,-1].astype(mx.float32).tolist();full_logits.append(arr)
        top_ids.extend(sorted(range(len(arr)),key=lambda i:arr[i],reverse=True)[:2]);captures.append({k:({kk:vv.tolist() if vv.ndim else float(vv.item()) for kk,vv in v.items()} if isinstance(v,dict) else v.tolist()) for k,v in capture.items()})
    shortlist=list(dict.fromkeys(top_ids))[:32]
    for c in 'ABCD':
        token=engine.tokenizer.encode(c,add_special_tokens=False)[0]
        if token not in shortlist:shortlist.append(token)
    weight=model.model.embed_tokens.weight if engine.config.get('tie_word_embeddings',True) else model.lm_head.weight
    probe_weight=weight[mx.array(shortlist)].astype(mx.float32);mx.eval(probe_weight)
    xs=[];train_rms=[];pairs=[];margins=[]
    for task,cap in zip(training,captures):
        h=mx.array(cap['early']);sketch=h@engine.projection;norm=engine.model.model.norm(h[None,None,:]).astype(mx.float32)[0,0];scores=(norm@probe_weight.T).tolist();order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True);margin=max(.05,scores[order[0]]-scores[order[1]]);rms=float(mx.sqrt(mx.mean(h**2)).item())
        xs.append(sketch.tolist()+[math.log1p(len(engine.prompt_ids(task['question']))),rms,margin]);train_rms.append(rms);pairs.append(order[:2]);margins.append(margin)
    risks={str(l):[] for l in engine.pool};influence={str(l):np.mean([c[str(l)]['influence'] for c in captures]) for l in engine.pool}
    for j,task in enumerate(training):
        ids=mx.array([engine.prompt_ids(task['question'])])
        for l in engine.pool:
            engine.reset();logits,_=engine.forward(ids,skip=[l]);prefill_passes+=1;effect=kl_effect(full_logits[j],logits[0,-1].astype(mx.float32).tolist());risks[str(l)].append(effect['risk']);records.append({'phase':'training','task_id':task['task_id'],'layer':l,**effect})
    ridge={l:policy.fit_ridge(xs,ys) for l,ys in risks.items()};leverage={};avg_rms=float(np.mean(train_rms))
    for l in engine.pool:
        mean=np.mean([np.asarray(c[str(l)]['residual'])/max(c[str(l)]['rms'],1e-5) for c in captures],axis=0)
        sensitivity=(mx.array(mean,dtype=mx.float32)@probe_weight.T).tolist();raw=[abs(sensitivity[a]-sensitivity[b])/margin*(rms/avg_rms) for (a,b),margin,rms in zip(pairs,margins,train_rms)]
        scale=float(np.median([(y+.001)/(x+.001) for x,y in zip(raw,risks[str(l)])]));leverage[str(l)]={'sensitivity':sensitivity,'scale':scale}
    profile={'shortlist':shortlist,'ridge':ridge,'leverage':leverage,'train_probe_rms':avg_rms,'thresholds':{'sketch':0.,'leverage':0.},
             'bi_static':sorted(sorted(engine.pool,key=lambda l:influence[str(l)])[:policy.MAX_SKIP]),'block_influence':influence,'pool':engine.pool,'max_skip':policy.MAX_SKIP}
    engine.prepare_profile(profile);trials=[]
    for family in ('sketch','leverage'):
        baseline={}
        for task in validation:
            engine.reset();logits,_=engine.forward(mx.array([engine.prompt_ids(task['question'])]));prefill_passes+=1;baseline[task['task_id']]=logits[0,-1].astype(mx.float32).tolist()
        options=[]
        for threshold in (0.,.05,.15,.5,2.):
            profile['thresholds'][family]=threshold;effects=[];counts=[]
            for task in validation:
                engine.reset();logits,_=engine.forward(mx.array([engine.prompt_ids(task['question'])]),adaptive='adaptive_'+family,prompt=task['question']);prefill_passes+=1
                effect=kl_effect(baseline[task['task_id']],logits[0,-1].astype(mx.float32).tolist());effects.append(effect);counts.append(len(engine.plan['skip']))
                records.append({'phase':'calibration','family':family,'threshold':threshold,'task_id':task['task_id'],'skip_layers':engine.plan['skip'],**effect})
            valid=all(not e['top1_changed'] for e in effects) and np.mean([e['kl'] for e in effects])<=.05
            trial={'family':family,'threshold':threshold,'mean_skips':float(np.mean(counts)),'mean_kl':float(np.mean([e['kl'] for e in effects])),'changed_top1':sum(e['top1_changed'] for e in effects),'eligible':bool(valid)};trials.append(trial)
            if valid:options.append((trial['mean_skips'],-threshold,threshold))
        profile['thresholds'][family]=max(options)[2] if options else 0.
    profile['calibration_trials']=trials;engine.prepare_profile(profile)
    return profile,records,{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'prefill_forward_passes':prefill_passes,'gpu_seconds':None,'scope':'Offline train/threshold calibration; no test labels or model weight training.'}
