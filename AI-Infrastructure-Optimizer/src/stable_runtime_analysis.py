"""Audit completed checkpoints and compare paired quality/real time, without tuning."""
import hashlib,json,random,re,statistics,sys
from pathlib import Path
from .stable_runtime_records import Store
from .stable_runtime_protocol import VARIANTS,option_complete
from .alsr_impact_scoring import grade
from .alsr_policy import estimated_matmul_flops,choose_by_risk
from .alsr_analysis import paired_ratio
from .controlled_statistics import describe,accuracy_comparison
from .adaptive_experiment import write_json
from .alsr_io import write_union
from .efficiency_repeat import frozen_check
from .solver300_dataset import task_key

METRICS=('wall_seconds','cpu_seconds','model_step_wall_seconds','model_step_cpu_seconds','decision_wall_inclusive_seconds','decision_cpu_seconds','model_step_except_gate_wall_seconds','tokenize_wall_seconds','cache_setup_wall_seconds','reset_wall_seconds','decode_wall_seconds','format_check_wall_seconds','integrity_check_wall_seconds','host_bookkeeping_wall_seconds','argmax_wait_wall_seconds','output_tokens','input_tokens','analytical_matmul_flops','host_created_input_tensors','scalar_token_reads')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def jobkey(j):return '/'.join(str(j[k]) for k in ('phase','dataset','repetition','task_id','variant'))

def analyze(folder):
    p=Path(folder).resolve();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for path,h in m['runtime_source_sha256'].items():assert sha(path)==h
    rows=Store(p/'checkpoints').rows();index={r['_key']:r for r in rows};assert len(index)==len(rows)
    assert all(jobkey(j) in index for j in m['schedule']);smoke=index['smoke'];assert smoke['native_skip_trap_positive_control'] and smoke['checks']['native_full_logits_max_abs_difference']==0
    for r in smoke['parity']:assert r['legacy']['token_ids']==r['lean']['token_ids'] and r['legacy']['first_logits_sha256']==r['lean']['first_logits_sha256']
    datasets=m['datasets'];tasks={t['task_id']:t for ts in datasets.values() for t in ts};ids={name:{t['task_id'] for t in ts} for name,ts in datasets.items()};assert len(ids['fresh64'])==len(ids['replay64'])==64 and not ids['fresh64']&ids['replay64']
    keys={name:{task_key(t['question']) for t in ts} for name,ts in datasets.items()};assert not keys['fresh64']&keys['replay64'];past=set();root=Path(__file__).resolve().parents[1]
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):past.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    import csv
    for path in m['preserved_sha256']:
        f=Path(path)
        if not f.is_relative_to(root/'data'):continue
        if f.suffix=='.json':walk(json.loads(f.read_text()))
        elif f.suffix=='.csv':
            with f.open() as stream:
                for r in csv.DictReader(stream):walk(r)
    assert not keys['fresh64']&past
    sketch=json.loads((root/'data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_m1_profile.json').read_text());config=m['model']['config'];ri={};allprimary=[]
    for r in rows:
        if r.get('phase') not in ('natural','fixed','profile'):continue
        ds,rep,id,v,phase=(r[k] for k in ('dataset','repetition','task_id','variant','phase'));assert id in ids[ds] and v in VARIANTS and rep in (1,2,3)
        key=(phase,ds,rep,id,v);assert key not in ri;ri[key]=r
        c=VARIANTS[v];skip=r['skip_layers'];calls=r['layer_calls'];nt=r['output_tokens'];expected=[0 if l in skip else nt for l in range(28)];assert calls==expected
        assert r['kv_offsets']==[0 if l in skip else r['input_tokens']+nt-1 for l in range(28)]
        if r['observed_attention_calls'] is not None:assert calls==r['observed_attention_calls']==r['observed_mlp_calls']
        if c['method']=='full':assert skip==[]
        elif c['method']=='stable1':assert skip==[24]
        else:
            d=r['gate_decision'];assert d['threshold']==sketch['thresholds']['sketch'];assert skip==choose_by_risk({int(k):v for k,v in d['risks'].items()},d['threshold'],sketch['pool'])
        assert len(r['token_ids'])==nt==len(r['work_shapes']);fs=[]
        for j,(q,s,n) in enumerate(r['work_shapes']):
            assert q==(r['input_tokens'] if j==0 else 1) and s==r['input_tokens']+j and n==28-len(skip)
            fs.append(dict(query_tokens=q,context_tokens=s,executed=list(range(n)),head_positions=1))
        assert estimated_matmul_flops(config,fs,r['gate_flops_estimate'])==r['analytical_matmul_flops']
        assert r['gpu_seconds'] is None and r['host_bookkeeping_wall_seconds']>=-1e-5
        if phase=='natural':
            assert r['grade']==grade(tasks[id],r['answer'])
            if r['completion_reason']=='request_option_complete':assert c['format_stop'] and option_complete(tasks[id]['question'],r['answer'])
            assert not r['native_audit_enabled'];allprimary.append(r)
        else:
            prepared=index['/'.join(map(str,('prepare',ds,0,id,'full')))];assert r['token_ids']==prepared['fixed_sequence']
            assert r['native_audit_enabled']==(phase=='profile')
    assert len(allprimary)==128*9*3
    parity=[];stopparity=[];stopchanges=[];transitions=[];failures=[];categories=[];repeats=[];summaries=[];comparisons=[];attribution=[]
    original=read_lines(Path(m['source'])/'GENERATION.jsonl');oi={(r['method'],r['task_id'],r['repetition']):r for r in original};originaldiff=0
    for ds,questions in ids.items():
        for rep in (1,2,3):
            for id in questions:
                for method in ('full','stable1','sketch'):
                    a=ri['natural',ds,rep,id,'legacy_'+method];b=ri['natural',ds,rep,id,'lean_'+method];parity.append(dict(dataset=ds,repetition=rep,task_id=id,method=method,tokens_equal=a['token_ids']==b['token_ids'],masks_equal=a['skip_layers']==b['skip_layers'],outputs_equal=a['answer']==b['answer']))
                    z=ri['natural',ds,rep,id,'lean_option_'+method];stopparity.append(dict(dataset=ds,repetition=rep,task_id=id,method=method,completed=z['completion_reason']=='request_option_complete',generated_prefix_equal=z['token_ids']==b['token_ids'][:len(z['token_ids'])],masks_equal=z['skip_layers']==b['skip_layers']))
                    stopchanges.append(dict(dataset=ds,repetition=rep,task_id=id,method=method,baseline_correct=b['grade']['correct'],option_correct=z['grade']['correct'],baseline_answer=b['answer'],option_answer=z['answer'],baseline_format=b['grade']['format_ok'],option_format=z['grade']['format_ok'],completion_reason=z['completion_reason'],possible_word_prefix=z['completion_reason']=='request_option_complete' and bool(re.match(r'^\s*[A-D][a-z]',b['answer'])),reference=tasks[id]['reference'],question=tasks[id]['question']))
                    if ds=='replay64':
                        oldmethod={'full':'baseline','sketch':'adaptive_sketch','stable1':'stable1'}[method];originaldiff+=a['token_ids']!=oi[oldmethod,id,rep]['token_ids']
                for v in VARIANTS:
                    r=ri['natural',ds,rep,id,v];c=VARIANTS[v];control='lean_option_full' if c['format_stop'] else c['backend']+'_full';b=ri['natural',ds,rep,id,control];bc,rc=b['grade']['correct'],r['grade']['correct']
                    tr=dict(dataset=ds,repetition=rep,task_id=id,variant=v,control=control,category=r['category'],transition=('Correct' if bc else 'Wrong')+' -> '+('Correct' if rc else 'Wrong'),baseline_answer=b['answer'],answer=r['answer'],baseline_correct=bc,correct=rc,format_ok=r['grade']['format_ok'],baseline_format_ok=b['grade']['format_ok'],skip_layers=r['skip_layers'],baseline_tokens=b['output_tokens'],output_tokens=r['output_tokens'],baseline_wall=b['wall_seconds'],wall_seconds=r['wall_seconds']);transitions.append(tr)
                    if rep==1 and (bc!=rc or (b['grade']['format_ok'] and not r['grade']['format_ok'])):failures.append(dict(tr,question=tasks[id]['question'],reference=tasks[id]['reference'],error_type=r['grade']['error_type']))
        for v in VARIANTS:
            for rep in (1,2,3):
                rr=[ri['natural',ds,rep,id,v] for id in questions];ff=[ri['fixed',ds,rep,id,v] for id in m['fixed_ids'][ds]] if not VARIANTS[v]['format_stop'] else []
                row=dict(dataset=ds,variant=v,repetition=rep,task_success=statistics.mean(r['grade']['correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),mcq_accuracy=statistics.mean(r['grade']['correct'] for r in rr if r['score_kind']=='choice'),mean_skipped_layers=statistics.mean(len(r['skip_layers']) for r in rr),mean_mlx_peak_bytes=statistics.mean(r['mlx_peak_bytes'] for r in rr),option_completions=sum(r['completion_reason']=='request_option_complete' for r in rr))
                row['transformer_block_invocations']=sum(sum(r['layer_calls']) for r in rr);row['layer_token_evaluations']=sum(q*n for r in rr for q,s,n in r['work_shapes'])
                mcq=[r for r in rr if r['score_kind']=='choice'];firstlabels=[re.match(r'^\s*([A-D])(?=$|[).:\s])',r['answer']) for r in mcq];row['initial_choice_label_accuracy']=sum(bool(match) and match.group(1)==tasks[r['task_id']]['reference'] for match,r in zip(firstlabels,mcq))/len(mcq)
                for metric in METRICS:row[metric]=sum(r[metric] for r in rr)
                for metric in ('wall_seconds','cpu_seconds','analytical_matmul_flops','output_tokens'):row['fixed_'+metric]=sum(r[metric] for r in ff) if ff else None
                repeats.append(row)
            rr=[r for r in repeats if r['dataset']==ds and r['variant']==v];summary=dict(dataset=ds,variant=v)
            for metric in rr[0]:
                if metric in ('dataset','variant','repetition'):continue
                for stat,value in describe([r[metric] for r in rr],draws=2000).items():summary[metric+'_'+stat]=value
            summaries.append(summary)
            for cat in sorted({tasks[id]['category'] for id in questions}):
                kk=[r for r in allprimary if r['dataset']==ds and r['variant']==v and r['category']==cat];categories.append(dict(dataset=ds,variant=v,category=cat,n_unique=len({r['task_id'] for r in kk}),task_success=statistics.mean(r['grade']['correct'] for r in kk),format_compliance=statistics.mean(r['grade']['format_ok'] for r in kk),wall_seconds_per_repeat=sum(r['wall_seconds'] for r in kk)/3,output_tokens_per_repeat=sum(r['output_tokens'] for r in kk)/3))
        pairs=[('legacy_stable1','legacy_full'),('legacy_sketch','legacy_full'),('lean_full','legacy_full'),('lean_stable1','legacy_stable1'),('lean_sketch','legacy_sketch'),('lean_stable1','lean_full'),('lean_sketch','lean_full'),('lean_stable1','legacy_full'),('lean_option_full','lean_full'),('lean_option_stable1','lean_option_full'),('lean_option_sketch','lean_option_full'),('lean_option_stable1','lean_stable1')]
        for target,control in pairs:
            difference=[[int(ri['natural',ds,rep,id,target]['grade']['correct'])-int(ri['natural',ds,rep,id,control]['grade']['correct']) for rep in (1,2,3)] for id in sorted(questions)];fmt=[[int(ri['natural',ds,rep,id,target]['grade']['format_ok'])-int(ri['natural',ds,rep,id,control]['grade']['format_ok']) for rep in (1,2,3)] for id in sorted(questions)];row=dict(dataset=ds,target=target,control=control,**accuracy_comparison(difference,draws=3000));row.update({'format_'+k:v for k,v in accuracy_comparison(fmt,draws=3000).items() if k!='inference_scope'})
            for phase,qq in [('natural',questions),('fixed',set(m['fixed_ids'][ds]))]:
                if phase=='fixed' and (VARIANTS[target]['format_stop'] or VARIANTS[control]['format_stop']):continue
                for metric in ('wall_seconds','cpu_seconds','analytical_matmul_flops'):
                    a={id:{rep:ri[phase,ds,rep,id,target][metric] for rep in (1,2,3)} for id in sorted(qq)};b={id:{rep:ri[phase,ds,rep,id,control][metric] for rep in (1,2,3)} for id in sorted(qq)}
                    for k,v in paired_ratio(a,b,draws=2500).items():
                        if k!='scope':row[phase+'_'+metric+'_'+k]=v
            comparisons.append(row)
        # Within-question observed length/time relation, not a causal regression.
        for backend in ('legacy','lean','lean_option'):
            delta_t=[];delta_w=[]
            for id in sorted(questions):
                delta_t.append(statistics.mean(ri['natural',ds,rep,id,backend+'_stable1']['output_tokens']-ri['natural',ds,rep,id,backend+'_full']['output_tokens'] for rep in (1,2,3)))
                delta_w.append(statistics.mean(ri['natural',ds,rep,id,backend+'_stable1']['wall_seconds']-ri['natural',ds,rep,id,backend+'_full']['wall_seconds'] for rep in (1,2,3)))
            xm,ym=statistics.mean(delta_t),statistics.mean(delta_w);xx=sum((x-xm)**2 for x in delta_t);yy=sum((y-ym)**2 for y in delta_w);xy=sum((x-xm)*(y-ym) for x,y in zip(delta_t,delta_w));attribution.append(dict(dataset=ds,backend=backend,token_delta=sum(delta_t),wall_delta_per_repeat=sum(delta_w),slope_seconds_per_output_token=xy/xx if xx else None,correlation=xy/(xx*yy)**.5 if xx and yy else None,warning='Post-hoc output-length association; not pure Python/GPU causal decomposition.'))
    profile=[];profile_hash_disagreements=0
    for ds in ids:
        for id in m['profile_ids'][ds]:
            for rep in (1,2,3):
                for method in ('full','stable1','sketch'):profile_hash_disagreements+=ri['profile',ds,rep,id,'legacy_'+method]['first_logits_sha256']!=ri['profile',ds,rep,id,'lean_'+method]['first_logits_sha256']
    for ds in ids:
        for v in VARIANTS:
            rr=[r for r in rows if r.get('phase')=='profile' and r['dataset']==ds and r['variant']==v]
            if not rr:continue
            profile.append(dict(dataset=ds,variant=v,profiled_requests=len(rr),actual_native_attention_mlp_counts_verified=all(r['observed_attention_calls']==r['observed_mlp_calls']==r['layer_calls'] for r in rr),mean_forward_plus_argmax_wall=statistics.mean(r['model_step_wall_seconds'] for r in rr),mean_argmax_wait_wall=statistics.mean(r['argmax_wait_wall_seconds'] for r in rr),mean_native_dispatch_wall=statistics.mean(r['detailed_native_dispatch_wall_seconds'] for r in rr) if VARIANTS[v]['backend']=='lean' else None,scope='Heavy instrumentation and first-logit CPU hash; excluded from primary latency. Wait includes queued model work, not GPU elapsed.'))
    verdict=[]
    for c in comparisons:
        if (c['target'],c['control']) not in [('lean_stable1','lean_full'),('lean_option_stable1','lean_option_full'),('lean_option_full','lean_full'),('lean_stable1','legacy_stable1')]:continue
        aa=next(r for r in summaries if r['dataset']==c['dataset'] and r['variant']==c['target']);bb=next(r for r in summaries if r['dataset']==c['dataset'] and r['variant']==c['control']);quality=c['accuracy_delta']>=0 and c['format_accuracy_delta']>=0;speed=c['natural_wall_seconds_ci95_high']<1;compute=c['natural_analytical_matmul_flops_ratio']<1
        verdict.append(dict(dataset=c['dataset'],target=c['target'],control=c['control'],quality_point_not_worse=quality,paired_latency_improvement=speed,estimated_natural_compute_reduction=compute,observed_efficiency_candidate=quality and speed and compute,strict_population_noninferiority_proven=c['conservative_exact_lower95']>=0,warning='Implementation vs same policy does not newly remove transformer operations; output-stop factor separate from layer skipping.'))
    out=p/('ANALYSIS_'+__import__('uuid').uuid4().hex);out.mkdir()
    for name,rr in {'RESULTS':summaries,'REPEATS':repeats,'COMPARISONS':comparisons,'TRANSITIONS':transitions,'FAILURES':failures,'CATEGORIES':categories,'PARITY':parity,'STOP_PREFIX_PARITY':stopparity,'STOP_TRANSITIONS':stopchanges,'LENGTH_ATTRIBUTION':attribution,'PROFILE':profile,'VERDICT':verdict}.items():write_union(out/(name+'.csv'),rr)
    audit={'existing_files_verified':len(m['preserved_sha256']),'fresh_exact_stem_past_overlap':0,'replay_and_fresh_disjoint':True,'model_policy_sources_frozen':True,'no_training_or_threshold_tuning':True,'actual_layer_counts_and_cache_verified':True,'actual_attention_mlp_audit_verified':all(r['actual_native_attention_mlp_counts_verified'] for r in profile),'native_logits_smoke_exact':True,'legacy_lean_token_disagreements':sum(not r['tokens_equal'] for r in parity),'legacy_lean_mask_disagreements':sum(not r['masks_equal'] for r in parity),'option_prefix_disagreements':sum(not r['generated_prefix_equal'] for r in stopparity),'replay_vs_previous_experiment_token_disagreements':originaldiff,'primary_requests':len(allprimary),'fixed_requests':sum(r.get('phase')=='fixed' for r in rows),'heavy_profile_requests':sum(r.get('phase')=='profile' for r in rows),'checkpoint_complete_records':len(rows),'GPU_elapsed_power_money':'unmeasured','FLOPs':'analytical dense matmul estimate; physical copy bandwidth unmeasured'}
    audit.update(profile_first_logit_hash_disagreements=profile_hash_disagreements,option_possible_word_prefix_observations=sum(r['possible_word_prefix'] for r in stopchanges))
    write_json(out/'AUDIT.json',audit);write_json(out/'SOURCE.json',{'experiment':str(p),'manifest_sha256':sha(p/'MANIFEST.json')});print(json.dumps({'output':str(out),'audit':audit,'candidates':[r for r in verdict if r['observed_efficiency_candidate']]},indent=2));return out

if __name__=='__main__':analyze(sys.argv[1])
