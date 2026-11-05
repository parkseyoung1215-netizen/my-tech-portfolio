"""Offline audit and question-cluster uncertainty for the actual MLX pilot."""
import hashlib,json,random,statistics,sys,math
from collections import defaultdict,Counter
from pathlib import Path
from difflib import SequenceMatcher
from . import alsr_policy as policy
from .alsr_dataset import grade
from .solver300_dataset import task_key
from .efficiency_repeat import frozen_check
from .controlled_analysis import read_decoded
from .controlled_statistics import describe,accuracy_comparison,quantile
from .tool_experiment import write_records
from .adaptive_experiment import write_json

def paired_ratio(target,baseline,draws=4000):
    ids=list(target);reps=sorted(target[ids[0]]);rng=random.Random(policy.SEED);boot=[]
    for _ in range(draws):
        ii=[rng.choice(ids) for _ in ids];rr=[rng.choice(reps) for _ in reps]
        boot.append(sum(target[i][r] for i in ii for r in rr)/sum(baseline[i][r] for i in ii for r in rr))
    return {'ratio':sum(v for vv in target.values() for v in vv.values())/sum(v for vv in baseline.values() for v in vv.values()),'ci95_low':quantile(boot,.025),'ci95_high':quantile(boot,.975),'scope':'Paired question and repetition resampling; same inputs kept together.'}

def analyze(prefix,output_dir=None):
    p=Path(prefix).resolve()
    def read(s):return read_decoded(f'{p}_{s}.csv')
    manifest=json.loads(Path(f'{p}_manifest.json').read_text());meta=json.loads(Path(f'{p}_metadata.json').read_text());profiles=json.loads(Path(f'{p}_profiles.json').read_text());frozen_check(manifest)
    assert hashlib.sha256(Path(manifest['setup_path']).read_bytes()).hexdigest()==manifest['setup_sha256']
    models=meta['models'];reps=list(range(1,meta['repeats']+1));tasks={t['task_id']:t for t in manifest['splits']['pilot']};fixed_ids=set(manifest['fixed_work_ids'])
    assert len(tasks)==32 and len(fixed_ids)==8
    splitkeys=[{task_key(t['question']) for t in tt} for tt in manifest['splits'].values()]
    assert all(not a&b for i,a in enumerate(splitkeys) for b in splitkeys[i+1:])
    past=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):past.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    for name in manifest['preserved_sha256']:
        path=Path(name)
        if not path.is_relative_to(p.parent):continue
        if path.suffix=='.json':walk(json.loads(path.read_text()))
        elif path.suffix=='.csv':
            for r in read_decoded(path):walk(r)
    assert not set.union(*splitkeys)&past
    rows,fixed,quality,cal=read('generation'),read('fixed_work'),read('quality'),read(meta.get('calibration_trace_suffix','calibration'));ri={};fi={}
    if meta.get('recovered_after_export_error'):
        for name,h in meta['original_trace_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
        assert all(r['recomputed_profile_exactly_matches_original'] for r in read('profile_recheck_status'))
    info={m['repo']:m for m in manifest['models']};calids={t['task_id'] for t in manifest['splits']['calibration']};trainids={t['task_id'] for t in manifest['splits']['training']}
    for mi,m in enumerate(models):
        profile=profiles[m];path=Path(f'{p}_m{mi}_profile.json');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['profile_hashes'][str(mi)] and json.loads(path.read_text())==profile
        n=info[m]['config']['num_hidden_layers'];assert profile['pool']==policy.pool_for(n) and profile['bi_static']==sorted(sorted(profile['pool'],key=lambda l:profile['block_influence'][str(l)])[:policy.MAX_SKIP])
        for r in cal:
            if r['model']!=m:continue
            assert r['task_id'] in (trainids if r['phase']=='training' else calids)
        for family in ('sketch','leverage'):
            trials=[]
            for trial in [r for r in profile['calibration_trials'] if r['family']==family]:
                rr=[r for r in cal if r['model']==m and r['phase']=='calibration' and r['family']==family and r['threshold']==trial['threshold']]
                assert {r['task_id'] for r in rr}==calids
                assert trial['changed_top1']==sum(r['top1_changed'] for r in rr) and abs(trial['mean_kl']-statistics.mean(r['kl'] for r in rr))<1e-12
                assert trial['mean_skips']==statistics.mean(len(r['skip_layers']) for r in rr)
                valid=not any(r['top1_changed'] for r in rr) and trial['mean_kl']<=.05;assert valid==trial['eligible']
                if valid:trials.append((trial['mean_skips'],-trial['threshold'],trial['threshold']))
            assert profile['thresholds'][family]==(max(trials)[2] if trials else 0.)
    for mode,rr,index in [('natural',rows,ri),('fixed',fixed,fi)]:
        for r in rr:
            m,rep,method,id=r['model'],r['repetition'],r['method'],r['task_id'];key=(m,rep,method,id);assert key not in index;index[key]=r
            assert id in tasks and method in policy.METHODS and rep in reps
            n=info[m]['config']['num_hidden_layers'];skip=r['skip_layers'];profile=profiles[m];forwards=r['forwards'];fcount=len(forwards)
            assert skip==sorted(set(skip)) and len(skip)<=4 and set(skip)<=set(profile['pool'])
            if method=='baseline':expected=[]
            elif method=='static':expected=policy.fixed_layers(n)
            elif method=='bi_static':expected=profile['bi_static']
            elif method=='random':expected=policy.random_layers(profile['pool'],4,tasks[id]['question'],policy.SEED+rep)
            else:
                d=r['gate_decision'];family='sketch' if method.endswith('sketch') else 'leverage';assert d['family']==family and d['threshold']==profile['thresholds'][family]
                risks={int(k):v for k,v in d['risks'].items()};expected=policy.choose_by_risk(risks,d['threshold'],profile['pool'])
                if method.startswith('random_match'):expected=policy.random_layers(profile['pool'],len(expected),tasks[id]['question'],policy.SEED+rep)
            assert skip==expected
            assert len(r['token_ids'])==r['output_tokens']==fcount
            assert r['layer_calls']==r['attention_calls']==r['mlp_calls']==[0 if l in skip else fcount for l in range(n)]
            assert r['kv_offsets']==[0 if l in skip else r['input_tokens']+fcount-1 for l in range(n)]
            for j,f in enumerate(forwards):
                assert f['skipped']==skip and f['executed']==[l for l in range(n) if l not in skip]
                assert f['query_tokens']==(r['input_tokens'] if j==0 else 1) and f['context_tokens']==r['input_tokens']+j
            assert r['executed_layer_token_evaluations']==sum(f['query_tokens']*len(f['executed']) for f in forwards)
            assert r['skipped_layer_token_evaluations']==sum(f['query_tokens']*len(skip) for f in forwards)
            assert r['analytical_matmul_flops']==policy.estimated_matmul_flops(info[m]['config'],forwards,r['gate_flops_estimate'])
            assert r['gpu_seconds'] is None and r['mlx_peak_bytes']>=r['mlx_active_bytes']
            if mode=='natural':assert r['grade']==grade(tasks[id],r['answer'])
            else:assert id in fixed_ids and r['output_tokens']==16
    assert set(ri)=={(m,rep,method,id) for m in models for rep in reps for method in policy.METHODS for id in tasks}
    assert set(fi)=={(m,rep,method,id) for m in models for rep in reps for method in policy.METHODS for id in fixed_ids}
    for key,r in fi.items():
        assert r['skip_layers']==ri[key]['skip_layers']
        mi=models.index(key[0]);sequences=json.loads(Path(f'{p}_m{mi}_fixed_sequences.json').read_text())
        assert r['token_ids']==sequences[key[3]]
    for m in models:
        for rep in reps:
            for id in tasks:
                for family in ('sketch','leverage'):assert len(ri[m,rep,'adaptive_'+family,id]['skip_layers'])==len(ri[m,rep,'random_match_'+family,id]['skip_layers'])
    qi={(r['model'],r['method'],r['task_id']):r for r in quality};assert len(qi)==32*len(models)*len(policy.METHODS)
    for key,r in qi.items():assert not r['quality_eval_gate_calls'] and r['quality_eval_skip_layers']==ri[key[0],1,key[1],key[2]]['skip_layers']
    smokes=read(meta.get('native_parity_suffix','native_parity'));assert all(r['native_full_logits_max_abs_difference']==0 and r['native_full_top1_equal'] and r['last_head_top1_equal'] and r['skipped_block_trap_passed'] and r['non_skipped_positive_control_raised'] for r in smokes)
    summaries=[]
    for m in models:
        for rep in reps:
            for method in policy.METHODS:
                rr=[ri[m,rep,method,id] for id in tasks];ff=[fi[m,rep,method,id] for id in fixed_ids];qq=[qi[m,method,id] for id in tasks]
                summaries.append({'model':m,'repetition':rep,'method':method,'requests':len(rr),'correct':sum(r['grade']['correct'] for r in rr),'accuracy':statistics.mean(r['grade']['correct'] for r in rr),
                   'mcq_accuracy':statistics.mean(r['grade']['correct'] for r in rr if r['score_kind']=='choice'),
                   'json_exact':statistics.mean(r['grade']['correct'] for r in rr if r['score_kind']=='json'),
                   'text_constraint_rate':statistics.mean(r['grade']['correct'] for r in rr if r['score_kind']=='constraints'),
                   'format_compliance':statistics.mean(r['grade']['format_ok'] for r in rr),'natural_wall_seconds':sum(r['wall_seconds'] for r in rr),'natural_cpu_seconds':sum(r['cpu_seconds'] for r in rr),
                   'output_tokens':sum(r['output_tokens'] for r in rr),'input_tokens':sum(r['input_tokens'] for r in rr),'gate_wall_seconds':sum(r['gate_wall_seconds'] for r in rr),'gate_cpu_seconds':sum(r['gate_cpu_seconds'] for r in rr),
                   'mean_skip_layers':statistics.mean(len(r['skip_layers']) for r in rr),'executed_layer_tokens':sum(r['executed_layer_token_evaluations'] for r in rr),'skipped_layer_tokens':sum(r['skipped_layer_token_evaluations'] for r in rr),
                   'natural_analytical_flops':sum(r['analytical_matmul_flops'] for r in rr),'mlx_peak_bytes':max(r['mlx_peak_bytes'] for r in rr),'natural_mean_wall_seconds':statistics.mean(r['wall_seconds'] for r in rr),
                   'fixed_wall_seconds':sum(r['wall_seconds'] for r in ff),'fixed_cpu_seconds':sum(r['cpu_seconds'] for r in ff),'fixed_analytical_flops':sum(r['analytical_matmul_flops'] for r in ff),
                   'reference_nll':statistics.mean(r['reference_nll'] for r in qq),'reference_quality_repeated':False,
                   'baseline_response_similarity':statistics.mean(SequenceMatcher(None,r['answer'],ri[m,rep,'baseline',r['task_id']]['answer']).ratio() for r in rr)})
    aggregate=[]
    for m in models:
        for method in policy.METHODS:
            rr=[r for r in summaries if r['model']==m and r['method']==method];a={'model':m,'method':method,'repeats':len(reps)}
            for k,v in rr[0].items():
                if isinstance(v,(int,float)) and not isinstance(v,bool) and k not in ('repetition','reference_nll'):a.update({k+'_'+s:val for s,val in describe([r[k] for r in rr]).items()})
            # Reference NLL was measured once on32 questions, not3 times.
            qq=[qi[m,method,id]['reference_nll'] for id in tasks]
            a.update(reference_nll_mean=statistics.mean(qq),reference_nll_repeat_sd=None,reference_nll_scope='One reference evaluation per question/method; repeated summary entries not independent observations.')
            a.update({'reference_nll_question_'+k:v for k,v in describe(qq).items()})
            aggregate.append(a)
    comparisons=[];promising=[]
    for m in models:
        for method in policy.METHODS[1:]:
            a=accuracy_comparison([[int(ri[m,rep,method,id]['grade']['correct'])-int(ri[m,rep,'baseline',id]['grade']['correct']) for rep in reps] for id in tasks])
            fmt=statistics.mean(int(ri[m,rep,method,id]['grade']['format_ok'])-int(ri[m,rep,'baseline',id]['grade']['format_ok']) for id in tasks for rep in reps)
            natural=paired_ratio({id:{rep:ri[m,rep,method,id]['wall_seconds'] for rep in reps} for id in tasks},{id:{rep:ri[m,rep,'baseline',id]['wall_seconds'] for rep in reps} for id in tasks})
            fw=paired_ratio({id:{rep:fi[m,rep,method,id]['wall_seconds'] for rep in reps} for id in fixed_ids},{id:{rep:fi[m,rep,'baseline',id]['wall_seconds'] for rep in reps} for id in fixed_ids})
            nll=describe([qi[m,method,id]['reference_nll']-qi[m,'baseline',id]['reference_nll'] for id in tasks]);flops=sum(fi[m,rep,method,id]['analytical_matmul_flops'] for id in fixed_ids for rep in reps)/sum(fi[m,rep,'baseline',id]['analytical_matmul_flops'] for id in fixed_ids for rep in reps)
            row={'model':m,'method':method,**a,'format_delta':fmt,'natural_wall_ratio':natural['ratio'],'natural_ratio_low':natural['ci95_low'],'natural_ratio_high':natural['ci95_high'],
                 'fixed_wall_ratio':fw['ratio'],'fixed_ratio_low':fw['ci95_low'],'fixed_ratio_high':fw['ci95_high'],'fixed_analytic_flops_ratio':flops,'reference_nll_delta':nll['mean'],'reference_nll_delta_low':nll['ci95_low'],'reference_nll_delta_high':nll['ci95_high']}
            comparisons.append(row);skip=any(ri[m,rep,method,id]['skip_layers'] for id in tasks for rep in reps)
            gates={'quality_point_within1pp':a['accuracy_delta']>=-.01 and fmt>=-.01 and nll['mean']<=.05,
                   'natural_speed10_confident':natural['ci95_high']<=.9,'fixed_speed10_confident':fw['ci95_high']<=.9,'real_skip_and_lower_estimated_compute':skip and flops<1}
            promising.append({'model':m,'method':method,'promising_for_expansion':all(gates.values()),'criteria':gates,'population_quality_noninferiority_confirmed':a['conservative_exact_lower95']>=-.01,
                 'scope':'Small pilot candidate, not a new technology success.32 prompts cannot generally establish1pp tolerance.'})
    matched=[]
    for m in models:
        for family in ('sketch','leverage'):
            method,control='adaptive_'+family,'random_match_'+family
            a=accuracy_comparison([[int(ri[m,rep,method,id]['grade']['correct'])-int(ri[m,rep,control,id]['grade']['correct']) for rep in reps] for id in tasks])
            natural=paired_ratio({id:{rep:ri[m,rep,method,id]['wall_seconds'] for rep in reps} for id in tasks},{id:{rep:ri[m,rep,control,id]['wall_seconds'] for rep in reps} for id in tasks})
            fw=paired_ratio({id:{rep:fi[m,rep,method,id]['wall_seconds'] for rep in reps} for id in fixed_ids},{id:{rep:fi[m,rep,control,id]['wall_seconds'] for rep in reps} for id in fixed_ids})
            matched.append({'model':m,'method':method,'control':control,**a,'natural_ratio':natural['ratio'],'natural_ratio_low':natural['ci95_low'],'natural_ratio_high':natural['ci95_high'],
                'fixed_ratio':fw['ratio'],'fixed_ratio_low':fw['ci95_low'],'fixed_ratio_high':fw['ci95_high'],'reference_nll_delta':statistics.mean(qi[m,method,id]['reference_nll']-qi[m,control,id]['reference_nll'] for id in tasks),
                'scope':'Same skip count and adaptive-probe overhead; random masks differ by repetition.'})
    categories=[]
    for m in models:
        for method in policy.METHODS:
            for cat in sorted({t['category'] for t in tasks.values()}):
                rr=[ri[m,rep,method,id] for id,t in tasks.items() if t['category']==cat for rep in reps]
                categories.append({'model':m,'method':method,'category':cat,'unique_questions':len(rr)/len(reps),'accuracy':statistics.mean(r['grade']['correct'] for r in rr),'format':statistics.mean(r['grade']['format_ok'] for r in rr),'mean_skip_layers':statistics.mean(len(r['skip_layers']) for r in rr)})
    failures=[{'model':r['model'],'method':r['method'],'task_id':r['task_id'],'question':r['question'],'reference':r['reference'],'answer':r['answer'],'baseline_answer':ri[r['model'],r['repetition'],'baseline',r['task_id']]['answer'],
               'skip_layers':r['skip_layers'],'grade':r['grade'],'baseline_grade':ri[r['model'],r['repetition'],'baseline',r['task_id']]['grade']} for r in rows if r['repetition']==1 and r['method']!='baseline' and (not r['grade']['correct'] or not r['grade']['format_ok'])]
    out=Path(output_dir).resolve() if output_dir else p.parent/p.name;out.mkdir(exist_ok=False)
    for name,rr in [('ALSR_RESULTS.csv',aggregate),('ALSR_REPEAT_METRICS.csv',summaries),('ALSR_COMPARISONS.csv',comparisons),('ALSR_MATCHED_COMPARISONS.csv',matched),('ALSR_CANDIDATE_VERDICT.csv',promising),('ALSR_CATEGORY_RESULTS.csv',categories),('ALSR_FAILURE_CASES.csv',failures)]:write_records(out/name,rr)
    audit={'preserved_files_unchanged':len(manifest['preserved_sha256']),'past_stems':len(past),'train_cal_pilot_disjoint':True,'past_overlap':0,'profile_hashes_verified':True,'threshold_selection_reconstructed':True,
      'native_full_forward_exact_in_smoke':True,'skipped_block_exception_trap_and_positive_control':True,'all_actual_block_attention_mlp_counts_verified':True,'all_cache_offsets_verified':True,
      'matched_random_same_skip_amount':True,'fixed_work_same_mask_as_prompt_only_natural':True,'no_reference_gate_calls':True,'analytic_flops_reconstructed':True,
      'generation_rows':len(rows),'fixed_rows':len(fixed),'reference_rows_once':len(quality),'GPU_elapsed_power_money':'unmeasured',
      'recovered_after_export_error':meta.get('recovered_after_export_error',False),
      'original_performance_not_regenerated_or_overwritten':True,
      'zero_skip_adaptive_response_disagreements':sum(not r['skip_layers'] and ('sketch' in r['method'] or 'leverage' in r['method']) and r['token_ids']!=ri[r['model'],r['repetition'],'baseline',r['task_id']]['token_ids'] for r in rows),
      'limits':'Counters measure materialized retained MLX graph calls, not hardware GPU instruction counts. Native full parity smoke does not prove global bitwise equivalence of last-head optimization.'}
    write_json(out/'ALSR_AUDIT.json',audit);print(json.dumps(audit,indent=2));print('PROMISING '+json.dumps([r for r in promising if r['promising_for_expansion']]));return out

if __name__=='__main__':analyze(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
