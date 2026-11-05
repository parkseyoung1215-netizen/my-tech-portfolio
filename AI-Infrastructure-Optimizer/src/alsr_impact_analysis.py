"""Offline audits, transition analysis, and held-out signal evaluation."""
import ast,hashlib,json,math,random,statistics,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from .alsr_dataset import grade as legacy_grade
from .alsr_impact_scoring import grade
from .alsr_policy import estimated_matmul_flops,predict_ridge,fixed_layers,random_layers,choose_by_risk,fit_ridge
from .alsr_impact_policy import METHODS,auc,select_threshold
from .alsr_impact_data import SEED
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .controlled_statistics import describe,accuracy_comparison,quantile
from .alsr_analysis import paired_ratio
from .solver300_dataset import task_key
from .efficiency_repeat import frozen_check

def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(values):
    v=np.asarray(values);order=np.argsort(v);out=np.empty(len(v),dtype=float);i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out
def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def metrics(rows,predictions):
    targets=[r['kl']+r['top1_changed'] for r in rows];labels=[r['top1_changed'] for r in rows]
    return dict(n=len(rows),top1_flip_rate=statistics.mean(labels),risk_mae=statistics.mean(abs(a-b) for a,b in zip(targets,predictions)),risk_spearman=correlation(rank(targets),rank(predictions)),flip_auroc=auc(labels,predictions))

def paired_signal(rows,new,control,draws=1500):
    ids=sorted({r['task_id'] for r in rows});groups={id:[i for i,r in enumerate(rows) if r['task_id']==id] for id in ids};y=np.array([r['top1_changed'] for r in rows]);target=np.array([r['kl']+r['top1_changed'] for r in rows]);a=np.array(new);b=np.array(control);rng=random.Random(SEED)
    def fast_auc(yy,ss):
        n=int(np.sum(yy));z=len(yy)-n
        return None if not n or not z else float((np.sum(rank(ss)[yy==1])+n-n*(n+1)/2)/(n*z))
    deltas=[];errors=[]
    for _ in range(draws):
        ii=[j for _ in ids for j in groups[rng.choice(ids)]];aa=fast_auc(y[ii],a[ii]);bb=fast_auc(y[ii],b[ii])
        if aa is not None and bb is not None:deltas.append(aa-bb)
        errors.append(float(np.mean(np.abs(target[ii]-a[ii])-np.abs(target[ii]-b[ii]))))
    aa,bb=fast_auc(y,a),fast_auc(y,b)
    return dict(auroc_delta=None if aa is None or bb is None else aa-bb,auroc_delta_ci95_low=quantile(deltas,.025) if deltas else None,auroc_delta_ci95_high=quantile(deltas,.975) if deltas else None,risk_mae_delta=float(np.mean(np.abs(target-a)-np.abs(target-b))),risk_mae_delta_ci95_low=quantile(errors,.025),risk_mae_delta_ci95_high=quantile(errors,.975),bootstrap_scope='Unique-question cluster; layer/position rows are not independent samples.')

def analyze(folder):
    out=Path(folder).resolve();data_root=Path(__file__).resolve().parents[1]/'data';m=json.loads((out/'MANIFEST.json').read_text());meta=json.loads((out/'METADATA.json').read_text());profile=json.loads((out/'PROFILE.json').read_text());frozen_check(m)
    for name,digest in m.get('diagnostic_source_sha256',{}).items():assert sha(name)==digest
    assert sha(out/'PROFILE.json')==meta['profile_sha256'];assert sha(Path(__file__).resolve().parents[1]/'ALSR_IMPACT_DESIGN.md')==m['design_sha256']
    ab=read(out/'LAYER_ABLATIONS.jsonl');ctx=read(out/'DIAGNOSTIC_CONTEXTS.jsonl');gen=read(out/'GENERATION.jsonl');fixed=read(out/'FIXED_WORK.jsonl');quality=read(out/'QUALITY.jsonl')
    tasks={t['task_id']:t for ts in m['splits'].values() for t in ts};evalids={t['task_id'] for t in m['splits']['evaluation']};splitids={s:{t['task_id'] for t in ts} for s,ts in m['splits'].items()}
    scoring=[]
    for r in gen:
        assert r['grade']==legacy_grade(tasks[r['task_id']],r['answer'])
        r['legacy_grade']=r['grade'];r['grade']=grade(tasks[r['task_id']],r['answer'])
        if r['legacy_grade']['correct']!=r['grade']['correct']:scoring.append(dict(method=r['method'],task_id=r['task_id'],repetition=r['repetition'],answer=r['answer'],legacy_correct=r['legacy_grade']['correct'],audited_correct=r['grade']['correct'],reason=r['grade']['error_type']))
    assert len(evalids)==64 and all(not a&b for i,a in enumerate(splitids.values()) for b in list(splitids.values())[i+1:])
    keys=[task_key(t['question']) for t in tasks.values()];assert len(keys)==len(set(keys));past=set()
    def walk(x):
        if isinstance(x,dict):
            if isinstance(x.get('question'),str):past.add(task_key(x['question']))
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    import csv
    for name in m['preserved_sha256']:
        p=Path(name)
        if p.suffix=='.json' and p.is_relative_to(data_root):walk(json.loads(p.read_text()))
        elif p.suffix=='.csv' and p.is_relative_to(data_root):
            with p.open() as f:
                for r in csv.DictReader(f):walk(r)
    assert not set(keys)&past
    assert set(profile['train_ids'])==splitids['training'] and set(profile['calibration_ids'])==splitids['calibration']
    for layer in range(28):
        rr=[r for r in ab if r['split']=='training' and r['layer']==layer];reconstructed=fit_ridge([r['features'] for r in rr],[r['kl']+r['top1_changed'] for r in rr]);assert reconstructed==profile['ridge'][str(layer)]
    means={str(l):statistics.mean(r['kl']+r['top1_changed'] for r in ab if r['split']=='training' and r['position']==0 and r['layer']==l) for l in range(28)}
    control_means={(l,p):statistics.mean(r['kl']+r['top1_changed'] for r in ab if r['split']=='training' and r['position']==p and r['layer']==l) for l in range(28) for p in range(2)}
    assert max(abs(means[l]-profile['train_mean_risk'][l]) for l in means)<1e-12
    assert profile['candidate']==min(range(2,27),key=lambda l:(means[str(l)],l))
    val=[dict(r,prediction=predict_ridge(r['features'],profile['ridge'][str(profile['candidate'])])) for r in ab if r['split']=='calibration' and r['position']==0 and r['layer']==profile['candidate']]
    threshold,trials=select_threshold(val);assert threshold==profile['threshold'] and trials==profile['trials']
    ci={(r['task_id'],r['position']):r for r in ctx};ai={(r['task_id'],r['position'],r['layer']):r for r in ab};assert len(ai)==len(ab)==len(ctx)*28
    for r in ab:
        assert r['task_id'] in splitids[r['split']];assert len(r['features'])==15
        assert r['layer_calls']==[int(l!=r['layer']) for l in range(28)]
        assert r['kl']>=0 and 0<=r['js']<=math.log(2)+1e-6
        assert r['top1_changed']==int(r['full_top1']!=r['skip_top1'])
        assert ci[r['task_id'],r['position']]['baseline_top1']==r['full_top1']
    assert all(r['probe_reference_max_abs_difference']==0 for r in ctx)
    gi={(r['method'],r['task_id'],r['repetition']):r for r in gen};fi={(r['method'],r['task_id'],r['repetition']):r for r in fixed}
    assert len(gi)==len(gen)==64*len(METHODS)*3 and len(fi)==16*len(METHODS)*3
    oldpath=data_root/'alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_m1_profile.json';assert sha(oldpath)==m['old_profile_sha256'];old=json.loads(oldpath.read_text());config=m['model']['config'];seqs=json.loads((out/'FIXED_SEQUENCES.json').read_text())
    for r in gen+fixed:
        id,method,rep=r['task_id'],r['method'],r['repetition'];skip=r['skip_layers'];fs=r['forwards'];assert id in evalids
        if method=='baseline':expected=[]
        elif method=='static':expected=fixed_layers(28)
        elif method=='random':expected=random_layers(old['pool'],4,tasks[id]['question'],SEED+rep)
        elif method=='stable1':expected=[profile['candidate']]
        elif method=='random1':expected=random_layers(range(2,27),1,tasks[id]['question'],SEED+rep)
        elif method=='prelayer_gate':
            d=r['gate_decision'];assert d['features']==ai[id,0,profile['candidate']]['features'];risk=predict_ridge(d['features'],profile['ridge'][str(profile['candidate'])]);assert risk==d['risk'];expected=[profile['candidate']] if risk<profile['threshold'] else []
        else:
            d=r['gate_decision'];assert d['threshold']==old['thresholds']['sketch'];expected=choose_by_risk({int(k):v for k,v in d['risks'].items()},d['threshold'],old['pool'])
        assert skip==expected
        assert r['layer_calls']==r['attention_calls']==r['mlp_calls']==[0 if l in skip else len(fs) for l in range(28)]
        assert r['kv_offsets']==[0 if l in skip else r['input_tokens']+r['output_tokens']-1 for l in range(28)]
        assert len(fs)==r['output_tokens'] and all(f['skipped']==skip and f['executed']==[l for l in range(28) if l not in skip] for f in fs)
        assert r['analytical_matmul_flops']==estimated_matmul_flops(config,fs,r['gate_flops_estimate'])
        if r['fixed_work']:
            assert r['token_ids']==seqs[id] and skip==gi[method,id,rep]['skip_layers']
        else:assert r['grade']==grade(tasks[id],r['answer'])
    assert len(quality)==64*len(METHODS) and all(not r['quality_eval_gate_calls'] and r['quality_eval_skip_layers']==gi[r['method'],r['task_id'],1]['skip_layers'] for r in quality)
    # Per-layer input-only predictors on untouched holdout, and depth-only controls.
    layers=[];signals=[];predictions=[];layer_categories=[];margin_relations=[]
    for l in range(28):
        for position in range(2):
            rr=[r for r in ab if r['split']=='evaluation' and r['layer']==l and r['position']==position]
            if not rr:continue
            layers.append(dict(layer=l,position=position,n=len(rr),train_position0_mean_risk=means[str(l)],mean_kl=statistics.mean(r['kl'] for r in rr),p90_kl=quantile([r['kl'] for r in rr],.9),max_kl=max(r['kl'] for r in rr),top1_agreement=1-statistics.mean(r['top1_changed'] for r in rr),mean_js=statistics.mean(r['js'] for r in rr),mean_full_margin=statistics.mean(r['full_margin'] for r in rr),mean_skip_margin=statistics.mean(r['skip_margin'] for r in rr)))
            margin_relations.append(dict(layer=l,position=position,full_final_margin_vs_kl_pearson=correlation([r['full_margin'] for r in rr],[r['kl'] for r in rr]),input_shortlist_margin_vs_kl_pearson=correlation([r['features'][12] for r in rr],[r['kl'] for r in rr]),scope='Full final margin is offline explanatory only, never a live feature.'))
            for cat in sorted({r['category'] for r in rr}):
                cc=[r for r in rr if r['category']==cat];layer_categories.append(dict(layer=l,position=position,category=cat,n=len(cc),mean_kl=statistics.mean(r['kl'] for r in cc),top1_agreement=1-statistics.mean(r['top1_changed'] for r in cc)))
            for signal,pred in [('prelayer_ridge',[predict_ridge(r['features'],profile['ridge'][str(l)]) for r in rr]),('layer_mean',[control_means[l,position]]*len(rr)),('input_margin_inverse',[1/(.01+max(0,r['features'][12])) for r in rr]),('previous_relative_change',[r['features'][10] for r in rr])]:
                signals.append(dict(layer=l,position=position,signal=signal,**metrics(rr,pred)))
                if signal=='prelayer_ridge':
                    for r,predicted in zip(rr,pred):predictions.append(dict(task_id=r['task_id'],layer=l,position=position,predicted_risk=predicted,observed_kl=r['kl'],observed_flip=r['top1_changed'],observed_risk=r['kl']+r['top1_changed']))
    pooled=[]
    for position in range(2):
        for scope,ll in [('all28',range(28)),('old_sketch_pool8',old['pool']),('candidate1',[profile['candidate']])]:
            rr=[r for r in ab if r['split']=='evaluation' and r['position']==position and r['layer'] in ll]
            for signal,pred in [('prelayer_ridge',[predict_ridge(r['features'],profile['ridge'][str(r['layer'])]) for r in rr]),('layer_mean',[control_means[r['layer'],position] for r in rr])]:pooled.append(dict(scope=scope,position=position,signal=signal,**metrics(rr,pred),warning='Pooled all-layer scores can reflect layer identity, not input adaptivity.'))
            if position==0 and scope=='old_sketch_pool8':
                pred=[gi['adaptive_sketch',r['task_id'],1]['gate_decision']['risks'][str(r['layer'])] for r in rr];pooled.append(dict(scope=scope,position=position,signal='frozen_old_sketch',**metrics(rr,pred),warning='Same pool and position; original historical training, not retrained.'))
    paired=[]
    for scope,ll in [('all28',range(28)),('candidate1',[profile['candidate']]),('old_sketch_pool8',old['pool'])]:
        rr=[r for r in ab if r['split']=='evaluation' and r['position']==0 and r['layer'] in ll];new=[predict_ridge(r['features'],profile['ridge'][str(r['layer'])]) for r in rr]
        controls=[('layer_mean',[means[str(r['layer'])] for r in rr])]
        if scope=='old_sketch_pool8':controls.append(('frozen_old_sketch',[gi['adaptive_sketch',r['task_id'],1]['gate_decision']['risks'][str(r['layer'])] for r in rr]))
        for label,control in controls:paired.append(dict(scope=scope,signal='prelayer_ridge',control=label,**paired_signal(rr,new,control)))
    # Repeat metrics and paired question-cluster inference.
    repeat=[];summary=[];comparison=[];transitions=[];categories=[];failures=[];agreements=[]
    for method in METHODS:
        for rep in range(1,4):
            rr=[gi[method,id,rep] for id in evalids];ff=[r for r in fixed if r['method']==method and r['repetition']==rep]
            changes=[(r,gi['baseline',r['task_id'],rep]) for r in rr]
            repeat.append(dict(method=method,repetition=rep,task_success=statistics.mean(r['grade']['correct'] for r in rr),mcq_accuracy=statistics.mean(r['grade']['correct'] for r in rr if r['score_kind']=='choice'),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),wall_seconds=sum(r['wall_seconds'] for r in rr),mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rr),cpu_seconds=sum(r['cpu_seconds'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr),output_tokens=sum(r['output_tokens'] for r in rr),mean_skipped_layers=statistics.mean(len(r['skip_layers']) for r in rr),analytical_matmul_flops=sum(r['analytical_matmul_flops'] for r in rr),fixed_wall_seconds=sum(r['wall_seconds'] for r in ff),fixed_cpu_seconds=sum(r['cpu_seconds'] for r in ff),fixed_analytical_matmul_flops=sum(r['analytical_matmul_flops'] for r in ff),gate_wall_seconds=sum(r['gate_wall_seconds'] for r in rr),mean_gate_wall_seconds=statistics.mean(r['gate_wall_seconds'] for r in rr),gate_cpu_seconds=sum(r['gate_cpu_seconds'] for r in rr),mean_mlx_peak_bytes=statistics.mean(r['mlx_peak_bytes'] for r in rr),correct_to_wrong=sum(b['grade']['correct'] and not r['grade']['correct'] for r,b in changes),wrong_to_correct=sum(not b['grade']['correct'] and r['grade']['correct'] for r,b in changes),different_responses=sum(r['answer']!=b['answer'] for r,b in changes)))
            for r,b in changes:
                bc,rc=b['grade']['correct'],r['grade']['correct'];effect=ai[r['task_id'],0,r['skip_layers'][0]] if len(r['skip_layers'])==1 else None
                transitions.append(dict(method=method,repetition=rep,task_id=r['task_id'],category=r['category'],transition=('Correct' if bc else 'Wrong')+' -> '+('Correct' if rc else 'Wrong'),baseline_answer=b['answer'],answer=r['answer'],skip_layers=r['skip_layers'],output_equal=r['answer']==b['answer'],initial_next_token_agreement=(not effect['top1_changed']) if effect else None,first_context_kl=effect['kl'] if effect else None,initial_token_equal=r['token_ids'][0]==b['token_ids'][0],content_error=r['grade']['error_type']))
                if method!='baseline' and (bc!=rc or r['answer']!=b['answer']) and rep==1:failures.append(dict(method=method,task_id=r['task_id'],question=r['question'],reference=r['reference'],baseline_answer=b['answer'],answer=r['answer'],baseline_correct=bc,final_correct=rc,skip_layers=r['skip_layers'],error_type=r['grade']['error_type'],first_context_top1_agree=(not effect['top1_changed']) if effect else None))
        these=[r for r in repeat if r['method']==method];row={'method':method}
        for key in these[0]:
            if key not in ('method','repetition'):
                for stat,value in describe([r[key] for r in these],draws=2000).items():row[key+'_'+stat]=value
        qq=[r for r in quality if r['method']==method];row['reference_nll_once']=statistics.mean(r['reference_nll'] for r in qq);row['quality_eval_wall_seconds_once']=sum(r['quality_eval_wall_seconds'] for r in qq);summary.append(row)
        for category in sorted({t['category'] for t in tasks.values()}):
            rr=[r for r in gen if r['method']==method and r['category']==category];categories.append(dict(method=method,category=category,n_unique_questions=len({r['task_id'] for r in rr}),success=statistics.mean(r['grade']['correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rr)))
        if method!='baseline':
            for control in ('baseline','adaptive_sketch'):
                if control==method:continue
                diffs=[[int(gi[method,id,rep]['grade']['correct'])-int(gi[control,id,rep]['grade']['correct']) for rep in range(1,4)] for id in sorted(evalids)];cc=dict(method=method,control=control,**accuracy_comparison(diffs,draws=3000))
                for mode,index,ids in [('natural',gi,evalids),('fixed',fi,set(m['fixed_ids']))]:
                    target={id:{rep:index[method,id,rep]['wall_seconds'] for rep in range(1,4)} for id in sorted(ids)};base={id:{rep:index[control,id,rep]['wall_seconds'] for rep in range(1,4)} for id in sorted(ids)}
                    for k,v in paired_ratio(target,base,draws=3000).items():
                        if k!='scope':cc[mode+'_wall_'+k]=v
                comparison.append(cc)
        if method in ('stable1','random1','prelayer_gate'):
            rr=[r for r in transitions if r['method']==method and r['initial_next_token_agreement'] is True]
            agreements.append(dict(method=method,scope='only requests actually skipping exactly1 layer with first top1 unchanged',n_repeat_observations=len(rr),unique_questions=len({r['task_id'] for r in rr}),response_changed=sum(not r['output_equal'] for r in rr),correct_destroyed=sum(r['transition']=='Correct -> Wrong' for r in rr),correct_repaired=sum(r['transition']=='Wrong -> Correct' for r in rr)))
    # Timing position correlation is descriptive, not a causal thermal/cache adjustment.
    order=[]
    for method in METHODS:
        rr=[r for r in gen if r['method']==method];order.append(dict(method=method,position_wall_pearson=correlation([r['position'] for r in rr],[r['wall_seconds'] for r in rr]),sequence_wall_pearson=correlation([r['sequence'] for r in rr],[r['wall_seconds'] for r in rr]),limitation='Output length and heterogeneous input confound these descriptive correlations.'))
    verdict=[];base=next(r for r in summary if r['method']=='baseline')
    for r in summary:
        if r['method']=='baseline':continue
        c=next(c for c in comparison if c['method']==r['method'] and c['control']=='baseline')
        qualityok=r['task_success_mean']>=base['task_success_mean']-.01 and r['format_compliance_mean']>=base['format_compliance_mean']-.01 and r['reference_nll_once']<=base['reference_nll_once']+.05
        speedok=c['natural_wall_ci95_high']<=.90 and c['fixed_wall_ci95_high']<=.90;compute=r['fixed_analytical_matmul_flops_mean']<base['fixed_analytical_matmul_flops_mean'] and r['mean_skipped_layers_mean']>0
        verdict.append(dict(method=r['method'],quality_point_within_target=qualityok,speed10_confident=speedok,true_skips_and_estimated_compute_reduction=compute,promising=qualityok and speedok and compute,population_1pp_noninferiority_proven=c['conservative_exact_lower95']>=-.01))
    legacy_comparison=[dict(method=method,legacy_success=statistics.mean(r['legacy_grade']['correct'] for r in gen if r['method']==method),audited_success=statistics.mean(r['grade']['correct'] for r in gen if r['method']==method),scope='Only JSON bool/int exact type check added; raw generation grades unchanged.') for method in METHODS]
    outputs={'LAYER_SUMMARY':layers,'LAYER_CATEGORY_SUMMARY':layer_categories,'LAYER_MARGIN_RELATION':margin_relations,'SIGNAL_METRICS':signals,'SIGNAL_POOLED':pooled,'SIGNAL_PAIRED':paired,'RISK_PREDICTIONS':predictions,'RESULTS':summary,'REPEAT_METRICS':repeat,'COMPARISONS':comparison,'TRANSITIONS':transitions,'CATEGORY_RESULTS':categories,'FAILURE_CASES':failures,'TOP1_VS_SEQUENCE':agreements,'ORDER_DIAGNOSTICS':order,'VERDICT':verdict,'STRICT_SCORE_AUDIT':scoring,'LEGACY_SCORE_COMPARISON':legacy_comparison}
    if meta.get('corrected_tie_rule'):
        ties=[dict(task_id=r['task_id'],split=r['split'],position=r['position'],layer=r['layer'],legacy_full_top1=r['legacy_full_top1'],canonical_full_top1=r['full_top1'],legacy_skip_top1=r['legacy_skip_top1'],canonical_skip_top1=r['skip_top1'],legacy_flip=r['legacy_top1_changed'],canonical_flip=r['top1_changed'],unchanged_kl=r['kl']) for r in ab if r['legacy_top1_changed']!=r['top1_changed']]
        original=read(Path(m['diagnostic_source'])/'GENERATION.jsonl');origfixed=read(Path(m['diagnostic_source'])/'FIXED_WORK.jsonl');original_metrics=[];phase_comparison=[]
        for method in METHODS:
            rr=[r for r in original if r['method']==method];ff=[r for r in origfixed if r['method']==method];current=next(r for r in summary if r['method']==method)
            original_metrics.append(dict(method=method,phase='Original tie-noisy training labels, not primary corrected analysis',task_success=statistics.mean(grade(tasks[r['task_id']],r['answer'])['correct'] for r in rr),wall_seconds_per_repeat=sum(r['wall_seconds'] for r in rr)/3,cpu_seconds_per_repeat=sum(r['cpu_seconds'] for r in rr)/3,fixed_wall_seconds_per_repeat=sum(r['wall_seconds'] for r in ff)/3))
            origindex={(r['task_id'],r['repetition']):r for r in rr};phase_comparison.append(dict(method=method,original_success=original_metrics[-1]['task_success'],corrected_success=current['task_success_mean'],changed_responses=sum(r['answer']!=origindex[r['task_id'],r['repetition']]['answer'] for r in gen if r['method']==method),original_wall=original_metrics[-1]['wall_seconds_per_repeat'],corrected_wall=current['wall_seconds_mean'],warning='Cross-phase timing differences confounded by caches/order/NLL instrumentation; not algorithm savings.'))
        outputs.update(TIE_LABEL_CHANGES=ties,ORIGINAL_PHASE_RESULTS=original_metrics,PHASE_COMPARISON=phase_comparison)
    for name,rows in outputs.items():write_union(out/(name+'.csv'),rows)
    no_skip_disagreements=sum(r['token_ids']!=gi['baseline',r['task_id'],r['repetition']]['token_ids'] for r in gen if r['method']!='baseline' and not r['skip_layers'])
    audit=dict(preserved_files_unchanged=len(m['preserved_sha256']),all_existing_hashes_verified=True,exact_stem_past_overlap=0,semantic_template_overlap_not_excluded=True,splits_disjoint=True,profile_training_and_threshold_reconstructed=True,gate_features_equal_preblock_diagnostic=True,full_vocabulary_effects=True,baseline_probe_logits_exact=True,all_actual_block_attention_mlp_and_kv_counts_verified=True,fixed_masks_and_tokens_identical=True,no_reference_gate_calls=True,ablation_rows=len(ab),diagnostic_contexts=len(ctx),generation_rows=len(gen),fixed_rows=len(fixed),reference_rows_once=len(quality),zero_skip_token_disagreements=no_skip_disagreements,all_costs_real_except_explicit_flops_estimates=True)
    audit.update(json_type_scoring_disagreements=len(scoring),raw_generation_scores_preserved=True)
    write_json(out/'AUDIT.json',audit);print(json.dumps({'output':str(out),'candidate':profile['candidate'],'threshold':profile['threshold'],'promising':[v['method'] for v in verdict if v['promising']],'audit':audit},indent=2));return outputs

if __name__=='__main__':analyze(sys.argv[1])
