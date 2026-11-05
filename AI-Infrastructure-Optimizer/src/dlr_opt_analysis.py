"""Audit real runs, preserve every failure, and separate modeled from measured work."""
import hashlib,json,statistics,sys
from collections import Counter
from pathlib import Path
from uuid import uuid4
from .dlr_policy import estimate,CANDIDATES,reuse_allowed
from .dlr_opt_data import score
from .stable_runtime_records import Store
from .efficiency_repeat import frozen_check
from .controlled_statistics import describe,accuracy_comparison
from .alsr_analysis import paired_ratio
from .alsr_io import write_union
from .adaptive_experiment import write_json
METHODS=('baseline','old_dlr','device','lightweight')
METRICS=('wall_seconds','model_wall_seconds','cpu_seconds','model_cpu_seconds','input_tokens','output_tokens','analytical_matmul_flops','limited_net_arithmetic_proxy','gate_arithmetic_estimate','gate_selection_element_count','gate_wall_seconds','gate_cpu_seconds','gate_graph_host_wall_seconds','gate_graph_host_cpu_seconds','pre_gate_sync_wall_seconds','gate_host_scalar_reads','explicit_pre_gate_evals','gate_queries','id_comparisons')

def analyze(folder):
    p=Path(folder).resolve();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for k in ('source_sha256','legacy_source_sha256'):
        for path,h in m[k].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path('DLR_OPTIMIZATION_DESIGN.md').read_bytes()).hexdigest()==m['design_sha256']
    rows=Store(p/'checkpoints').rows();assert len(rows)==len(m['schedule'])==1728
    tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};index={};events=[];cases=[]
    for r in rows:
        phase,ds,rep,id,method=(r[k] for k in ('phase','dataset','repetition','task_id','method'));key=(phase,ds,rep,id,method);assert key not in index;index[key]=r;n=r['output_tokens']
        assert r['grade']==score(tasks[id],r['answer']);assert len(r['token_ids'])==len(r['work_shapes'])==n and r['model_calls']==1
        assert r['kv_offsets']==[r['input_tokens']+n-1]*28 and r['attention_calls']==[n]*28
        assert all(r['mlp_calls'][i]+r['reused_mlp_calls'][i]==n for i in range(28))
        assert all(r['reused_mlp_calls'][i]==0 for i in range(28) if i not in CANDIDATES)
        assert sum(x[3] for x in r['work_shapes'])==sum(r['mlp_calls'])
        assert estimate(m['model']['config'],r['work_shapes'])==r['analytical_matmul_flops']
        assert r['limited_net_arithmetic_proxy']==r['analytical_matmul_flops']+r['gate_arithmetic_estimate']
        assert r['gate_arithmetic_estimate']==r['gate_queries']*(5*m['model']['config']['hidden_size']+4)
        if method in ('baseline','lightweight','device'):assert r['gate_host_scalar_reads']==r['explicit_pre_gate_evals']==0
        if method=='baseline':assert not sum(r['reused_mlp_calls']) and not r['events']
        if method=='lightweight':
            assert r['gate_queries']==0 and r['id_comparisons']==n-1
            assert all(s[3] in (24,28) for s in r['work_shapes'])
            assert sum(28-s[3] for s in r['work_shapes'])==sum(r['reused_mlp_calls'])
            assert all(not (a[3]==24 and b[3]==24) for a,b in zip(r['work_shapes'],r['work_shapes'][1:]))
        elif method=='old_dlr':assert r['gate_host_scalar_reads']==len(r['events'])
        elif method=='device':assert r['device_counter_read_batches']==int(n>1) and r['gate_queries']==4*(n-1)
        for e in r['events']:
            if method=='device':assert e['gpu_projection_mac_branches']==(0 if e['reused'] else 3)
            if method=='old_dlr':assert e['reused']==reuse_allowed(e['distance'],0,'dlr')
            events.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=id,method=method,**e))
    summaries=[];repeats=[];comparisons=[];pareto=[];groups=[];verdicts=[]
    for phase in ('natural','fixed'):
        for ds,ts in m['datasets'].items():
            ids=[t['task_id'] for t in ts]
            for method in METHODS:
                for rep in (1,2,3):
                    rr=[index[phase,ds,rep,id,method] for id in ids];bb=[index[phase,ds,rep,id,'baseline'] for id in ids]
                    row=dict(phase=phase,dataset=ds,method=method,repetition=rep,requests=len(rr),content_accuracy=statistics.mean(r['grade']['content_correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),output_agreement=statistics.mean(r['token_ids']==b['token_ids'] for r,b in zip(rr,bb)),answer_text_agreement=statistics.mean(r['answer']==b['answer'] for r,b in zip(rr,bb)),reused_mlp_calls=sum(sum(r['reused_mlp_calls']) for r in rr),executed_mlp_calls=sum(sum(r['mlp_calls']) for r in rr),attention_calls=sum(sum(r['attention_calls']) for r in rr),model_calls=len(rr),mlx_peak_bytes=max(r['mlx_peak_bytes'] for r in rr),rss_lifetime_peak_bytes=max(r['process_lifetime_rss_peak_bytes'] for r in rr))
                    for metric in METRICS:row[metric]=sum(r[metric] for r in rr)
                    row['mean_seconds_per_output_token']=row['wall_seconds']/row['output_tokens'];row['tokens_per_second']=1/row['mean_seconds_per_output_token']
                    repeats.append(row)
                    for r,b in zip(rr,bb):
                        if phase=='fixed':
                            ref=index['natural',ds,1,r['task_id'],'baseline'];assert r['token_ids']==ref['token_ids'] and r['input_tokens']==ref['input_tokens']
                        cases.append(dict(phase=phase,dataset=ds,method=method,repetition=rep,task_id=r['task_id'],category=r['category'],question=r['question'],reference=tasks[r['task_id']]['reference'],answer=r['answer'],baseline_answer=b['answer'],correct=r['grade']['content_correct'],baseline_correct=b['grade']['content_correct'],format_ok=r['grade']['format_ok'],output_match=r['token_ids']==b['token_ids'],transition=('Correct' if b['grade']['content_correct'] else 'Wrong')+'->'+('Correct' if r['grade']['content_correct'] else 'Wrong'),wall_seconds=r['wall_seconds'],cpu_seconds=r['cpu_seconds'],output_tokens=r['output_tokens'],reused_mlp_calls=sum(r['reused_mlp_calls']),sequence=r['sequence'],position=r['position'],completion_reason=r['completion_reason']))
                rr=[r for r in repeats if r['phase']==phase and r['dataset']==ds and r['method']==method];summary=dict(phase=phase,dataset=ds,method=method,requests_per_repeat=len(ids))
                metrics=('content_accuracy','format_compliance','output_agreement','answer_text_agreement','reused_mlp_calls','executed_mlp_calls','attention_calls','model_calls','mlx_peak_bytes','rss_lifetime_peak_bytes','mean_seconds_per_output_token','tokens_per_second')+METRICS
                for metric in metrics:
                    for key,val in describe([r[metric] for r in rr],draws=1000).items():summary[metric+'_'+key]=val
                summaries.append(summary)
                if method=='baseline':continue
                controls=['baseline']+(['old_dlr'] if method in ('device','lightweight') else [])
                for control in controls:
                    diff=[[int(index[phase,ds,rep,id,method]['grade']['content_correct'])-int(index[phase,ds,rep,id,control]['grade']['content_correct']) for rep in (1,2,3)] for id in ids];cmp=dict(phase=phase,dataset=ds,method=method,control=control,**accuracy_comparison(diff,draws=2000))
                    for metric in ('wall_seconds','model_wall_seconds','cpu_seconds','analytical_matmul_flops','limited_net_arithmetic_proxy','output_tokens'):
                        target={id:{rep:index[phase,ds,rep,id,method][metric] for rep in (1,2,3)} for id in ids};base={id:{rep:index[phase,ds,rep,id,control][metric] for rep in (1,2,3)} for id in ids}
                        for key,val in paired_ratio(target,base,draws=2000).items():
                            if key!='scope':cmp[metric+'_'+key]=val
                    cmp['repeat_wall_ratios']=[sum(index[phase,ds,rep,id,method]['wall_seconds'] for id in ids)/sum(index[phase,ds,rep,id,control]['wall_seconds'] for id in ids) for rep in (1,2,3)];comparisons.append(cmp)
            for method in METHODS:
                for category in sorted({t['category'] for t in ts}):
                    rr=[c for c in cases if c['phase']==phase and c['dataset']==ds and c['method']==method and c['category']==category];groups.append(dict(phase=phase,dataset=ds,method=method,category=category,unique_questions=len(rr)//3,accuracy=statistics.mean(c['correct'] for c in rr),format_compliance=statistics.mean(c['format_ok'] for c in rr),output_match=statistics.mean(c['output_match'] for c in rr),correct_to_wrong=sum(c['transition']=='Correct->Wrong' for c in rr)/3,wrong_to_correct=sum(c['transition']=='Wrong->Correct' for c in rr)/3))
            ss=[s for s in summaries if s['phase']==phase and s['dataset']==ds]
            for s in ss:
                dominated=any(o['content_accuracy_mean']>=s['content_accuracy_mean'] and o['wall_seconds_mean']<=s['wall_seconds_mean'] and (o['content_accuracy_mean']>s['content_accuracy_mean'] or o['wall_seconds_mean']<s['wall_seconds_mean']) for o in ss)
                pareto.append(dict(phase=phase,dataset=ds,method=s['method'],accuracy=s['content_accuracy_mean'],wall_seconds=s['wall_seconds_mean'],point_pareto=not dominated,not_statistical_dominance=True))
    # Overall totals retain proper request weighting; do not average cohort accuracies equally.
    for phase in ('natural','fixed'):
        for method in METHODS:
            aggregate=[]
            for rep in (1,2,3):
                rr=[r for r in repeats if r['phase']==phase and r['method']==method and r['repetition']==rep and r['dataset']!='all72'];total=sum(r['requests'] for r in rr)
                row=dict(phase=phase,dataset='all72',method=method,repetition=rep,requests=total)
                for metric in ('content_accuracy','format_compliance','output_agreement','answer_text_agreement'):row[metric]=sum(r[metric]*r['requests'] for r in rr)/total
                for metric in ('reused_mlp_calls','executed_mlp_calls','attention_calls','model_calls')+METRICS:row[metric]=sum(r[metric] for r in rr)
                for metric in ('mlx_peak_bytes','rss_lifetime_peak_bytes'):row[metric]=max(r[metric] for r in rr)
                row['mean_seconds_per_output_token']=row['wall_seconds']/row['output_tokens'];row['tokens_per_second']=1/row['mean_seconds_per_output_token'];aggregate.append(row)
            repeats.extend(aggregate);summary=dict(phase=phase,dataset='all72',method=method,requests_per_repeat=72)
            for metric in ('content_accuracy','format_compliance','output_agreement','answer_text_agreement','reused_mlp_calls','executed_mlp_calls','attention_calls','model_calls','mlx_peak_bytes','rss_lifetime_peak_bytes','mean_seconds_per_output_token','tokens_per_second')+METRICS:
                for k,v in describe([r[metric] for r in aggregate],draws=1000).items():summary[metric+'_'+k]=v
            summaries.append(summary)
    for ds in m['datasets']:
        base=next(s for s in summaries if s['phase']=='natural' and s['dataset']==ds and s['method']=='baseline')
        for method in METHODS[1:]:
            natural=next(c for c in comparisons if c['phase']=='natural' and c['dataset']==ds and c['method']==method and c['control']=='baseline');fixed=next(c for c in comparisons if c['phase']=='fixed' and c['dataset']==ds and c['method']==method and c['control']=='baseline')
            passed=natural['accuracy_delta']==0 and natural['any_repeat_loss_questions']==0 and base['content_accuracy_mean']>0 and fixed['wall_seconds_ratio']<=.95 and all(r<=.95 for r in fixed['repeat_wall_ratios']) and fixed['limited_net_arithmetic_proxy_ratio']<1
            verdicts.append(dict(dataset=ds,method=method,equal_observed_accuracy=natural['accuracy_delta']==0,no_any_observed_correct_loss=natural['any_repeat_loss_questions']==0,baseline_task_accuracy_nonzero=base['content_accuracy_mean']>0,equal_stream_wall_gain_at_least_5pct=fixed['wall_seconds_ratio']<=.95,all_three_repeats_gain_at_least_5pct=all(r<=.95 for r in fixed['repeat_wall_ratios']),equal_stream_net_proxy_reduced=fixed['limited_net_arithmetic_proxy_ratio']<1,cpu_ratio=fixed['cpu_seconds_ratio'],wall_ci_entirely_at_least_5pct_better=fixed['wall_seconds_ci95_high']<=.95,observed_minimum_pass=passed,population_accuracy_noninferiority_proven=False))
    out=p/('ANALYSIS_'+uuid4().hex);out.mkdir()
    for name,rr in [('DLR_OPTIMIZATION_RESULTS',summaries),('DLR_OPTIMIZATION_REPEATS',repeats),('DLR_OPTIMIZATION_COMPARISONS',comparisons),('DLR_OPTIMIZATION_CASES',cases),('DLR_OPTIMIZATION_FAILURES',[c for c in cases if c['phase']=='natural' and (not c['correct'] or not c['output_match'])]),('DLR_OPTIMIZATION_EVENTS',events),('DLR_OPTIMIZATION_GROUPS',groups),('DLR_OPTIMIZATION_PARETO',pareto),('DLR_OPTIMIZATION_VERDICTS',verdicts)]:write_union(out/(name+'.csv'),rr)
    write_json(out/'AUDIT.json',{'real_requests':len(rows),'fixed_stream_all_counts_and_input_tokens_match':True,'all_attention_kv_offsets_correct':True,'all_device_projection_counters_match_branch':True,'original_files_unchanged':len(m['preserved_sha256']),'core_sources_unchanged':True,'no_reference_used_by_gate':True,'no_paid_api_or_upload':True,'gate_gpu_duration_separately_measured':False,'flops_scope':'analytical dense matmul + approx gate arithmetic, not all hardware work; selection/memory excluded','fresh_kernel_arithmetic_differs_from_native':True,'freshness_scope':m['freshness']});write_json(out/'SOURCE.json',{'experiment':str(p)});print('DLR OPT ANALYSIS '+str(out),flush=True)
    return out

if __name__=='__main__':analyze(sys.argv[1])
