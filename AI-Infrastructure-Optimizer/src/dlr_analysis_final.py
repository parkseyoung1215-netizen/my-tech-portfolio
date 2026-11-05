"""Immutable real-request audit, equal-stream checks, paired uncertainty and report."""
import csv,hashlib,json,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
from uuid import uuid4
from .dlr_policy import METHODS,CANDIDATES,reuse_allowed,estimate
from .dlr_data import score
from .stable_runtime_records import Store
from .efficiency_repeat import frozen_check
from .adaptive_experiment import write_json
from .alsr_io import write_union
from .alsr_analysis import paired_ratio
from .controlled_statistics import accuracy_comparison,describe

METRICS=('wall_seconds','cpu_seconds','model_wall_seconds','model_cpu_seconds','input_tokens','output_tokens','analytical_matmul_flops','gate_wall_seconds','gate_cpu_seconds','pre_gate_sync_wall_seconds','gate_elements')
def analyze(folder):
    p=Path(folder).resolve();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for path,h in m['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    rows=Store(p/'checkpoints').rows();assert len(rows)==len(m['schedule'])==960
    tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};idx={};events=[]
    for r in rows:
        phase,ds,rep,id,method=(r[k] for k in ('phase','dataset','repetition','task_id','method'));key=(phase,ds,rep,id,method);assert key not in idx;idx[key]=r
        assert r['grade']==score(tasks[id],r['answer']);n=r['output_tokens'];assert n==len(r['token_ids'])==len(r['work_shapes']) and r['model_calls']==1
        assert r['kv_offsets']==[0 if method=='static' and i==24 else r['input_tokens']+n-1 for i in range(28)]
        assert r['attention_calls']==[0 if method=='static' and i==24 else n for i in range(28)]
        reused=Counter(e['layer'] for e in r['events'] if e['reused'])
        for i in range(28):assert r['mlp_calls'][i]+r['reused_mlp_calls'][i]==r['attention_calls'][i] and r['reused_mlp_calls'][i]==reused[i]
        for e in r['events']:
            assert e['layer'] in CANDIDATES and e['step']>0 and e['reused']==reuse_allowed(e['distance'],0,method)
            events.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=id,method=method,**e))
        assert estimate(m['model']['config'],r['work_shapes'])==r['analytical_matmul_flops']
        if method in ('baseline','static'):assert not r['events'] and not sum(r['reused_mlp_calls'])
        if phase=='fixed':
            base=next(x for x in rows if x['phase']=='natural' and x['dataset']==ds and x['repetition']==1 and x['task_id']==id and x['method']=='baseline')
            assert r['token_ids']==base['token_ids'] and r['input_tokens']==base['input_tokens']
    repeats=[];summaries=[];comparisons=[];cases=[];groups=[];pareto=[]
    for phase in ('natural','fixed'):
        for ds in m['datasets']:
            ids=[t['task_id'] for t in m['datasets'][ds] if phase=='natural' or t['task_id'] in m['fixed_ids'][ds]]
            for method in METHODS:
                selected=[idx[phase,ds,rep,id,method] for rep in (1,2,3) for id in ids]
                for rep in (1,2,3):
                    rr=[idx[phase,ds,rep,id,method] for id in ids]
                    row=dict(phase=phase,dataset=ds,method=method,repetition=rep,requests=len(rr),content_accuracy=statistics.mean(r['grade']['content_correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),model_calls=len(rr),reused_mlp_calls=sum(sum(r['reused_mlp_calls']) for r in rr),attention_calls=sum(sum(r['attention_calls']) for r in rr),executed_mlp_calls=sum(sum(r['mlp_calls']) for r in rr),gate_queries=sum(len(r['events']) for r in rr),mlx_peak_bytes=max(r['mlx_peak_bytes'] for r in rr),rss_lifetime_peak_bytes=max(r['process_lifetime_rss_peak_bytes'] for r in rr),cached_activation_payload_bytes=max(r['cached_activation_payload_bytes'] for r in rr))
                    for metric in METRICS:row[metric]=sum(r[metric] for r in rr)
                    row['generated_tokens_per_second']=row['output_tokens']/row['wall_seconds'];repeats.append(row)
                rr=[r for r in repeats if r['phase']==phase and r['dataset']==ds and r['method']==method];summary=dict(phase=phase,dataset=ds,method=method,requests_per_repeat=len(ids))
                for metric in ('content_accuracy','format_compliance','model_calls','reused_mlp_calls','attention_calls','executed_mlp_calls','gate_queries','mlx_peak_bytes','rss_lifetime_peak_bytes','cached_activation_payload_bytes','generated_tokens_per_second')+METRICS:
                    for key,value in describe([r[metric] for r in rr],draws=1000).items():summary[metric+'_'+key]=value
                summaries.append(summary)
                for r in selected:
                    base=idx[phase,ds,r['repetition'],r['task_id'],'baseline'];a=r['grade']['content_correct'];b=base['grade']['content_correct']
                    cases.append(dict(phase=phase,dataset=ds,repetition=r['repetition'],task_id=r['task_id'],method=method,category=r['category'],question=r['question'],reference=tasks[r['task_id']]['reference'],answer=r['answer'],baseline_answer=base['answer'],correct=a,baseline_correct=b,format_ok=r['grade']['format_ok'],transition=('Correct' if b else 'Wrong')+'->'+('Correct' if a else 'Wrong'),token_stream_changed=r['token_ids']!=base['token_ids'],reused_mlp_calls=sum(r['reused_mlp_calls']),gate_queries=len(r['events']),wall_seconds=r['wall_seconds'],cpu_seconds=r['cpu_seconds'],output_tokens=r['output_tokens'],completion_reason=r['completion_reason'],sequence=r['sequence'],position=r['position']))
                for category in sorted({r['category'] for r in selected}):
                    rr=[r for r in selected if r['category']==category];groups.append(dict(phase=phase,dataset=ds,method=method,category=category,unique_questions=len(rr)//3,accuracy=statistics.mean(r['grade']['content_correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rr),reuse_count=sum(sum(r['reused_mlp_calls']) for r in rr)))
                if method=='baseline':continue
                diff=[[int(idx[phase,ds,rep,id,method]['grade']['content_correct'])-int(idx[phase,ds,rep,id,'baseline']['grade']['content_correct']) for rep in (1,2,3)] for id in ids]
                cmp=dict(phase=phase,dataset=ds,method=method,**accuracy_comparison(diff,draws=2000))
                for metric in ('wall_seconds','cpu_seconds','analytical_matmul_flops','output_tokens','mlx_peak_bytes'):
                    target={id:{rep:idx[phase,ds,rep,id,method][metric] for rep in (1,2,3)} for id in ids};base={id:{rep:idx[phase,ds,rep,id,'baseline'][metric] for rep in (1,2,3)} for id in ids}
                    for k,v in paired_ratio(target,base,draws=2000).items():
                        if k!='scope':cmp[metric+'_'+k]=v
                cmp['noninferiority_one_pp_proven']=cmp['conservative_exact_lower95']>=-.01
                cmp['observed_candidate_only']=cmp['accuracy_delta']>=0 and cmp['wall_seconds_ci95_high']<1 and cmp['analytical_matmul_flops_ratio']<1
                comparisons.append(cmp)
            ss=[r for r in summaries if r['phase']==phase and r['dataset']==ds]
            for r in ss:
                dominated=any(o['content_accuracy_mean']>=r['content_accuracy_mean'] and o['wall_seconds_mean']<=r['wall_seconds_mean'] and (o['content_accuracy_mean']>r['content_accuracy_mean'] or o['wall_seconds_mean']<r['wall_seconds_mean']) for o in ss)
                pareto.append(dict(phase=phase,dataset=ds,method=r['method'],accuracy=r['content_accuracy_mean'],wall_seconds=r['wall_seconds_mean'],point_pareto=not dominated,not_statistical_dominance=True))
    out=p/('ANALYSIS_'+uuid4().hex);out.mkdir()
    for name,records in [('DLR_RESULTS',summaries),('DLR_REPEATS',repeats),('DLR_COMPARISONS',comparisons),('DLR_CASES',cases),('DLR_FAILURES',[r for r in cases if r['phase']=='natural' and (not r['correct'] or r['transition']=='Correct->Wrong')]),('DLR_REUSE_EVENTS',events),('DLR_GROUPS',groups),('DLR_PARETO',pareto)]:write_union(out/(name+'.csv'),records)
    write_json(out/'AUDIT.json',{'real_requests':len(rows),'all_core_checks_passed':True,'fixed_streams_identical':True,'prior_files_unchanged':len(m['preserved_sha256']),'source':str(p),'reused_events':sum(e['reused'] for e in events),'gate_queries':len(events),'evaluation_not_used_for_thresholds':True,'semantic_scope':'closed MCQ/typed JSON/color aliases; not full natural-language adjudication','weights_hashes_checked_on_load':True,'score_source_sha256':m['source_sha256']})
    verdict=[]
    for method in METHODS[1:]:
        natural=[r for r in comparisons if r['method']==method and r['phase']=='natural'];fixed=[r for r in comparisons if r['method']==method and r['phase']=='fixed']
        verdict.append(dict(method=method,no_observed_accuracy_loss_both_cohorts=all(r['accuracy_delta']>=0 for r in natural),one_pp_noninferiority_proven=all(r['noninferiority_one_pp_proven'] for r in natural),equal_stream_model_matmul_reduced_both_cohorts=all(r['analytical_matmul_flops_ratio']<1 for r in fixed),equal_stream_wall_ci_below_one_both_cohorts=all(r['wall_seconds_ci95_high']<1 for r in fixed),equal_stream_cpu_ci_below_one_both_cohorts=all(r['cpu_seconds_ci95_high']<1 for r in fixed),actual_temporal_reuse_events=sum(e['reused'] for e in events if e['method']==method),strict_success_established=all(r['noninferiority_one_pp_proven'] for r in natural) and all(r['wall_seconds_ci95_high']<1 and r['cpu_seconds_ci95_high']<1 and r['analytical_matmul_flops_ratio']<1 for r in fixed),is_dlr=method in ('dlr','conservative')))
    write_json(out/'DLR_VERDICT.json',verdict)
    write_json(out/'SOURCE.json',{'experiment':str(p)});print('DLR ANALYSIS '+str(out));return out

if __name__=='__main__':analyze(sys.argv[1])
