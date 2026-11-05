"""Immutable actual-request audit and small-sample paired uncertainty."""
import hashlib,json,statistics,sys
from pathlib import Path
from uuid import uuid4
from .oracle_policy import RATES,validate
from .oracle_data import score
from .dlr_policy import estimate
from .stable_runtime_records import Store
from .efficiency_repeat import frozen_check
from .controlled_statistics import describe,accuracy_comparison
from .alsr_analysis import paired_ratio
from .alsr_io import write_union
from .adaptive_experiment import write_json
CONDITIONS=[('baseline',None),('existing',None)]+[(m,r) for m in ('ideal','random','overhead') for r in RATES]
METRICS=('wall_seconds','cpu_seconds','model_wall_seconds','model_cpu_seconds','input_tokens','output_tokens','analytical_matmul_flops','gate_arithmetic_estimate','gate_wall_seconds','gate_cpu_seconds','gate_queries','pre_gate_sync_wall_seconds','cache_python_wall_seconds','route_lookup_wall_seconds')
def analyze(folder):
    p=Path(folder).resolve();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for k in ('core_sha256','legacy_sha256'):
        for f,h in m[k].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};tables={id:json.loads((p/'tables'/id/'TABLES.json').read_text()) for id in tasks};rows=Store(p/'checkpoints').rows();assert len(rows)==4000;index={}
    for receipt in p.glob('*_BUILD.json'):
        for id,h in json.loads(receipt.read_text())['table_sha256'].items():assert hashlib.sha256((p/'tables'/id/'TABLES.json').read_bytes()).hexdigest()==h
    for id,table in tables.items():
        for typ in ('oracle','random'):
            for rate in RATES:validate(table[typ][str(rate)],rate)
    cases=[];summaries=[];repeats=[];comparisons=[];verdicts=[]
    for r in rows:
        phase,ds,rep,id,method,rate=(r[k] for k in ('phase','dataset','repetition','task_id','method','rate'));k=(phase,ds,rep,id,method,rate);assert k not in index;index[k]=r;n=r['output_tokens'];assert n==len(r['token_ids'])==len(r['work_shapes'])
        assert r['grade']==score(tasks[id],r['answer']);assert r['attention_calls']==[n]*28 and r['kv_offsets']==[r['input_tokens']+n-1]*28
        assert all(r['mlp_calls'][i]+r['reused_mlp_calls'][i]==n for i in range(28));assert sum(r['mlp_calls'])==sum(f[3] for f in r['work_shapes']);assert estimate(m['model']['config'],r['work_shapes'])==r['analytical_matmul_flops']
        if method in ('ideal','random','overhead'):
            plan=tables[id]['random' if method=='random' else 'oracle'][str(rate)];assert sum(r['reused_mlp_calls'])==sum(map(len,plan[:n]));assert r['reused_mlp_calls']==[sum(i in s for s in plan[:n]) for i in range(28)]
            if method!='overhead':assert r['gate_queries']==0 and r['gate_wall_seconds']==0
        if method=='baseline':assert sum(r['reused_mlp_calls'])==0
        assert r['gate_arithmetic_estimate']==r['gate_queries']*(5*m['model']['config']['hidden_size']+4)
        if phase=='fixed':assert r['token_ids']==next(x['token_ids'] for x in rows if x['phase']=='natural' and x['dataset']==ds and x['task_id']==id and x['method']=='baseline' and x['repetition']==1)
    natural_overhead_mismatches=0;fixed_overhead_prediction_mismatches=0
    for ds,ts in m['datasets'].items():
        for t in ts:
            for rep in range(1,6):
                for rate in RATES:
                    ideal=index['natural',ds,rep,t['task_id'],'ideal',rate];over=index['natural',ds,rep,t['task_id'],'overhead',rate]
                    natural_overhead_mismatches+=ideal['token_ids']!=over['token_ids']
                    ideal=index['fixed',ds,rep,t['task_id'],'ideal',rate];over=index['fixed',ds,rep,t['task_id'],'overhead',rate]
                    fixed_overhead_prediction_mismatches+=ideal['predicted_token_ids']!=over['predicted_token_ids']
    for phase in ('natural','fixed'):
        for ds,ts in m['datasets'].items():
            ids=[t['task_id'] for t in ts]
            for method,rate in CONDITIONS:
                for rep in range(1,6):
                    rr=[index[phase,ds,rep,id,method,rate] for id in ids];bb=[index[phase,ds,rep,id,'baseline',None] for id in ids];skipped=sum(sum(r['reused_mlp_calls']) for r in rr);decode_slots=sum(28*(r['output_tokens']-1) for r in rr);all_tokens=sum(28*(r['input_tokens']+r['output_tokens']-1) for r in rr)
                    row=dict(phase=phase,dataset=ds,method=method,rate=rate,repetition=rep,requests=len(rr),content_accuracy=statistics.mean(r['grade']['content_correct'] for r in rr),format_compliance=statistics.mean(r['grade']['format_ok'] for r in rr),output_agreement=statistics.mean(r['token_ids']==b['token_ids'] for r,b in zip(rr,bb)),reused_mlp_calls=skipped,executed_mlp_calls=sum(sum(r['mlp_calls']) for r in rr),attention_calls=sum(sum(r['attention_calls']) for r in rr),achieved_decode_skip_fraction=skipped/decode_slots if decode_slots else 0,achieved_all_layer_token_skip_fraction=skipped/all_tokens,mlx_peak_bytes=max(r['mlx_peak_bytes'] for r in rr),rss_lifetime_peak_bytes=max(r['process_lifetime_rss_peak_bytes'] for r in rr),cache_logical_bytes=max(r.get('cache_logical_bytes',r.get('cached_activation_payload_bytes',0)) for r in rr))
                    for metric in METRICS:
                        vals=[r[metric] for r in rr];row[metric]=None if any(v is None for v in vals) else sum(vals)
                    row['mean_seconds_per_output_token']=row['wall_seconds']/row['output_tokens'];row['saved_mlp_matmul_flops_estimate']=skipped*6*m['model']['config']['hidden_size']*m['model']['config']['intermediate_size'];row['limited_net_arithmetic_proxy']=row['analytical_matmul_flops']+row['gate_arithmetic_estimate'];repeats.append(row)
                    for r,b in zip(rr,bb):
                        correct=r['grade']['content_correct'];bc=b['grade']['content_correct'];cases.append(dict(phase=phase,dataset=ds,repetition=rep,task_id=r['task_id'],method=method,rate=rate,category=r['category'],question=r['question'],reference=tasks[r['task_id']]['reference'],answer=r['answer'],baseline_answer=b['answer'],correct=correct,baseline_correct=bc,format_ok=r['grade']['format_ok'],output_match=r['token_ids']==b['token_ids'],transition=('Correct' if bc else 'Wrong')+'->'+('Correct' if correct else 'Wrong'),output_tokens=r['output_tokens'],baseline_output_tokens=b['output_tokens'],wall_seconds=r['wall_seconds'],cpu_seconds=r['cpu_seconds'],reused_mlp_calls=sum(r['reused_mlp_calls']),sequence=r['sequence'],position=r['position'],completion_reason=r['completion_reason']))
                rr=[r for r in repeats if r['phase']==phase and r['dataset']==ds and r['method']==method and r['rate']==rate];s=dict(phase=phase,dataset=ds,method=method,rate=rate,unique_questions=len(ids))
                metrics=METRICS+('content_accuracy','format_compliance','output_agreement','reused_mlp_calls','executed_mlp_calls','attention_calls','achieved_decode_skip_fraction','achieved_all_layer_token_skip_fraction','mlx_peak_bytes','rss_lifetime_peak_bytes','cache_logical_bytes','mean_seconds_per_output_token','saved_mlp_matmul_flops_estimate','limited_net_arithmetic_proxy')
                for metric in metrics:
                    for key,val in describe([r[metric] for r in rr],draws=1000).items():s[metric+'_'+key]=val
                summaries.append(s)
                if method=='baseline':continue
                controls=[('baseline',None)]+([('ideal',rate)] if method=='overhead' else [])
                for control,cr in controls:
                    pairs=[[int(index[phase,ds,rep,id,method,rate]['grade']['content_correct'])-int(index[phase,ds,rep,id,control,cr]['grade']['content_correct']) for rep in range(1,6)] for id in ids];c=dict(phase=phase,dataset=ds,method=method,rate=rate,control=control,control_rate=cr,**accuracy_comparison(pairs,draws=1500))
                    for metric in ('wall_seconds','cpu_seconds','model_wall_seconds','analytical_matmul_flops','output_tokens'):
                        t={id:{rep:index[phase,ds,rep,id,method,rate][metric] for rep in range(1,6)} for id in ids};b={id:{rep:index[phase,ds,rep,id,control,cr][metric] for rep in range(1,6)} for id in ids}
                        for key,val in paired_ratio(t,b,draws=1500).items():
                            if key!='scope':c[metric+'_'+key]=val
                    c['repeat_wall_ratios']=[sum(index[phase,ds,rep,id,method,rate]['wall_seconds'] for id in ids)/sum(index[phase,ds,rep,id,control,cr]['wall_seconds'] for id in ids) for rep in range(1,6)];comparisons.append(c)
    for ds in m['datasets']:
        base=next(s for s in summaries if s['phase']=='natural' and s['dataset']==ds and s['method']=='baseline')
        for rate in RATES[1:]:
            natural=next(c for c in comparisons if c['phase']=='natural' and c['dataset']==ds and c['method']=='ideal' and c['rate']==rate and c['control']=='baseline');fixed=next(c for c in comparisons if c['phase']=='fixed' and c['dataset']==ds and c['method']=='ideal' and c['rate']==rate and c['control']=='baseline');s=next(s for s in summaries if s['phase']=='natural' and s['dataset']==ds and s['method']=='ideal' and s['rate']==rate)
            passed=base['content_accuracy_mean']>0 and natural['accuracy_delta']>=0 and natural['any_repeat_loss_questions']==0 and s['format_compliance_mean']>=base['format_compliance_mean'] and fixed['wall_seconds_ratio']<=.95 and fixed['wall_seconds_ci95_high']<1 and all(r<1 for r in fixed['repeat_wall_ratios']) and s['reused_mlp_calls_mean']>0
            verdicts.append(dict(dataset=ds,rate=rate,conditional_opportunity=passed,no_observed_accuracy_loss=natural['accuracy_delta']>=0,individual_loss_questions=natural['any_repeat_loss_questions'],format_not_lower=s['format_compliance_mean']>=base['format_compliance_mean'],baseline_quality_nonzero=base['content_accuracy_mean']>0,matched_wall_saving=1-fixed['wall_seconds_ratio'],matched_ci_high=fixed['wall_seconds_ci95_high'],all_five_repeats_faster=all(r<1 for r in fixed['repeat_wall_ratios']),population_noninferiority_proven=False))
    common=[rate for rate in RATES[1:] if all(v['conditional_opportunity'] for v in verdicts if v['rate']==rate)];category='A' if common else 'B' if any(v['conditional_opportunity'] for v in verdicts) else 'C'
    out=p/('ANALYSIS_'+uuid4().hex);out.mkdir()
    for name,records in [('ORACLE_RESULTS',summaries),('ORACLE_REPEATS',repeats),('ORACLE_COMPARISONS',comparisons),('ORACLE_CASES',cases),('ORACLE_FAILURES',[c for c in cases if c['phase']=='natural' and (not c['correct'] or not c['output_match'])]),('ORACLE_VERDICTS',verdicts)]:write_union(out/(name+'.csv'),records)
    write_json(out/'AUDIT.json',{'actual_primary_requests':len(rows),'original_files_unchanged':len(m['preserved_sha256']),'table_quotas_and_native_mlp_counts_checked':True,'all_attention_kv_offsets_checked':True,'fixed_streams_identical':True,'oracle_uses_evaluation_labels_and_future':True,'mathematical_oracle_optimality_proven':False,'actual_gpu_kernel_count_flops_power_money_measured':False,'offline_table_generation_seconds':sum(t['builder_wall_seconds'] for t in tables.values()),'offline_table_generation_cpu_seconds':sum(t['builder_cpu_seconds'] for t in tables.values()),'offline_trace_model_calls':sum(t['trace_model_calls'] for t in tables.values()),'offline_candidate_model_calls':sum(t['candidate_model_calls'] for t in tables.values())})
    write_json(out/'OVERHEAD_EQUIVALENCE.json',{'natural_pairs':600,'natural_token_mismatches':natural_overhead_mismatches,'fixed_prediction_pairs':600,'fixed_prediction_mismatches':fixed_overhead_prediction_mismatches,'note':'Fixed forced texts alone cannot prove equivalent computation, so actual per-step predictions are compared.'})
    write_json(out/'CLASSIFICATION.json',{'classification':category,'common_qualifying_rates':common,'qualified_conditions':[v for v in verdicts if v['conditional_opportunity']],'scope':'Only privileged best-of-three schedules on this small task set; not a deployable algorithm or exhaustive DLR impossibility result','population_accuracy_retention_proven':False});write_json(out/'SOURCE.json',{'experiment':str(p)});print('ORACLE ANALYSIS '+str(out),flush=True);return out

if __name__=='__main__':analyze(sys.argv[1])
