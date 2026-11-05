"""Frozen-checkpoint ACS analysis, with content/format/legacy scores kept separate."""
import hashlib,json,statistics,sys
from pathlib import Path
from collections import defaultdict
from .stable_runtime_records import Store
from .acs_policy import METHODS,boundary,threshold,limit,request_kind
from .acs_scoring import score
from .alsr_policy import estimated_matmul_flops
from .alsr_analysis import paired_ratio
from .controlled_statistics import describe,accuracy_comparison
from .efficiency_repeat import frozen_check
from .solver300_dataset import task_key
from .alsr_io import write_union
from .adaptive_experiment import write_json
METRICS=('wall_seconds','cpu_seconds','input_tokens','output_tokens','visible_output_tokens','model_calls','forward_calls','analytical_matmul_flops','decision_wall_seconds','probability_wall_seconds','probability_query_calls','probability_elements_scanned','model_step_wall_seconds','first_step_wall_seconds','decode_steps_wall_seconds')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(folder):
    p=Path(folder).resolve();m=json.loads((p/'MANIFEST.json').read_text());frozen_check(m)
    for f,h in m['runtime_source_sha256'].items():assert sha(f)==h
    assert sha(Path(__file__).resolve().parents[1]/'ACS_DESIGN.md')==m['design_sha256']
    rows=Store(p/'checkpoints').rows();assert len(rows)==128*len(METHODS)*3;idx={};tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};ids={ds:{t['task_id'] for t in ts} for ds,ts in m['datasets'].items()};assert not ids['fresh64']&ids['replay64']
    past=set()
    def walk(obj):
        if isinstance(obj,dict):
            if isinstance(obj.get('question'),str):past.add(task_key(obj['question']))
            for v in obj.values():walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    import csv
    data_root=Path(__file__).resolve().parents[1]/'data'
    for name in m.get('freshness_reference_preserved_sha256',m['preserved_sha256']):
        f=Path(name)
        if not f.is_relative_to(data_root):continue
        if f.suffix=='.json':walk(json.loads(f.read_text()))
        elif f.suffix=='.csv':
            with f.open() as stream:
                for r in csv.DictReader(stream):walk(r)
    assert not {task_key(tasks[id]['question']) for id in ids['fresh64']}&past
    probability_events=[]
    for r in rows:
        ds,rep,id,method=(r[k] for k in ('dataset','repetition','task_id','method'));key=(ds,rep,id,method);assert key not in idx;idx[key]=r;assert id in ids[ds] and rep in (1,2,3) and method in METHODS
        n=r['output_tokens'];assert r['model_calls']==1 and r['forward_calls']==n and r['layer_calls']==[n]*28 and r['kv_offsets']==[r['input_tokens']+n-1]*28
        assert r['token_cap']==limit(tasks[id]['question'],method);assert r['score']==score(tasks[id],r['answer']);fs=[]
        for j,(q,s,count) in enumerate(r['work_shapes']):
            assert count==28 and q==(r['input_tokens'] if j==0 else 1) and s==r['input_tokens']+j;fs.append(dict(query_tokens=q,context_tokens=s,executed=list(range(28)),head_positions=1))
        assert len(fs)==n and estimated_matmul_flops(m['model']['config'],fs)==r['analytical_matmul_flops']
        assert r['end_to_end_tokens_per_second']==n/r['wall_seconds'];assert r['native_audit'] is False
        count=0
        for e in r['stop_events']:
            assert 1<=e['position']<=n
            if e['selected_probability'] is not None:
                count+=1;assert method in ('confidence','adaptive') and e['candidate'];assert 0<=e['selected_probability']<=1.000001 and 0<=e['eos_competitor_mass']<=1.000001;assert e['threshold']==threshold(r['question'],r['input_tokens'],method);assert e['stop']==(e['selected_probability']>=e['threshold'])
                probability_events.append(dict(dataset=ds,repetition=rep,task_id=id,method=method,**e,stopped_content_correct=r['score']['content_correct'] if e['stop'] else None,stopped_incomplete=r['score']['incomplete_output'] if e['stop'] else None))
            elif e['stop']:assert method=='boundary' and e['candidate']
        assert count==r['probability_query_calls'];assert r['probability_elements_scanned']==count*m['model']['config']['vocab_size']
        if r['completion_reason'] not in ('eos','max_tokens'):assert r['stop_events'][-1]['stop'] and boundary(r['question'],r['answer'],adaptive=method=='adaptive')
    native=[]
    for f in (p/'sessions').glob('*.native_audits.json'):
        for r in json.loads(f.read_text())['records']:
            a=r['result'];assert a['native_attention_calls']==a['native_mlp_calls']==a['layer_calls'];native.append({'session_file':f.name,'method':r['method'],'forwards':a['forward_calls'],'actual_native_calls':sum(a['layer_calls'])})
    assert native
    parity=[];cases=[];failures=[];repeats=[];summary=[];groups=[];comparisons=[];legacy=[];verdict=[]
    bins=lambda n:'1-4' if n<=4 else '5-16' if n<=16 else '17-32' if n<=32 else '33+'
    baseline_lengths={id:statistics.mean(idx[ds,rep,id,'baseline']['output_tokens'] for rep in (1,2,3)) for ds,qq in ids.items() for id in qq}
    for ds,qq in ids.items():
        for method in METHODS:
            for rep in (1,2,3):
                rr=[idx[ds,rep,id,method] for id in sorted(qq)];row=dict(dataset=ds,method=method,repetition=rep,content_accuracy=statistics.mean(r['score']['content_correct'] for r in rr),format_compliance=statistics.mean(r['score']['format_ok'] for r in rr),task_success=statistics.mean(r['score']['task_success'] for r in rr),incomplete_rate=statistics.mean(r['score']['incomplete_output'] for r in rr),legacy_accuracy=statistics.mean(r['score']['legacy_grade']['correct'] for r in rr),legacy_format=statistics.mean(r['score']['legacy_grade']['format_ok'] for r in rr),mean_mlx_peak_bytes=statistics.mean(r['mlx_peak_bytes'] for r in rr),transformer_block_calls=sum(sum(r['layer_calls']) for r in rr),layer_token_evaluations=sum(q*n for r in rr for q,s,n in r['work_shapes']),mean_request_tps=statistics.mean(r['end_to_end_tokens_per_second'] for r in rr))
                for metric in METRICS:row[metric]=sum(r[metric] for r in rr)
                row['aggregate_tokens_per_second']=row['output_tokens']/row['wall_seconds'];row['aggregate_decode_tps']=sum(r['output_tokens']-1 for r in rr)/row['decode_steps_wall_seconds'] if row['decode_steps_wall_seconds']>0 else None;repeats.append(row)
                for r in rr:
                    b=idx[ds,rep,r['task_id'],'baseline'];prefix=r['token_ids']==b['token_ids'][:len(r['token_ids'])];parity.append(dict(dataset=ds,repetition=rep,task_id=r['task_id'],method=method,prefix_equal=prefix))
                    case=dict(dataset=ds,repetition=rep,task_id=r['task_id'],method=method,category=r['category'],difficulty=r['difficulty'],baseline_length_bin=bins(baseline_lengths[r['task_id']]),reference_length_bin=bins(r['reference_tokens']),baseline_content_correct=b['score']['content_correct'],content_correct=r['score']['content_correct'],format_ok=r['score']['format_ok'],task_success=r['score']['task_success'],incomplete_output=r['score']['incomplete_output'],transition=('Correct' if b['score']['content_correct'] else 'Wrong')+' -> '+('Correct' if r['score']['content_correct'] else 'Wrong'),completion_reason=r['completion_reason'],output_tokens=r['output_tokens'],baseline_tokens=b['output_tokens'],wall_seconds=r['wall_seconds'],baseline_wall_seconds=b['wall_seconds'],probability_queries=r['probability_query_calls'],answer=r['answer'],baseline_answer=b['answer'],reference=r['reference'],question=r['question'],error_type=r['score']['error_type'])
                    cases.append(case)
                    if rep==1 and (not r['score']['content_correct'] or not r['score']['format_ok']):failures.append(case)
            rr=[r for r in repeats if r['dataset']==ds and r['method']==method];s=dict(dataset=ds,method=method)
            for metric in rr[0]:
                if metric in ('dataset','method','repetition'):continue
                for k,v in describe([r[metric] for r in rr],draws=2000).items():s[metric+'_'+k]=v
            summary.append(s);legacy.append(dict(dataset=ds,method=method,historical_score=s['legacy_accuracy_mean'],new_content=s['content_accuracy_mean'],historical_format=s['legacy_format_mean'],new_format=s['format_compliance_mean'],full_task_success=s['task_success_mean'],note='Historical scores retained; sentence abbreviation/fact completeness and content-vs-format are separate audited metrics.'))
            for groupkey in ('category','difficulty','baseline_length_bin','reference_length_bin'):
                for val in sorted({r[groupkey] for r in cases if r['dataset']==ds and r['method']==method}):
                    kk=[r for r in cases if r['dataset']==ds and r['method']==method and r[groupkey]==val];groups.append(dict(dataset=ds,method=method,group_key=groupkey,group=val,unique_questions=len({r['task_id'] for r in kk}),content_accuracy=statistics.mean(r['content_correct'] for r in kk),format_compliance=statistics.mean(r['format_ok'] for r in kk),incomplete_rate=statistics.mean(r['incomplete_output'] for r in kk),wall_seconds_per_repeat=sum(r['wall_seconds'] for r in kk)/3,output_tokens_per_repeat=sum(r['output_tokens'] for r in kk)/3))
        pairs=[(x,'baseline') for x in METHODS if x!='baseline']+[('adaptive',x) for x in ('fixed8','fixed16','boundary','confidence')]
        for target,control in pairs:
            a=[[(int(idx[ds,rep,id,target]['score']['content_correct'])-int(idx[ds,rep,id,control]['score']['content_correct'])) for rep in (1,2,3)] for id in sorted(qq)];f=[[(int(idx[ds,rep,id,target]['score']['format_ok'])-int(idx[ds,rep,id,control]['score']['format_ok'])) for rep in (1,2,3)] for id in sorted(qq)]
            joint=[[int(idx[ds,rep,id,target]['score']['task_success'])-int(idx[ds,rep,id,control]['score']['task_success']) for rep in (1,2,3)] for id in sorted(qq)]
            c=dict(dataset=ds,target=target,control=control,**accuracy_comparison(a,draws=3000));c.update({'format_'+k:v for k,v in accuracy_comparison(f,draws=3000).items() if k!='inference_scope'});c.update({'task_'+k:v for k,v in accuracy_comparison(joint,draws=3000).items() if k!='inference_scope'})
            for metric in ('wall_seconds','cpu_seconds','analytical_matmul_flops','output_tokens'):
                aa={id:{rep:idx[ds,rep,id,target][metric] for rep in (1,2,3)} for id in sorted(qq)};bb={id:{rep:idx[ds,rep,id,control][metric] for rep in (1,2,3)} for id in sorted(qq)}
                for k,v in paired_ratio(aa,bb,draws=2500).items():
                    if k!='scope':c[metric+'_'+k]=v
            comparisons.append(c)
            if control=='baseline':
                s=next(s for s in summary if s['dataset']==ds and s['method']==target);b=next(s for s in summary if s['dataset']==ds and s['method']=='baseline');quality=c['accuracy_delta']>=0 and c['format_accuracy_delta']>=0 and c['task_accuracy_delta']>=0 and s['incomplete_rate_mean']<=b['incomplete_rate_mean'];speed=c['wall_seconds_ci95_high']<1;compute=c['analytical_matmul_flops_ratio']<1 and s['transformer_block_calls_mean']<b['transformer_block_calls_mean'];prefix_ok=all(r['prefix_equal'] for r in parity if r['dataset']==ds and r['method']==target);verdict.append(dict(dataset=ds,method=target,quality_point_not_worse=quality,latency_ci_improved=speed,model_work_reduced=compute,observed_candidate=quality and speed and compute and prefix_ok,population_noninferiority_proven=c['conservative_exact_lower95']>=0,extra_probability_reductions_not_in_matmul_estimate=target in ('confidence','adaptive')))
    pareto=[]
    for s in summary:
        competitors=[t for t in summary if t['dataset']==s['dataset'] and t['method']!=s['method']];dom=[t['method'] for t in competitors if t['content_accuracy_mean']>=s['content_accuracy_mean'] and t['format_compliance_mean']>=s['format_compliance_mean'] and t['wall_seconds_mean']<=s['wall_seconds_mean'] and (t['content_accuracy_mean']>s['content_accuracy_mean'] or t['format_compliance_mean']>s['format_compliance_mean'] or t['wall_seconds_mean']<s['wall_seconds_mean'])];pareto.append(dict(dataset=s['dataset'],method=s['method'],point_pareto=not dom,dominated_by=dom,note='Observed means, not statistical dominance or a full cap sweep'))
    probability_audit=[]
    for ds in ids:
        for method in ('confidence','adaptive'):
            for low,high in ((0,.75),(.75,.9),(.9,.99),(.99,1.000001)):
                ee=[e for e in probability_events if e['dataset']==ds and e['method']==method and e['stop'] and low<=e['selected_probability']<high]
                if ee:probability_audit.append(dict(dataset=ds,method=method,token_probability_bin=f'{low}-{high}',repeat_events=len(ee),unique_questions=len({e['task_id'] for e in ee}),mean_selected_token_probability=statistics.mean(e['selected_probability'] for e in ee),stopped_content_accuracy=statistics.mean(e['stopped_content_correct'] for e in ee),stopped_incomplete_rate=statistics.mean(e['stopped_incomplete'] for e in ee),scope='Stopped candidates only; token likelihood is not an answer-correctness probability. Rejected candidate content not graded.'))
    out=p/('ANALYSIS_'+__import__('uuid').uuid4().hex);out.mkdir()
    for name,rr in {'RESULTS':summary,'REPEATS':repeats,'COMPARISONS':comparisons,'CASES':cases,'FAILURES':failures,'GROUPS':groups,'PROBABILITY_EVENTS':probability_events,'TOKEN_PROBABILITY_AUDIT':probability_audit,'PREFIX_PARITY':parity,'LEGACY_VS_CONTENT':legacy,'VERDICT':verdict,'PARETO':pareto,'NATIVE_AUDIT':native}.items():write_union(out/(name+'.csv'),rr)
    audit=dict(existing_files_unchanged=len(m['preserved_sha256']),new_stem_past_overlap=0,replay_fresh_disjoint=True,frozen_policy_and_no_eval_tuning=True,all_model_calls_one_per_request=True,all_28_layers_and_cache_verified=True,native_attention_mlp_audit_verified=True,actual_probability_queries=len(probability_events),probabilities_in_range=True,confidence_is_not_calibrated_correctness_probability=True,prefix_disagreements=sum(not r['prefix_equal'] for r in parity),requests=len(rows),unique_questions=128,repeats=3,old_replay_difficulty_unrated=True,fresh_challenge_abbreviation_cases=8,source_score_preserved=True,GPU_elapsed_power_money='unmeasured')
    audit.update(explicit_replay=m.get('explicit_replay',False),freshness_scope='Original pre-ACS holdout provenance; explicit replays are not newly unused problems.')
    write_json(out/'AUDIT.json',audit);write_json(out/'SOURCE.json',{'experiment':str(p),'manifest_sha256':sha(p/'MANIFEST.json')});print(json.dumps({'output':str(out),'audit':audit,'candidates':[v for v in verdict if v['observed_candidate']]},indent=2));return out

if __name__=='__main__':analyze(sys.argv[1])
