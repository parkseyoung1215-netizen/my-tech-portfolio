"""Post-run limited semantic fact audit. No stopping policy or original score changes."""
import json,re,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .controlled_statistics import accuracy_comparison
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .acs_policy import METHODS

def audit_fact(task,answer,original):
    coverage=original['content_correct'];incomplete=original['incomplete_output'];synonym=False;additions=[]
    if task['score_kind']=='constraints':
        required=list(task['required'])+re.findall(r'\bis (?:a|an) ([A-Za-z]+)[.!]',task['question']);flags=[]
        for fact in required:
            if fact.casefold()=='doctor':
                direct=bool(re.search(r'\bdoctor\b',answer,re.I));equivalent=bool(re.search(r'\bphysician\b',answer,re.I));flags.append(direct or equivalent);synonym|=equivalent and not direct
            else:flags.append(fact.casefold() in answer.casefold())
        coverage=all(flags);incomplete=not coverage or not bool(re.search(r'[.!?]["\']?$',answer.strip()))
        # Explicit observed additions: not a general hallucination classifier.
        additions=[word for word in ('dedicated','diligently','exceptional','compassionate','patients','community') if re.search(r'\b'+word+r'\b',answer,re.I) and not re.search(r'\b'+word+r'\b',task['question'],re.I)]
    return dict(audited_fact_coverage=bool(coverage),audited_incomplete=bool(incomplete),synonym_recognized=synonym,observed_unrequested_terms=additions,instruction_fidelity_known_failure=bool(additions),scope='Limited doctor/physician fact equivalence and identified added words; not complete semantic/fidelity adjudication.')

def run(folder):
    out=Path(folder);p=Path(json.loads((out/'SOURCE.json').read_text())['experiment']);m=json.loads((p/'MANIFEST.json').read_text());tasks={t['task_id']:t for ts in m['datasets'].values() for t in ts};cases=read_decoded(out/'CASES.csv');literal=read_decoded(out/'RESULTS.csv');original_comp=read_decoded(out/'COMPARISONS.csv');annot=[];index={}
    for c in cases:
        a=audit_fact(tasks[c['task_id']],c['answer'],{'content_correct':c['content_correct'],'incomplete_output':c['incomplete_output']});row=dict(c,**a);annot.append(row);index[c['dataset'],c['method'],c['task_id'],c['repetition']]=row
    results=[];comparisons=[];verdict=[]
    for ds in ('replay64','fresh64'):
        ids=sorted({c['task_id'] for c in cases if c['dataset']==ds})
        for method in METHODS:
            rr=[r for r in annot if r['dataset']==ds and r['method']==method];old=next(r for r in literal if r['dataset']==ds and r['method']==method)
            results.append(dict(dataset=ds,method=method,literal_content_accuracy=old['content_accuracy_mean'],audited_fact_coverage=statistics.mean(r['audited_fact_coverage'] for r in rr),format_compliance=old['format_compliance_mean'],audited_fact_and_format=statistics.mean(r['audited_fact_coverage'] and r['format_ok'] for r in rr),audited_incomplete_rate=statistics.mean(r['audited_incomplete'] for r in rr),identified_fidelity_failure_rate=statistics.mean(r['instruction_fidelity_known_failure'] for r in rr),synonym_changes=sum(r['content_correct']!=r['audited_fact_coverage'] for r in rr)/3,wall_seconds=old['wall_seconds_mean'],cpu_seconds=old['cpu_seconds_mean'],note='Post-run grading sensitivity; original scores/model timings preserved.'))
        pairs=[(x,'baseline') for x in METHODS if x!='baseline']+[('adaptive',x) for x in ('fixed8','fixed16','boundary','confidence')]
        for target,control in pairs:
            dd=[[int(index[ds,target,id,rep]['audited_fact_coverage'])-int(index[ds,control,id,rep]['audited_fact_coverage']) for rep in (1,2,3)] for id in ids];delta=accuracy_comparison(dd,draws=3000);orig=next(r for r in original_comp if r['dataset']==ds and r['target']==target and r['control']==control);c=dict(dataset=ds,target=target,control=control,**delta,wall_seconds_ratio=orig['wall_seconds_ratio'],wall_seconds_ci95_low=orig['wall_seconds_ci95_low'],wall_seconds_ci95_high=orig['wall_seconds_ci95_high'],cpu_seconds_ratio=orig['cpu_seconds_ratio'],analytical_matmul_flops_ratio=orig['analytical_matmul_flops_ratio']);comparisons.append(c)
            if control=='baseline':
                a=next(r for r in results if r['dataset']==ds and r['method']==target);b=next(r for r in results if r['dataset']==ds and r['method']=='baseline');quality=delta['accuracy_delta']>=0 and a['format_compliance']>=b['format_compliance'] and a['audited_fact_and_format']>=b['audited_fact_and_format'] and a['audited_incomplete_rate']<=b['audited_incomplete_rate'] and a['identified_fidelity_failure_rate']<=b['identified_fidelity_failure_rate'];candidate=quality and c['wall_seconds_ci95_high']<1 and c['analytical_matmul_flops_ratio']<1
                verdict.append(dict(dataset=ds,method=target,audited_fact_quality_not_worse=quality,limited_observed_candidate=candidate,population_noninferiority_proven=delta['conservative_exact_lower95']>=0,global_semantic_correctness_proven=False,scoring_audit_is_posthoc=True))
    write_union(out/'CONTENT_AUDIT_CASES.csv',annot);write_union(out/'AUDITED_RESULTS.csv',results);write_union(out/'AUDITED_COMPARISONS.csv',comparisons);write_union(out/'AUDITED_VERDICT.csv',verdict);details(out);pareto(out)
    write_json(out/'CONTENT_AUDIT_SCOPE.json',{'posthoc':True,'reason':'All8 doctor cases used physician in Baseline; literal matching hid fact loss from stops at Dr.','equivalence_assumption':'doctor and physician treated as equivalent requested role for this limited audit','policy_or_threshold_or_model_output_changes':False,'original_scores_overwritten':False,'identified_fidelity_additions':'Dedicated/diligently and some invented patient/community details remain quality failures; audit is not full semantic evaluation','not_a_new_blind_holdout':True})
    print(json.dumps({'audited_results':results,'verdict':verdict},indent=2))

def details(out):
    out=Path(out);annot=read_decoded(out/'CONTENT_AUDIT_CASES.csv');index={(r['dataset'],r['method'],r['task_id'],r['repetition']):r for r in annot};audited_failures=[];audited_groups=[]
    for r in annot:
        base=index[r['dataset'],'baseline',r['task_id'],r['repetition']];r['audited_baseline_fact_coverage']=base['audited_fact_coverage'];r['audited_transition']=('Correct' if base['audited_fact_coverage'] else 'Wrong')+' -> '+('Correct' if r['audited_fact_coverage'] else 'Wrong')
        if r['repetition']==1 and (not r['audited_fact_coverage'] or not r['format_ok'] or r['instruction_fidelity_known_failure']):audited_failures.append(r)
    for ds in ('replay64','fresh64'):
        for method in METHODS:
            for field in ('category','difficulty','baseline_length_bin','reference_length_bin'):
                for value in sorted({r[field] for r in annot if r['dataset']==ds and r['method']==method}):
                    rr=[r for r in annot if r['dataset']==ds and r['method']==method and r[field]==value];audited_groups.append(dict(dataset=ds,method=method,group_key=field,group=value,unique_questions=len({r['task_id'] for r in rr}),content_accuracy=statistics.mean(r['audited_fact_coverage'] for r in rr),format_compliance=statistics.mean(r['format_ok'] for r in rr),incomplete_rate=statistics.mean(r['audited_incomplete'] for r in rr),wall_seconds_per_repeat=sum(r['wall_seconds'] for r in rr)/3,output_tokens_per_repeat=sum(r['output_tokens'] for r in rr)/3))
    write_union(out/'AUDITED_TRANSITIONS.csv',annot);write_union(out/'AUDITED_FAILURES.csv',audited_failures);write_union(out/'AUDITED_GROUPS.csv',audited_groups)

def pareto(folder):
    out=Path(folder);rows=read_decoded(out/'AUDITED_RESULTS.csv');result=[]
    for r in rows:
        dd=[x['method'] for x in rows if x['dataset']==r['dataset'] and x['method']!=r['method'] and x['audited_fact_coverage']>=r['audited_fact_coverage'] and x['format_compliance']>=r['format_compliance'] and x['wall_seconds']<=r['wall_seconds'] and (x['audited_fact_coverage']>r['audited_fact_coverage'] or x['format_compliance']>r['format_compliance'] or x['wall_seconds']<r['wall_seconds'])];result.append(dict(dataset=r['dataset'],method=r['method'],point_pareto=not dd,dominated_by=dd,scope='Limited posthoc fact audit and measured timing; no statistical dominance guarantee.'))
    write_union(out/'AUDITED_PARETO.csv',result)

if __name__=='__main__':run(sys.argv[1])
