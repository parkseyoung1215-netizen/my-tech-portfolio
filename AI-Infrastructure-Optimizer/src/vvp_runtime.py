"""Budgeted batch allocation from pre-verification inputs only."""
import math,random,time
if __package__:
    from .vvp_model import predict,estimate_cpu
    from .controlled_runtime import choose
else:
    from vvp_model import predict,estimate_cpu
    from controlled_runtime import choose
METHODS=('none','full','selective','rcav','improved','vvp')
BUDGETS=(0,10,25,50,75,100)

def allocate(primaries,method,budget,profiles,predictor,arrival):
    if method not in METHODS or budget not in BUDGETS:raise ValueError('Unknown condition')
    n=len(primaries);cap=math.floor(n*budget/100)
    if not cap or method=='none':return {'selected':[],'call_cap':cap,'predicted_cpu_quota':0.,'selected_estimated_cpu':0.}
    predictions={p['task_id']:(predict(p['question'],p,predictor) if method=='vvp' else {'estimated_cpu_seconds':estimate_cpu(p['question'],p,predictor)}) for p in primaries}
    quota=budget/100*sum(r['estimated_cpu_seconds'] for r in predictions.values())
    eligible=[]
    for p in primaries:
        id=p['task_id']
        if method=='full' or method=='vvp' and predictions[id]['expected_value']>0 or method not in ('full','vvp') and choose(p['question'],p,method,profiles)['verify']:eligible.append(id)
    priority={id:i for i,id in enumerate(arrival)}
    if method=='vvp':eligible.sort(key=lambda id:(-predictions[id]['value_per_estimated_cpu'],priority[id]))
    else:eligible.sort(key=priority.get)
    selected=[];spent=0.
    for id in eligible:
        cost=predictions[id]['estimated_cpu_seconds']
        if len(selected)<cap and spent+cost<=quota+1e-12:selected.append(id);spent+=cost
    # Only selection uses value order. Execute in common arrival order for fairness.
    selected.sort(key=priority.get)
    return {'selected':selected,'call_cap':cap,'predicted_cpu_quota':quota,'selected_estimated_cpu':spent}

def execute(primaries,method,budget,client,profiles,predictor,arrival):
    start,cpu=time.perf_counter(),time.process_time()
    plan=allocate(primaries,method,budget,profiles,predictor,arrival);selection=set(plan['selected'])
    planning_wall=time.perf_counter()-start;planning_cpu=time.process_time()-cpu
    calls=[];results={};lookup={p['task_id']:p for p in primaries}
    for id in plan['selected']:
        p=lookup[id];a,v=client.verify(p['question'],p['answer']);calls.append({**a,'task_id':id});results[id]=(a,v)
    rows=[]
    for p in primaries:
        id=p['task_id'];a,v=results.get(id,(None,None));selected=v is not None and v['prediction'] is not None
        rows.append({'task_id':id,'question':p['question'],'primary_answer':p['answer'],'primary_prediction':p['prediction'],
            'primary_source':p['source'],'primary_format_compliant':p['format_compliant'],
            'final_answer':a['answer'] if selected else p['answer'],'final_prediction':v['prediction'] if selected else p['prediction'],
            'final_format_compliant':v['format_compliant'] if selected else p['format_compliant'],
            'verification_executed':id in selection,'verification_answer':a['answer'] if a else '',
            'verification_prediction':v['prediction'] if v else None,'call_id':a['call_id'] if a else ''})
    return {'plan':plan,'rows':rows,'calls':calls,'planning_wall_seconds':planning_wall,'planning_cpu_seconds':planning_cpu}
