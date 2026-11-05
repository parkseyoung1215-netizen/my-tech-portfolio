"""Independent guarded replay; identical client/prompts and original RCAV profile."""
import time
if __package__:
    from .rcav_runtime import verifier_prompt
    from . import rcav_policy as base
    from . import rcav_guard_policy as guard
    from .skipwork_runtime import preflight
else:
    from rcav_runtime import verifier_prompt
    import rcav_policy as base
    import rcav_guard_policy as guard
    from skipwork_runtime import preflight
METHODS=('rcav','guarded','selective')

def execute_block(questions,method,client,base_profile,guard_profile,selective_profile):
    if method not in METHODS:raise ValueError('Unknown method')
    start,cpu=time.perf_counter(),time.process_time();rows=[];calls=[]
    for q in questions:
        request_start=time.perf_counter();p=client.primary(q);calls.extend(p['calls']);chars=len(verifier_prompt(q,p['answer']))
        if method=='guarded':d=guard.decide(q,p,chars,base_profile,guard_profile)
        elif method=='rcav':d=base.decide(q,p,chars,base_profile,.05)
        else:
            old=preflight(q,'selective',selective_profile);d={'verify':not old['skip_verifier'],'reason':old['reason']}
        row={'question':q,'method':method,'model':client.model,'decision':d,'verification_prompt_chars':chars,
            'primary_answer':p['answer'],'primary_prediction':p['prediction'],'primary_format_compliant':p['format_compliant'],
            'primary_source':p['source'],'primary_parse_mode':p['parse_mode'],'tool_calls':p['tool_calls'],'tool_error':p['tool_error'],
            'final_answer':p['answer'],'final_prediction':p['prediction'],'final_format_compliant':p['format_compliant'],
            'verification_executed':False,'verification_answer':'','verification_prediction':None,'verification_format_compliant':None,
            'verification_model_seconds':None,'verification_preserved_primary':False,'call_ids':[a['call_id'] for a in p['calls']]}
        if d['verify']:
            a,v=client.verify(q,p['answer']);calls.append(a);row['call_ids'].append(a['call_id'])
            row.update(verification_executed=True,verification_answer=a['answer'],verification_prediction=v['prediction'],
                verification_format_compliant=v['format_compliant'],verification_model_seconds=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None,
                verification_preserved_primary=v['prediction'] is None)
            if v['prediction'] is not None:row.update(final_answer=a['answer'],final_prediction=v['prediction'],final_format_compliant=v['format_compliant'])
        row.update(service_seconds=time.perf_counter()-request_start,completion_latency_seconds=time.perf_counter()-start)
        rows.append(row)
    return {'rows':rows,'calls':calls,'processing_seconds':time.perf_counter()-start,'router_cpu_seconds':time.process_time()-cpu}
