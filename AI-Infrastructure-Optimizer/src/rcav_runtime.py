"""Local model client and verification decision boundary; no evaluation labels."""
import hashlib,json,time,urllib.request
from uuid import uuid4
if __package__:
    from .efficiency_protocol import extract
    from .solver300_runtime import features
    from .safe_calculator import calculate,parse_expression
    from .skipwork_runtime import preflight
    from . import rcav_policy as policy
else:
    from efficiency_protocol import extract
    from solver300_runtime import features
    from safe_calculator import calculate,parse_expression
    from skipwork_runtime import preflight
    import rcav_policy as policy

def verifier_prompt(question,candidate):
    f=features(question)
    return (question+'\nCandidate response: '+json.dumps(candidate,ensure_ascii=False)+
        '\nIndependently check the candidate and solve the original request. The candidate may be wrong. '+
        ('Return only the correct option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the correct integer.')+' Do not explain.')

class LocalClient:
    def __init__(self,model):self.model=model
    def send(self,prompt,stage):
        payload={'model':self.model,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':64}}
        body=json.dumps(payload).encode();start=time.perf_counter()
        a={'call_id':uuid4().hex,'stage':stage,'model':self.model,'request_payload':payload,
            'request_sha256':hashlib.sha256(body).hexdigest(),'answer':'','status':'api_error','error':'',
            'input_tokens':None,'output_tokens':None,'eval_seconds':None,'prefill_seconds':None,'total_model_seconds':None,
            'load_seconds':None,'done_reason':None,'format_compliant':False,'parse_mode':'unresolved'}
        try:
            req=urllib.request.Request('http://localhost:11434/api/generate',data=body,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
            a.update(answer=r['response'].strip(),status='generated',input_tokens=r.get('prompt_eval_count'),output_tokens=r.get('eval_count'),done_reason=r.get('done_reason'))
            for key,server in [('eval_seconds','eval_duration'),('prefill_seconds','prompt_eval_duration'),('total_model_seconds','total_duration'),('load_seconds','load_duration')]:a[key]=r[server]/1e9 if server in r else None
        except Exception as exc:a['error']=f'{type(exc).__name__}: {exc}'
        a['elapsed_seconds']=time.perf_counter()-start
        return a
    def primary(self,question):
        f=features(question);start=time.perf_counter()
        if f['tool_supported']:
            try:answer=str(calculate(parse_expression(question)));error=''
            except Exception as exc:answer='';error=f'{type(exc).__name__}: {exc}'
            return {'answer':answer,**extract(question,answer,f['answer_kind']),'source':'tool' if not error else 'tool_error',
                'calls':[],'tool_calls':1,'tool_error':error,'tool_seconds':time.perf_counter()-start}
        instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
        a=self.send(question+'\n'+instruction+' Do not explain.','primary');p=extract(question,a['answer'],f['answer_kind'])
        a.update(format_compliant=p['format_compliant'],parse_mode=p['parse_mode'])
        return {'answer':a['answer'],**p,'source':'model','calls':[a],'tool_calls':0,'tool_error':'','tool_seconds':0.}
    def verify(self,question,candidate):
        a=self.send(verifier_prompt(question,candidate),'verify');v=extract(question,a['answer'],features(question)['answer_kind'])
        a.update(format_compliant=v['format_compliant'],parse_mode=v['parse_mode'])
        return a,v

def execute_block(questions,method,client,profile,selective_profile):
    if method not in policy.METHODS:raise ValueError('Unknown method')
    start,cpu=time.perf_counter(),time.process_time();rows=[];calls=[]
    for q in questions:
        request_start=time.perf_counter();p=client.primary(q);calls.extend(p['calls'])
        chars=len(verifier_prompt(q,p['answer']))
        # Everything passed to the policy exists before verify() is called.
        if method in policy.WEIGHTS:
            decision=policy.decide(q,p,chars,profile,policy.WEIGHTS[method]);do_verify=decision['verify']
        else:
            old=preflight(q,method,selective_profile)
            decision={'verify':not old['skip_verifier'],'reason':old['reason']};do_verify=decision['verify']
        decision_at=time.perf_counter()
        row={'question':q,'method':method,'model':client.model,'decision':decision,'verification_prompt_chars':chars,
            'primary_answer':p['answer'],'primary_prediction':p['prediction'],'primary_format_compliant':p['format_compliant'],
            'primary_source':p['source'],'primary_parse_mode':p['parse_mode'],'tool_calls':p['tool_calls'],
            'tool_error':p['tool_error'],'tool_seconds':p['tool_seconds'],
            'final_answer':p['answer'],'final_prediction':p['prediction'],'final_format_compliant':p['format_compliant'],
            'verification_executed':False,'verification_answer':'','verification_prediction':None,
            'verification_format_compliant':None,'verification_model_seconds':None,'verification_client_seconds':0.,
            'verification_preserved_primary':False,'call_ids':[a['call_id'] for a in p['calls']],
            'decision_elapsed_before_verify_seconds':decision_at-request_start}
        if do_verify:
            a,v=client.verify(q,p['answer']);calls.append(a);row['call_ids'].append(a['call_id'])
            seconds=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None
            row.update(verification_executed=True,verification_answer=a['answer'],verification_prediction=v['prediction'],
                verification_format_compliant=v['format_compliant'],verification_model_seconds=seconds,
                verification_client_seconds=a['elapsed_seconds'],verification_preserved_primary=v['prediction'] is None)
            if v['prediction'] is not None:row.update(final_answer=a['answer'],final_prediction=v['prediction'],final_format_compliant=v['format_compliant'])
        row.update(service_seconds=time.perf_counter()-request_start,completion_latency_seconds=time.perf_counter()-start)
        rows.append(row)
    return {'rows':rows,'calls':calls,'processing_seconds':time.perf_counter()-start,'router_cpu_seconds':time.process_time()-cpu}
