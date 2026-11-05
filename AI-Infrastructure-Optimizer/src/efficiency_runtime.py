"""Same omission policies/prompts, corrected protocol, expanded actual measurements."""
import hashlib,json,time
import urllib.request
from uuid import uuid4
if __package__:
    from .skipwork_runtime import MODEL,METHODS,preflight,make_profile,batch_plan,features
    from .safe_calculator import calculate,parse_expression
    from .efficiency_protocol import extract,extract_batch
else:
    from skipwork_runtime import MODEL,METHODS,preflight,make_profile,batch_plan,features
    from safe_calculator import calculate,parse_expression
    from efficiency_protocol import extract,extract_batch

def send(prompt,stage,budget=64):
    payload={'model':MODEL,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':budget}}
    body=json.dumps(payload).encode();start=time.perf_counter()
    a={'call_id':uuid4().hex,'stage':stage,'model':MODEL,'request_payload':payload,
       'request_sha256':hashlib.sha256(body).hexdigest(),'answer':'','status':'api_error','error':'',
       'input_tokens':None,'output_tokens':None,'eval_seconds':None,'prefill_seconds':None,
       'total_model_seconds':None,'load_seconds':None,'done_reason':None,
       'format_compliant':False,'parse_mode':'unresolved'}
    try:
        req=urllib.request.Request('http://localhost:11434/api/generate',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
        a.update(answer=r['response'].strip(),status='generated',input_tokens=r.get('prompt_eval_count'),output_tokens=r.get('eval_count'),done_reason=r.get('done_reason'))
        for key,server in [('eval_seconds','eval_duration'),('prefill_seconds','prompt_eval_duration'),('total_model_seconds','total_duration'),('load_seconds','load_duration')]:
            a[key]=r[server]/1e9 if server in r else None
    except Exception as error:a['error']=f'{type(error).__name__}: {error}'
    a['elapsed_seconds']=time.perf_counter()-start
    return a

def primary(question,cap=False):
    f=features(question);start=time.perf_counter()
    if f['tool_supported']:
        try:answer=str(calculate(parse_expression(question)));p=extract(question,answer,f['answer_kind']);error=''
        except Exception as exc:answer='';p=extract(question,'',f['answer_kind']);error=f'{type(exc).__name__}: {exc}'
        return {'answer':answer,**p,'source':'tool' if not error else 'tool_error','calls':[],
            'tool_calls':1,'tool_error':error,'tool_seconds':time.perf_counter()-start}
    instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
    a=send(question+'\n'+instruction+' Do not explain.','primary',8 if cap and f['answer_kind']=='choice' else 64)
    p=extract(question,a['answer'],f['answer_kind']);a.update(format_compliant=p['format_compliant'],parse_mode=p['parse_mode'])
    return {'answer':a['answer'],**p,'source':'model','calls':[a],'tool_calls':0,'tool_error':'','tool_seconds':0.}

def verify(question,candidate):
    f=features(question)
    prompt=(question+'\nCandidate response: '+json.dumps(candidate,ensure_ascii=False)+
            '\nIndependently check the candidate and solve the original request. The candidate may be wrong. '+
            ('Return only the correct option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the correct integer.')+
            ' Do not explain.')
    a=send(prompt,'verify');p=extract(question,a['answer'],f['answer_kind'])
    a.update(format_compliant=p['format_compliant'],parse_mode=p['parse_mode'])
    return a,p

def execute_block(questions,method,profile):
    start,cpu=time.perf_counter(),time.process_time()
    decisions=[preflight(q,method,profile) for q in questions]
    rows=[];calls=[]
    for q,d in zip(questions,decisions):
        p=primary(q,method=='budget_cap');calls.extend(p['calls'])
        row={'question':q,'method':method,'decision':d,'primary_answer':p['answer'],'primary_prediction':p['prediction'],
            'primary_format_compliant':p['format_compliant'],'primary_source':p['source'],'primary_parse_mode':p['parse_mode'],
            'tool_calls':p['tool_calls'],'tool_error':p['tool_error'],'tool_seconds':p['tool_seconds'],
            'final_answer':p['answer'],'final_prediction':p['prediction'],'final_format_compliant':p['format_compliant'],
            'final_source':p['source'],'verification_executed':False,'verification_answer':'','verification_prediction':None,
            'verification_format_compliant':None,'verification_parse_mode':None,'batch_envelope_error':'',
            'verification_preserved_primary':False,'call_ids':[a['call_id'] for a in p['calls']],'completion_latency_seconds':None}
        if method!='batch_verify':
            if not d['skip_verifier']:
                a,v=verify(q,p['answer']);calls.append(a);row['call_ids'].append(a['call_id'])
                row.update(verification_executed=True,verification_answer=a['answer'],verification_prediction=v['prediction'],
                    verification_format_compliant=v['format_compliant'],verification_parse_mode=v['parse_mode'],verification_preserved_primary=v['prediction'] is None)
                if v['prediction'] is not None:row.update(final_answer=a['answer'],final_prediction=v['prediction'],final_format_compliant=v['format_compliant'],final_source='verifier')
            row['completion_latency_seconds']=time.perf_counter()-start
        rows.append(row)
    if method=='batch_verify':
        items=[{'id':f'Q{i}','question':r['question'],'candidate':r['primary_answer']} for i,r in enumerate(rows)]
        prompt=('Independently verify each request and candidate below. Return ONLY a JSON object with an "answers" object '
                'mapping every exact id to its correct single option letter or integer string. No explanations.\n'+json.dumps(items,ensure_ascii=False))
        a=send(prompt,'batch_verify',128);calls.append(a)
        decoded=extract_batch(a['answer'],questions,[d['answer_kind'] for d in decisions])
        a.update(format_compliant=decoded['format_compliant'],parse_mode='batch_fenced' if decoded['envelope_fenced'] else 'batch_json')
        completed=time.perf_counter()-start
        for row,v in zip(rows,decoded['items']):
            row.update(verification_executed=True,verification_answer=v['raw_value'],verification_prediction=v['prediction'],
                verification_format_compliant=v['format_compliant'],verification_parse_mode=v['parse_mode'],
                batch_envelope_error=decoded['error'],verification_preserved_primary=v['prediction'] is None,completion_latency_seconds=completed)
            row['call_ids'].append(a['call_id'])
            if v['prediction'] is not None:row.update(final_answer=v['raw_value'],final_prediction=v['prediction'],final_format_compliant=v['format_compliant'],final_source='batch_verifier')
    return {'rows':rows,'calls':calls,'processing_seconds':time.perf_counter()-start,'router_cpu_seconds':time.process_time()-cpu}
