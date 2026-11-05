"""Pre-solve omission policies, actual local inference, unique-call accounting."""
from collections import defaultdict
import hashlib,json,time
import urllib.request
from uuid import uuid4
if __package__:
    from .solver300_runtime import features
    from .solver300_protocol import content_extract
    from .safe_calculator import parse_expression,calculate
else:
    from solver300_runtime import features
    from solver300_protocol import content_extract
    from safe_calculator import parse_expression,calculate
MODEL='qwen2.5:1.5b'
METHODS=('baseline','early_exit','selective','budget_cap','batch_verify')
VERSION='skipwork-v1'


def send(prompt,stage,budget=64):
    payload={'model':MODEL,'prompt':prompt,'stream':False,'options':{'temperature':0,'seed':42,'num_predict':budget}}
    body=json.dumps(payload).encode();start=time.perf_counter()
    a={'call_id':uuid4().hex,'stage':stage,'model':MODEL,'request_payload':payload,
       'request_sha256':hashlib.sha256(body).hexdigest(),'answer':'','status':'api_error','error':'',
       'input_tokens':None,'output_tokens':None,'eval_seconds':None,'load_seconds':None,'done_reason':None}
    try:
        req=urllib.request.Request('http://localhost:11434/api/generate',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
        a.update(answer=r['response'].strip(),status='generated',input_tokens=r.get('prompt_eval_count'),output_tokens=r.get('eval_count'),
                 eval_seconds=r['eval_duration']/1e9 if 'eval_duration'in r else None,
                 load_seconds=r['load_duration']/1e9 if 'load_duration'in r else None,done_reason=r.get('done_reason'))
    except Exception as error:a['error']=f'{type(error).__name__}: {error}'
    a['elapsed_seconds']=time.perf_counter()-start
    return a


def primary(question,cap=False):
    f=features(question)
    if f['tool_supported']:
        start=time.perf_counter()
        try:
            value=str(calculate(parse_expression(question)))
            return {'answer':value,'prediction':value,'source':'tool','calls':[],'tool_calls':1,'tool_error':'','tool_seconds':time.perf_counter()-start}
        except Exception as error:
            return {'answer':'','prediction':None,'source':'tool_error','calls':[],'tool_calls':1,
                    'tool_error':f'{type(error).__name__}: {error}','tool_seconds':time.perf_counter()-start}
    else:tool_error=''
    instruction='Return only the option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the calculated integer.'
    a=send(question+'\n'+instruction+' Do not explain.','primary',8 if cap and f['answer_kind']=='choice' else 64)
    p,mode=content_extract(a['answer'],f['answer_kind'])
    return {'answer':a['answer'],'prediction':p,'source':'model','calls':[a],
            'tool_calls':int(f['tool_supported']),'tool_error':tool_error,'tool_seconds':None,'parse_mode':mode}


def verify(question,candidate):
    f=features(question)
    prompt=(question+'\nCandidate response: '+json.dumps(candidate,ensure_ascii=False)+
            '\nIndependently check the candidate and solve the original request. The candidate may be wrong. '+
            ('Return only the correct option letter A, B, C or D.' if f['answer_kind']=='choice' else 'Return only the correct integer.')+
            ' Do not explain.')
    a=send(prompt,'verify');p,mode=content_extract(a['answer'],f['answer_kind'])
    return a,p,mode


def make_profile(calibration_rows):
    groups=defaultdict(list)
    for r in calibration_rows:groups[r['inferred_category']].append(r)
    return {'version':VERSION,'min_samples':20,'skip_rule':'easy and calibration corrected <= harmed; exact tool proof always skips',
        'categories':{c:{'samples':len(rs),'primary_correct':sum(r['primary_correct'] for r in rs),
            'final_correct':sum(r['final_correct'] for r in rs),
            'corrected':sum(not r['primary_correct'] and r['final_correct'] for r in rs),
            'harmed':sum(r['primary_correct'] and not r['final_correct'] for r in rs)} for c,rs in groups.items()}}


def preflight(question,method,profile):
    if method not in METHODS:raise ValueError('Unknown method')
    f=features(question)
    skip=False;reason='verify_all'
    if method=='early_exit' and f['tool_supported']:skip=True;reason='safe_integer_tool_proof'
    if method=='selective':
        if profile.get('version')!=VERSION:raise ValueError('Wrong profile version')
        metric=profile.get('categories',{}).get(f['category'],{})
        if f['tool_supported']:skip=True;reason='safe_integer_tool_proof'
        elif f['difficulty']=='easy' and metric.get('samples',0)>=20 and metric['corrected']<=metric['harmed']:
            skip=True;reason='nonpositive_calibrated_verifier_uplift'
    return {**f,'skip_verifier':skip,'reason':reason,'primary_budget':8 if method=='budget_cap' and f['answer_kind']=='choice' else 64}


def batch_plan(questions,max_items=4,max_chars=2600):
    groups=defaultdict(list)
    for i,q in enumerate(questions):
        f=features(q);groups[(f['category'],f['answer_kind'])].append(i)
    batches=[]
    for indices in groups.values():
        current=[];size=0
        for i in indices:
            if current and (len(current)>=max_items or size+len(questions[i])>max_chars):batches.append(current);current=[];size=0
            current.append(i);size+=len(questions[i])
        if current:batches.append(current)
    return batches


def execute_block(questions,method,profile):
    """All requests arrive at block start; latency includes within-block waiting.

    No task labels enter this function. Omission decisions are made before primaries.
    Each shared batch call is counted exactly once in calls, never divided into fake per-request calls.
    """
    start,cpu=time.perf_counter(),time.process_time()
    decisions=[preflight(q,method,profile) for q in questions]
    rows=[];calls=[]
    for i,(q,d) in enumerate(zip(questions,decisions)):
        p=primary(q,method=='budget_cap');calls.extend(p['calls'])
        row={'question':q,'method':method,'decision':d,'primary_answer':p['answer'],'primary_prediction':p['prediction'],
             'primary_source':p['source'],'tool_calls':p['tool_calls'],'tool_error':p['tool_error'],
             'final_answer':p['answer'],'final_prediction':p['prediction'],'verification_executed':False,
             'verification_answer':'','verification_prediction':None,'verification_parse_mode':None,
             'call_ids':[a['call_id'] for a in p['calls']],'completion_latency_seconds':None}
        if method!='batch_verify':
            if not d['skip_verifier']:
                a,pred,mode=verify(q,p['answer']);calls.append(a);row['call_ids'].append(a['call_id'])
                row.update(verification_executed=True,verification_answer=a['answer'],verification_prediction=pred,verification_parse_mode=mode)
                if pred is not None:row.update(final_answer=a['answer'],final_prediction=pred)
            row['completion_latency_seconds']=time.perf_counter()-start
        rows.append(row)
    if method=='batch_verify':
        items=[{'id':f'Q{i}','question':r['question'],'candidate':r['primary_answer']} for i,r in enumerate(rows)]
        prompt=('Independently verify each request and candidate below. Return ONLY a JSON object with an "answers" object '
                'mapping every exact id to its correct single option letter or integer string. No explanations.\n'+json.dumps(items,ensure_ascii=False))
        a=send(prompt,'batch_verify',128);calls.append(a)
        try:
            decoded=json.loads(a['answer']);answers=decoded['answers']
            if not isinstance(answers,dict) or set(answers)!={f'Q{i}' for i in range(len(rows))}:raise ValueError('Batch ids missing/extra')
        except (ValueError,KeyError,TypeError):answers={}
        completed=time.perf_counter()-start
        for i,r in enumerate(rows):
            value=answers.get(f'Q{i}');text=str(value) if isinstance(value,(str,int)) and not isinstance(value,bool) else ''
            pred,mode=content_extract(text,decisions[i]['answer_kind'])
            r.update(verification_executed=True,verification_answer=text,verification_prediction=pred,verification_parse_mode=mode,
                     completion_latency_seconds=completed)
            r['call_ids'].append(a['call_id'])
            if pred is not None:r.update(final_answer=text,final_prediction=pred)
    return {'rows':rows,'calls':calls,'processing_seconds':time.perf_counter()-start,
            'router_cpu_seconds':time.process_time()-cpu}
