"""Four preregistered strategies. Runtime accepts no evaluation label or task object."""
import ast
import hashlib
import json
import re
import time
import urllib.request
from uuid import uuid4

if __package__:
    from .safe_calculator import parse_expression, calculate, ToolDeclined
    from .solver300_protocol import content_extract
    from .adaptive_router import lower_accuracy
else:
    from safe_calculator import parse_expression, calculate, ToolDeclined
    from solver300_protocol import content_extract
    from adaptive_router import lower_accuracy

SMALL,LARGE = "qwen2.5:0.5b","qwen2.5:1.5b"
STRATEGIES = ("fixed","rule","confidence","hybrid")
VERSION = "solver-choice-v1"


def features(question):
    try:
        tree = parse_expression(question)
        literals = [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)]
        hard = any(isinstance(n,ast.Mult) for n in ast.walk(tree)) or max(literals)>100
        return {"category":"arithmetic","difficulty":"hard" if hard else "easy",
                "answer_kind":"integer","tool_supported":True}
    except ToolDeclined:
        pass
    low = question.lower()
    if "```python" in low: category="coding"
    elif low.startswith("read the passage:"): category="comprehension"
    elif low.startswith("reason "): category="logic"
    elif low.startswith(("what ","which ","how ","when ","where ","can ","does ")): category="knowledge"
    else: category="unknown"
    hard = category in ("logic","unknown") or len(question)>320
    if category=="coding": hard = len(re.findall(r"\b(for|if|def)\b",question))>=2 or len(question)>320
    kind = "choice" if all(re.search(rf"(?m)^{c}\) ",question) for c in "ABCD") else "unsupported"
    return {"category":category,"difficulty":"hard" if hard else "easy",
            "answer_kind":kind,"tool_supported":False}


def generate(model, question):
    f = features(question)
    instruction = ("Return only the best option letter A, B, C or D." if f['answer_kind']=='choice'
                   else "Calculate the expression and return only the final integer.")
    payload = {"model":model,"prompt":question+"\n"+instruction+" Do not add explanations.",
               "stream":False,"options":{"temperature":0,"seed":42,"num_predict":64}}
    body = json.dumps(payload).encode('utf-8')
    record = {"model":model,"client_request_id":uuid4().hex,
              "request_sha256":hashlib.sha256(body).hexdigest(),"request_payload":payload,
              "answer":"","status":"api_error","error":"","input_tokens":None,
              "output_tokens":None,"eval_seconds":None,"load_seconds":None,"done_reason":None}
    start = time.perf_counter()
    try:
        request = urllib.request.Request("http://localhost:11434/api/generate",data=body,
                                         headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
        record.update(answer=result['response'].strip(),status="generated",
                      input_tokens=result.get('prompt_eval_count'),output_tokens=result.get('eval_count'),
                      eval_seconds=result['eval_duration']/1e9 if 'eval_duration' in result else None,
                      load_seconds=result['load_duration']/1e9 if 'load_duration' in result else None,
                      done_reason=result.get('done_reason'))
    except Exception as error:
        record['error']=f"{type(error).__name__}: {error}"
    record['elapsed_seconds']=time.perf_counter()-start
    return record


def make_profile(calibration_rows):
    groups={}
    for r in calibration_rows:
        groups.setdefault(r['inferred_category'],{}).setdefault(r['model'],[]).append(r)
    return {"version":VERSION,"threshold":0.8,"min_samples":20,
            "categories":{category:{model:{"samples":len(rs),"correct":sum(r['correct'] for r in rs),
                "accuracy":sum(r['correct'] for r in rs)/len(rs),
                "lower_bound":lower_accuracy(sum(r['correct'] for r in rs),len(rs))}
                for model,rs in models.items()} for category,models in groups.items()}}


def choose(question,strategy,profile):
    if strategy not in STRATEGIES: raise ValueError("Unknown strategy")
    f=features(question)
    if f['answer_kind']=='unsupported': return {**f,"source":"none","model":None,"reason":"unsupported_request"}
    if strategy in ('rule','hybrid') and f['tool_supported']:
        return {**f,"source":"tool","model":None,"reason":"exact_integer_expression"}
    if strategy=='fixed': model,reason=LARGE,'fixed_large'
    elif strategy=='rule':
        small=f['category'] in ('knowledge','comprehension') and f['difficulty']=='easy'
        model,reason=(SMALL,'short_knowledge_or_passage_rule') if small else (LARGE,'rule_conservative_large')
    else:
        if profile.get('version')!=VERSION: raise ValueError("Profile version mismatch")
        metric=profile.get('categories',{}).get(f['category'],{}).get(SMALL,{})
        allowed=(metric.get('samples',0)>=20 and metric.get('lower_bound',0)>=0.8 and f['difficulty']=='easy')
        model,reason=(SMALL,'calibrated_small_lower_bound') if allowed else (LARGE,'small_not_calibrated_or_hard')
    return {**f,"source":"model","model":model,"reason":reason}


def execute(question,strategy,profile):
    start,cpu=time.perf_counter(),time.process_time()
    row={"question":question,"strategy":strategy,"source":"none","model":"","answer":"",
         "prediction":None,"status":"rejected","reason":"","attempts":[],"tool_calls":0,
         "tool_error":"","fallback":False,"fallback_seconds":0.0,"format_errors":0}
    try:
        decision=choose(question,strategy,profile)
        row['decision']=decision
        row['reason']=decision['reason']
        if decision['source']=='none': return row
        if decision['source']=='tool':
            row['tool_calls']=1
            try:
                value=calculate(parse_expression(question))
                row.update(source='tool',answer=str(value),prediction=str(value),status='answered')
                return row
            except Exception as error:
                row['tool_error']=f"{type(error).__name__}: {error}"
                row['fallback']=True
                decision={**decision,'source':'model','model':LARGE}
        row['source']='model'
        candidates=[decision['model']]+([LARGE] if decision['model']==SMALL else [])
        for index,model in enumerate(candidates):
            a=generate(model,question)
            prediction,mode=content_extract(a['answer'],decision['answer_kind'])
            a['prediction']=prediction
            a['parse_mode']=mode
            a['format_compliant']=bool(re.fullmatch('[A-D]' if decision['answer_kind']=='choice' else '[+-]?[0-9]+',a['answer']))
            if a['status']=='generated':
                a['status']='answered' if prediction is not None else 'format_error'
            row['attempts'].append(a)
            row['format_errors']+=a['status']=='format_error'
            if index>0 or row['tool_error']:
                row['fallback']=True;row['fallback_seconds']+=a['elapsed_seconds']
            if prediction is not None:
                row.update(answer=a['answer'],prediction=prediction,model=model,status='answered')
                break
        else: row['reason']='all_candidates_failed_format_or_api'
        return row
    finally:
        row['elapsed_seconds']=time.perf_counter()-start
        row['router_cpu_seconds']=time.process_time()-cpu
