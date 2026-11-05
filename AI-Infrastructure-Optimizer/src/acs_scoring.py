"""Offline content/format/completeness audit, keeping the historical grade separately."""
import re
from .alsr_impact_scoring import grade as legacy_grade
from .acs_policy import boundary,request_kind,requested_keys

def json_equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(json_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(json_equal(x,y) for x,y in zip(a,b))
    return a==b

def score(task,answer):
    old=legacy_grade(task,answer);content=old['correct'];form=old['format_ok'];incomplete=False
    if task['score_kind']=='json':
        content=json_equal(old['prediction'],task['expected_object']);incomplete=not boundary(task['question'],answer)
    elif task['score_kind']=='constraints':
        required=list(task['required'])
        required+=re.findall(r'\bis (?:a|an) ([A-Za-z]+)[.!]',task['question'])
        s=answer.strip();content=all(v.casefold() in s.casefold() for v in required)
        protected=re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof)\.',r'\1<dot>',s,flags=re.I);protected=re.sub(r'(?<=\d)\.(?=\d)','<dot>',protected)
        form=bool(s) and '\n' not in s and bool(re.search(r'[.!?]["\']?$',s)) and len(re.findall(r'[.!?](?:["\']?\s|$)',protected))==1
        incomplete=not content or not bool(re.search(r'[.!?]["\']?$',s))
    return {'legacy_grade':old,'content_correct':bool(content),'format_ok':bool(form),'task_success':bool(content and form),'incomplete_output':bool(incomplete),'error_type':'none' if content and form else 'format_only' if content else 'incomplete_or_missing_facts' if incomplete else 'content_error'}
