"""Posthoc format diagnostics. Never changes runtime policy or official scores."""
import csv,json,re,sys
from pathlib import Path
from collections import Counter
if __package__:
    from .solver300_protocol import content_extract
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    from solver300_protocol import content_extract
    from tool_experiment import write_records
    from adaptive_experiment import write_json

def recover_option(question,response):
    """Question + response only, no expected answer. Accept exact option text."""
    match=re.fullmatch(r'([A-D])\)[ \t]*(.*)',response.strip(),re.I)
    if not match:return None
    letter,body=match.group(1).upper(),match.group(2)
    choices=dict(re.findall(r'(?m)^([A-D])\) (.*)$',question))
    if letter not in choices or (body and body!=choices[letter]):return None
    return letter

def diagnose(prefix):
    prefix=Path(prefix).resolve()
    def read(name):
        with Path(f'{prefix}_{name}.csv').open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    rows,calls=read('evaluation'),read('calls');ci={a['call_id']:a for a in calls}
    output=[];wrappers=Counter();batch_valid=0;batch_correct=0
    for r in rows:
        recovered=r['final_prediction'] or recover_option(r['question'],r['final_answer'])
        vr=None;envelope='not_batch'
        if r['method']=='batch_verify':
            a=ci[json.loads(r['call_ids'])[-1]];text=a['answer'].strip();envelope='no_fence'
            if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4];envelope='json_code_fence'
            try:
                answers=json.loads(text)['answers']
                items=json.loads(json.loads(a['request_payload'])['prompt'].split('\n',1)[1])
                if not isinstance(answers,dict) or set(answers)!={item['id'] for item in items}:raise ValueError('ids')
                id=next(item['id'] for item in items if item['question']==r['question'])
                value=answers[id]
                if isinstance(value,(str,int)) and not isinstance(value,bool):vr=content_extract(str(value),r['task_format'])[0]
            except (ValueError,KeyError,TypeError,StopIteration):pass
            wrappers[envelope]+=1;batch_valid+=vr is not None;batch_correct+=vr is not None and vr==r['expected']
        output.append({'method':r['method'],'category':r['category'],'question':r['question'],'expected':r['expected'],
            'official_prediction':r['final_prediction'],'official_correct':r['final_correct']=='True',
            'format_recovered_final_prediction':recovered,'format_recovered_final_correct':recovered is not None and recovered==r['expected'],
            'batch_envelope':envelope,'batch_wrapped_verifier_prediction':vr,
            'batch_wrapped_verifier_correct':vr is not None and vr==r['expected']})
    summaries=[]
    for m in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in output if r['method']==m]
        summaries.append({'method':m,'official_correct':sum(r['official_correct'] for r in rr),
            'diagnostic_format_recovered_correct':sum(r['format_recovered_final_correct'] for r in rr),
            'format_recovered_requests':sum(not r['official_prediction'] and r['format_recovered_final_prediction'] is not None for r in rr)})
    diagnostic={'posthoc_only':True,'official_scores_unchanged':True,'format_diagnostics':summaries,
        'batch_envelopes_by_request':dict(wrappers),'batch_wrapped_verifier_valid':batch_valid,
        'batch_wrapped_verifier_correct':batch_correct,
        'warning':'These parses were not used in execution. No timing or rerun performance is inferred from them.'}
    write_records(Path(f'{prefix}_format_diagnostics.csv'),output)
    write_json(Path(f'{prefix}_format_diagnostics.json'),diagnostic)
    print(json.dumps(diagnostic,ensure_ascii=False,indent=2))

if __name__=='__main__':diagnose(sys.argv[1])
