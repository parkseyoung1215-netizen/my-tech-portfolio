"""Small fresh train12/calibration8/pilot32; optional future expansion is separate."""
import hashlib,json,random,re
from .solver300_dataset import ROOT,prior_keys,task_key
SEED=88112026

def build(data_dir):
    seen=prior_keys(data_dir);rng=random.Random(SEED);out={}
    for split,n,base in [('training',12,17101),('calibration',8,18101),('pilot',32,19101)]:
        rows=[]
        def add(cat,q,ref,kind='exact',**extra):
            k=task_key(q)
            if k in seen:raise ValueError('ALSR duplicate stem')
            seen.add(k);rows.append(dict(question=q,reference=ref,category=cat,score_kind=kind,task_id=hashlib.sha256(q.encode()).hexdigest(),**extra))
        def mcq(cat,q,a,wrong):
            opts=[str(a)]+[str(v) for v in wrong];assert len(set(opts))==4;rng.shuffle(opts);letter='ABCD'[opts.index(str(a))]
            add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.',letter,'choice')
        scored=n if split!='pilot' else 24
        for i in range(scored//4):
            k=base+i;a,b=rng.randint(13,49),rng.randint(3,9)
            mcq('arithmetic',f'Compute the integer in worksheet {k}: ({a} + {b}) * 2.',(a+b)*2,[a+b,a+b*2,(a+b)*2+2])
            names=[f'Rena{k}',f'Somu{k}',f'Tavi{k}',f'Ula{k}'];rank=i%4
            mcq('logic',f'Puzzle {k}: The list from youngest to oldest is '+', '.join(names)+f'. Who is position {rank+1} in that order?',names[rank],[x for x in names if x!=names[rank]])
            mcq('coding',f'Python case {k}:\n```python\na = [{a}, {a+3}, {a+8}]\nprint(a[-2])\n```\nWhat is printed?',a+3,[a,a+8,'Error'])
            mcq('reading',f'Note {k}: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter?','Navo',['Mira','Oren','Nobody'])
        if split=='pilot':
            for i in range(4):
                k=base+100+i;obj={'item':f'code{k}','count':i+3}
                add('structured_output',f'Convert this record to a JSON object with exactly keys item and count. item=code{k}; count={i+3}. Return only JSON.',json.dumps(obj,separators=(',',':')),'json',expected_object=obj)
            for i in range(4):
                k=base+200+i;who=f'Lira{k}';where=f'Room {i+2}'
                add('text_quality',f'Rewrite as one concise sentence preserving the person and room: "The visitor is {who}. The visitor waits in {where}." Do not add facts.',f'{who} waits in {where}.','constraints',required=[who,where])
        rng.shuffle(rows);out[split]=rows
    return out

def grade(task,response):
    from .efficiency_protocol import extract,unwrap,unique_object
    text=response.strip()
    if task['score_kind']=='choice':
        m=re.fullmatch(r'(?:FINAL_ANSWER\s*:\s*)?([A-D])[.)]?',text,re.I);prediction=m.group(1).upper() if m else extract(task['question'],text,'choice')['prediction']
        if prediction is None:
            clean=text.replace('**','');options=dict(re.findall(r'(?m)^([A-D])\) (.*)$',task['question']))
            explicit=re.fullmatch(r'(?:The correct (?:answer|option) is|Answer:)\s*(.+?)[.!]?',clean,re.I)
            if explicit:
                body=explicit.group(1).strip();label=re.fullmatch(r'([A-D])(?:[).:]\s*(.*))?',body,re.I)
                if label:
                    letter=label.group(1).upper();tail=label.group(2)
                    if tail is None or not tail or options.get(letter)==tail:prediction=letter
                else:
                    matches=[letter for letter,value in options.items() if value==body]
                    if len(matches)==1:prediction=matches[0]
            if re.fullmatch('[A-D]',clean,re.I):prediction=clean.upper()
        return {'correct':prediction==task['reference'],'format_ok':bool(re.fullmatch('[A-D]',text)),'prediction':prediction,'error_type':'none' if prediction==task['reference'] else 'content' if prediction else 'format_or_unresolved'}
    if task['score_kind']=='json':
        try:
            body,fenced=unwrap(text);obj=json.loads(body,object_pairs_hook=unique_object);ok=obj==task['expected_object']
            return {'correct':ok,'format_ok':not fenced,'prediction':obj,'error_type':'format_only' if ok and fenced else 'none' if ok else 'content'}
        except (ValueError,TypeError):return {'correct':False,'format_ok':False,'prediction':None,'error_type':'invalid_json'}
    ok=all(v.casefold() in text.casefold() for v in task['required']);one_sentence=bool(text) and '\n' not in text and len(re.findall(r'[.!?](?:\s|$)',text))<=1
    return {'correct':ok and one_sentence,'format_ok':one_sentence,'prediction':text,'error_type':'none' if ok and one_sentence else 'constraint_failure',
            'limits':'Constraint compliance, not a semantic/hallucination oracle.'}
