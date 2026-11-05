"""Frozen synthetic holdout, aliases defined before any inference."""
import csv,hashlib,json,random,re
from pathlib import Path
from .solver300_dataset import task_key
from .dlr_policy import SEED
from .acs_scoring import json_equal

def recorded_keys(data):
    seen=set()
    def walk(o):
        if isinstance(o,dict):
            if isinstance(o.get('question'),str):seen.add(task_key(o['question']))
            for v in o.values():walk(v)
        elif isinstance(o,list):
            for v in o:walk(v)
    for p in Path(data).rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.json':walk(json.loads(p.read_text()))
        elif p.suffix=='.csv':
            with p.open(newline='') as f:
                for r in csv.DictReader(f):walk(r)
    return seen

def fresh(data):
    seen=recorded_keys(data);rows=[];rng=random.Random(SEED)
    def add(q,cat,kind,ref,**kw):
        k=task_key(q)
        if k in seen:raise ValueError('Previously recorded DLR question stem')
        seen.add(k);rows.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,category=cat,score_kind=kind,reference=ref,**kw))
    for j in range(16):
        a,b=rng.sample(range(17,87),2);tag=64100+j
        if j<4:q=f'Parcel ledger {tag}: a shelf contains {a} parcels and receives {b} more. How many are there now?';ans=a+b;wrong=[a-b,a+b+1,a*b];cat='arithmetic'
        elif j<8:q=f'Trace card {tag}: Python executes x = [{a}, {b}, 7]; print(x[-2]). What is printed?';ans=b;wrong=[a,7,a+b];cat='coding'
        elif j<12:q=f'Queue note {tag}: Vena{tag} precedes Wori{tag}, and Wori{tag} precedes Xelu{tag}. Who is last?';ans=f'Xelu{tag}';wrong=[f'Vena{tag}',f'Wori{tag}','Cannot be determined'];cat='logic'
        else:q=f'Pickup record {tag}: Joro{tag} hands a key to Kenu{tag}, who keeps it. Who now holds the key?';ans=f'Kenu{tag}';wrong=[f'Joro{tag}','Nobody','Cannot be determined'];cat='reading'
        opts=list(map(str,[ans]+wrong));rng.shuffle(opts);add(q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.',cat,'choice','ABCD'[opts.index(str(ans))])
    for j in range(8):
        code=f'dlr{65200+j}';obj={'code':code,'quantity':j+21};add(f'Return only a JSON object with keys code and quantity. Values: code="{code}", quantity={j+21}.','structured_output','json',json.dumps(obj),expected_object=obj)
        color=('red','blue','green','yellow')[j%4];add(f'Label note {66300+j}: The only scarf in the box is {color}. What color is the scarf? Return only the color word.','reading_short','short',color,aliases=[color])
    rng.shuffle(rows);return rows

def score(task,answer):
    text=answer.strip();kind=task['score_kind'];prediction=None;correct=False;form=False
    if kind=='choice':
        form=bool(re.fullmatch('[ABCD]',text));letters=re.findall(r'(?<!\w)[ABCD](?!\w)',text)
        if form:prediction=text
        elif re.match(r'^[ABCD][).:\s]',text):prediction=text[0]
        elif len(set(letters))==1:prediction=letters[0]
        correct=prediction==task['reference']
    elif kind=='json':
        def pairs(items):
            obj={}
            for k,v in items:
                if k in obj:raise ValueError('duplicate key')
                obj[k]=v
            return obj
        try:
            prediction=json.loads(text,object_pairs_hook=pairs);form=isinstance(prediction,dict)
        except ValueError:
            start=text.find('{')
            if start>=0:
                try:prediction=json.JSONDecoder(object_pairs_hook=pairs).raw_decode(text[start:])[0]
                except ValueError:pass
        correct=json_equal(prediction,task['expected_object'])
    elif kind=='short':
        aliases={x.casefold() for x in task['aliases']};form=text.casefold() in aliases
        # Unambiguous single color, permitting a grammatical answer but rejecting contradiction.
        words=set(re.findall(r'\b(?:red|blue|green|yellow)\b',text.casefold()));correct=len(words)==1 and bool(words&aliases) and bool(re.fullmatch(r'(?:the scarf (?:in the box )?is |it is |the color is |color: )?(red|blue|green|yellow)[.!]?',text.casefold()))
        prediction=next(iter(words)) if len(words)==1 else None
    else:raise ValueError('Unsupported DLR task format')
    return dict(content_correct=bool(correct),format_ok=bool(form),content_and_format=bool(correct and form),prediction=prediction,metric_scope='MCQ extraction, typed exact JSON, closed color aliases; not a general semantic judge')
