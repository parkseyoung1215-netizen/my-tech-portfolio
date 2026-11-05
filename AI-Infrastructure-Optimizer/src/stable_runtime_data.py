"""Independent exact-stem holdout; no model calls and no reference-aware routing."""
import hashlib,json,random
from .solver300_dataset import prior_keys,task_key
SEED=2026110521

def build(data_dir):
    seen=prior_keys(data_dir);rng=random.Random(SEED);rows=[]
    def add(cat,q,ref,kind='choice',**kw):
        k=task_key(q)
        if k in seen:raise ValueError('Duplicate historical or current stem')
        seen.add(k);rows.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,reference=ref,category=cat,score_kind=kind,**kw))
    def mcq(cat,q,a,wrong):
        opts=[str(a)]+list(map(str,wrong));assert len(set(opts))==4;rng.shuffle(opts)
        add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.','ABCD'[opts.index(str(a))])
    for i in range(12):
        a,b=rng.sample(range(21,80),2);n=rng.randint(3,8);k=41100+i
        if i%2==0:mcq('arithmetic',f'A shop ticket lists {n} identical pens at {a} credits each. The customer gives {n*a+b} credits. How many credits of change are due?',b,[a,n*a,n*a-b])
        else:mcq('arithmetic',f'A rover at distance marker {a} travels forward {b} units and then backward {n} units. At which distance marker does it stop?',a+b-n,[a+b+n,a-b-n,a+n])
        names=[f'Rilo{k}',f'Tema{k}',f'Wesi{k}']
        if i%2==0:mcq('logic',f'Badge check {k}: If a person has a bronze badge, the person may enter. {names[0]} has a bronze badge. Which statement must be true?',f'{names[0]} may enter',[f'{names[0]} cannot enter','Everyone has a bronze badge','Nobody may enter'])
        else:mcq('logic',f'Shelf {k} holds exactly red, green, and white books. The red book is left of the green book. The white book is right of the green book. Which book is farthest right?','White',['Red','Green','Cannot be determined'])
        if i%3==0:mcq('coding',f'Python test {k}:\n```python\nv = {{"oak": {a}, "elm": {b}}}\nprint(v["elm"])\n```\nWhat is printed?',b,[a,a+b,'Error'])
        elif i%3==1:mcq('coding',f'Python test {k}:\n```python\nv = [{a}, {b}, {n}]\nv.reverse()\nprint(v[0])\n```\nWhat is printed?',n,[a,b,'Error'])
        else:mcq('coding',f'Python test {k}:\n```python\nx = {n}\nprint(x * 2 + 1)\n```\nWhat is printed?',2*n+1,[2*n,n+1,2*n-1])
        if i%2==0:mcq('reading',f'Dispatch record {k}: {names[0]} gives a sealed envelope to {names[1]}. {names[1]} then gives it to {names[2]}. No other transfer occurs. Who holds it at the end?',names[2],[names[0],names[1],'Nobody'])
        else:mcq('reading',f'Appointment {k}: The meeting was first scheduled for Monday but a notice changes it to Thursday. The notice is final. On which day will the meeting occur?','Thursday',['Monday','Tuesday','Friday'])
    for i in range(8):
        code=f'tag{42100+i}';amount=i+12;obj={'code':code,'amount':amount}
        add('structured_output',f'Write one JSON object from this inventory row: code is "{code}" and amount is the integer {amount}. Use exactly those two keys. Return only JSON.',json.dumps(obj,separators=(',',':')),'json',expected_object=obj)
        person=f'Talu{43100+i}';place=f'Dock {i+21}'
        add('text_quality',f'Express these facts in exactly one sentence without extra facts: "{person} repairs boats. {person} works at {place}."',f'{person} repairs boats at {place}.','constraints',required=[person,place])
    rng.shuffle(rows);return rows
