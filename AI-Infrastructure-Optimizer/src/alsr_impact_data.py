"""Fresh, deterministic layer-impact splits; references never enter model/gate calls."""
import hashlib,json,random
from .solver300_dataset import prior_keys,task_key
SEED=2026110517

def build(data_dir):
    seen=prior_keys(data_dir);rng=random.Random(SEED);out={}
    for split,count,openn,base in [('training',6,4,28100),('calibration',2,4,29100),('evaluation',12,8,30100)]:
        rows=[]
        def add(cat,q,ref,kind='choice',**kw):
            key=task_key(q)
            if key in seen:raise ValueError('Existing layer-impact stem')
            seen.add(key);rows.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,reference=ref,category=cat,score_kind=kind,**kw))
        def mcq(cat,q,a,wrong):
            opts=[str(a)]+list(map(str,wrong));assert len(set(opts))==4;rng.shuffle(opts)
            add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.','ABCD'[opts.index(str(a))])
        for j in range(count):
            k=base+j;a,b,c=rng.sample(range(11,70),3);x=rng.randint(2,7)
            if j%3==0:mcq('arithmetic',f'A tray contains {a} red and {b} blue beads. After removing {x} red beads, how many beads remain?',a+b-x,[a+b,a+b+x,a-x])
            elif j%3==1:mcq('arithmetic',f'Parcel {k} contains {x} packs of {a} cards and {b} loose cards. What is the total?',x*a+b,[x*(a+b),a+b,x*a-b])
            else:mcq('arithmetic',f'In ledger {k}, an account starts at {a}, receives {b}, then pays {c}. What is its balance?',a+b-c,[a+b+c,a-b-c,a-b+c])
            names=[f'Nemi{k}',f'Pavo{k}',f'Selu{k}']
            if j%2==0:mcq('logic',f'Queue {k}: {names[0]} stands immediately before {names[1]}, and {names[1]} immediately before {names[2]}. The queue contains only these three people. Who is in the middle?',names[1],[names[0],names[2],'Cannot be determined'])
            else:mcq('logic',f'Rules for set {k}: All daxes are merins. No merin is blue. Object {names[0]} is a dax. Which conclusion follows?',f'{names[0]} is not blue',[f'{names[0]} is blue',f'{names[0]} is not a merin','Nothing follows'])
            if j%3==0:mcq('coding',f'Python snippet {k}:\n```python\nx = [{a}, {b}, {c}]\nx.append({x})\nprint(len(x))\n```\nWhat is printed?',4,[3,5,'Error'])
            elif j%3==1:mcq('coding',f'Python snippet {k}:\n```python\nx = {a}\ny = x + {x}\nx = 0\nprint(y)\n```\nWhat is printed?',a+x,[0,a,x])
            else:mcq('coding',f'Python snippet {k}:\n```python\nx = [{a}, {b}, {c}]\nprint(x[0] + x[2])\n```\nWhat is printed?',a+c,[a+b,b+c,a+b+c])
            owner,borrower=names[:2]
            if j%2==0:mcq('reading',f'Memo {k}: {owner} owns a silver compass. {owner} lends the compass to {borrower} for one day; ownership does not change. Who owns the compass during that day?',owner,[borrower,names[2],'Nobody'])
            else:mcq('reading',f'Log {k}: A package arrives on Tuesday. {owner} stores it until Friday, when {borrower} collects it. On which day is it collected?','Friday',['Tuesday','Wednesday','Saturday'])
        for j in range(openn):
            k=base+100+j;obj={'label':f'unit{k}','enabled':bool(j%2)}
            add('structured_output',f'Encode this as JSON with exactly label and enabled: label is unit{k}; enabled is {str(bool(j%2)).lower()}. Return only JSON.',json.dumps(obj,separators=(',',':')),'json',expected_object=obj)
            person=f'Vera{k}';place=f'Gate {j+11}'
            add('text_quality',f'Combine into one concise sentence without adding facts: "{person} is a courier. {person} is waiting at {place}."',f'{person}, a courier, is waiting at {place}.','constraints',required=[person,place])
        rng.shuffle(rows);out[split]=rows
    return out
