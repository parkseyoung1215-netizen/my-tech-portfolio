"""New ACS holdout with a larger non-MCQ share; references remain offline."""
import hashlib,json,random
from .solver300_dataset import prior_keys,task_key
SEED=2026110603

def build(data_dir):
    seen=prior_keys(data_dir);rng=random.Random(SEED);rows=[]
    def add(cat,q,ref,kind='choice',difficulty='easy',**kw):
        k=task_key(q)
        if k in seen:raise ValueError('ACS duplicate stem')
        seen.add(k);rows.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,reference=ref,category=cat,score_kind=kind,difficulty=difficulty,**kw))
    def mcq(cat,q,answer,wrong,difficulty):
        opts=[str(answer)]+list(dict.fromkeys(str(x) for x in wrong if str(x)!=str(answer)))
        while len(opts)<4:opts.append(str(90000+len(opts)))
        opts=opts[:4];rng.shuffle(opts);add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.','ABCD'[opts.index(str(answer))],difficulty=difficulty)
    for i in range(8):
        a,b=rng.sample(range(12,65),2);c=rng.randint(2,8);k=51100+i;level='easy' if i<4 else 'moderate'
        if i<4:mcq('arithmetic',f'Inventory ticket {k}: A rack has {a} tools and receives {b} more. What is the new count?',a+b,[a-b,a+b+1,a*b],level)
        else:mcq('arithmetic',f'Batch ticket {k}: {c} bins each contain {a} bolts. Then {b} bolts are taken away from the combined total. How many remain?',c*a-b,[c*a+b,c*(a-b),a+b-c],level)
        names=[f'Daro{k}',f'Envi{k}',f'Fomu{k}']
        if i<4:mcq('logic',f'Order note {k}: {names[0]} finishes before {names[1]}; {names[1]} finishes before {names[2]}. Who finishes first?',names[0],[names[1],names[2],'Cannot be determined'],level)
        else:mcq('logic',f'Rule note {k}: Every lum is a vek. Every vek can swim. {names[0]} is a lum. What must follow?',f'{names[0]} can swim',[f'{names[0]} cannot swim',f'{names[0]} is not a vek','No conclusion follows'],level)
        if i<4:mcq('coding',f'Python card {k}:\n```python\na = [{a}, {b}]\nprint(a[1])\n```\nWhat is printed?',b,[a,a+b,'Error'],level)
        else:mcq('coding',f'Python card {k}:\n```python\na = [{a}, {b}, {c}]\nx = a.pop()\nprint(len(a) + x)\n```\nWhat is printed?',2+c,[3+c,a+b,c,'Error'],level)
        if i<4:mcq('reading',f'Receipt {k}: {names[0]} buys a violet bag from {names[1]}. The sale transfers ownership to the buyer. Who owns the bag after the sale?',names[0],[names[1],names[2],'Nobody'],level)
        else:mcq('reading',f'Receipt {k}: {names[0]} hands a token to {names[1]}. {names[1]} hands it to {names[2]}. {names[2]} keeps it. Who finally holds the token?',names[2],[names[0],names[1],'Nobody'],level)
    for i in range(16):
        code=f'acs{52100+i}';obj={'code':code,'quantity':i+3}
        if i>=8:obj['status']='ready' if i%2 else 'on hold'
        keys='code and quantity' if i<8 else 'code, quantity and status';values=f'code="{code}", quantity={i+3}'+(f', status="{obj["status"]}"' if i>=8 else '')
        add('structured_output',f'Return only a JSON object with keys {keys}. Values: {values}.',json.dumps(obj,separators=(',',':')),'json','easy' if i<8 else 'moderate',expected_object=obj)
        who=('Dr. ' if i>=8 else '')+f'Leno{53100+i}';role='doctor' if i>=8 else 'artist';place=f'Studio {i+31}'
        add('text_quality',f'Use exactly one sentence to preserve all these facts without adding facts: "{who} is a {role}. {who} works at {place}."',f'{who} is a {role} working at {place}.','constraints','easy' if i<8 else 'moderate',required=[who,role,place])
    rng.shuffle(rows);return rows
