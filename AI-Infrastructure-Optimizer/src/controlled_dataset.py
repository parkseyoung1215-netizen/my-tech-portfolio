"""Fresh300 mixed synthetic tasks; no prior stem/option-shuffle reuse."""
import random
if __package__:
    from .solver300_dataset import ROOT,prior_keys,task_key
    from .verify_routing200 import reference_answer
else:
    from solver300_dataset import ROOT,prior_keys,task_key
    from verify_routing200 import reference_answer
SEED=11052026

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);rows=[]
    def add(c,q,a,kind='choice',code=''):
        key=task_key(q)
        if key in seen:raise ValueError('Controlled dataset duplicates stored stem')
        seen.add(key);rows.append({'question':q,'category':c,'expected':str(a),'task_format':kind,'reference_code':code})
    def choice(c,q,a,wrong,code=''):
        opts=[str(a)]+[str(v) for v in wrong]
        if len(opts)!=4 or len(set(opts))!=4:raise ValueError('Duplicate choices')
        rng.shuffle(opts);add(c,q+'\n'+'\n'.join(f'{k}) {v}' for k,v in zip('ABCD',opts)),'ABCD'[opts.index(str(a))],code=code)
    for i in range(60):
        while True:
            a,b,c=rng.randint(6101,9900),rng.randint(19,83),rng.randint(423,601)
            q=f'({a} + {c}) - {b}' if i%3==0 else f'{a} * {b}' if i%3==1 else f'({a} - {c}) * ({b} + 3)'
            if task_key(q) not in seen:break
        add('arithmetic',q,reference_answer(q),'integer')
    elements=['Hydrogen','Helium','Lithium','Beryllium','Boron','Carbon','Nitrogen','Oxygen','Fluorine','Neon','Sodium','Magnesium','Aluminium','Silicon','Phosphorus','Sulfur','Chlorine','Argon','Potassium','Calcium']
    for z,name in enumerate(elements,1):
        choice('knowledge',f'What is the atomic number of {name}?',z,[z+1,z+3,z+5])
    for n in range(5,25):
        prop=('faces','edges','vertices')[n%3]
        answer=n+2 if prop=='faces' else 3*n if prop=='edges' else 2*n
        choice('knowledge',f'How many {prop} does a prism whose base is an {n}-sided polygon have?',answer,[answer+1,answer+2,answer+4])
    for op in ('AND','OR','XOR','NAND','NOR'):
        for x,y in [(False,False),(False,True),(True,False),(True,True)]:
            value=(x and y) if op=='AND' else (x or y) if op=='OR' else (x!=y) if op=='XOR' else not(x and y) if op=='NAND' else not(x or y)
            choice('knowledge',f'For a Boolean {op} gate with inputs {str(x).lower()} and {str(y).lower()}, which output is correct?',str(value),[str(not value),'Both True and False','Undefined'])
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for i in range(12):
        k=2401+i;v=151+i;today=days[i%7];offset=31+i;answer=days[(i-offset)%7]
        choice('logic',f'Reason about a weekly calendar: date marker {k} is a {today}. Which weekday was {offset} days before this marker?',answer,rng.sample([d for d in days if d!=answer],3))
        names=[f'Anu{k}',f'Bela{k}',f'Ciro{k}',f'Daro{k}'];rank=i%4+1
        rel=[f'{names[j+1]} finishes after {names[j]}' for j in range(3)];rng.shuffle(rel)
        choice('logic','Reason about finishing order: '+'; '.join(rel)+f'. Who finishes at rank {rank}?',names[rank-1],[n for n in names if n!=names[rank-1]])
        bits=[rng.randrange(2) for _ in range(13)];extra=i%3+1
        choice('logic',f'Reason about bit parity marker {k}: bits={bits}. Append {extra} additional 1 bits. Is the resulting count of 1 bits odd or even?','Odd' if (sum(bits)+extra)%2 else 'Even',['Even' if (sum(bits)+extra)%2 else 'Odd','Neither','Unknown'])
        nodes=[f'V{k}{c}' for c in 'abcd']
        choice('logic',f'Reason about directed links: {nodes[0]}->{nodes[1]}, {nodes[1]}->{nodes[2]}, {nodes[2]}->{nodes[3]}. Start at {nodes[0]}. Which node is reached after exactly three links?',nodes[3],nodes[:3])
        choice('logic',f'Reason about set membership: X={[k,k+2,k+4]}, Y={[k+2,k+6]}, Z={[k+2,k+8]}. Which integer is in all three sets?',k+2,[k,k+4,k+8])
        codes=[(f'values = [n + 2 for n in range({v}, {v+9}, 3)]\nprint(sum(values))',str(sum(n+2 for n in range(v,v+9,3))),[str(v*3),str(v+9),'Error']),
            (f"s = 'oak{k},elm,pine'\nprint(len(s.split(',')))",'3',['2','4','Error']),
            (f'a = [{v}, {v+6}, {v+11}, {v+19}]\nprint(a[1:][1])',str(v+11),[str(v+6),str(v+19),'Error']),
            (f"d = {{'a': {v}, 'b': 5}}\nprint(d.get('c', 11) + d['b'])",'16',['11',str(v+5),'Error']),
            (f'print([n * 2 for n in range({v}, {v+5}) if n % 2 == 0])',str([n*2 for n in range(v,v+5) if n%2==0]),[str([n for n in range(v,v+5) if n%2==0]),'[]','Error'])]
        for code,a,w in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,w,code)
    for i in range(10):
        k=3401+i;name=f'Eni{k}';other=f'Faro{k}';amount=51+i;refund=17+i
        choice('comprehension',f'Read the passage: Reservation {k} changed from Room Birch to Room Ash and then to Room Willow. What is the final reserved room?','Room Willow',['Room Birch','Room Ash','All rooms'])
        choice('comprehension',f'Read the passage: Receipt {k} lists payment {amount}, refund {refund}, and shipping fee 4. The fee is not refunded. What amount is explicitly labeled refund?',refund,[amount,'4',amount-refund])
        choice('comprehension',f'Read the passage: {name} invented a tune. {other} recorded the tune but did not invent it. Who invented the tune?',name,[other,'Both people','Nobody'])
        choice('comprehension',f'Read the passage: Rule {k} closes all halls at 18:00. Hall Iris closes at 20:00 except Friday, when it closes at 17:00. Today is Friday. When does Hall Iris close?','17:00',['18:00','20:00','Never'])
        choice('comprehension',f'Read the passage: Parcel {k} was queued, sent, returned, and finally sent again. What is its final state?','Sent',['Queued','Returned','Unknown'])
        choice('comprehension',f'Read the passage: Badge {k} is granted only if a form arrives. No form arrived, and no badge was granted. Was a badge granted?','No',['Yes','Both yes and no','Not stated'])
    if len(rows)!=300:raise ValueError('Expected300')
    rng.shuffle(rows);return rows
