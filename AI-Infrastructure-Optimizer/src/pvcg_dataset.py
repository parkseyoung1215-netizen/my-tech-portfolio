"""Fresh calibration100/evaluation300. Context lookup is not general-knowledge recall."""
import random
from .solver300_dataset import ROOT,task_key,prior_keys
from .verify_routing200 import reference_answer
SEED=78112026

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);splits=[]
    for phase,n,base in [('validation',20,12101),('test',60,13101)]:
        rows=[]
        def add(cat,q,a,kind='choice',code=''):
            key=task_key(q)
            if key in seen:raise ValueError('PVCG duplicate question stem')
            seen.add(key);rows.append({'question':q,'expected':str(a),'category':cat,'task_format':kind,'reference_code':code})
        def choice(cat,q,a,wrong,code=''):
            opts=[str(a)]+[str(x) for x in wrong]
            if len(set(opts))!=4:raise ValueError('Duplicate choices')
            rng.shuffle(opts);add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts)),'ABCD'[opts.index(str(a))],code=code)
        for i in range(n):
            while True:
                a,b,c=rng.randint(51001,71000),rng.randint(101,199),rng.randint(701,999)
                q=f'{a} + ({b} * {c})' if i%2 else f'({a} + {b}) * 3 - {c}'
                if task_key(q) not in seen:break
            add('arithmetic',q,reference_answer(q),'integer')
        for i in range(n):
            k=base+i;names=[f'S{k}{x}' for x in 'ABCD'];slots=['archive','search','queue','monitor'];rng.shuffle(slots);target=i%4
            text='; '.join(f'{name} handles {slot}' for name,slot in zip(names,slots))
            choice('context_lookup',f'Which service handles {slots[target]} in this supplied catalog? Catalog: {text}.',names[target],[x for x in names if x!=names[target]])
        for i in range(n//4):
            k=base+i;v=(501 if phase=='validation' else 601)+i
            nodes=[f'V{k}{x}' for x in 'abcdef'];start=i%3
            choice('logic',f'Reason about a directed path: '+', '.join(f'{nodes[j]}->{nodes[j+1]}' for j in range(5))+f'. Starting at {nodes[start]}, where do exactly two forward edges end?',nodes[start+2],[nodes[start],nodes[start+1],nodes[5]])
            bits=[rng.randrange(2) for _ in range(9)];pos=i%9;count=sum(bits)+(1 if bits[pos]==0 else -1)
            choice('logic',f'Reason about a bit vector in record {k}: {bits}. Toggle the bit at zero-based index {pos}. How many 1 bits remain?',count,[count+1,count+2,count-1])
            a,b,c=k,k+2,k+5
            choice('logic',f'Reason about set difference: X={{{a}, {b}, {c}}}, Y={{{b}, {c}, {k+9}}}. Which value is in X but not Y?',a,[b,c,k+9])
            names=[f'Yara{k}',f'Zeno{k}',f'Umi{k}',f'Vela{k}'];rank=i%4
            choice('logic',f'Reason about a queue: from front to back it contains '+', '.join(names)+f'. After removing the first person, who occupies position {(rank%3)+1}?',names[1+rank%3],[x for x in names if x!=names[1+rank%3]])
            codes=[(f'a = [{v}, {v+4}, {v+9}]\nb = a\nb.append({v+15})\nprint(len(a))','4',['3','1','Error']),
                   (f"d = {{'p': {v}, 'q': {v+3}}}\nd['r'] = d['p'] - d['q']\nprint(d['r'])",'-3',['3',str(v),'Error']),
                   (f'print(sum(n for n in range({v}, {v+7}) if n % 3 == 0))',str(sum(x for x in range(v,v+7) if x%3==0)),[str(sum(x for x in range(v,v+7) if x%3==0)+3),'0','Error']),
                   (f"s = 'cedar|pine|oak|elm{k}'\nprint(s.split('|')[1])",'pine',['cedar','oak',f'elm{k}'])]
            for code,a,wrong in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,wrong,code)
            choice('comprehension',f'Read the passage: Order {k} specifies 12 green crates and 7 red crates. The later correction changes only green crates to 9. How many red crates are ordered?','7',['12','9','19'])
            choice('comprehension',f'Read the passage: In agreement {k}, Nira owns a silver pen. Nira lets Olan use it for one day without transferring ownership. Who owns the pen?','Nira',['Olan','Both equally','Nobody'])
            choice('comprehension',f'Read the passage: Booking {k} is approved unless payment fails. Payment failed and an explicit note says approval was revoked. What is its current approval state?','Revoked',['Approved','Pending','Not specified'])
            choice('comprehension',f'Read the passage: Record {k} says Aven inspected, Boro repaired and Cira shipped the device. Who repaired it?','Boro',['Aven','Cira','All three'])
        counts={}
        for r in rows:
            i=counts.get(r['category'],0);counts[r['category']]=i+1;r['cohort']='A' if i<n//2 else 'B'
        rng.shuffle(rows);splits.append(rows)
    return tuple(splits)
