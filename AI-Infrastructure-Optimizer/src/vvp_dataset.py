"""New validation100 and final test300. Historical evaluations are never training."""
import random
if __package__:
    from .solver300_dataset import ROOT,prior_keys,task_key
    from .verify_routing200 import reference_answer
else:
    from solver300_dataset import ROOT,prior_keys,task_key
    from verify_routing200 import reference_answer
SEED=58112026

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);splits=[]
    http=[(100,'Continue'),(101,'Switching Protocols'),(200,'OK'),(201,'Created'),(202,'Accepted'),(203,'Non-Authoritative Information'),(204,'No Content'),(205,'Reset Content'),(206,'Partial Content'),(300,'Multiple Choices'),(301,'Moved Permanently'),(302,'Found'),(303,'See Other'),(304,'Not Modified'),(307,'Temporary Redirect'),(308,'Permanent Redirect'),(400,'Bad Request'),(401,'Unauthorized'),(403,'Forbidden'),(404,'Not Found')]
    tags=[('p','A paragraph'),('a','A hyperlink'),('h1','A top-level heading'),('ul','An unordered list'),('ol','An ordered list'),('li','A list item'),('table','A table'),('tr','A table row'),('td','A table data cell'),('th','A table header cell'),('form','A form'),('input','An input control'),('button','A button'),('img','An image'),('video','A video player'),('audio','An audio player'),('textarea','A multiline text control'),('select','An option selection control'),('option','A selectable option'),('br','A line break')]
    keywords=[('def','Define a function'),('class','Define a class'),('return','Return from a function'),('yield','Produce a generator value'),('try','Start a block with exception handling'),('except','Handle a matching exception'),('finally','Specify a cleanup clause'),('with','Use a context manager'),('lambda','Create an anonymous function'),('pass','Provide an empty statement'),('break','Exit the innermost loop'),('continue','Start the next loop iteration'),('global','Declare a module-level name binding'),('nonlocal','Declare an enclosing function name binding'),('del','Delete a binding or item'),('assert','Express a debugging assertion'),('import','Import a module'),('raise','Raise an exception'),('for','Iterate over an iterable'),('while','Repeat while a condition is true')]
    prefixes=[('yotta',24),('zetta',21),('exa',18),('peta',15),('tera',12),('giga',9),('mega',6),('kilo',3),('hecto',2),('deca',1),('deci',-1),('centi',-2),('milli',-3),('micro',-6),('nano',-9),('pico',-12),('femto',-15),('atto',-18),('zepto',-21),('yocto',-24)]
    facts=[]
    for code,a in http:facts.append((f'Which standard HTTP reason phrase corresponds to status code {code}?',a,rng.sample([v for _,v in http if v!=a],3)))
    for tag,a in tags:facts.append((f'Which purpose matches the HTML element <{tag}>?',a,rng.sample([v for _,v in tags if v!=a],3)))
    for key,a in keywords:facts.append((f'Which description matches the Python keyword {key}?',a,rng.sample([v for _,v in keywords if v!=a],3)))
    for name,power in prefixes:facts.append((f'Which power of ten is represented by the SI prefix {name}?',f'10^{power}',[f'10^{power+1}',f'10^{power-2}',f'10^{power+4}']))
    rng.shuffle(facts);days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for split,n,start in [('validation100',20,0),('test300',60,20)]:
        rows=[]
        def add(category,q,a,kind='choice',code=''):
            k=task_key(q)
            if k in seen:raise ValueError('VVP old/split duplicate stem')
            seen.add(k);rows.append({'question':q,'expected':str(a),'category':category,'task_format':kind,'reference_code':code})
        def choice(category,q,a,wrong,code=''):
            opts=[str(a)]+[str(v) for v in wrong]
            if len(set(opts))!=4:raise ValueError('VVP duplicate options')
            rng.shuffle(opts);add(category,q+'\n'+'\n'.join(f'{k}) {v}' for k,v in zip('ABCD',opts)),'ABCD'[opts.index(str(a))],code=code)
        for i in range(n):
            while True:
                a,b,c=rng.randint(11001,16000),rng.randint(23,91),rng.randint(701,999)
                q=f'({a} - {c}) + {b}' if i%3==0 else f'({a} + {c}) * {b}' if i%3==1 else f'({a} * {b}) - ({c} + 7)'
                if task_key(q) not in seen:break
            add('arithmetic',q,reference_answer(q),'integer')
        for q,a,wrong in facts[start:start+n]:choice('knowledge',q,a,wrong)
        for j in range(n//5):
            k=j+(4101 if split=='validation100' else 5101);v=j+(181 if split=='validation100' else 221)
            today=days[j%7];offset=41+j
            choice('logic',f'Reason about a weekly calendar: event {k} happens on {today}. Which weekday is {offset} days after it?',days[(j+offset)%7],rng.sample([d for d in days if d!=days[(j+offset)%7]],3))
            names=[f'Ira{k}',f'Jori{k}',f'Kimi{k}',f'Lano{k}',f'Melo{k}'];rank=j%5+1
            rel=[f'{names[i]} finishes before {names[i+1]}' for i in range(4)];rng.shuffle(rel)
            a=names[rank-1];choice('logic','Reason about a race: '+'; '.join(rel)+f'. Who finishes in position {rank}?',a,rng.sample([x for x in names if x!=a],3))
            bits=[rng.randrange(2) for _ in range(15)];drop=bits[0]
            choice('logic',f'Reason about bit parity {k}: bits={bits}. Remove the first bit. Is the count of remaining 1 bits odd or even?','Odd' if (sum(bits)-drop)%2 else 'Even',['Even' if (sum(bits)-drop)%2 else 'Odd','Unknown','Neither'])
            nodes=[f'R{k}{x}' for x in 'abcde'];edges=', '.join(f'{nodes[i]}->{nodes[i+1]}' for i in range(4))
            choice('logic',f'Reason about directed links: {edges}. From {nodes[0]}, which node is reached after exactly three links?',nodes[3],[nodes[0],nodes[1],nodes[4]])
            choice('logic',f'Reason about set membership: X={[k,k+3,k+7]}, Y={[k+3,k+8]}, Z={[k+3,k+11]}. Which number belongs to the intersection of all sets?',k+3,[k,k+7,k+11])
            codes=[(f'def f(a):\n    return sum(range(a, a+10, 3))\nprint(f({v}))',str(sum(range(v,v+10,3))),[str(v*4),str(v+10),'Error']),
                (f"text = 'violet{k} blue green amber'\nprint(len(text.split()))",'4',['3','5','Error']),
                (f'a = [{v}, {v+3}, {v+8}, {v+14}]\nprint(a[::-1][2])',str(v+3),[str(v+8),str(v+14),'Error']),
                (f"d = {{'n': {v}, 'x': 6}}\nprint(d.get('missing', 13) + d['x'])",'19',['13',str(v+6),'Error']),
                (f'print([n-2 for n in range({v}, {v+6}) if n % 2 == 1])',str([n-2 for n in range(v,v+6) if n%2]),[str([n for n in range(v,v+6) if n%2]),'[]','Error'])]
            for code,a,w in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,w,code)
            who,other=f'Pela{k}',f'Qori{k}';pay=71+j;refund=21+j
            choice('comprehension',f'Read the passage: Request {k} reserved Hall A, then Hall B, then cancelled Hall B and reserved Hall C. Which hall is finally reserved?','Hall C',['Hall A','Hall B','No hall'])
            choice('comprehension',f'Read the passage: Receipt {k} explicitly records payment {pay}, refund {refund}, and a separate deposit 8. Which amount is the refund?',refund,[pay,'8',pay-refund])
            choice('comprehension',f'Read the passage: {who} composed the first melody; {other} composed a second melody and performed the first. Who composed the first melody?',who,[other,'Both composed it','Nobody'])
            choice('comprehension',f'Read the passage: Timetable {k} closes all rooms at 18:00 except Room Alder at 21:00. On Saturday Alder closes at 16:00. It is Saturday. When does Alder close?','16:00',['18:00','21:00','Never'])
            choice('comprehension',f'Read the passage: Item {k} changed from drafted to submitted to withdrawn, then was submitted again. What is its latest state?','Submitted',['Drafted','Withdrawn','Unknown'])
        counts={}
        for r in rows:
            i=counts.get(r['category'],0);counts[r['category']]=i+1;r['cohort']='A' if i<n//2 else 'B'
        rng.shuffle(rows);splits.append(rows)
    return tuple(splits)
