"""Fresh VRG calibration100/test300; no re-labeled historical evaluation training."""
import random
from pathlib import Path
from .solver300_dataset import ROOT,prior_keys,task_key
from .verify_routing200 import reference_answer
SEED=68112026

def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir)
    http=[(405,'Method Not Allowed'),(406,'Not Acceptable'),(407,'Proxy Authentication Required'),(408,'Request Timeout'),(409,'Conflict'),(410,'Gone'),(411,'Length Required'),(412,'Precondition Failed'),(413,'Content Too Large'),(414,'URI Too Long'),(415,'Unsupported Media Type'),(416,'Range Not Satisfiable'),(417,'Expectation Failed'),(421,'Misdirected Request'),(422,'Unprocessable Content'),(425,'Too Early'),(426,'Upgrade Required'),(428,'Precondition Required'),(429,'Too Many Requests'),(431,'Request Header Fields Too Large')]
    functions=[('abs','Return absolute magnitude'),('len','Return the number of items'),('sum','Add items of an iterable'),('min','Return the smallest item'),('max','Return the largest item'),('sorted','Return a new sorted list'),('enumerate','Pair a count with iterable values'),('zip','Iterate over tuples from multiple iterables'),('reversed','Return a reverse iterator'),('round','Round a number to specified precision'),('all','Test whether every item is truthy'),('any','Test whether at least one item is truthy'),('int','Construct an integer'),('str','Construct a string'),('bool','Construct a Boolean value'),('list','Construct a mutable sequence'),('dict','Construct a mapping'),('set','Construct a mutable set'),('tuple','Construct an immutable sequence'),('divmod','Return quotient and remainder')]
    errors=[('ZeroDivisionError','Division by zero'),('IndexError','Sequence index out of range'),('KeyError','Mapping key not found'),('TypeError','Operation applied to an inappropriate type'),('ValueError','Appropriate type but inappropriate value'),('NameError','Local or global name not found'),('AttributeError','Attribute lookup or assignment fails'),('ImportError','Import operation fails'),('ModuleNotFoundError','Imported module cannot be located'),('FileNotFoundError','Requested file does not exist'),('PermissionError','Insufficient access rights'),('StopIteration','Iterator has no remaining items'),('AssertionError','An assert statement fails'),('NotImplementedError','Required operation is not implemented'),('OverflowError','Arithmetic result is too large'),('RecursionError','Maximum recursion depth exceeded'),('MemoryError','Operation runs out of memory'),('SyntaxError','Parser encounters invalid syntax'),('IndentationError','Incorrect indentation'),('UnicodeEncodeError','Unicode encoding fails')]
    attrs=[('href on an anchor','Specify the hyperlink destination'),('src on an image','Specify the image resource URL'),('alt on an image','Supply replacement text'),('id','Give an element its identifier'),('class','Assign element classes'),('title','Provide advisory information'),('style','Specify inline CSS declarations'),('lang','Specify the element language'),('dir','Specify text directionality'),('hidden','Mark content as not currently relevant'),('disabled on a button','Disable the control'),('checked on a checkbox','Specify default checkedness'),('readonly on a text input','Prevent user editing of the value'),('required on an input','Require a value for form submission'),('multiple on a select','Allow multiple selections'),('placeholder on an input','Provide an input hint'),('download on an anchor','Indicate a download hyperlink'),('target on an anchor','Specify the target navigable'),('rel on an anchor','Specify the relationship to the linked resource'),('colspan on a table cell','Specify how many columns the cell spans')]
    facts=[]
    for code,a in http:facts.append((f'Which registered HTTP description belongs to code {code}?',a,[v for _,v in http if v!=a]))
    for f,a in functions:facts.append((f'What is the main role of Python built-in {f}()?',a,[v for _,v in functions if v!=a]))
    for f,a in errors:facts.append((f'Which situation is associated with Python {f}?',a,[v for _,v in errors if v!=a]))
    for f,a in attrs:facts.append((f'What is the purpose of HTML attribute {f}?',a,[v for _,v in attrs if v!=a]))
    rng.shuffle(facts);splits=[];days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for phase,n,start,base in [('validation',20,0,8101),('test',60,20,9101)]:
        rows=[]
        def add(cat,q,a,kind='choice',code=''):
            k=task_key(q)
            if k in seen:raise ValueError('VRG duplicate stem')
            seen.add(k);rows.append({'question':q,'expected':str(a),'category':cat,'task_format':kind,'reference_code':code})
        def choice(cat,q,a,wrong,code=''):
            opts=[str(a)]+[str(v) for v in wrong];assert len(set(opts))==4
            rng.shuffle(opts);add(cat,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts)),'ABCD'[opts.index(str(a))],code=code)
        for i in range(n):
            while True:
                a,b,c=rng.randint(21001,41000),rng.randint(31,99),rng.randint(101,599)
                q=f'({a} + {b}) - {c}' if i%2 else f'({a} - {c}) * {b}'
                if task_key(q) not in seen:break
            add('arithmetic',q,reference_answer(q),'integer')
        for q,a,wrong in facts[start:start+n]:choice('knowledge',q,a,rng.sample(wrong,3))
        for i in range(n//4):
            k=base+i;v=301+i if phase=='validation' else 401+i;d=i%7;offset=53+i
            choice('logic',f'Reason about schedule {k}: today is {days[d]}. What weekday is {offset} days later?',days[(d+offset)%7],rng.sample([x for x in days if x!=days[(d+offset)%7]],3))
            names=[f'Avo{k}',f'Bex{k}',f'Cyd{k}',f'Dov{k}'];rank=i%4
            choice('logic',f'Reason about heights: {names[0]} is taller than {names[1]}, who is taller than {names[2]}, who is taller than {names[3]}. Who is {rank+1} in descending height?',names[rank],rng.sample([x for x in names if x!=names[rank]],3))
            lo,hi=k,k+8+i%5;count=sum(x%3==0 for x in range(lo,hi+1))
            choice('logic',f'Reason about integer constraints: among integers from {lo} through {hi}, including both ends, how many are divisible by 3?',count,[count+1,count+2,count-1])
            choice('logic',f'Reason about permissions {k}: every member of group L has access to vault Q. Navo{k} is a member of L. What follows?',f'Navo{k} has access to Q',[f'Navo{k} has no access to Q','Nobody has access to Q','Access cannot be inferred'])
            codes=[(f'a = [{v}, {v+2}, {v+5}, {v+9}]\nprint(a[1:][::-1][-1])',str(v+2),[str(v+5),str(v+9),'Error']),
                   (f"d = {{'a': {v}, 'b': 7}}\nx = d['b']\ndel d['b']\nprint(x + len(d))",'8',['7',str(v+7),'Error']),
                   (f'print(sum(x % 4 for x in range({v}, {v+5})))',str(sum(x%4 for x in range(v,v+5))),[str(sum(x%4 for x in range(v,v+5))+1),'0','Error']),
                   (f"s = 'alpha{k}:beta:gamma'\nprint(s.split(':')[-2])",'beta',['gamma',f'alpha{k}','Error'])]
            for code,a,wrong in codes:choice('coding','What does this Python program output?\n```python\n'+code+'\n```',a,wrong,code)
            choice('comprehension',f'Read the passage: Ticket {k} moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state?','Open',['Pending','Closed','Deleted'])
            choice('comprehension',f'Read the passage: In shipment {k}, Orin packed the parcel, Sava labelled it, and Tavi delivered it. Who labelled the parcel?','Sava',['Orin','Tavi','Nobody'])
            choice('comprehension',f'Read the passage: Sensor {k} uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used?','Blue',['Red','Green','No channel'])
            choice('comprehension',f'Read the passage: Invoice {k} lists subtotal {v}, tax 17, shipping 9 and discount 6. What is the shipping amount?','9',['17','6',str(v)])
        counts={}
        for r in rows:
            j=counts.get(r['category'],0);counts[r['category']]=j+1;r['cohort']='A' if j<n//2 else 'B'
        rng.shuffle(rows);splits.append(rows)
    return tuple(splits)
