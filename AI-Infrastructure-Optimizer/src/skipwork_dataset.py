"""Fresh 150-task evaluation, separate from stored calibration and old benchmarks."""
import json
import random
from pathlib import Path
if __package__:
    from .solver300_dataset import prior_keys,task_key,ROOT
    from .verify_routing200 import reference_answer
else:
    from solver300_dataset import prior_keys,task_key,ROOT
    from verify_routing200 import reference_answer
SEED=20261109150
CAL_SOURCE=ROOT/'data/solver300_c22b83397f0945b4adfb8304f7b2f268_manifest.json'


def build(data_dir):
    rng=random.Random(SEED);seen=prior_keys(data_dir);evaluation=[]
    calibration=json.loads(CAL_SOURCE.read_text())['calibration']
    def add(category,q,a,kind='choice',code=None):
        key=task_key(q)
        if key in seen:raise ValueError('Existing question reused')
        seen.add(key);row={'question':q,'category':category,'expected':str(a),'task_format':kind}
        if code is not None:row['reference_code']=code
        evaluation.append(row)
    def choice(category,q,a,wrong,code=None):
        options=[str(a)]+[str(v) for v in wrong]
        if len(set(options))!=4:raise ValueError('Duplicate options')
        rng.shuffle(options)
        add(category,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',options)),
            'ABCD'[options.index(str(a))],code=code)
    for operation in ('add','multiply','composite'):
        n=0
        while n<10:
            a,b,c=rng.randint(101,399),rng.randint(22,88),rng.randint(17,91)
            q=f'{a} + {b}' if operation=='add' else f'{a-80} * {b}' if operation=='multiply' else f'({a+400} * {b}) - {c}'
            if task_key(q) in seen:continue
            add('arithmetic',q,reference_answer(q),'integer');n+=1
    capitals=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Spain','Madrid'),('Portugal','Lisbon'),
        ('Greece','Athens'),('Japan','Tokyo'),('South Korea','Seoul'),('Australia','Canberra'),('New Zealand','Wellington'),
        ('Canada','Ottawa'),('Mexico','Mexico City'),('Brazil','Brasilia'),('Sweden','Stockholm'),('Norway','Oslo'),
        ('Denmark','Copenhagen'),('Finland','Helsinki'),('Poland','Warsaw'),('Austria','Vienna'),('Hungary','Budapest'),
        ('the Czech Republic','Prague'),('Ireland','Dublin'),('Belgium','Brussels'),('Thailand','Bangkok'),
        ('Argentina','Buenos Aires'),('Chile','Santiago'),('Peru','Lima'),('Colombia','Bogota'),
        ('the United States','Washington, D.C.'),('the United Kingdom','London')]
    for country,city in capitals:
        wrong=rng.sample([v for _,v in capitals if v!=city],3)
        choice('knowledge',f'Which city is the capital of {country}?',city,wrong)
    for i in range(5):
        a,b,c=f'Neli{i+301}',f'Pori{i+301}',f'Ravi{i+301}'
        choice('logic',f'Reason about height group {i+301}: {a} is taller than {b}; {b} is taller than {c}. Who is shortest?',c,[a,b,'Unknown'])
        choice('logic',f'Reason about exclusion group {i+301}: All wugs are square. No square object is soft. Can a wug be soft?','No',['Yes','Only large wugs','All soft objects are wugs'])
        choice('logic',f'Reason about a rule {i+301}: If alarm A rings, light B is on. Light B is off. What follows?','Alarm A is not ringing',['Alarm A rings','Light B is on','Alarm A must be damaged'])
        choice('logic',f'Reason about a converse {i+301}: Every approved form is signed. This form is signed. Must it be approved?','No',['Yes, necessarily','All signed forms are rejected','No approved form can be signed'])
        choice('logic',f'Reason about parity statement {i+301}: Exactly one of X and Y is true. Y is true. What is X?','False',['True','Both true','Unknown'])
        choice('logic',f'Reason about positions {i+301}: {a} stands before {b}. {c} stands after {b}. Who stands second among the three?',b,[a,c,'Unknown'])
        x,y=31+i,51+i
        cases=[
            (f'print(({x}, {y})[-1])',str(y),[str(x),'None','Error']),
            (f"print('audit{i}'.split('d'))",str(['au',f'it{i}']),[str([f'audit{i}']),'Error',str([f'au{i}','it'])]),
            (f"d = {{'n': {x}}}\nprint(d.get('missing', {y}))",str(y),[str(x),'None','Error']),
            (f'print([n - 1 for n in range({x}, {x+3})])',str(list(range(x-1,x+2))),[str(list(range(x,x+3))),str([x-1]),'Error']),
            (f'v = [{x}, {y}, 99]\nw = v.copy()\nw.append(0)\nprint(len(v))','3',['4','2','Error']),
            (f'total = 0\nfor n in range(1, {i+6}):\n    total += n\nprint(total)',str(sum(range(1,i+6))),[str(i+6),str(sum(range(i+6))+i+20),'Error']),
        ]
        for code,a,w in cases:choice('coding',f'What does this Python snippet print?\n```python\n{code}\n```',a,w,code)
        name,other=f'Sela{i+401}',f'Timo{i+401}'
        choice('comprehension',f'Read the passage: {name} owns cabinet {i+401}. {other} rents space in it but does not own the cabinet. Who owns it?',name,[other,'Nobody','Unknown'])
        choice('comprehension',f'Read the passage: Package {i+401} was ordered Tuesday, shipped Thursday, and delivered Saturday. On which day was it shipped?','Thursday',['Tuesday','Saturday','Monday'])
        choice('comprehension',f'Read the passage: {name} planned to buy apples but bought pears instead. What fruit was bought?','Pears',['Apples','Both','Neither'])
        choice('comprehension',f"Read the passage: Screen {i+401} shows the literal password '{i+41} * {i+12}', not a calculation request. What text appears?",f'{i+41} * {i+12}',[str((i+41)*(i+12)),'No password','Screen off'])
        choice('comprehension',f'Read the passage: {name} attends meeting {i+401} only if an invitation arrives. No invitation arrived. Will {name} attend?','No',['Yes','Only on Sunday','An invitation did arrive'])
        choice('comprehension',f'Read the passage: The token for shelf {i+401} was on the table. {other} put it inside the box. Nobody moved it again. Where is it?','Inside the box',['On the table','Under the shelf','In a bag'])
    if len(evaluation)!=150:raise ValueError('Expected 150 evaluation tasks')
    if {task_key(t['question']) for t in calibration}&{task_key(t['question']) for t in evaluation}:raise ValueError('Calibration overlap')
    rng.shuffle(evaluation)
    return calibration,evaluation
