"""New closed tasks; evaluation labels are used only by the privileged builder/scorer."""
import hashlib,json,random
from .dlr_data import recorded_keys
from .dlr_opt_data import score
from .solver300_dataset import task_key
from .oracle_policy import SEED
def fresh(data):
    seen=recorded_keys(data);rng=random.Random(SEED);short=[];long=[]
    def add(target,q,category,kind,ref,cap=32,**kw):
        key=task_key(q)
        if key in seen:raise ValueError('Oracle new stem already recorded')
        seen.add(key);target.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,category=category,score_kind=kind,reference=ref,token_cap=cap,**kw))
    for j in range(4):
        k=78100+j;a,b=rng.sample(range(19,89),2)
        if j==0:q=f'Oracle study parcel record {k}: {a} parcels are placed beside {b} parcels. How many parcels are there in total?';ans=a+b;wrong=[a*b,a-b,a+b+1];cat='arithmetic'
        elif j==1:q=f'Oracle study Python record {k}: x=[{a}, {b}]; print(x[0]). What is printed?';ans=a;wrong=[b,a+b,'Error'];cat='coding'
        elif j==2:q=f'Oracle study queue {k}: Yera{k} is before Zobi{k}. Zobi{k} is before Aven{k}. Who is first?';ans=f'Yera{k}';wrong=[f'Zobi{k}',f'Aven{k}','Cannot be determined'];cat='logic'
        else:q=f'Oracle study handover {k}: Benu{k} gives a key to Ciro{k}, who keeps it. Who now holds the key?';ans=f'Ciro{k}';wrong=[f'Benu{k}','Nobody','Cannot be determined'];cat='reading'
        options=list(map(str,[ans]+wrong));rng.shuffle(options);add(short,q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',options))+'\nReturn only the option letter.',cat,'choice','ABCD'[options.index(str(ans))])
    for j in range(2):
        obj={'code':f'oracle{79200+j}','quantity':j+41};add(short,f'Return only a JSON object with keys code and quantity. Values: code="{obj["code"]}", quantity={j+41}.','structured_output','json',json.dumps(obj),expected_object=obj)
        color=('blue','green')[j];add(short,f'Oracle study clothing record {80300+j}: There is a scarf, and its color is {color}. What color is the scarf? Return only the color word.','reading_short','short',color,aliases=[color])
    for j,words in enumerate((['beep']*24,['la']*24,['ruby','stone','tide','vale','wood','zeal']*4,['acorn','brook','cloud','dove','fern','gate']*4)):
        target=' '.join(words);add(long,f'Oracle study transcription {81400+j}. Copy exactly these 24 words in the same order, separated by single spaces, without explanation:\n{target}','long_repetitive' if j<2 else 'long_varied','copy_words',target,cap=64,target_words=words)
    rng.shuffle(short);return short,long
