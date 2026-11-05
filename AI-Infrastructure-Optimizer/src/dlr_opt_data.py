"""New exact-stem holdout and predeclared high/low reuse stress tasks."""
import hashlib,json,random,re
from .dlr_data import recorded_keys,score as short_score
from .dlr_policy import SEED as PREV_SEED
from .solver300_dataset import task_key
SEED=2026110605

def build(data):
    seen=recorded_keys(data);rows=[];rng=random.Random(SEED)
    def add(q,cat,kind,ref,**kw):
        k=task_key(q)
        if k in seen:raise ValueError('Duplicate DLR optimization stem')
        seen.add(k);rows.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,category=cat,score_kind=kind,reference=ref,token_cap=32,**kw))
    for j in range(16):
        a,b=rng.sample(range(23,94),2);k=71100+j
        if j<4:q=f'Packing sheet {k}: {a} envelopes are stored, then {b} additional envelopes arrive. What is the total?';ans=a+b;wrong=[a-b,a*b,a+b+2];cat='arithmetic'
        elif j<8:q=f'Python inspection {k}: a=[{a}, {b}, 9]; a.reverse(); print(a[1]). What is printed?';ans=b;wrong=[a,9,a+b];cat='coding'
        elif j<12:q=f'Sequence memo {k}: Ravu{k} is ahead of Seno{k}, and Seno{k} is ahead of Tilu{k}. Who is furthest ahead?';ans=f'Ravu{k}';wrong=[f'Seno{k}',f'Tilu{k}','Cannot be determined'];cat='logic'
        else:q=f'Bag tracking {k}: Ume{k} hands a blue bag to Vira{k}. Vira{k} holds it and does not pass it on. Who holds the bag now?';ans=f'Vira{k}';wrong=[f'Ume{k}','Nobody','Cannot be determined'];cat='reading'
        opts=list(map(str,[ans]+wrong));rng.shuffle(opts);add(q+'\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',opts))+'\nReturn only the option letter.',cat,'choice','ABCD'[opts.index(str(ans))])
    for j in range(8):
        code=f'opt{72400+j}';obj={'code':code,'quantity':j+31};add(f'Return only a JSON object with keys code and quantity. Values: code="{code}", quantity={j+31}.','structured_output','json',json.dumps(obj),expected_object=obj)
        color=('red','blue','green','yellow')[j%4];add(f'Clothing record {73500+j}: One scarf is stored in this crate. Its color is {color}. What color is the scarf? Return only the color word.','reading_short','short',color,aliases=[color])
    rng.shuffle(rows)
    long={}
    for label,words in [('long_high',['echo','ha','yes','go']),('long_low',['amber birch cedar dune elm frost grove haze','jade kite lake mint nest opal pine quartz','reed sand teal umber vine wave yarn zinc','arch beam clay dawn east fern gold hill'])]:
        tasks=[]
        for j,w in enumerate(words):
            target=' '.join([w]*40) if label=='long_high' else ' '.join((w.split()*5))
            q=f'Literal transcription check {74600+j+(10 if label=="long_low" else 0)}. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation:\n{target}'
            if task_key(q) in seen:raise ValueError('Duplicate long prompt')
            seen.add(task_key(q));tasks.append(dict(task_id=hashlib.sha256(q.encode()).hexdigest(),question=q,category=label,score_kind='copy_words',reference=target,target_words=target.split(),token_cap=96))
        long[label]=tasks
    return rows,long

def score(task,answer):
    if task['score_kind']!='copy_words':return short_score(task,answer)
    words=answer.strip().split();target=task['target_words'];exact=words==target
    return dict(content_correct=exact,format_ok=answer.strip()==' '.join(target),content_and_format=exact,prediction=words,word_count=len(words),word_prefix_fraction=sum(a==b for a,b in zip(words,target))/len(target),metric_scope='Exact literal copying, not semantic open generation')
