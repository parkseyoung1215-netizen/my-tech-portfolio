"""Small multinomial transition predictor and calibrated CPU regression; no AI calls."""
import ast,math,re
if __package__:
    from .rcav_guard_policy import family
    from .solver300_runtime import features
else:
    from rcav_guard_policy import family
    from solver300_runtime import features
VERSION='vvp-v1'
REGULARIZATIONS=(.01,.1,1.);TEMPERATURES=(.75,1.,1.5)

def observed(question,p):
    f=features(question);d={'bias':1.,'question_length':min(len(question)/400,5),'answer_length':min(len(p['answer'])/100,5),
        'number_count':min(len(re.findall(r'\d+',question))/20,5),'unresolved':float(p['prediction'] is None),
        'noncompliant':float(not p['format_compliant']),'tool':float(p['source']=='tool'),'hard':float(f['difficulty']=='hard'),
        'category='+f['category']:1.,'family='+family(question):1.}
    match=re.search(r'```python\s*\n([\s\S]*?)\n```',question)
    try:tree=ast.parse(match.group(1)) if match else None
    except SyntaxError:tree=None
    d['ast_nodes']=min(sum(1 for _ in ast.walk(tree))/60,5) if tree is not None else 0.
    return d

def softmax(v):
    e=[math.exp(x-max(v)) for x in v];z=sum(e);return [x/z for x in e]
def vector(d,keys):return [d.get(k,0.) for k in keys]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def label(pc,fc):return 0 if not pc and fc else 1 if pc and not fc else 2
def distribution(w,x,temp,tool):
    p=softmax([dot(a,x)/temp for a in w])
    if tool:p[0]=0.;p[2]=1-p[1]
    return p

def train(rows,validation):
    keys=sorted({k for r in rows for k in r['features']});xs=[vector(r['features'],keys) for r in rows];ys=[r['label'] for r in rows];n=len(rows);dim=len(keys)
    best=None;trials=[]
    for reg in REGULARIZATIONS:
        w=[[0.]*dim for _ in range(3)]
        for epoch in range(220):
            grad=[[0.]*dim for _ in range(3)]
            for x,y in zip(xs,ys):
                probs=softmax([dot(a,x) for a in w])
                for c in range(3):
                    error=probs[c]-(y==c)
                    for j,v in enumerate(x):grad[c][j]+=error*v/n
            rate=.25/(1+epoch/100)
            for c in range(3):
                for j in range(dim):w[c][j]-=rate*(grad[c][j]+reg*w[c][j]*(keys[j]!='bias'))
        for temp in TEMPERATURES:
            loss=sum(-math.log(max(1e-12,distribution(w,vector(r['features'],keys),temp,r['features']['tool'])[r['label']])) for r in validation)/len(validation)
            trials.append({'l2':reg,'temperature':temp,'validation_logloss':loss})
            if best is None or loss<best[0]:best=(loss,reg,temp,[a[:] for a in w])
    # Regress log(measured total verifier CPU); missing measurements are not fabricated.
    if any(r['cpu_seconds'] is None or r['cpu_seconds']<=0 for r in rows+validation):raise ValueError('Measured positive CPU required for VVP fit')
    targets=[math.log(r['cpu_seconds']) for r in rows];cw=[0.]*dim;cw[keys.index('bias')]=sum(targets)/n
    for epoch in range(260):
        g=[0.]*dim
        for x,y in zip(xs,targets):
            error=dot(cw,x)-y
            for j,v in enumerate(x):g[j]+=error*v/n
        for j in range(dim):cw[j]-=.12/(1+epoch/100)*(g[j]+.1*cw[j]*(keys[j]!='bias'))
    ratios=sorted(r['cpu_seconds']/math.exp(max(-12,min(5,dot(cw,vector(r['features'],keys))))) for r in validation)
    scale=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
    return {'version':VERSION,'feature_keys':keys,'weights':best[3],'temperature':best[2],'l2':best[1],
        'cost_weights':cw,'cost_scale':scale,'cost_floor':min(r['cpu_seconds'] for r in rows),
        'training_rows':len(rows),'validation_rows':len(validation),'selection_trials':trials,
        'cost_target':'measured verifier Ollama plus client Python CPU seconds','class_order':['repair','damage','unchanged']}

def predict(question,p,model):
    if model.get('version')!=VERSION:raise ValueError('VVP model version')
    x=vector(observed(question,p),model['feature_keys']);pr=distribution(model['weights'],x,model['temperature'],p['source']=='tool' and p['prediction'] is not None)
    cost=max(model['cost_floor'],math.exp(max(-12,min(5,dot(model['cost_weights'],x))))*model['cost_scale'])
    return {'p_repair':pr[0],'p_damage':pr[1],'p_unchanged':pr[2],'expected_value':pr[0]-pr[1],
        'estimated_cpu_seconds':cost,'value_per_estimated_cpu':(pr[0]-pr[1])/cost}

def estimate_cpu(question,p,model):
    x=vector(observed(question,p),model['feature_keys'])
    return max(model['cost_floor'],math.exp(max(-12,min(5,dot(model['cost_weights'],x))))*model['cost_scale'])
