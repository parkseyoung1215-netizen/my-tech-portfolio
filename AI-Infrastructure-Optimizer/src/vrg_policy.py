"""Post-verification acceptance. Public interfaces never receive reference labels."""
import math,re
from .vvp_content_audit import interpret
from .efficiency_protocol import extract
from .solver300_runtime import features
from .vvp_model import observed,vector,dot,softmax
VERSION='vrg-1'

def check_public(p,v):
    forbidden={'expected','correct','primary_correct','final_correct','revision_correct','label'}
    if forbidden.intersection(p) or forbidden.intersection(v):raise ValueError('Reference labels are forbidden at the acceptance boundary')

def parse(q,answer):
    kind=features(q)['answer_kind'];old=extract(q,answer,kind);p,mode=interpret(q,answer,kind)
    return {'prediction':p,'format_compliant':old['format_compliant'],'parse_mode':mode}

def attributes(q,p,v):
    check_public(p,v)
    d={'initial:'+k:x for k,x in observed(q,p).items()}
    d.update({'revision:'+k:x for k,x in observed(q,dict(v,source='model')).items()})
    d.update(same=float(p['prediction']==v['prediction']),both_resolved=float(p['prediction'] is not None and v['prediction'] is not None),
             resolved_gain=float(p['prediction'] is None and v['prediction'] is not None),
             length_change=max(-5,min(5,(len(v['answer'])-len(p['answer']))/100)),
             initial_uncertainty=float(bool(re.search(r'not sure|cannot determine|uncertain',p['answer'],re.I))))
    return d

def heuristic(q,p,v):
    check_public(p,v)
    if v['prediction'] is None:return False,'revision_unresolved'
    if p['prediction']==v['prediction']:return True,'same_content'
    if p['source']=='tool' and p['prediction'] is not None:return False,'preserve_proven_tool'
    if p['prediction'] is None:return True,'recover_unresolved_initial'
    return False,'preserve_resolved_initial'

def train(rows,validation):
    keys=sorted({k for r in rows for k in attributes(r['question'],r['primary'],r['revision'])});xs=[vector(attributes(r['question'],r['primary'],r['revision']),keys) for r in rows]
    ys=[r['label'] for r in rows];n=len(xs);best=None;trials=[]
    for reg in (.01,.1,1.):
        w=[[0.]*len(keys) for _ in range(3)]
        for epoch in range(200):
            g=[[0.]*len(keys) for _ in range(3)]
            for x,y in zip(xs,ys):
                pr=softmax([dot(a,x) for a in w])
                for c in range(3):
                    err=(pr[c]-(c==y))/n
                    for j,z in enumerate(x):g[c][j]+=err*z
            rate=.2/(1+epoch/100)
            for c in range(3):
                for j in range(len(keys)):w[c][j]-=rate*(g[c][j]+reg*w[c][j]*(not keys[j].endswith(':bias')))
        for temp in (.75,1.,1.5):
            loss=sum(-math.log(max(1e-12,softmax([dot(a,vector(attributes(r['question'],r['primary'],r['revision']),keys))/temp for a in w])[r['label']])) for r in validation)/len(validation)
            trials.append({'l2':reg,'temperature':temp,'validation_logloss':loss})
            if best is None or loss<best[0]:best=(loss,reg,temp,[a[:] for a in w])
    return {'version':VERSION,'keys':keys,'weights':best[3],'temperature':best[2],'l2':best[1],'trials':trials,'training_rows':len(rows),'validation_rows':len(validation)}

def learned(q,p,v,profile):
    if profile['version']!=VERSION:raise ValueError('VRG version mismatch')
    x=vector(attributes(q,p,v),profile['keys']);pr=softmax([dot(a,x)/profile['temperature'] for a in profile['weights']]);ev=pr[0]-pr[1]
    safe,reason=heuristic(q,p,v)
    if reason in ('revision_unresolved','preserve_proven_tool'):accept=False
    elif reason=='same_content':accept=True
    else:accept=ev>0
    return accept,{'p_repair':pr[0],'p_damage':pr[1],'expected_gain':ev,'reason':'positive_expected_gain' if accept else reason}
