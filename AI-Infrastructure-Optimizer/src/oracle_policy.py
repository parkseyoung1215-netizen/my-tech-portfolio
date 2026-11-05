"""Privileged offline schedules; no ground truth in runtime table lookup."""
import math,random
RATES=(0,10,20,30,40,50)
SEED=2026110606
def quota(step,rate,n=28):return (n*rate*step)//100-(n*rate*(step-1))//100 if step else 0
def random_plan(cap,rate,seed,n=28):
    rng=random.Random(seed);plan=[[]];previous=set()
    for step in range(1,cap):
        selected=sorted(rng.sample([i for i in range(n) if i not in previous],quota(step,rate,n)));plan.append(selected);previous=set(selected)
    return plan
def drift_plan(outputs,rate):
    import numpy as np
    cap,n,_=outputs.shape;anchors=outputs[0].copy();previous=set();plan=[[]];risks=[]
    for step in range(1,cap):
        diff=np.sqrt(np.sum((outputs[step]-anchors)**2,axis=1)/(np.sum(outputs[step]**2,axis=1)+1e-8))
        selected=sorted(sorted((i for i in range(n) if i not in previous),key=lambda i:(float(diff[i]),i))[:quota(step,rate,n)]);plan.append(selected);risks.append(diff.tolist())
        for i in range(n):
            if i not in selected:anchors[i]=outputs[step,i]
        previous=set(selected)
    return plan,risks
def validate(plan,rate,n=28):
    assert plan[0]==[]
    for step,s in enumerate(plan):
        assert s==sorted(set(s)) and all(0<=i<n for i in s) and len(s)==quota(step,rate,n)
        if step:assert not set(s)&set(plan[step-1])
def choose_candidate(candidates):
    """LABEL-AWARE offline selection; never used as a runtime policy."""
    return max(candidates,key=lambda c:(int(c['grade']['content_correct']),int(c['grade']['format_ok']),int(c['baseline_token_match']),-c['length_difference'],-c['index']))['index']
