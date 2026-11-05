"""Pre-block signal math; no evaluation references or counterfactual outcomes accepted."""
import math
from .alsr_policy import predict_ridge
METHODS=('baseline','adaptive_sketch','static','random','stable1','random1','prelayer_gate')
THRESHOLDS=(0,.01,.025,.05,.1,.2,.5,1,2)

def select_threshold(records):
    options=[];trials=[]
    for threshold in THRESHOLDS:
        selected=[r for r in records if r['prediction']<threshold]
        valid=not any(r['top1_changed'] for r in selected) and (sum(r['kl'] for r in selected)/max(1,len(selected)))<=.02
        trials.append(dict(threshold=threshold,selected=len(selected),flips=sum(r['top1_changed'] for r in selected),mean_kl=sum(r['kl'] for r in selected)/max(1,len(selected)),eligible=valid))
        if valid:options.append((len(selected),-threshold,threshold))
    return max(options)[2],trials

def accept(features,profile):
    risk=predict_ridge(features,profile['ridge'][str(profile['candidate'])])
    return risk<profile['threshold'],risk

def distribution_effect(full,skipped):
    import numpy as np
    a=np.asarray(full,dtype=np.float64);b=np.asarray(skipped,dtype=np.float64)
    la=a-np.logaddexp.reduce(a);lb=b-np.logaddexp.reduce(b);pa=np.exp(la);pb=np.exp(lb);lm=np.logaddexp(la,lb)-math.log(2)
    aa=np.argpartition(a,-2)[-2:];aa=aa[np.argsort(a[aa])[::-1]];bb=np.argpartition(b,-2)[-2:];bb=bb[np.argsort(b[bb])[::-1]]
    x=int(aa[0]);other=float(np.max(np.delete(b,x)))
    return dict(kl=max(0,float(np.sum(pa*(la-lb)))),js=max(0,float(.5*np.sum(pa*(la-lm))+.5*np.sum(pb*(lb-lm)))),top1_changed=int(x!=int(bb[0])),full_top1=x,skip_top1=int(bb[0]),full_margin=float(a[aa[0]]-a[aa[1]]),skip_margin=float(b[bb[0]]-b[bb[1]]),full_top1_skip_margin=float(b[x]-other))

def auc(labels,scores):
    pos=[s for y,s in zip(labels,scores) if y];neg=[s for y,s in zip(labels,scores) if not y]
    if not pos or not neg:return None
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
