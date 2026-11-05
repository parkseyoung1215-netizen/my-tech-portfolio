"""Support-aware calibrated filter on top of the frozen Selective gate; no AI calls."""
import math,statistics
from collections import defaultdict
from .solver300_runtime import features
from .controlled_runtime import choose
VERSION='pvcg-1'
FORBIDDEN={'expected','label','correct','primary_correct','final_correct','revision','verification_answer','verification_prediction','verification_result'}

def public(question,primary):
    if FORBIDDEN&primary.keys():raise ValueError('Gold or future verification information at PVCG boundary')
    f=features(question);state='unresolved' if primary['prediction'] is None else 'resolved' if primary['format_compliant'] else 'noncompliant'
    return f['category'],state

def fit_groups(rows,old):
    groups=defaultdict(list)
    for r in rows:
        q,p=r['question'],r['primary'];cat,state=public(q,p)
        if not choose(q,p,'selective',old)['verify']:continue
        groups[cat+'|*'].append(r);groups[cat+'|'+state].append(r)
    result={}
    for key,rr in groups.items():
        gains=sum(r['gain']>0 for r in rr);damage=sum(r['gain']<0 for r in rr);n=len(rr)
        # Symmetric neutral Dirichlet prior (repair1, damage1, unchanged2).
        mean=(gains-damage)/(n+4);second=(gains+damage+2)/(n+4)
        costs=[r['verifier_cpu_seconds'] for r in rr if r['verifier_cpu_seconds'] is not None and r['verifier_cpu_seconds']>0]
        result[key]={'samples':n,'repair':gains,'damage':damage,'mean_gain':mean,'standard_error':math.sqrt(max(0,second-mean**2)/(n+5)),
                     'median_cpu':statistics.median(costs) if costs else None}
    return result

def decide(question,primary,profile,old):
    cat,state=public(question,primary)
    if profile['version']!=VERSION:raise ValueError('PVCG version mismatch')
    if primary['source']=='tool' and primary['prediction'] is not None:return {'verify':False,'reason':'proven_tool'}
    if not choose(question,primary,'selective',old)['verify']:return {'verify':False,'reason':'legacy_selective_skips'}
    key=cat+'|'+state;g=profile['groups'].get(key)
    if not g or g['samples']<profile['min_support']:key=cat+'|*';g=profile['groups'].get(key)
    if not g or g['samples']<profile['min_support'] or g['median_cpu'] is None:return {'verify':False,'reason':'insufficient_training_support','group':key}
    lower=g['mean_gain']-profile['uncertainty_weight']*g['standard_error'];threshold=profile['cpu_penalty']*g['median_cpu']
    verify=lower>threshold
    return {'verify':verify,'reason':'positive_supported_value' if verify else 'insufficient_expected_value','group':key,
       'predicted_gain':g['mean_gain'],'uncertainty_adjusted_gain':lower,'estimated_cpu_seconds':g['median_cpu'],'cost_threshold':threshold}

def train(rows,validation,old):
    groups=fit_groups(rows,old);trials=[];best=None
    for weight in (0.,.5,1.):
        for penalty in (0.,.05,.2):
            profile={'version':VERSION,'min_support':10,'groups':groups,'uncertainty_weight':weight,'cpu_penalty':penalty}
            chosen=[decide(r['question'],r['primary'],profile,old)['verify'] for r in validation]
            correct=sum(r['final_correct'] if use else r['primary_correct'] for r,use in zip(validation,chosen))
            costs=[r['verifier_cpu_seconds'] for r,use in zip(validation,chosen) if use]
            if any(v is None for v in costs):raise ValueError('Calibration CPU missing; do not invent measured costs')
            cpu=sum(costs);trial={'uncertainty_weight':weight,'cpu_penalty':penalty,'validation_correct':correct,'selected_verifications':sum(chosen),'validation_measured_verifier_cpu_sum':cpu}
            trials.append(trial);score=(correct,-cpu,weight,penalty)
            if best is None or score>best[0]:best=(score,profile)
    return dict(best[1],trials=trials,training_rows=len(rows),validation_rows=len(validation),selection='Max calibration accuracy; ties lower measured CPU, then more conservative parameters. No test tuning.')
