"""Canonical greedy tie audit from saved full-vocabulary top10s; no raw overwrite."""
import math,statistics
from .alsr_policy import fit_ridge,predict_ridge
from .alsr_impact_policy import select_threshold

def distribution_effect(full,skipped):
    import numpy as np
    from .alsr_impact_policy import distribution_effect as original_effect
    result=original_effect(full,skipped);a=np.asarray(full);b=np.asarray(skipped);f=int(np.argmax(a));s=int(np.argmax(b))
    result.update(full_top1=f,skip_top1=s,top1_changed=int(f!=s),full_top1_skip_margin=float(b[f]-np.max(np.delete(b,f))))
    return result

def canonical_top(distribution):
    pairs=list(zip(distribution['top_ids'],distribution['top_probabilities']));best=max(v for _,v in pairs)
    if all(v==best for _,v in pairs):raise ValueError('Saved top-k saturates with ties; full logits must be remeasured')
    return min(i for i,v in pairs if v==best)

def correct(rows,contexts):
    index={(r['task_id'],r['position']):r for r in contexts};out=[]
    for r in rows:
        context=index[r['task_id'],r['position']];full=context['baseline_top1'];assert canonical_top(context['baseline_distribution'])==full
        skip=canonical_top(r['skip_distribution']);margin=r['full_top1_skip_margin'];status='original_exact'
        if full!=r['full_top1']:
            d=r['skip_distribution'];pairs=list(zip(d['top_ids'],d['top_probabilities']));found=[v for i,v in pairs if i==full]
            if found:margin=math.log(found[0])-math.log(max(v for i,v in pairs if i!=full));status='reconstructed_from_saved_top10_probability_ratio'
            else:margin=None;status='requires_targeted_recheck_outside_saved_top10'
        out.append(dict(r,legacy_full_top1=r['full_top1'],legacy_skip_top1=r['skip_top1'],legacy_top1_changed=r['top1_changed'],full_top1=full,skip_top1=skip,top1_changed=int(full!=skip),full_top1_skip_margin=margin,margin_status=status))
    return out

def fit_profile(rows,shortlist):
    training=[r for r in rows if r['split']=='training'];ridge={};means={}
    for l in range(28):
        rr=[r for r in training if r['layer']==l];ridge[str(l)]=fit_ridge([r['features'] for r in rr],[r['kl']+r['top1_changed'] for r in rr]);means[str(l)]=statistics.mean(r['kl']+r['top1_changed'] for r in rr if r['position']==0)
    candidate=min(range(2,27),key=lambda l:(means[str(l)],l));cal=[dict(r,prediction=predict_ridge(r['features'],ridge[str(candidate)])) for r in rows if r['split']=='calibration' and r['position']==0 and r['layer']==candidate];threshold,trials=select_threshold(cal)
    return {'candidate':candidate,'ridge':ridge,'train_mean_risk':means,'threshold':threshold,'trials':trials,'shortlist':shortlist,'train_ids':sorted({r['task_id'] for r in training}),'calibration_ids':sorted({r['task_id'] for r in cal}),'feature_scope':'current block INPUT only; preceding block change; no current/future block output, full logits, references','tie_rule':'canonical greedy argmax = smallest token index among exact max ties; fixed before corrected generation'}
