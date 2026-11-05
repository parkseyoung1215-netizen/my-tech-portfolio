"""Prespecified inference on unique questions and paired block/repetition costs."""
import math,random,statistics

def quantile(v,p):
    v=sorted(v);x=(len(v)-1)*p;i=int(x)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)

def describe(values,draws=4000):
    if not values or any(v is None for v in values):return {'mean':None,'sample_std':None,'min':None,'max':None,'ci95_low':None,'ci95_high':None}
    rng=random.Random(11052026);n=len(values)
    boot=[sum(rng.choice(values) for _ in range(n))/n for i in range(draws)]
    return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if n>1 else None,
        'min':min(values),'max':max(values),'ci95_low':quantile(boot,.025),'ci95_high':quantile(boot,.975)}

def cp_lower(k,n,tail=.025):
    if k==0:return 0.
    if k==n:return tail**(1/n)
    lo,hi=0.,1.
    for _ in range(65):
        p=(lo+hi)/2;prob=sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1))
        if prob>tail:hi=p
        else:lo=p
    return (lo+hi)/2

def cp_upper(k,n,tail=.025):
    return 1-cp_lower(n-k,n,tail)

def accuracy_comparison(question_pairs,draws=4000):
    """Each item is a question's list of paired differences across five repeats."""
    n=len(question_pairs);gain_all=sum(all(d>0 for d in ds) for ds in question_pairs);loss_any=sum(any(d<0 for d in ds) for ds in question_pairs)
    differences=[statistics.mean(ds) for ds in question_pairs];rng=random.Random(11052026)
    boot=[sum(rng.choice(differences) for _ in range(n))/n for i in range(draws)]
    return {'accuracy_delta':statistics.mean(differences),'paired_bootstrap_low':quantile(boot,.025),'paired_bootstrap_high':quantile(boot,.975),
        'all_repeat_gain_questions':gain_all,'any_repeat_loss_questions':loss_any,
        'conservative_exact_lower95':cp_lower(gain_all,n)-cp_upper(loss_any,n),
        'inference_scope':'Unique-question clusters; frozen primaries. Exact lower uses all-repeat gains minus any-repeat losses with .025 tails.'}

def paired_cost(matrix,draws=4000):
    """Matrix[block][repeat] contains target minus control; resample both axes."""
    if not matrix or any(v is None for row in matrix for v in row):return {'delta':None,'ci95_low':None,'ci95_high':None}
    rng=random.Random(11052026);b=len(matrix);r=len(matrix[0]);samples=[]
    for _ in range(draws):
        bs=[rng.randrange(b) for i in range(b)];rs=[rng.randrange(r) for i in range(r)]
        samples.append(sum(matrix[i][j] for i in bs for j in rs)/r)
    return {'delta':sum(sum(row)/r for row in matrix),'ci95_low':quantile(samples,.025),'ci95_high':quantile(samples,.975)}
