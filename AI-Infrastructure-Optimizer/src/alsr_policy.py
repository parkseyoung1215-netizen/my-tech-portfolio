"""Project-local hypotheses, not established novel methods. No test labels accepted."""
import hashlib,math,random
SEED=88112026
METHODS=('baseline','static','random','bi_static','adaptive_sketch','random_match_sketch','adaptive_leverage','random_match_leverage')
MAX_SKIP=4

def pool_for(n):return sorted(set(round(4+i*(n-9)/7) for i in range(8)))
def fixed_layers(n):return pool_for(n)[1::2]
def random_layers(pool,k,prompt,seed):
    salt=int(hashlib.sha256(prompt.encode()).hexdigest()[:12],16);return sorted(random.Random(seed+salt).sample(list(pool),k))
def choose_by_risk(risks,threshold,pool,budget=MAX_SKIP):
    return sorted(sorted((l for l in pool if max(0,risks[l])<threshold),key=lambda l:(risks[l],l))[:budget])

def estimated_matmul_flops(config,forwards,gate_flops=0):
    """Analytical dense MAC*2 estimate, NOT hardware/GPU measurement."""
    d=config['hidden_size'];i=config['intermediate_size'];v=config['vocab_size'];kv=config['num_key_value_heads']*(d//config['num_attention_heads']);total=float(gate_flops)
    for f in forwards:
        t,s=f['query_tokens'],f['context_tokens'];per_layer=t*(4*d*d+4*d*kv+6*d*i)+4*t*s*d
        total+=len(f['executed'])*per_layer+2*d*v*f.get('head_positions',1)
    return total

def fit_ridge(xs,ys):
    import numpy as np
    x=np.asarray(xs,dtype=float);y=np.asarray(ys,dtype=float);scale=np.std(x,axis=0);scale[scale<1e-6]=1.;mean=np.mean(x,axis=0);z=(x-mean)/scale;z=np.column_stack([np.ones(len(x)),z])
    reg=np.eye(z.shape[1]);reg[0,0]=0.;w=np.linalg.solve(z.T@z+reg,z.T@y)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'weights':w.tolist()}
def predict_ridge(x,model):
    z=[1.]+[(a-m)/s for a,m,s in zip(x,model['mean'],model['scale'])]
    return max(0,sum(a*b for a,b in zip(z,model['weights'])))
