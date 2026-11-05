"""Pre-evaluation hypotheses; no references or correctness labels accepted."""
import math
METHODS=('baseline','static','dlr','conservative')
CANDIDATES=(8,12,16,24)
SEED=2026110604
THRESHOLDS={'dlr':.5,'conservative':.2}
def reuse_allowed(distance,age,method):
    return method in THRESHOLDS and math.isfinite(distance) and 0<=distance<=THRESHOLDS[method] and age==0

def estimate(config,work):
    """Dense model matmul FLOPs only, MAC=2. No gate/nonlinear/normalization FLOPs."""
    d=config['hidden_size'];m=config['intermediate_size'];v=config['vocab_size'];kv=config['num_key_value_heads']*(d//config['num_attention_heads'])
    return sum(t*(a*(4*d*d+4*d*kv)+b*6*d*m)+a*4*t*s*d+2*d*v for t,s,a,b in work)
