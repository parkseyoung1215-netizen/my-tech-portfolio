"""Supplement CPU uncertainty from the existing real traces, without overwriting them."""
import json,sys
from pathlib import Path
from .alsr_analysis import paired_ratio
from .alsr_io import write_union

def analyze(folder):
    p=Path(folder);rows=[]
    for scope,name in [('natural','GENERATION'),('fixed','FIXED_WORK')]:
        raw=[json.loads(s) for s in (p/(name+'.jsonl')).read_text().splitlines()];index={(r['method'],r['task_id'],r['repetition']):r for r in raw};ids=sorted({r['task_id'] for r in raw});methods=sorted({r['method'] for r in raw})
        base={id:{rep:index['baseline',id,rep]['cpu_seconds'] for rep in range(1,4)} for id in ids}
        for method in methods:
            if method=='baseline':continue
            target={id:{rep:index[method,id,rep]['cpu_seconds'] for rep in range(1,4)} for id in ids}
            rows.append(dict(method=method,workload=scope,**paired_ratio(target,base,draws=3000),measurement='Python process CPU time, not GPU/device total compute'))
    write_union(p/'CPU_COMPARISONS.csv',rows)

if __name__=='__main__':analyze(sys.argv[1])
