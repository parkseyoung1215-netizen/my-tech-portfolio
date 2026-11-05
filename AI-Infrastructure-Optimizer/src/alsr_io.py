"""Normalize heterogeneous trace columns without changing the legacy CSV writer."""
from .tool_experiment import write_records

def write_union(path,rows,fields=None):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r)) if rows else list(fields or ['status'])
    normalized=[{k:r.get(k) for k in keys} for r in rows]
    return write_records(path,normalized,keys)
