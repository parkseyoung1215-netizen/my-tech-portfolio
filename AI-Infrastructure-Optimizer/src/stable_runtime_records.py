"""Append-only per-request checkpoints; never overwrite successful or partial attempts."""
import json,os,time
from pathlib import Path
from uuid import uuid4

class Store:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def load(self,key):
        folder=self.root/key
        valid=[]
        for p in sorted(folder.glob('*.json')) if folder.exists() else []:
            try:
                row=json.loads(p.read_text())
                if row.get('_complete') is True and row.get('_key')==key:valid.append(row)
            except (ValueError,OSError):pass
        if len(valid)>1:raise RuntimeError('Multiple complete checkpoint attempts: '+key)
        return valid[0] if valid else None
    def save(self,key,row):
        if self.load(key) is not None:raise FileExistsError(key)
        folder=self.root/key;folder.mkdir(parents=True,exist_ok=True);path=folder/(uuid4().hex+'.json');record=dict(row,_complete=True,_key=key)
        with path.open('x') as f:json.dump(record,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        return record
    def rows(self):
        out=[]
        for p in self.root.rglob('*.json'):
            try:r=json.loads(p.read_text())
            except (ValueError,OSError):continue
            if r.get('_complete') is True:out.append(r)
        return out
