"""Explicit full inference replay; never describe reused questions as unseen."""
import json,sys
from pathlib import Path
from uuid import uuid4
from .oracle_experiment import ROOT,sha,run
from .adaptive_experiment import write_json

def replay(manifest):
    source=Path(manifest).resolve();m=json.loads(source.read_text())
    for key in ('core_sha256','legacy_sha256'):
        for f,h in m[key].items():assert sha(f)==h,'Source differs; not an exact protocol replay'
    assert sha(ROOT/'ORACLE_DESIGN.md')==m['design_sha256']
    preserved={str(p.resolve()):sha(p) for base in ('src','tests','data') for p in (ROOT/base).rglob('*') if p.is_file()};preserved.update({str(p.resolve()):sha(p) for p in ROOT.iterdir() if p.is_file()})
    out=ROOT/'data'/('oracle_replay_'+uuid4().hex);out.mkdir();m.update(preserved_sha256=preserved,replay_from=str(source),new_exact_stems=False,replay_scope='Same already-used questions; recompute Oracle tables and all real model requests; no fresh-data claim')
    write_json(out/'MANIFEST.json',m);write_json(Path(str(out)+'_manifest.json'),{'datasets':m['datasets'],'manifest':str(out/'MANIFEST.json'),'reused_questions':True});run(out)

if __name__=='__main__':replay(sys.argv[1])
