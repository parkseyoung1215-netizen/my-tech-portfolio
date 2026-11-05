"""Read-only checkpoint status; does not import MLX or call a model."""
import collections,json,sys
from pathlib import Path
from .stable_runtime_records import Store
if __name__=='__main__':
    p=Path(sys.argv[1]).resolve();m=json.loads((p/'MANIFEST.json').read_text());rows=Store(p/'checkpoints').rows();print(json.dumps({'saved_complete_records':len(rows),'phases':dict(collections.Counter(r.get('phase','smoke') for r in rows)),'scheduled_main_records':len(m['schedule']),'resume_command':'.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment '+str(p)},indent=2))
