"""Verify this public snapshot; not independent proof of historical executions."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    inventory = json.loads((ROOT / "publication/FILES.json").read_text())
    for item in inventory["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        raw = path.read_bytes()
        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], item["path"]
    expected = {item["path"] for item in inventory["files"]} | {"publication/FILES.json"}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and not any(part in {".git", "__pycache__", ".venv", ".alsr-venv"} for part in p.relative_to(ROOT).parts) and p.suffix != ".pyc"}
    assert actual == expected, {"extra": sorted(actual - expected), "missing": sorted(expected - actual)}
    lineage = json.loads((ROOT / "publication/LINEAGE.json").read_text())
    for item in lineage["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["public_sha256"], item["path"]
    print(f"PASS: {len(inventory['files'])} public files and {len(lineage['files'])} source-lineage entries. No claim of original private-byte identity.")
if __name__ == "__main__": main()
