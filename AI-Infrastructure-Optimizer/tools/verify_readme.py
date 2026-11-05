"""Recompute documented facts directly from the preserved CSV copies."""
import csv, json, math, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    facts = json.loads((ROOT / "docs/README_FACTS.json").read_text())["facts"]
    text = (ROOT / "README.md").read_text() + (ROOT / "docs/RESEARCH_LEDGER.md").read_text()
    cache = {}
    for fact in facts:
        path = fact["csv"]
        if path not in cache:
            with (ROOT / path).open(newline="") as stream: cache[path] = list(csv.DictReader(stream))
        rows = [row for row in cache[path] if all(str(row.get(k, "")) == str(v) for k, v in fact["where"].items())]
        assert len(rows) == fact["matched_rows"], fact["id"]
        values = [(float(row[fact["column"]].lower() == "true") if row[fact["column"]].lower() in ("true", "false") else float(row[fact["column"]])) for row in rows]
        agg = fact["aggregation"]
        assert agg != "unique" or len(values) == 1
        value = values[0] if agg == "unique" else statistics.mean(values) if agg == "mean" else sum(values)
        assert math.isclose(value, fact["value"], rel_tol=1e-12, abs_tol=1e-12), fact["id"]
        rendered = f"{value * fact['multiplier']:.{fact['decimals']}f}"
        assert rendered == fact["rendered"] and rendered in text, fact["id"]
    print(f"PASS: {len(facts)} documented numeric facts recomputed from {len(cache)} CSV files.")
if __name__ == "__main__": main()
