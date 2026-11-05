# Publication provenance and limitations

This update is confined to `AI-Infrastructure-Optimizer/`. Other portfolio
projects are unchanged. Previous versions of changed public project files are
also retained under `docs/history/83145a4/`; Git history remains intact.
The original notebook workspace and all research data there are unchanged.

## Public copies

Research Python, tests, CSV, JSON/JSONL, Markdown and plotted PNG/SVG files are
included. Model weights, virtual environments, credentials, caches and lock files
are excluded. Public copies replace private absolute project paths with relative
paths and redact other private home paths; Markdown project links are made
relative to their reports. Measured numeric values are not changed.
`publication/LINEAGE.json` maps original private source hashes to public hashes.

Existing public files that were unchanged in the private source since the first
publication are retained byte for byte. Their earlier hash rebasing is documented
in the preserved `PUBLICATION_HASH_HISTORY.json`. This update does NOT rewrite
historical measurement fingerprints to pretend sanitized files are original bytes.
Original run receipts describe the original machine, not a fresh public rerun.

Privacy edits can invalidate private historical byte hashes and absolute paths.
Use the explicitly described public verification/test tools. The declared public
lineage is a publication record, not independent proof that an experiment ran.
Model replay additionally needs the recorded local checkpoint and package versions;
weights and temporary notebook environments are not bundled.

All performance claims concern actual stored local measurements. Unit-test mocks
are not performance evidence. GPU FLOPs/power/money were not measured; analytical
FLOPs and resource proxies are labelled. Oracle tables intentionally used labels
and future activations and are not deployable algorithms. Adaptive Precision has
not been started and is not presented as a completed study.
