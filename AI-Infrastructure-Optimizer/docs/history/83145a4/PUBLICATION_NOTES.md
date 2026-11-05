# Public snapshot and provenance

This is an additive project subdirectory. Existing portfolio files are not modified.
The private working project remains unchanged. All Python sources, tests, CSV measurements,
JSON manifests/profiles and Markdown reports are included. The English README is new;
the previous README is retained as `docs/README.original.md`.

## Privacy and integrity

Private absolute project paths were replaced with `.` in nine original data files.
No questions, answers, model outputs, measured timings, resource counters, token counts,
selection decisions or experimental scores were edited. Fictional names and toy password
strings in questions are synthetic benchmark inputs, not real credentials.

Path changes alter artifact hashes. Historical integrity maps and manifest/profile hash
references in the public copy were refreshed in dependency order for portable offline audits.
These hashes describe the published copy, not the original machine's historical byte identity.
`PUBLICATION_HASH_HISTORY.json` preserves the recorded hashes, public hashes and per-file
lineage without exposing private paths. Do not treat rebased integrity checks as independent
attestation of original execution. Private originals are retained unchanged.

## Exclusions and limits

Virtual environments, caches, credentials, private keys, agent settings and model weights
are excluded. Python dependencies are standard-library only. No paid API is required.
All performance numbers are existing local measurements; tests are not performance data.
Money, power, GPU utilization and dedicated model memory were not measured.
