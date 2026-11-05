# Repeated efficiency verification

Run: `.venv/bin/python -B -u src/efficiency_repeat.py`

Test: `.venv/bin/python -B -m unittest discover -s tests -v`

The previous experiment files remain immutable. New independent modules implement
the same five policies with corrected gold-independent parsing. No new routing
algorithm. The previous 150 tasks are repeated three times, temperature 0, seed 42,
same local qwen2.5:1.5b. This measures replay reliability, not new-holdout accuracy.
Separate calibration 100 tasks are rerun once, then the existing uplift gate is
frozen before evaluation. Answers never enter omission decisions.

Strict raw output compliance and content correctness are separate. Fully fenced
JSON, unambiguous exact option notation/text, and single FINAL_ANSWER wrappers
can be read without accepting them as compliant with the bare-output contract.
Conflicting options, duplicate JSON keys and missing/extra IDs fail closed.
No regex search for a matching gold answer; unresolved content remains unresolved.

Measured: unique requests, server input/output token counts, client block wall time,
completion latency including waiting, server prefill/generation/total/load times,
Ollama process CPU counter deltas, Python CPU, boundary RSS samples. Shared batch
calls counted once. Per-repeat and mean/min/max/sample-SD CSVs are saved exclusively
with a UUID. Monetary cost, energy and FLOPs remain unmeasured. Three repeats of
the same tasks are not 450 independent tasks or a statistical noninferiority test.

The baseline still includes exhaustive verification and is not the strongest old
production Hybrid. Calibration/warmup and full evaluation-loop wall time are logged
separately. Warm-cache, order, system load and thermal effects remain limitations.
Original official results are retained; reinterpreting old selected final text is
separately labeled and never presented as a new timed run.
