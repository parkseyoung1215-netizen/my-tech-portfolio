# Pre-solve omission research

Run locally: `.venv/bin/python -B -u src/skipwork_experiment.py`.
Test: `.venv/bin/python -B -m unittest discover -s tests -v`.
Requires an already installed `qwen2.5:1.5b` on localhost Ollama. No paid API.

Independent files preserve all previous experiments. Every run uses exclusive,
UUID-named outputs. A repeated run refuses previously evaluated question stems;
prepare a new independent dataset rather than treating a repeated test as fresh.

## Frozen design

150 new questions, 30 each arithmetic, knowledge, logic, coding, comprehension.
100 separate stored calibration questions estimate verifier corrected/harmed counts.
Gold labels enter calibration statistics and post-execution scoring, never runtime
questions or policy decisions. All omission decisions precede primary generation.

* Baseline: previous Hybrid solution path (safe calculator or 1.5B), then always
  independently verify with 1.5B. This deliberately constructed ablation baseline
  is not claimed to be the best prior production router.
* Early Exit: skip model verification for syntactically proven safe integer math.
* Selective Verification: also omit easy-category verification if >=20 calibration
  examples show nonpositive marginal benefit (corrected <= harmed). Hard or
  insufficiently calibrated requests retain verification. This exploratory point
  estimate is not an accuracy guarantee or a confidence bound.
* Budget Cap hypothesis: cap primary multiple-choice output at 8 vs 64 tokens,
  retaining verification. Calls can remain equal while generation work changes.
* Batch Verify hypothesis: pre-group by inferred type, <=4 requests / 2600 input
  question characters; verify a group in one JSON call (128 output tokens).
  Reduced calls do not imply reduced tokens, latency, or compute.

The latter hypotheses differ from earlier model-choice/calculator-routing controls.
The same 150 questions and primary/verification protocol apply to all arms except
the explicit output-cap/batch interventions. A malformed verifier preserves the
primary prediction; no hidden retry. Tool failure remains unresolved when skipped.

## Accounting

All block requests arrive together. Per-request completion latency includes queue
time, including the full wait for a shared batch verifier. Block wall time measures
total processing separately. Rotate method order across blocks. Shared model calls
and their server-reported tokens appear once in calls.csv. No artificial per-item
token division. Log raw prompts, responses, request hashes, UUIDs and frozen plans.

Ollama process CPU deltas and RSS boundary samples are actual measurements, but
do not measure GPU FLOPs, peak/exclusive RAM, energy or monetary cost. Input tokens
do not distinguish cached prefill. Calibration overhead is recorded separately;
the initial warmup is outside evaluation. Single-run timings are exploratory.

Answer parsing accepts a bare option/integer or one unambiguous FINAL_ANSWER line
using the existing protocol; no gold-driven response rescue. Log unresolved format
and content errors separately. Posthoc correct-to-correct verification is described
as redundant by outcome, not knowably unnecessary before execution.

Dataset caveats: English synthetic templates, capital-city-heavy knowledge, code
output prediction rather than code authoring. Novel stems are checked against all
stored CSV/JSON questions and option-shuffled duplicates, not semantic independence
of every template. Calibration topics differ from fresh knowledge topics.
