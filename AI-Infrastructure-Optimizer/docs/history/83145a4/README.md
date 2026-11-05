# AI Infrastructure Optimizer

A local research prototype for choosing an efficient way to answer a request: a small language model, a larger
model, or a bounded arithmetic tool. It compares accuracy, answer coverage, latency, model calls, tokens and
observed CPU/RSS usage using existing Ollama models and no paid inference API.

**Latest finding:** on a new 300-task evaluation, Hybrid reached **80.33% accuracy** versus **75.33% for Rule-Based**,
with **100% answer coverage** for both. Hybrid used fewer model calls and less request-processing time. Calibration
overhead was not recovered in the first batch, and Hybrid regressed on six individual questions.

## Goals

- Preserve or improve answer quality while avoiding unnecessary model inference.
- Apply a calculator only when the whole request is explicitly supported arithmetic.
- Select models from request-time features and separate calibration evidence, never evaluation labels.
- Report refusals, failures, trade-offs and unmeasured costs without inflating performance claims.

## Current implementation

| Component | Files |
|---|---|
| Ollama benchmarks | `src/benchmark.py`, `src/benchmark_v2.py` |
| Explicit integer answer validation | `src/answer_validator.py` |
| Measured model selection and execution | `src/model_router.py`, `src/routing_runner.py` |
| Calibration and conservative fallback | `src/adaptive_router.py`, `src/adaptive_experiment.py` |
| Verification-gated Coverage Routing | `src/coverage_policy.py`, `src/coverage_runner.py` |
| Direct safe arithmetic | `src/safe_calculator.py`, `src/tool_router.py` |
| Mixed-task evaluation and scoring audit | `src/mixed_experiment.py`, `src/mixed_reliability_audit.py` |
| Fixed / Rule / Confidence / Hybrid study | `src/solver300_runtime.py`, `src/solver300_experiment.py` |
| Cost and cache examples | `src/cost_analyzer.py` — illustrative prices, not measured Ollama costs |

The calculator parses an AST and explicitly computes allowed integer `+`, `-`, `*`, parentheses and unary signs.
It does not execute expression source code. Input length, node count, depth, literal size and intermediate values
are bounded. Unsupported text, units, decimals, division, powers and function calls are not interpreted as arithmetic.

## Latest 300-task experiment

There are 60 tasks each in arithmetic, general knowledge, logic, Python output prediction and reading comprehension.
Evaluation excludes earlier question stems, reordered-choice duplicates and commutative arithmetic duplicates.
Calibration uses 96 earlier mixed tasks plus four new explicit reading tasks, separate from the 300 evaluation tasks.
Policies and the answer contract were frozen before evaluation.

| Strategy | Accuracy | Coverage | Mean latency | Request time total | 0.5B / 1.5B calls | Tool calls | Input / output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 1.5B | 70.00% | 100% | 67.93 ms | 20.380 s | 0 / 300 | 0 | 26,288 / 746 |
| Rule-Based | 75.33% | 100% | 67.76 ms | 20.327 s | 120 / 167 | 60 | 27,513 / 741 |
| Confidence-Based | 70.00% | 100% | 72.18 ms | 21.655 s | 0 / 300 | 0 | 26,288 / 746 |
| Hybrid | **80.33%** | **100%** | **48.50 ms** | **14.549 s** | **0 / 240** | 60 | **23,088 / 480** |

Rule-Based uses the tool for arithmetic, 0.5B for short knowledge/reading requests and 1.5B for other requests.
Confidence-Based permits 0.5B only with at least 20 calibration samples, a 95% Wilson lower accuracy bound of at
least 0.8, and an easy request. Hybrid combines that model selection with direct arithmetic. Small-model format/API
failure can fall back to 1.5B; semantic errors are not detected by looking at the expected answer.

No 0.5B category passed calibration in this run. Confidence-Based consequently behaved like Fixed, while Hybrid
used the tool for 60 arithmetic requests and 1.5B for the other 240. This does not demonstrate activation of several
reliable small-model routes. Rule-Based made 47 additional fallback calls after format failures.

Observed Hybrid versus Rule: +5 percentage points accuracy, approximately 28.4% less request time and 47 fewer
model calls. Six questions changed from correct to wrong, while 21 changed from wrong to correct. One-run paired
bootstrap intervals favor Hybrid but do not establish independent repeated-run or production superiority.

Calibration required **200 model calls, 19.630 seconds, 16,908 input tokens and 1,173 output tokens**. Charging the
entire calibration time to Hybrid gives about **34.178 seconds** for calibration plus its first 300 requests, versus
20.327 seconds for Rule's evaluation requests. First-batch break-even was not demonstrated.

- [Full assessment](data/solver300_c22b83397f0945b4adfb8304f7b2f268_assessment.md)
- [Per-category measurements](data/solver300_c22b83397f0945b4adfb8304f7b2f268_summary.csv)
- [Hybrid–Rule comparison](data/solver300_c22b83397f0945b4adfb8304f7b2f268_comparison.csv)
- [Raw requests, decisions, responses and fallbacks](data/solver300_c22b83397f0945b4adfb8304f7b2f268_evaluation.csv)

## Earlier studies and reliability lessons

- Direct tool routing answered all 300 supported arithmetic expressions correctly without model calls. This is an
  arithmetic-domain result, not 100% accuracy on general AI tasks.
  [Report](data/tool_routing_73d749e404ea4347bbba5883b485a75e_assessment.md)
- A mixed-task evaluation initially scored Tool-Assisted at 45% under a strict `FINAL_ANSWER` contract. An offline,
  gold-independent content audit scored the same responses at 83%. **The original score was preserved.** Missing
  prefixes explained 38 otherwise-correct answers; four ownership labels were flagged as ambiguous. Non-arithmetic
  paths were identical across strategies and the relative gains were arithmetic-specific.
  [Reliability audit](data/mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_assessment.md)
- Coverage Routing improved accepted answer quality but missed 95% coverage and increased retry time.
  [Negative result](data/coverage_951686cca01344ed8b732fc1e66d7cab_assessment.md)

These studies use different data and protocols. Their scores are not a directly comparable model-training
improvement curve. Earlier Korean-language reports are preserved alongside this English overview.

## Setup and offline tests

Tested with Python 3.14. Python dependencies are standard-library only. Model binaries are not included.

```bash
git clone https://github.com/parkseyoung1215-netizen/my-tech-portfolio.git
cd my-tech-portfolio/AI-Infrastructure-Optimizer
python3 -m venv .venv
.venv/bin/python -B -m unittest discover -s tests -v
```

The public snapshot is checked with the complete **123-test** suite. Tests use temporary directories and mocks
where appropriate; mock responses are not published experimental measurements.

Run a tool-only request without Ollama:

```bash
.venv/bin/python -B src/tool_router.py --question '(140 * 93) - 28'
```

## Local model execution

For model-backed requests, run Ollama locally and make `qwen2.5:0.5b` and `qwen2.5:1.5b` available yourself.
The application does not install models automatically. Both measured models were Q4_K_M quantized versions.

Use the saved Hybrid profile:

```python
import json
from pathlib import Path
from src.solver300_runtime import execute

profile = json.loads(Path(
    'data/solver300_c22b83397f0945b4adfb8304f7b2f268_profile.json'
).read_text())
result = execute('27 * 48', 'hybrid', profile)
print(result['answer'])
```

Experiment entry points include `src/solver300_experiment.py`, `src/tool_experiment.py` and `src/mixed_experiment.py`.
They require the existing local Ollama models. Fresh-data builders deliberately stop when their fixed question bank
would reuse recorded tasks. Running them over this complete archive is **not** a fresh benchmark. Inspect saved
manifests and use the runtime APIs for individual requests, or prepare a separately reviewed new question bank.
Do not delete historical results to bypass duplicate checks. Experiment writers use unique filenames and exclusive
creation to prevent overwrites.

## Measurement limits and publication integrity

- Request times include routing, parsing, tools and fallback. Warm-up, grading and `ps` sampling are excluded.
- CPU counters cover Ollama processes; latest experiments record router Python CPU separately. RSS is a boundary
  sample, not dedicated model memory or a continuous peak. Memory savings were not established.
- Monetary cost, power, GPU utilization and FLOPs are **unmeasured**. Tokens are not converted into money saved.
- Data is English, synthetic, multiple-choice and template-heavy. Coding tests output prediction rather than free-form
  software construction. Production concurrency, multilingual tasks and long documents remain untested.
- This is a privacy-cleaned snapshot. Private paths were normalized and public integrity references were refreshed
  transparently. Original fingerprints are retained in [PUBLICATION_HASH_HISTORY.json](PUBLICATION_HASH_HISTORY.json).
  See [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md). Numerical experimental results were not changed.

## Layout

```text
src/      Runtime strategies, validators, calculator and experiment/audit scripts
tests/    Offline unit and integration tests
data/     CSV measurements, profiles, manifests, resource logs and Markdown reports
docs/     Original project README
```
