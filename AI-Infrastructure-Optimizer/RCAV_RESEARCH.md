# Risk-Cost Adaptive Verification (RCAV)

An exploratory expected-value policy; no claim of a new general technology.
All previous code/results remain immutable. New independent modules only.

Run `.venv/bin/python -B -u src/rcav_experiment.py` after checking local Ollama.
Test `.venv/bin/python -B -m unittest discover -s tests -q`.
Default generator refuses evaluation/calibration stems already stored in data;
repeated trials within a run reuse the frozen holdout by design.

## Prespecified decision and experiment

Fresh calibration100 (20/category), fresh evaluation150 (30/category), mutually
disjoint and checked against old stems, commutative expressions and shuffled MC
options. Five task categories; synthetic templates remain semantically related.
Trusted code/arithmetic references are independently checked before inference.

All models' profiles freeze before any evaluation. Runtime sees only question,
actual primary response, parser status, inferred request features and a frozen
profile. No evaluation gold or future verification outcome. Training labels are
used solely for calibration; no evaluation-dependent threshold tuning.

For a calibrated stratum, estimate primary error probability p, conditional repair
probability r and conditional damage probability h. Global Beta(1,1) rates and
empirical prior strength10 shrink small groups; n<8 falls back to category n>=10
or global. Proven calculator answers have risk0; unresolved responses have valid
answer failure risk1. This is not semantic certainty or a calibrated confidence
guarantee. Delta = p*r-(1-p)*h estimates net error reduction.

Fit nonnegative-slope linear regression from verification prompt characters to
actual calibration verifier prefill+generation seconds; floor at observed minimum.
Cref = calibration median verification time. Verify iff Delta > lambda*Cpred/Cref.
Main lambda .05; sensitivity lambda 0 and .15 are fixed in advance and all reported.
Lambda is an arbitrary declared quality/time preference, not an optimized money
conversion. Lambda0 is the no-cost-term ablation.

Compare exhaustive Baseline, existing Selective rule recalibrated on the same new
training data, and all three RCAV weights. Three fixed-seed trials of identical150
tasks, rotating method order and shuffling blocks before inference. Existing local
1.5B required, 0.5B tested if installed; no downloads, paid API or install actions.
Both models calibrate separately. Warmups and calibration measured separately.

Unique calls, server tokens and durations, block wall time, per-request service and
queued completion latency, Ollama process CPU and boundary RSS are measured.
CPU/latency/token savings are not money, energy or FLOPs savings. Pareto frontiers
use observed accuracy, coverage and mean model time/wall/CPU axes; tiny timing
differences and shared caches can change empirical dominance. Repeats do not make
the150 questions independent450 samples.

Failure analysis separates risk forecast error, verification damage, formatting
failure and time forecast residuals. Skipped verification has no same-execution
ground truth cost; matching Baseline candidates can supply a separately labeled
paired reference, never an input to decisions. No hidden shadow inference.
