# RCAV Controlled Efficiency Validation: frozen design

No new routing algorithm. Preserve all previous code/data. Use old RCAV150 and
fresh300 (60/type), five fixed-seed repeats per installed existing model. Fresh
knowledge is atomic-number, prism and Boolean-gate knowledge, a synthetic STEM
bias; this is not a representative business workload or semantic-template split.

Use original frozen RCAV/Selective profiles and original1.5B improved guard.
For0.5B only, fit the same guard rules on its historical training100 plus the
existing guard training100, never evaluation data. No policy threshold tuning.

Generate each model's450 initial solutions once, including the unchanged exact
calculator path for90 arithmetic items. Store raw answers, request IDs, metadata
and hashes; never generate primaries during verification. Every method receives
the exact same answer. Independently execute actual needed verification calls:
no verifier-response replay or hypothetical calls passed off as execution.

Freeze profiles, cache and schedule before verification. First four repeats
rotate a random base order so every method occupies every position for every
10-task block. Fifth repeat repeats a position; block order is shuffled each
repeat. Interleave old/fresh blocks. Record full schedule, position and sequence.

Unload the two experimental models before each repeat, inspect /api/ps, then
warm up the current model with a neutral prompt. Log control calls and warmup
costs separately. This observes runner removal, not a guarantee about every OS,
GPU or intra-repeat prefix cache. Cache-hit tokens are unmeasured. No arbitrary
num_keep option or unverified cache-disable claim. Analyze paired cost differences
by target execution position and by repeat.

Measure actual verifier calls/tokens, server prefill/generation/total time, full
verification-stage wall time, Ollama CPU counter delta, Python process CPU and
RSS every100ms plus endpoints. Report observed sampled maximum, not true peak.
Sampling overhead remains and is not estimated/subtracted. Initial-generation,
calibration, load/unload and warmup costs stay separate. Adding the shared primary
cost to each method is labeled accounting, not separately executed pipelines.

Success criteria fixed before measurements: absolute accuracy tolerance1%p.
Require conservative exact95% paired lower bound >=-1%p and no lower answer
coverage. The bound uses all-repeat gains versus any-repeat losses on unique
questions, with Clopper-Pearson tails .025 each. Require paired block-cluster
bootstrap95% upper deltas for measured CPU and wall time vs Selective <0,
and nonpositive CPU/wall deltas in each of four target positions. Require these
conditions on both old and fresh data and observed runner resets. Failure or
insufficient evidence on any condition means no success declaration.

Bootstrap4000 draws, keeping all five repeats of a block together. Accuracy
intervals also include ordinary paired question-cluster bootstrap for readability.
Repeated frozen answers are not independent2250 questions. These statistical
estimates condition on fixed primaries/profiles and do not capture training or
template-family uncertainty. No money, energy or GPU-FLOPs estimates.
