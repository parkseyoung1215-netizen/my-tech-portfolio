# Zero-Overhead Oracle: prespecified local capability probe

Preserve every existing study. Existing Qwen2.5-1.5B-Instruct BF16, 28 native
MLX blocks, no custom replacement GEMV, unchanged greedy chat template/EOS.
No paid API or upload. Seed 2026110606. Offline-only HF weights, hash checked.

Twenty tasks:8 existing (6 MCQ,2 JSON),8 previously unused (4 MCQ,2 JSON,2
closed factual color answers),4 new long literal-copy tasks (2 repetitive,
2 varied;24 words). Short cap32/long cap64 for EVERY method. No exclusions
based on correctness. Existing scoring functions reused without changes.
This is a small hardware/quality pilot, NOT a1% accuracy noninferiority study.

Conditions: Baseline and unchanged Existing DLR once per question/repeat;
Ideal, Random Skip, Oracle+Actual Overhead at decode MLP target rates
0/10/20/30/40/50%.20 distinct conditions,5 repeats, natural and fixed-stream
modes:20*20*5*2=4000 real requests. Order shuffled within question blocks.
Baseline/Existing are not measured six times and called independent curves.

MLP-only temporal reuse returns most recent fresh residual, never zeroes it.
Attention/KV remain fresh. Prefill is NEVER reused. All28 MLPs available to
the table; original DLR retains its4 candidate limit and .5 threshold. Thus
Existing DLR is NOT a matched50% method; report achieved rate honestly.
Each layer may be reused at most one consecutive step. Per-step skip budgets
are floor(28*r*step)-floor(28*r*(step-1)), giving exact cumulative rounded
decode quotas. Global layer-token/FLOP reductions differ from this target.

Offline Oracle construction: run native full model to the COMMON CAP without
breaking at EOS, collecting its actual per-layer MLP outputs. This explicitly
uses future Baseline tokens/activations, including an after-EOS extension.
Generate three schedules per nonzero rate: lowest baseline-MLP-output drift
relative to each layer's latest fresh baseline anchor, and two random schedules.
Rank actual offline candidate generations lexicographically by content correct,
format correct, exact Baseline token match, smaller length difference, then
fixed candidate index. This USES EVALUATION REFERENCES and privileged future
outputs. Freeze the chosen table BEFORE timed trials. It is a label-aware
best-of-three optimistic comparator, not a deployable gate or mathematically
optimal Oracle. Failure does NOT rule out every possible DLR schedule.
Do not back off the requested rate to preserve quality. No primary tuning.
Record all candidates, costs, tokens, references and construction timings.

Random Skip means randomly selected MLP residual reuse at the SAME per-step
quota/cooldown, not zero-residual layer deletion. Its seed is independent of
Oracle selection. Ideal/Random/Overhead all cache28 output anchors even at0%,
so0% vs Baseline measures cache/lookup overhead. Tables are precompiled outside
timing; compilation/load cost is separately recorded, not claimed nonexistent.

Oracle+Actual Overhead applies exactly the Ideal table but really executes the
previous DLR distance/eval/item logic on the original4 candidate inputs when
eligible, discarding those decisions. Output cache management is shared with
Ideal; four extra input anchors are charged to E. Runtime references/future
tokens are not inputs to this added gate. E-vs-Ideal tests gate+input-cache
overhead, not a hypothetical deduction from measurements.

Measure synchronized wall, process CPU (not GPU time), model-step time, Python
cache/route dispatch time, distance/eval/item time, input/output tokens incl EOS,
native MLP/attention calls, skipped layer-tokens, logical cache payload and MLX
peak/RSS lifetime peak. GPU kernel count/GPU timestamps/power/money unmeasured.
Analytical dense matmul MAC=2 estimates and approximate gate arithmetic are
separate. Do NOT call Python dispatch duration actual GPU memory-copy time.
No activation host export in primary inference. A small separately labelled
intrusive profiling pass synchronizes Attention/MLP/cache phases; its timings
are excluded from primary comparisons and cannot reconstruct normal overlap.

Fixed-stream mode executes every method on exactly the first Baseline token
sequence, including its EOS step. Its forced text is NOT a quality result;
record actual per-step predictions too. Natural output changes/cap failures
remain in data. Oracle construction, model load/warmup/profile costs separate.

Report mean, SD, paired question/repetition bootstrap95%CI and five-repeat
direction. Define meaningful conditional opportunity as >=5% matched-stream
mean reduction, CI upper ratio<1, all5 repeat ratios<1, no observed content or
format loss, and no individual Baseline-correct loss; real MLP bypass >0.
Do not count a cohort with all-zero Baseline quality as a quality success.
A if such opportunity reproduces on all three cohorts at a common rate;
B if only subset/rate qualifies; C if none. Always scope conclusion to these
privileged selected schedules/small tasks; even A is not a deployable method
or general population accuracy proof. Include all failures, rates and Random.
