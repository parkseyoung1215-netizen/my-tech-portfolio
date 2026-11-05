# Dynamic Layer Reuse: frozen pilot protocol

Free local Qwen2.5-1.5B-Instruct BF16, existing MLX environment and weights.
No existing experimental file is edited; every run uses a UUID directory and
exclusive append-only checkpoints. No upload, paid API, or dependency download.

All methods use the same chat template, greedy argmax, EOS, and 32-token budget.
Baseline executes all 28 blocks. Static skips block 24 (the prior Stable1
candidate, not selected on this evaluation). DLR and Conservative DLR always
execute attention and update all KV caches. They may replace MLP output at
layers 8,12,16,24 with the most recent freshly computed MLP output for that
layer in the SAME request. This is approximate temporal activation reuse, NOT
lossless KV caching, a complete-layer reuse method, or an invention claim.

Compare normalized current MLP input to the cached input: relative L2 distance
sqrt(sum((x-old)^2)/(sum(x^2)+1e-8)). A host branch before MLP actually avoids
its three projections. Reuse only at decode (never prefill), distance <= .5
(DLR) or <= .2 (Conservative), maximum one consecutive reused token. Cache
anchor changes only on fresh computation. All comparisons use float32.
These thresholds are hypotheses, not calibrated safety probabilities. Never
use answer labels, future outputs, or computed candidate MLP output to gate.

Eight development prompts, separate from evaluation, run first. They cannot
prove accuracy; report zero reuse or slowdowns rather than tune on evaluation.
Freeze these policies even if the pilot fails. A forced-reuse diagnostic is
an implementation positive control, excluded from performance/accuracy claims.

Main pilot: 32 existing questions (24 MCQ,8 JSON),32 new questions
(16 MCQ,8 JSON,8 short factual responses), three repetitions, four methods,
randomized question/method order, seed 2026110604. Eight items per cohort also
run a fixed-token-stream experiment with the first Baseline token stream.
The latter uses exactly equal input token positions and output counts: no
quality claim from teacher-forcing, but distinguishes internal savings from
changed generated length. Main natural-generation results are reported in full.

Primary metrics: content accuracy and format separately; wall time with MLX
synchronization, process CPU time (NOT GPU time), real attention/MLP counts,
input/output tokens, gate time, MLX peak/active and process lifetime RSS high
water mark. Cached activation payload is reported separately. Never infer
memory improvement from small noisy process high water changes. FLOPs are a
dense-matmul analytical estimate (MAC=2), not actual hardware FLOPs; explicitly
exclude nonlinearities, norms and elementwise gate operations from matmul
estimate, and give gate element count separately. Gate-inclusive wall/CPU
remain primary. No measured GPU FLOPs/power/money.

No training or threshold fitting. New questions are checked against recursively
recorded prior JSON/CSV question stems. Parameterized synthetic templates may
still overlap conceptually; no external generalization claim. New short-answer
aliases are fixed before model execution. Typed JSON equality rejects duplicate
keys/bool-number confusion. Content extraction and format are separate; retain
the entire output. Existing constraints are excluded to avoid repeating the
doctor/physician grading problem. Evaluation references live only in offline
scoring, never in runtime's signature/input.

Use paired question-cluster uncertainty and three-repetition mean/SD. Zero
observed accuracy differences do not prove population noninferiority. A useful
candidate must preserve accuracy, reduce equal-stream dense work AND measured
gate-inclusive time; merely shorter responses or fewer calls do not qualify.
Report all variants, losses, instability and overhead. This small study cannot
establish 1 percentage point noninferiority or a novel technique.

Related primary sources checked before implementation:
- https://arxiv.org/abs/2512.16843 (layer-wise activation caching)
- https://arxiv.org/abs/2506.06295 (diffusion-model activation reuse; different setting)
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0)
- https://docs.ollama.com/api/generate (no public per-block execution hook)
