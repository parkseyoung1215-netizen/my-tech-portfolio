# ALSR small pilot: prespecified protocol

Local Apple M4 16GiB. Existing Ollama Qwen2.5 Instruct Q4_K_M remains untouched.
Ollama public generation API/types show no layer skip control. Isolated MLX environment
loads pinned official HF weights (actual dtype recorded); no Ollama/HF speed comparison.
Attempted read-only MLX GGUF load failed `gguf_tensor_to_f16`; do not assume conversion.

Train12 new prompts, calibration8, pilot32 (24 MCQ,4 JSON,4 constrained sentence tasks).
No old normalized question stems. Last8 evaluate structure/constraints, not full human
semantic quality. Human references also score teacher-forced NLL; no LLM judge.
No safe-calculator route: ALSR really generates all answers inside the model.

Eight methods: native-equivalent manual baseline, fixed4 middle skips, random4,
training BI/cosine static4 (ShortGPT-inspired partial baseline, not full reproduction),
adaptive early-state output-risk sketch regressor, matched-random same count,
adaptive residual-to-output-shortlist leverage, matched-random same count.
Candidate8 middle blocks, first2 and last blocks protected. Do not claim global novelty:
early exit, LayerSkip, ShortGPT, SkipDecode, CLaSp, SkipGPT, residual gating are prior art.
LayerSkip trained early-exit checkpoints/self-speculation not reproduced with Qwen.

Two hypotheses go beyond raw hidden-state-difference thresholds:
(1) frozen low-dimensional early-state sketch predicts per-block final logit disturbance
from train-only single-block ablations (ridge); (2) train mean residuals projected onto
a small output-head shortlist estimate decision-margin leverage, conditioned on an
early-state probe pair. Shortlist from TRAIN model logits only, never eval labels.
These are project-local combinations with unconfirmed novelty and unknown efficacy.

Adaptive plan chosen once immediately after first2 prefill blocks, then frozen for
prefill remainder and ALL decode tokens. No re-entering a skipped cache; skipped KV
offset stays0, all executed layers stay aligned. Tokenwise dynamic skipping not claimed.
No compute-then-discard: skipped block callable/attention/MLP is never invoked.
Counter logs plus forced exception-on-call smoke test; materialize all retained logits.

Calibrate risk threshold in {0,.05,.15,.5,2}; max skip4. Choose largest mean actual
skip count with calibration top1 agreement100% and mean KL<=.05 vs full, tie conservative.
Train labels are baseline/ablation logit effects, not evaluation gold. Model weights frozen.

Three actual repeats, common prompts/chat template/greedy/max16 MCQ or48 open tokens,
EOS respected; randomized method order. Include tokenize/decode/synchronization/gates
in generation wall. Same8 prespecified inputs also fixed16-token teacher-forced workload,
including gate overhead: prevent early-EOS/shorter-answer fake speedup. Human-reference
NLL once/method/question with PROMPT-ONLY mask frozen, not a gate using reference tokens;
NLL/evaluation overhead separate from service latency. Repeated fixed prompts not new data.

Initial targets: <=1pp correctness/format loss, mean reference NLL delta<=.05 nats,
>=10% synchronized generation wall reduction,
actual skipped graph blocks and lower analytical dense matmul FLOPs, repeat stability.
32-task pilot resolution3.125pp cannot establish1pp population noninferiority. Treat
uncertainty honestly; no expansion for quality/speed failures or zero-skip degeneracy.
Only promising fixed-profile candidates could justify a separate larger validation.

Measure Python CPU, synchronized wall, MLX resettable peak allocated memory, current RSS
endpoints and process lifetime RSS peak separately. GPU elapsed unavailable unless an
actual profiling counter is obtained: synchronization wall is NOT GPU elapsed. FLOPs
are analytical MAC*2, exclude norms/nonlinear/softmax/RoPE; no FLOP-counter claim.
All model weights remain resident: skipping does not imply lower parameter memory.
Power/money unmeasured. No paid APIs. Old code/results preserved; new UUID artifacts.
