"""Render measured ALSR pilot results, limitations and primary-source comparisons."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .vvp_report import table,percent,number
from . import alsr_policy as policy
NAMES={'baseline':'Baseline','static':'Static4','random':'Random4','bi_static':'BI-Static4','adaptive_sketch':'Adaptive-Sketch','random_match_sketch':'MatchedRandom-S','adaptive_leverage':'Adaptive-Leverage','random_match_leverage':'MatchedRandom-L'}

def plots(out):
    deps=Path(os.environ.get('VVP_PLOT_DEPS','/private/tmp/vvp_plot_deps_20261105'))
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/alsr_matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_decoded(out/'ALSR_RESULTS.csv');comparisons=read_decoded(out/'ALSR_COMPARISONS.csv');models=list(dict.fromkeys(r['model'] for r in rows));methods=list(NAMES);colors=['#777777','#c56a24','#d59d20','#2e8f66','#6d53ab','#a79bbd','#2279a5','#8db3c9']
    short=['Base','Static4','Random4','BI4','Sketch','Match-S','Leverage','Match-L']
    for kind in ('COMPARISON','EFFICIENCY'):
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        for axs,m in zip(axes,models):
            rr=[next(r for r in rows if r['model']==m and r['method']==s) for s in methods];x=list(range(8))
            if kind=='COMPARISON':
                metrics=[('accuracy',100,'Overall task success (%)'),('natural_mean_wall_seconds',1000,'Natural generation wall per request (ms)'),('fixed_wall_seconds',1000/8,'Fixed16-token workload per request (ms)')]
                for ax,(metric,scale,label) in zip(axs,metrics):
                    vals=[r[metric+'_mean']*scale for r in rr];lo=[r[metric+'_ci95_low']*scale for r in rr];hi=[r[metric+'_ci95_high']*scale for r in rr]
                    ax.bar(x,vals,color=colors,yerr=[[max(0,v-l) for v,l in zip(vals,lo)],[max(0,h-v) for v,h in zip(vals,hi)]],capsize=2);ax.set_ylabel(label)
                axs[0].set_ylim(0,100)
            else:
                axs[0].bar(x,[r['mean_skip_layers_mean'] for r in rr],color=colors);axs[0].set_ylabel('Mean skipped whole blocks per request')
                cr={r['method']:r for r in comparisons if r['model']==m};axs[1].bar(x,[1 if s=='baseline' else cr[s]['fixed_analytic_flops_ratio'] for s in methods],color=colors);axs[1].set_ylabel('Analytical dense FLOPs / baseline (fixed16)');axs[1].axhline(1,color='black',ls='--',lw=.8)
                axs[2].bar(x,[r['mlx_peak_bytes_mean']/2**30 for r in rr],color=colors);axs[2].set_ylabel('Measured MLX peak allocated memory (GiB)')
            for ax in axs:ax.set_xticks(x,short,rotation=35,ha='right');ax.set_title(m);ax.grid(axis='y',alpha=.2)
        fig.suptitle('ALSR small pilot:32 prompts x3 repeats; 8-prompt fixed workload subset\nConditional repeat-mean CI; FLOPs analytical, GPU elapsed/power/money unmeasured')
        fig.tight_layout()
        for ext in ('png','svg'):
            with (out/f'ALSR_{kind}.{ext}').open('xb') as f:fig.savefig(f,format=ext,dpi=180,bbox_inches='tight')
        plt.close(fig)
    with (out/'ALSR_PLOT_METADATA.json').open('x') as f:json.dump({'matplotlib':matplotlib.__version__,'sources':['ALSR_RESULTS.csv','ALSR_COMPARISONS.csv'],'no_synthetic_performance':True},f,indent=2)

def report(prefix,test_result,output_dir=None):
    p=Path(prefix).resolve();out=Path(output_dir).resolve() if output_dir else p.parent/p.name
    def read(s):return read_decoded(f'{p}_{s}.csv')
    def result(s):return read_decoded(out/s)
    manifest=json.loads(Path(f'{p}_manifest.json').read_text());meta=json.loads(Path(f'{p}_metadata.json').read_text());setup=json.loads(Path(manifest['setup_path']).read_text());profiles=json.loads(Path(f'{p}_profiles.json').read_text());audit=json.loads((out/'ALSR_AUDIT.json').read_text())
    rows=result('ALSR_RESULTS.csv');comparisons=result('ALSR_COMPARISONS.csv');verdict=result('ALSR_CANDIDATE_VERDICT.csv');categories=result('ALSR_CATEGORY_RESULTS.csv');matched=result('ALSR_MATCHED_COMPARISONS.csv');promising=[r for r in verdict if r['promising_for_expansion']]
    total_wall=number(meta.get('total_trial_wall_seconds'))+'초' if meta.get('total_trial_wall_seconds') is not None else '미측정(최종 메타데이터 저장 전 오류)'
    any_compute=any(r['fixed_analytic_flops_ratio']<1 for r in comparisons)
    txt=['# Adaptive Layer Skipping Research (ALSR)',
      '\n## 결론\n'+('일부 후보가 사전 파일럿 기준에서 확대 검토 대상으로 나타났다. 다만 작은 표본의 결과이며 품질1%p 일반화 유지나 새로운 기술의 성공을 확증한 것은 아니다.' if promising else '**계층 계산을 직접 생략하는 환경은 구현했지만, 품질 유지와 지연10% 감소를 함께 충족하는 확대 대상은 확인하지 못했다. 유리한 사례만 골라 성공으로 표현하지 않는다.**'),
      f'\n실험 `{p.name}`. 두 모델, 8방식, 신규32문제, 3회 실제 반복으로 생성 {meta["generation_rows"]}건을 측정했다. 동일16토큰 부하의 별도8문제 하위 실험 {meta["fixed_work_rows"]}건, 참조 NLL {meta["quality_rows"]}건(문제/방식당1회)을 분리했다. 전체 MLX 작업 벽시계는 {total_wall}이다. 요청별 생성/고정 부하 CPU·wall 실측 로그는 모두 보존되어 있다. 유료 추론 API는 사용하지 않았다.',
      '\n## 1. 환경·구현 가능성\n'+
      '실제 장비는 **Apple M4 / 통합 메모리16GiB / macOS15.7.4 / arm64**다. MLX가 보고한 권장 작업 메모리와 물리 메모리를 확인하고, 모델을 하나씩 로드했다. 기존 `.venv`는 변경하지 않고 `.alsr-venv`를 새로 만들었다. 공개 HF 체크포인트는 `/private/tmp/alsr_models_20261105`에 별도 다운로드했으며 기존 Ollama 모델을 대체하거나 삭제하지 않았다.\n\n'+
      '로컬 GGUF 메타데이터를 직접 읽어 두 모델이 Qwen2.5 Instruct / Qwen2ForCausalLM 계열임을 확인했다. Ollama의 [공개 generate API](https://docs.ollama.com/api/generate)와 [Options/Runner 소스](https://github.com/ollama/ollama/blob/main/api/types.go)에서 계층 생략 제어를 확인하지 못해 지원한다고 가정하지 않았다. `num_gpu` 같은 배치/장치 옵션을 계산 생략으로 해석하지 않는다. 읽기 전용 MLX GGUF 로드는 `gguf_tensor_to_f16 failed`로 실패해 Q4 파일을 임의 변환하지 않았다.\n\n'+
      '대신 공식 HF Instruct 가중치를 MLX로 로드했다. 실제 dtype은 아래와 같다. 따라서 **Ollama Q4와 HF BF16의 속도·메모리를 직접 비교하지 않는다.** 모든 ALSR 방식은 같은 BF16 체크포인트를 공유한다. 모델은 [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 모두 모델 카드의 Apache-2.0이고, 기존 GGUF 내 라이선스도 같은 표제를 확인했다. [MLX-LM](https://github.com/ml-explore/mlx-lm)은 MIT 라이선스다.\n']
    loads=read(meta.get('loads_suffix','loads'))
    txt.append(table(['모델','HF revision','층','hidden','FFN','attention/KV heads','실행 dtype','로드 활성 MLX GiB'],[[i['repo'],i['revision'],i['config']['num_hidden_layers'],i['config']['hidden_size'],i['config']['intermediate_size'],f'{i["config"]["num_attention_heads"]}/{i["config"]["num_key_value_heads"]}',next(r['dtype'] for r in loads if r['model']==i['repo']),number(next(r['active_bytes'] for r in loads if r['model']==i['repo'])/2**30)] for i in manifest['models']]))
    txt.append('\n패키지: '+', '.join(k+'='+v for k,v in setup['packages'].items())+f'. MLX 메모리 한도 {meta["memory_limit_bytes"]/2**30:.3f}GiB, allocator cache128MiB를 동일하게 적용했다. 모델 가중치는 학습·수정하지 않았다.\n')
    parity=read(meta.get('native_parity_suffix','native_parity'))
    txt.append('\n기준 구현 정합성 관측(복구 실행에서는 보조 재검사):\n'+table(['모델','전체 logits max 차이','마지막 위치 head max 차이','마지막 위치 top1 일치','실제 생략 trap/정상 호출 대조'],[[r['model'],r['native_full_logits_max_abs_difference'],r['last_head_max_abs_difference'],r['last_head_top1_equal'],r['skipped_block_trap_passed'] and r['non_skipped_positive_control_raised']] for r in parity]))
    txt.append('\n모든 시간 비교 방식은 공통으로 마지막 위치만 출력 head에 투영한다. native 전체 logits 경로와의 정합성은 스모크에서 확인했으나 BF16의 head 행렬 크기에 따른 반올림 차이는 위처럼 존재하므로 모든 입력의 bitwise 동등을 주장하지 않는다.\n')
    if meta.get('recovered_after_export_error'):
        txt.append('\n### 최종 저장 오류와 보존 복구\n모든3회 반복이 끝난 뒤 보정 trace의 학습행/보정행 열이 달라 legacy CSV writer가 중단했다. 원래1536개 생성·384개 고정 부하·512개 참조 평가와 모델별 profile/고정 token sequence는 이미 저장돼 있었다. 이 성능 출력은 재생성하거나 덮어쓰지 않았다. 부분 보정 CSV도 보존했다.\n\n원래 모델 로드·학습/보정·warmup/준비 비용과 전체 job 종료 clock은 누락돼 **미측정**으로 남겼다. 별도의 postrun 보조 재검사에서 같은 모델/학습12/보정8로 원래 profile을 정확히 재현했고 native/trap/cache를 재검사했다. 해당 trace와 비용은 `_recheck` 파일에 분리했으며 원래 비용인 것처럼 대체하지 않았다. 위 로드 활성 메모리/dtype 표는 이 보조 재검사의 관측값이다. 새 `alsr_run` 진입점은 합집합 열을 만드는 exporter로 같은 저장 오류를 수정하며 원래 모델 코드와 파일을 변경하지 않는다.\n')
        costs=read(meta.get('calibration_cost_suffix','calibration_cost'))
        txt.append(table(['모델','후속 보조 재검사 prefill 수','재검사 CPU초','재검사 wall초','원래 학습/보정 비용'],[[r['model'],r['prefill_forward_passes'],number(r['cpu_seconds']),number(r['wall_seconds']),'미측정'] for r in costs]))
    txt.append('\n## 2. 선행 기법과 차이·새로움의 범위\n'+table(['기법','선행 내용','이번 구현과의 관계'],[
      ['Early exit','중간 층에서 출력하고 나머지 suffix를 생략','이번에는 마지막 층/출력 head를 남기고 중간 블록을 생략'],
      ['[ShortGPT](https://arxiv.org/abs/2403.03853)','입출력 표현 유사성의 Block Influence로 정적 제거','BI-Static4는 학습12입력·후보8층만 쓰는 부분 기준선, 논문 전체 재현 아님'],
      ['[LayerSkip](https://arxiv.org/abs/2404.16710)','layer dropout/early-exit 학습 및 self-speculative decoding','이번 Qwen은 해당 학습을 하지 않았고 full-model correction도 없음'],
      ['[SkipDecode](https://arxiv.org/abs/2307.02628)','토큰별 early exit와 batching/KV cache 정합성 문제','이번 계획은 입력별 고정이며 토큰별 재진입을 구현하지 않음'],
      ['[CLaSp](https://arxiv.org/abs/2505.24196)','완전한 검증 단계 hidden states를 이용한 self-speculative layer 계획','이번에는 현재 입력의 첫2층과 과거 학습 통계만 사용'],
      ['[SkipGPT](https://proceedings.mlr.press/v267/zhao25ab.html)','token-aware routing, attention/MLP 분리 정책과 학습','이번에는 전체 블록 단위 생략, 모델 fine-tuning 없음'],
      ['[GateSkip](https://github.com/Thiggel/GateSkip)','학습된 residual-stream gate 등 유사한 경량 동적 제어','가벼운 게이트라는 발상 자체의 새로움을 주장하지 않음'],
      ['[2026 Qwen 감사 preprint](https://arxiv.org/abs/2608.28846)','동일 Qwen2.5 규모에서도 skipping/early-exit 비교가 이미 존재','모델 크기나 연구 방향 자체도 새로운 발명이 아님']]))
    txt.append('\n아래 두 규칙은 프로젝트 내 추가 가설이다. 제한된 문헌 조사로 정확히 같은 조합의 선행 부재를 증명할 수 없으며 **전 세계적으로 새로운 기술이라는 근거는 확보하지 못했다.** 성능 효과 역시 실험 전 가정하지 않았다.\n')
    txt.append('\n## 3. 두 판단 규칙과 실제 생략 구현\n'+
      '**Adaptive-Sketch:** 학습12입력의 단일 층 제거 실험에서 최종 next-token 분포 KL과 top1 변경을 측정한다. 첫2층 마지막 prompt 토큰의8차원 고정 random sketch, 길이·RMS·짧은 출력 후보의 margin으로 층별 위험을 예측하는 ridge 회귀를 맞춘다. 온라인에는 추가 모델 호출이 없고 후보 전체 층을 먼저 계산해 본 뒤 버리지 않는다.\n\n'+
      '**Adaptive-Leverage:** 학습 시 얻은 후보층의 평균 residual을 작은 출력 head shortlist에 투영해 둔다. 온라인 첫2층 probe의 상위2후보 margin과 RMS로 정규화해 해당 층의 예상 출력 결정 영향도를 추정한다. 전체 vocab projection 대신 학습에서 고정한 작은 shortlist를 사용한다. 이는 실제 출력 기여의 보장된 상한이나 인과적 증명이 아니다.\n\n'+
      '두 방식 모두 raw hidden 변화량 하나의 임계값과는 다르지만, 출력 민감도/학습된 라우터라는 기존 사고방식의 조합이다. 학습은 모델 가중치가 아닌 경량 회귀·통계만 수행한다. 학습12개와 보정8개는 모두 객관식이고, 평가의 JSON/문장8개는 출력 유형까지 이동하는 OOD 조건이다. 이 한계는 작은 보정 집합으로 다양한 생성 품질을 추정하는 데 불리하다.\n\n'+
      '시드 **88112026**, 후보8개 중 최대4층. 앞2층은 항상 실행해 probe를 얻고, 그 뒤 **프롬프트별 생략 계획을 고정**한다. 남은 prefill과 모든 decode 토큰에 같은 계획을 사용해 skipped KV offset0, 실행된 cache의 offset 정합성을 유지한다. 토큰마다 다시 켜는 cache backfill이나 KV 근사 복사를 했다고 주장하지 않는다.\n\n'+
      '보정 임계값0/.05/.15/.5/2 중8개 보정 입력의 다음 토큰 top1이 모두 유지되고 평균 KL<=.05인 설정에서 평균 생략 수가 최대인 값을 선택했다. 동률은 작은 임계값을 택했다. **첫 토큰 보존이 전체 생성 품질 보존을 뜻하지 않는다.** 최종 평가 정답이나 평가 결과로 설정을 조절하지 않았다.\n\n'+
      '전체 블록 호출 전 분기하므로 생략 층의 attention·MLP 자체를 호출하지 않는다. 실제 모듈 wrapper의 호출 카운터, retained graph의 `mx.eval`, GPU 동기화, cache offset 검사, 호출 시 예외가 나는 trap과 정상 호출 양성 대조를 사용했다. 이 카운터는 GPU의 하드웨어 instruction counter와는 다르다. 계측용 Python wrapper가 모든 방식에 들어간 연구용 구현이므로 최적화된 배포 서버의 절대 성능과 동일하지 않다.\n')
    txt.append('\n## 4. 비교 조건·데이터·계측\n8방식: Baseline, 사전 Static4, Random4, 학습 BI-Static4, 적응형2종, 각 적응형과 동일 생략량의 matched random2종이다. Matched random도 같은 adaptive probe를 실제 계산해 생략 수를 얻고 그 overhead를 부담한다. 단순히 무작위 기준선에 공짜 생략량을 주지 않았다.\n\n'+
      '새 학습12·보정8·평가32를 분리했고 과거 정규화 stem 중복을 검사했다. 평가32개는 산술/논리/코딩/읽기 각6개 객관식과 JSON4개·문장 제약4개다. **계산 도구는 사용하지 않고 모델이 모든 답변을 직접 생성**했다. 전체32개 과제 성공률과 객관식24개 정확도를 분리한다. 문장 제약 성공은 완전한 의미·사실성 평가가 아니다.\n\n'+
      '모든 방식은 같은 chat template, greedy decoding, EOS, 객관식 최대16·나머지 최대48토큰을 사용했다. 반복3회에 문제/방식 순서를 무작위화해 저장했다. 모델 순서는 메모리를 고려해0.5B→1.5B로 고정했으므로 모델 간 절대 속도를 열/순서 영향과 완전히 분리하지 못했다. 자연 생성 서비스 시간에는 tokenization·gate·모든 모델 forward·decode·동기화가 포함된다. 모델 로드/학습/채점/로그 저장은 단계별로 분리했다. CPU는 Python process_time이며 모델의 모든 GPU 시간을 포함하지 않는다.\n\n'+
      '추가8입력(객관식4·JSON2·문장2)에는 미리 기록한 Baseline token sequence를 EOS로 padding해 **동일16토큰** 부하를 모든 방식에 입력했다. 이는 자연 생성 품질 시험이 아닌 같은 일의 계산 비용 비교다. 계획은 prompt만으로 정하고 미래 고정 토큰을 보지 않으며 자연 생성과 같은 mask인지 검사했다. 짧은 출력/EOS 때문에 빨라진 것을 순수 layer 효율로 해석하지 않는다.\n\n'+
      '참조 NLL은 사람 작성 reference의 teacher forcing으로 문제/방식당1회만 계산했다. 자연 생성의 prompt-only mask를 그대로 고정하고 이 단계에서는 gate 호출이0인지 검사했다. 참조 토큰이 계획 선택에 들어가지 않는다. NLL 채점 비용은 서비스 지연에서 제외하고 별도로 보존했다. 동등한 다른 정답 표현의 NLL도 달라질 수 있어 단독 의미 품질 판정이 아니다.\n\n'+
      'GPU elapsed counter를 얻지 못했다. MLX 동기화 wall은 GPU+host의 실제 경과 시간이며 GPU 시간으로 바꾸어 쓰지 않았다. `xcrun --find xctrace`는 이 환경에서 도구를 찾지 못했다. MLX peak allocated memory는 요청마다 reset해 관측했으며 프로세스 RSS 전후와 lifetime 최대 RSS는 별도 저장했다. lifetime RSS를 방식별 peak처럼 비교하지 않는다. **전력·금전 비용은 미측정**이다.\n')
    txt.append('\n## 5. 주요 결과: 3회 평균\n전체 성공률에는 JSON/문장 제약도 포함하므로 객관식 정확도와 동일한 지표가 아니다. 시간은32요청 합계, 고정 부하는8요청 합계다.\n'+table(['모델','방식','평균 생략층','전체 성공률32','객관식 정확도24','형식 준수','자연 생성초','고정16토큰초','CPU초','MLX peak GiB','참조 NLL¹'],[[r['model'].replace('Qwen/Qwen2.5-',''),NAMES[r['method']],number(r['mean_skip_layers_mean'],2),percent(r['accuracy_mean']),percent(r['mcq_accuracy_mean']),percent(r['format_compliance_mean']),number(r['natural_wall_seconds_mean']),number(r['fixed_wall_seconds_mean']),number(r['natural_cpu_seconds_mean']),number(r['mlx_peak_bytes_mean']/2**30),number(r['reference_nll_mean'])] for r in rows]))
    txt.append('\n¹ NLL은 문제별1회, 총32개의 평균이며3회 반복 측정이 아니다.\n')
    txt.append('\n## 6. 계산량·지연 비교와 불확실성\nFLOPs는 dense matmul의 MAC×2 분석값이다. 층당 attention projection `T*(4d²+4d*d_kv)`, MLP `6T*d*I`, attention QK/AV `4T*S*d`, 출력 head `2*d*V*출력위치수`를 더했다. 작은 gate matmul도 추정에 포함한다. norm·softmax·RoPE·비선형 함수와 하드웨어 fusion/causal 효율은 빠져 있어 실제 GPU FLOP counter가 아니다. 레이어 수 비율을 전체 FLOPs 비율로 단순 대체하지 않았다.\n')
    txt.append(table(['모델','방식','성공률 차이 pp','question95 CI pp','자연 wall비','자연95 CI','고정 wall비','고정95 CI','고정 분석 FLOPs비','NLL 차이'],[[r['model'].replace('Qwen/Qwen2.5-',''),NAMES[r['method']],number(100*r['accuracy_delta']),f'[{100*r["paired_bootstrap_low"]:.2f},{100*r["paired_bootstrap_high"]:.2f}]',number(r['natural_wall_ratio']),f'[{r["natural_ratio_low"]:.3f},{r["natural_ratio_high"]:.3f}]',number(r['fixed_wall_ratio']),f'[{r["fixed_ratio_low"]:.3f},{r["fixed_ratio_high"]:.3f}]',number(r['fixed_analytic_flops_ratio']),number(r['reference_nll_delta'])] for r in comparisons]))
    txt.append('\n고정 token 부하에서는 실제 실행 블록이 줄었는지를 확인하고, 자연 생성은 토큰 수가 달라지는 효과를 함께 본다. 모든 방식은 같은 수의 생성 요청을 처리했으므로 모델 호출 수 감소를 성과로 삼지 않았다. 모델 가중치는 계속 상주하며 layer skipping이 parameter memory 감소를 뜻하지 않는다.\n\n'+
      '평균·표본 SD·반복 CI는 CSV에 있다. 정확도 차이는32문제 단위의 반복 묶음 bootstrap4,000회와 보수적 전이 하한, 속도비는 문제와 반복을 대응시킨 bootstrap4,000회를 사용했다.3회 반복을96개 새 문제처럼 세지 않는다.32문제에서 한 문제는3.125%p, 객관식24개에서는4.167%p로1%p 목표를 정밀하게 확인하기 어렵다. 고정 부하도8입력의 작은 하위 실험이다.\n')
    txt.append('\n동일 생략량/동일 probe overhead의 무작위 대조군과 비교:\n'+table(['모델','적응형','대조군','성공률 차이 pp','95 CI pp','자연 wall비','고정 wall비','NLL 차이'],[[r['model'].replace('Qwen/Qwen2.5-',''),NAMES[r['method']],NAMES[r['control']],number(100*r['accuracy_delta']),f'[{100*r["paired_bootstrap_low"]:.2f},{100*r["paired_bootstrap_high"]:.2f}]',number(r['natural_ratio']),number(r['fixed_ratio']),number(r['reference_nll_delta'])] for r in matched]))
    txt.append('\n탐색적 관측: 1.5B의 Adaptive-Sketch는 같은 생략량의 무작위 대조군보다 전체 과제 성공률이20.83%p 높았고 문제 bootstrap95 CI는[10.42,31.25]%p였다. 그러나 다중 비교 보정이 없고 보수적 전이 하한은 통과하지 못했으며0.5B에서는 같은 이점이 재현되지 않았다. 이 신호를 새로운 기술이나 확증된 일반화 우위로 부르지 않는다.\n')
    txt.append('\n## 7. 유형별 품질·실패 사례\n'+table(['모델','방식','유형','고유 문제 수','성공률','형식 준수','평균 생략층'],[[r['model'].replace('Qwen/Qwen2.5-',''),NAMES[r['method']],r['category'],r['unique_questions'],percent(r['accuracy']),percent(r['format']),number(r['mean_skip_layers'],2)] for r in categories]))
    failures=result('ALSR_FAILURE_CASES.csv');examples=[]
    for m in meta['models']:
        for method in ('static','bi_static','adaptive_sketch','adaptive_leverage'):
            candidates=[r for r in failures if r['model']==m and r['method']==method and r['baseline_grade']['correct'] and not r['grade']['correct']]
            for r in candidates[:2]:examples.append([m.replace('Qwen/Qwen2.5-',''),NAMES[method],r['task_id'][:12],r['question'].replace('\n',' ').replace('|','/'),r['reference'],r['baseline_answer'].replace('\n',' '),r['answer'].replace('\n',' '),r['skip_layers'],r['grade']['error_type']])
    txt.append('\n첫 반복에서 기준 답변이 맞았으나 수정 모델이 실패한 사후 예시(학습/임계값 조정에 쓰지 않음):\n'+table(['모델','방식','ID','질문','참조','기준 출력','생략 출력','생략층','오류'],examples))
    txt.append('\nMCQ는 정답을 보지 않는 고정 추출 규칙으로 letter/해당 보기 본문/명시적인 답 표현을 인식하고 형식 준수는 별도 측정했다. JSON은 객체 일치와 중복 키를 검사하며 문장4개는 이름·방과 한 문장 제약만 검사한다. 보수적인 parser가 더 긴 정답 설명을 미해결로 분류할 수 있고, 제약 통과가 사실성·문체의 완전한 보장은 아니다. 참조 NLL과 원문 실패 사례를 함께 제공한다.\n\n'+
      '실패 가능 요인은 학습/보정 표본이 작음, 단일 토큰 logit 효과를 전체 생성 품질의 대리목표로 사용함, 학습에 없던 열린 출력 유형, 여러 층을 동시에 생략할 때의 상호작용, 짧은 요청에서의 GPU-host 동기화 overhead다. 이는 설계와 관측에 근거한 진단이며 모델 내부 원인을 확정하거나 실패를 일반 법칙으로 주장하지 않는다.\n')
    txt.append('\n## 8. 파일럿 판정과 확대 여부\n사전 목표는 성공률/형식 손실1%p 이내, NLL 증가 .05nats 이내, 자연 생성과 고정 부하 각각10% 이상 지연 감소, 실제 생략·분석 계산량 감소다. 아래는 점추정 품질 조건과 속도 CI를 함께 확인한 **확대 검토 여부**이며 신기술 성공 판정이 아니다.\n'+table(['모델','방식','확대 검토 후보','사전 파일럿 조건','일반화 품질1pp 비열등 확정'],[[r['model'].replace('Qwen/Qwen2.5-',''),NAMES[r['method']],r['promising_for_expansion'],json.dumps(r['criteria']),r['population_quality_noninferiority_confirmed']] for r in verdict]))
    txt.append('\n'+('유망 후보는 '+', '.join(r['model']+' '+r['method'] for r in promising)+'다. 이는 독립적인 더 큰 자료로 검증할 후보이며 작은 파일럿만으로 확증 성공이라 하지 않는다.' if promising else '동시 조건을 충족하는 후보가 없어 대규모 실험으로 확대하지 않았다. 설정을 평가 결과에 맞춰 바꾸거나 유리한 데이터만 선택하지 않았다.')+'\n')
    txt.append('\n## 9. 마지막 세 질문을 분리한 결론\n'+
      '1. **실제 계산량 감소에 성공했는가?** '+('일부 방식에서는 그렇다. 실제 블록·attention·MLP 미호출과 KV 정합성 검사를 통과했고 동일 token 부하의 분석 matmul FLOPs도 감소했다.' if any_compute else '이번 조건에서는 유효한 계층/분석 계산량 감소를 확인하지 못했다.')+' 다만 GPU 하드웨어 FLOPs·전력·금전 비용 감소를 직접 측정한 것은 아니다.\n'+
      '2. **품질을 유지하면서 속도도 개선했는가?** '+('일부 파일럿 후보가 있으나 품질1%p의 일반화 유지와 안정적 배포 성능은 확증하지 못했다.' if promising else '이번 파일럿에서 두 조건을 함께 충족하는 확대 후보는 확인하지 못했다.')+' 단순히 출력이 짧아졌거나 생략 층 수가 늘어난 것만으로 성공을 선언하지 않는다.\n'+
      '3. **기존 기술과 구별되는 새로운 발견이 있었는가?** 독립적인 새 기술·전 세계적 새로움·기존 대표 기법 대비 우위를 주장할 근거는 없다. 두 판단 규칙의 로컬 적용과 실제 호출 계측을 추가한 파일럿이다.\n')
    txt.append('\n## 10. 보존·테스트·재현\n'+f'실행 전 파일 **{audit["preserved_files_unchanged"]}개 해시 유지**. 과거 stem 중복0, 학습/보정/평가 분리, profile hash 및 보정 선택 재구성, 모든 블록/attention/MLP 호출·KV offset·matched random 생략량·NLL gate0 검사 통과. 생략0인데 기준 token 응답과 달랐던 adaptive 기록은 {audit["zero_skip_adaptive_response_disagreements"]}개다(수치적 평가 순서 가능성도 있어 생략 효과로 단정하지 않음). 기존 Ollama GGUF의 SHA256도 종료 시 확인했다.\n\n'+
      f'자동 테스트: **{test_result}**. 원래 환경에는 MLX를 설치하지 않았고 기존 테스트는 고정 역사 입력의 임시 스냅샷으로 실행해 실제 데이터를 지우지 않았다. 실측표는 테스트 모의 응답으로 만들지 않았다. GPU native parity/trap/cache smoke와 실제 반복 로그는 별도 증거다.\n\n'+
      '재현 명령(프로젝트 루트, Metal 접근 필요):\n\n```bash\n.alsr-venv/bin/python -B -u -m src.alsr_prepare\n'+
      f'.alsr-venv/bin/python -B -u -m src.alsr_run {manifest["setup_path"]}\n'+
      f'.venv/bin/python -B -m src.alsr_analysis data/{p.name} data/{p.name}_reanalysis\n'+
      '.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json\n'+
      f'.venv/bin/python -B -m src.alsr_report data/{p.name} "<실제 테스트 결과>" data/{p.name}_reanalysis\n```\n\n'+
      '새 동일 데이터 생성은 이미 저장된 stem이 있으면 중단한다. 원래 split·모델 revision·프롬프트·threshold·seed·schedule은 manifest/profile에 보존했다. 분석 출력은 새 폴더를 사용하며 기존 결과를 덮어쓰지 않는다. 임시 HF 캐시가 없어졌으면 setup의 고정 revision으로 다시 받아야 한다.\n\n'+
      '[결과](ALSR_RESULTS.csv) · [반복 지표](ALSR_REPEAT_METRICS.csv) · [기준 비교](ALSR_COMPARISONS.csv) · [동일 생략량 비교](ALSR_MATCHED_COMPARISONS.csv) · [실패 사례](ALSR_FAILURE_CASES.csv) · [감사](ALSR_AUDIT.json) · [비교 그림](ALSR_COMPARISON.png) · [계산량/메모리 그림](ALSR_EFFICIENCY.png)\n\n'+
      f'원시 실제 실행: `../{p.name}_generation.csv`, `../{p.name}_fixed_work.csv`, `../{p.name}_quality.csv`. 보조 trace: `../{p.name}_{meta.get("calibration_trace_suffix","calibration")}.csv`, `../{p.name}_{meta.get("native_parity_suffix","native_parity")}.csv`. 원래 부분 calibration CSV도 그대로 보존돼 있다.\n\n'+
      '한계: 영어 합성32문제, 보정8개, 일부 출력 유형 OOD, parser 의존, 정확한 전력/GPU 시간 미측정, Python 계측기 overhead, 비전용 노트북 자원/열/캐시, greedy/BF16/Qwen2.5 두 모델만의 결과다. 다른 모델·장문·실서비스로 일반화하거나 논문의 성능을 재현했다고 주장하지 않는다.\n')
    with (out/'ALSR_RESEARCH_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(txt)+'\n')
    plots(out);print(out/'ALSR_RESEARCH_REPORT.md')

if __name__=='__main__':report(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
