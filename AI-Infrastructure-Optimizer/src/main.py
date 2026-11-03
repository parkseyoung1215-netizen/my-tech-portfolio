
from cost_analyzer import analyze_cost, simulate_cache_savings


# 가상의 기업 AI 사용 기록
requests = [
    {
        "model": "model_a",
        "prompt": "What is artificial intelligence?",
        "input_tokens": 1000,
        "output_tokens": 500,
    },
    {
        "model": "model_a",
        "prompt": "What is artificial intelligence?",
        "input_tokens": 1000,
        "output_tokens": 500,
    },
    {
        "model": "model_b",
        "prompt": "Explain Python programming.",
        "input_tokens": 2000,
        "output_tokens": 1000,
    },
    {
        "model": "model_a",
        "prompt": "Explain machine learning.",
        "input_tokens": 1500,
        "output_tokens": 700,
    },
]


# 비용 분석 실행
result = analyze_cost(requests)


print("\n===== AI COST ANALYZER =====")

print(f"Total cost: ${result['total_cost']:.8f}")

print(f"Duplicate groups: {result['duplicate_groups']}")

print(f"Duplicate requests: {result['duplicate_count']}")

print("============================")

# 캐싱 최적화 실험
cache_result = simulate_cache_savings(requests)

print("\n===== CACHE OPTIMIZATION =====")

print(f"Original cost: ${cache_result['original_cost']:.8f}")

print(f"Optimized cost: ${cache_result['optimized_cost']:.8f}")

print(f"Money saved: ${cache_result['savings']:.8f}")

print(f"Savings rate: {cache_result['savings_rate']}%")

print(f"Cache hits: {cache_result['cache_hits']}")

print("==============================")
