
from collections import Counter


# 모델별 가격 (USD per 1 million tokens)
MODEL_PRICES = {
    "model_a": {
        "input": 1.00,
        "output": 4.00,
    },
    "model_b": {
        "input": 0.20,
        "output": 0.80,
    },
}


def calculate_cost(model, input_tokens, output_tokens):
    """Calculate the estimated cost of one AI request."""

    if model not in MODEL_PRICES:
        raise ValueError(f"Unknown model: {model}")

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    prices = MODEL_PRICES[model]

    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000

    return input_cost + output_cost


def find_duplicate_requests(requests):
    """Find requests with exactly matching normalized prompts."""

    prompt_counts = Counter(
        " ".join(request["prompt"].lower().split())
        for request in requests
    )

    return {
        prompt: count
        for prompt, count in prompt_counts.items()
        if count > 1
    }


def analyze_cost(requests):
    """Analyze total AI usage cost."""

    total_cost = 0.0

    for request in requests:
        total_cost += calculate_cost(
            request["model"],
            request["input_tokens"],
            request["output_tokens"],
        )

    duplicates = find_duplicate_requests(requests)

    return {
        "total_cost": round(total_cost, 8),
        "duplicate_groups": duplicates,
        "duplicate_count": sum(
            count - 1 for count in duplicates.values()
        ),
    }



def is_cache_safe(request):
    """Check whether a request is safe to reuse."""

    required_fields = [
        "tenant_id",
        "user_id",
        "model_version",
        "system_prompt_version",
        "temperature",
    ]

    if not all(field in request for field in required_fields):
        return False

    if request["cacheable"] is not True:
        return False

    if request["temperature"] != 0:
        return False

    if request.get("tools_enabled", False):
        return False

    if request.get("requires_fresh_data", False):
        return False

    return True


def simulate_cache_savings(requests):
    """Simulate conservative, scope-aware caching."""

    cache = set()
    original_cost = 0.0
    optimized_cost = 0.0
    cache_hits = 0
    rejected_requests = 0

    for request in requests:

        cost = calculate_cost(
            request["model"],
            request["input_tokens"],
            request["output_tokens"],
        )

        original_cost += cost

        if not is_cache_safe(request):
            optimized_cost += cost
            rejected_requests += 1
            continue

        key = (
            request["tenant_id"],
            request["user_id"],
            request["model"],
            request["model_version"],
            request["system_prompt_version"],
            request["temperature"],
            request["prompt"],
        )

        if key in cache:
            cache_hits += 1
        else:
            cache.add(key)
            optimized_cost += cost

    savings = original_cost - optimized_cost

    savings_rate = (
        savings / original_cost * 100
        if original_cost > 0 else 0
    )

    return {
        "original_cost": round(original_cost, 8),
        "optimized_cost": round(optimized_cost, 8),
        "savings": round(savings, 8),
        "savings_rate": round(savings_rate, 2),
        "cache_hits": cache_hits,
        "rejected_requests": rejected_requests,
    }
