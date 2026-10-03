import numpy as np

def calculate_semantic_drift(original_intent: str, processed_state: str) -> float:
    """
    Simulates the semantic divergence between the original user intent 
    and the intermediate AI processing state.
    """
    # 임시로 벡터화 및 코사인 유사도 계산 로직 시뮬레이션
    # (실제 구현 시 sentence-transformers 등을 연동하여 고차원 벡터 비교)
    print(f"[*] Analyzing Intent: '{original_intent}'")
    print(f"[*] Comparing with State: '{processed_state}'")
    
    # 예시 유사도 측정값 반환 (0.0 ~ 1.0)
    # 값이 낮을수록 의미 손실(Divergence)이 큼을 의미
    similarity_score = 0.88  # 예시 값
    divergence_score = 1.0 - similarity_score
    
    return divergence_score

if __name__ == "__main__":
    original = "Analyze the financial risks and summarize key points."
    processed = "Summarize financial risks briefly."
    
    drift = calculate_semantic_drift(original, processed)
    print(f"[Result] Semantic Divergence Score: {drift:.4f}")
    
    if drift > 0.15:
        print("[!] Warning: Significant semantic drift detected!")
    else:
        print("[+] Intent well preserved.")
