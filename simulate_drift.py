import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load a lightweight, state-of-the-art sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def measure_semantic_divergence(original_intent: str, processed_state: str) -> float:
    """
    Computes real semantic divergence between the user's original instruction 
    and the AI's intermediate or final processing state using high-dimensional embeddings.
    """
    print(f"[*] Analyzing Intent: '{original_intent}'")
    print(f"[*] Comparing with State: '{processed_state}'")
    
    # 1. Encode sentences into high-dimensional dense vectors
    intent_embedding = model.encode(original_intent, convert_to_tensor=True)
    state_embedding = model.encode(processed_state, convert_to_tensor=True)
    
    # 2. Calculate Cosine Similarity (ranging from -1.0 to 1.0, typically 0.0 to 1.0 for semantic text)
    similarity = util.cos_sim(intent_embedding, state_embedding).item()
    
    # 3. Derive Divergence Score (1.0 - Similarity)
    # Higher divergence means greater loss of original intent.
    divergence_score = 1.0 - similarity
    
    return float(divergence_score)

if __name__ == "__main__":
    # Test cases with varying degrees of semantic shift
    original_intent = "Analyze the financial risks and summarize key points for the executive board."
    
    # Case A: Well-preserved intent
    processed_state_good = "Summarize financial risks for executives."
    
    # Case B: Significant semantic drift / loss of intent
    processed_state_drifted = "Make some coffee and check the weather."
    
    print("\n--- Test Case 1: Minor Variation ---")
    div_good = measure_semantic_divergence(original_intent, processed_state_good)
    print(f"[Result] Divergence Score: {div_good:.4f}")
    
    print("\n--- Test Case 2: Severe Drift ---")
    div_bad = measure_semantic_divergence(original_intent, processed_state_drifted)
    print(f"[Result] Divergence Score: {div_bad:.4f}")
    
    if div_bad > 0.3:
        print("\n[!] Alert: Severe semantic divergence detected! Intent is heavily corrupted.")
