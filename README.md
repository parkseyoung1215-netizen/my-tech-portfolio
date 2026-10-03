# my-tech-portfolio

# AI-Driven Future Service Concept: "MindBridge AI"


> Author: Park Seyoung  
> Date: 2026-10-03  


# Semantic Alignment & Intent Preservation

A deep-dive research framework investigating how Large Language Models (LLMs) process human language, measure semantic loss, and track information divergence during token and context transformation.

---

##  Research Hypothesis & Core Problem

- **The Problem:** LLMs perform probabilistic token prediction, which can lead to semantic drift and loss of original human intent between input instruction and internal representation.
- **The Goal:** Moving beyond black-box generation by measuring how much original meaning is preserved or lost across internal model states.

---

##  Research Architecture: Semantic Divergence & Information Flow

To analyze how human intent transforms and suffers from semantic loss during model processing, we propose a structural information-tracking pipeline:

```text
[ User Instruction (Original Intent) ] 
                 │
                 ▼
[ Tokenization & Embedding Space ]
                 │
                 ▼
[ Semantic Representation & Context Extraction ]
                 │
                 ▼
[ Divergence & Loss Measurement (Comparing Input vs. AI State) ]
                 │
                 ▼
[ Empirical Analysis: Tracking Semantic Drift ]

---

##  Experimental Design & Methodology (Planned)

To validate the semantic divergence hypothesis, we plan to implement a lightweight evaluation framework:

1. **Baseline Setup:** Use open-source embedding models to capture high-dimensional vector states of input prompts.
2. **Perturbation Tracking:** Introduce controlled variations (paraphrasing, ambiguity) into instructions to observe how internal representations shift.
3. **Similarity Metrics:** Apply cosine similarity and semantic distance algorithms between input vectors and intermediate hidden states to quantify "Semantic Drift".
4. **Visualization:** Plot divergence scores across processing steps to identify threshold points where intent loss sharply increases.

---

##  Repository Structure

- `README.md`: Research conceptualization, hypothesis, and structural architecture.
- `simulate_drift.py`: Prototype simulation script for measuring semantic divergence and intent drift.

##  Getting Started (Prototype)

To run the simulation script locally:

```bash
python simulate_drift.py

