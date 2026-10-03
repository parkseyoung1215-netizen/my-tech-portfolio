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
- `evaluate_drift.py`: Scores a labeled test set and reports how the divergence score changes with drift level.
- `testset.csv`: Instruction pairs labeled with six drift levels (0 = identical, 5 = unrelated).
- `results/`: Scores and the divergence-by-level plot from the latest run.

##  Getting Started (Prototype)

To run the simulation script locally:

```bash
python simulate_drift.py
```

## Experiment 1: Does an embedding-based divergence score catch intent drift?

`testset.csv` pairs each original instruction with variants at six drift levels
(0 = identical, 1 = reworded, 2 = part dropped, 3 = condition flipped,
4 = related but different task, 5 = unrelated). `evaluate_drift.py` scores every
pair with `1 - cosine similarity` of sentence embeddings (all-MiniLM-L6-v2),
the same score used in `simulate_drift.py`.

Run: `python3 evaluate_drift.py`

![Divergence score vs. drift level](results/drift_by_level.png)

| Level | Meaning | Mean score |
|---|---|---|
| 0 | identical | 0.00 |
| 1 | reworded | 0.13 |
| 2 | part dropped | 0.20 |
| 3 | condition flipped | 0.21 |
| 4 | different task | 0.54 |
| 5 | unrelated | 0.97 |

Spearman correlation between level and score: 0.884 (30 pairs, 5 per level).

**Finding:** the score separates unrelated tasks well, but it barely reacts to
flipped conditions. For example, "without changing the database schema" turned
into "by redesigning the database schema" scored 0.017. At a 0.3 threshold, 4 of
5 flipped-condition cases are missed. Embeddings measure topic similarity, not
whether a constraint was followed.

**Limitations:** only 30 pairs so far, so these numbers show direction, not a
final measurement. Next: a larger test set and a contradiction-aware scorer.


## Experiment 2: Does a contradiction-aware scorer catch what embeddings miss?

`evaluate_nli.py` adds a second scorer on the same 30 pairs: a natural-language-inference
model (`cross-encoder/nli-MiniLM2-L6-H768`) reads the new state and the original
instruction and returns P(entailment). The NLI score is `1 - P(entailment)`.

Run: `python3 evaluate_nli.py` (after `python3 evaluate_drift.py`)

![Embedding vs NLI](results/compare_embedding_vs_nli.png)

Share of pairs flagged at threshold 0.3:

| Level | Meaning | Embedding | NLI |
|---|---|---|---|
| 0 | identical | 0% | 0% |
| 1 | reworded | 0% | 0% |
| 2 | part dropped | 0% | 100% |
| 3 | condition flipped | 20% | 100% |
| 4 | different task | 80% | 100% |
| 5 | unrelated | 100% | 100% |

Level 3 (condition flipped), score per pair:

| Variant | Embedding | NLI |
|---|---|---|
| "ignore the risks, summarize only the positive points" | 0.211 | 0.993 |
| accept the meeting instead of declining | 0.059 | 0.999 |
| "redesign the database schema" instead of leaving it unchanged | 0.017 | 0.999 |
| flight direction reversed, price limit removed | 0.192 | 0.996 |
| short summary changed to a detailed expert review | 0.573 | 0.999 |

**Findings:**
- NLI separates preserved instructions (levels 0-1) from violated ones (levels 2-5)
  with no overlap at any threshold from 0.3 to 0.9, and it catches all five flipped-condition
  cases that the embedding score mostly missed.
- NLI does not rank severity: levels 2 to 5 all score close to 1.0. Embeddings track
  severity but miss flipped constraints. The two scorers are complementary.
- One flipped case ("no price limit") was judged neutral rather than contradictory
  (P(contradiction) = 0.005), so `1 - P(entailment)` does not tell neutral from contradiction.

**Limitations:** 30 hand-written pairs (5 per level), and the level 1 paraphrases are
easy ones, so the 0% false-alarm rate is not yet established. Next: a larger test set
with harder paraphrases, multi-constraint instructions and longer text, and a combined
score that uses both scorers.

