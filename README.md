# my-tech-portfolio

# AI-Driven Future Service Concept: "MindBridge AI"


> Author: Park Seyoung  
> Date: 2026-10-03  


## Core Research Direction: Semantic Alignment & Intent Preservation

> **The Goal:** Measuring and preserving human intent through AI processing pipelines to ensure transparency, self-limitation awareness, and verifiable safety.

- **The Problem:** LLMs often suffer from semantic loss or divergence between user instructions and predicted actions because token-based probabilistic prediction doesn't 100% guarantee intent preservation.
- **The Approach:** Designing a structured validation flow (`User Instruction` $\rightarrow$ `Semantic Understanding` $\rightarrow$ `Consistency Check` $\rightarrow$ `Action`) to catch discrepancies before execution.

---

###  System Architecture: Intent Verification Pipeline

To bridge the gap between user intention and model execution, we propose a modular validation pipeline prior to tool invocation:

```text
[ User Instruction ] 
        │
        ▼
[ Semantic Understanding & Intent Extraction ]
        │
        ▼
[ Consistency Check (Semantic Divergence Measurement) ]
        │
        ├── 🟢 High Match (> 95%) ──> [ Execute Action ]
        ├── 🟡 Moderate Match ────> [ Ask Human Confirmation ]
        └── 🔴 Low Match / Risk ──> [ Block & Report Divergence ]

---

### Research Architecture: Semantic Divergence & Information Flow

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


