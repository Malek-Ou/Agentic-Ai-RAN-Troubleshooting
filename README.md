# Agentic-RAG-RAN-AIOps

> An Agentic AI platform for anomaly detection and root cause analysis in 4G LTE Radio Access Networks (RAN), combining hybrid ML-based KPI monitoring, multi-agent reasoning, Knowledge Graphs, RAG, and Case-Based Reasoning.
<p align="center">
  <img src="demo/1 Light Mode.png" width="900">
</p>
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Diagnostic Pipeline](#diagnostic-pipeline)
- [Anomaly Detection](#anomaly-detection)
- [Knowledge Graph](#knowledge-graph)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Research Areas](#research-areas)
- [License](#license)

---

## Overview

Modern 4G LTE networks generate large volumes of heterogeneous KPI, alarm, and technical data. Traditional troubleshooting approaches rely heavily on manual analysis and predefined rules, making Root Cause Analysis (RCA) time-consuming and operator-dependent.

**RAN AIOps Copilot** combines machine learning, structured network knowledge, technical documentation, and agentic reasoning to support automated network diagnosis.

The system provides two investigation entry points:

- **ML-Assisted Investigation**: detected KPI anomalies are directly forwarded to the diagnostic pipeline as structured inputs.
- **Interactive Investigation**: NOC engineers submit natural-language troubleshooting queries.

Both entry points converge into a shared diagnostic workflow.

---

## Architecture

```text
                         ┌──────────────────────────┐
                         │       Input Layer        │
                         │                          │
                         │  Natural Language Query  │
                         │           │              │
                         │  ML KPI Anomaly          │
                         └───────────┬──────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        Intent Classification    │
                    │                                 │
                    │ Diagnostic / Definition /       │
                    │ Recommendation / Chitchat       │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
             ┌─────────────────────────────────────────────┐
             │       Agentic Diagnostic Pipeline           │
             │                                             │
             │  1. Symptom Identification                  │
             │              │                              │
             │  2. CBR Memory Retrieval                    │
             │              │                              │
             │  3. ReAct Retrieval                         │
             │       ┌──────┼──────────┐                   │
             │       │      │          │                   │
             │      KG     RAG     Web Search              │
             │       │      │          │                   │
             │       └──────┼──────────┘                   │
             │              │                              │
             │  4. Root Cause Analysis                     │
             │              │                              │
             │  5. Case Persistence                        │
             └──────────────┬──────────────────────────────┘
                            │
                            ▼
             ┌─────────────────────────────────────────────┐
             │          Explainable Diagnosis              │
             │                                             │
             │ Root Cause │ Evidence │ Confidence │        │
             │ Recommendations │ Causal Chain              │
             └────────────────┬────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │     HITL Validation  │
                    │                      │
                    │ Validate / Correct / │
                    │ Reject / Skip        │
                    └──────────┬───────────┘
                               │
                         Validated Cases
                               │
                               ▼
                         CBR Memory
```

---

## Diagnostic Pipeline

The diagnostic workflow is implemented as a LangGraph state graph with shared state. Two investigation modes are supported: user query and `ml_anomaly` (pre-structured metadata injected directly from the ML pipeline).

| Node | Component | Main Responsibility |
|------|-----------|---------------------|
| 1 | **Symptom Identification Agent** | Builds a structured symptom context from a natural-language query or ML anomaly metadata |
| 2 | **CBR Memory Manager** | Retrieves similar validated or corrected historical cases |
| 3 | **Retrieval Agent** | Performs ReAct-based evidence retrieval using Knowledge Graph, RAG, and web search |
| 4 | **Root Cause Analysis Agent** | Synthesizes available evidence into an explainable RCA hypothesis |
| 5 | **Memory Save** | Persists the generated case with `pending` validation status |

---

## Anomaly Detection

Each KPI series is dispatched to the appropriate detection method based on its temporal characteristics.

| Condition | Method | Rationale |
|-----------|--------|-----------|
| Low residual variability | **SARIMA** | Stable temporal structure and daily seasonality |
| Moderate residual variability | **Prophet** | Robust to outliers and trend changes |
| High residual variability | **STL + IQR** | Non-parametric detection for irregular series |
| Event-based KPIs | **Modified Z-Score / MAD** | Suitable for ratio-based and non-seasonal behavior |

---

## Knowledge Graph

The system implements a three-layer causal graph over structured operational data. Unlike vector retrieval which extracts semantic evidence from technical documents the Knowledge Graph provides explicit causal reasoning over structured operational data. While RAG retrieves documents describing possible root causes, the KG encodes causal relationships between KPI anomalies, alarms, and root cause categories. This complementarity motivates the use of two separate knowledge stores.


#### Graph Layers

| Layer | Description |
|-------|-------------|
| **OSS Knowledge** | Represents OSS alarms and their relationships with root causes. |
| **Network Knowledge** | Represents eNodeBs, cells, KPIs, KPI categories, anomalies, and time windows. |
| **Causal Inference** | Associates KPI patterns with possible root causes using posterior probabilities derived from documentation and field statistics. |

#### Main Node Types

| Node | Role |
|------|------|
| `:ENodeB` | Base station |
| `:Cell` | Network observation unit |
| `:KPI` | Network performance metric |
| `:KPICategory` | KPI classification |
| `:Anomaly` | ML-detected KPI anomaly |
| `:TimeWindow` | Temporal information |
| `:RootCause` | Potential network root cause |
| `:OSSAlarm` | OSS alarm |

#### Main Relationships

```text
(ENodeB)-[:HAS_CELL]->(Cell)
(Cell)-[:HAS_ANOMALY]->(Anomaly)
(Anomaly)-[:INVOLVES_KPI]->(KPI)
(KPI)-[:BELONGS_TO]->(KPICategory)
(Anomaly)-[:OCCURS_AT]->(TimeWindow)
(OSSAlarm)-[:CAUSES]->(RootCause)
(KPI)-[:HAS_ROOT_CAUSE]->(RootCause)
```

---

---

## Key Features

- **Hybrid KPI Anomaly Detection** using SARIMA, Prophet, STL + IQR, and Modified Z-Score / MAD
- **Intent Classification** for routing natural-language requests
- **Multi-Agent Diagnostic Pipeline** orchestrated with LangGraph
- **ReAct Retrieval** for adaptive knowledge-source selection with deterministic KG pre-query
- **Three-Layer Knowledge Graph** with Bayesian posterior scoring over 283,564 anomaly observations
- **Hybrid RAG** with dense + BM25 retrieval and RRF fusion
- **Cross-Encoder Reranking** for final context precision
- **Huawei-Aware Chunking** preserving alarm entries and MML commands
- **Case-Based Reasoning** using validated historical investigations
- **Web Search Fallback** via Tavily for novel fault patterns
- **Explainable Root Cause Analysis** with evidence traceability and confidence breakdown, streamed via Server-Sent Events
- **Human-In-The-Loop Validation** with Validate / Correct / Reject / Skip actions
- **RAGAS Evaluation** on a 12-case golden set
- **Pipeline Observability** using Arize Phoenix

---

## Tech Stack

### AI / Machine Learning

| Component | Technology |
|-----------|------------|
| Anomaly Detection | SARIMA, Prophet, STL + IQR, Modified Z-Score / MAD |
| LLM Reasoning (cloud) | llama-3.3-70b-instruct, llama-3.1-8b-instruct via OpenRouter |
| LLM Reasoning (local fallback) | qwen2.5:7b, qwen2.5:3b-instruct via Ollama |
| Agentic Loops | ReAct (Reasoning + Acting) |
| Language | Python 3.10+ |

### Agentic AI

| Component | Technology |
|-----------|------------|
| Workflow Orchestration | LangGraph |
| LLM & Retrieval Components | LangChain |

### Knowledge & Retrieval

| Component | Technology |
|-----------|------------|
| Knowledge Graph | Neo4j |
| Vector Database | Qdrant (INT8 quantization, on-disk payload) |
| Embedding Model | nomic-embed-text v1.5 (768 dim, 8,192 tok context) |
| Sparse Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Web Search Fallback | Tavily |

### Evaluation

| Component | Technology |
|-----------|------------|
| RAG Evaluation Framework | RAGAS |
| LLM-as-a-Judge | meta-llama/llama-3.3-70b-instruct via OpenRouter |

### Observability

| Component | Technology |
|-----------|------------|
| LLM & Pipeline Tracing | Arize Phoenix |

### Development

| Component | Technology |
|-----------|------------|
| Version Control | Git / GitHub |
| Containerization | Docker |
| Project Management | Jira |

---

## Research Areas

- Generative AI
- Agentic AI & Multi-Agent Systems
- Retrieval-Augmented Generation (RAG)
- Knowledge Graphs
- Case-Based Reasoning (CBR)
- Explainable AI (XAI)
- Network AIOps
- Time-Series Anomaly Detection

---

## License

This project was developed as part of a Final Year Engineering Project (PFE).

The repository contains only non-confidential research and software components. Industrial data and proprietary materials are not included.
