# Agentic-Ai-RAN-Troubleshooting

> An Agentic AI platform for anomaly detection and root cause analysis in 4G LTE Radio Access Networks (RAN), combining hybrid ML-based KPI monitoring, multi-agent reasoning, Knowledge Graphs, RAG, and Case-Based Reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Diagnostic Pipeline](#diagnostic-pipeline)
- [Anomaly Detection](#anomaly-detection)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Repository Scope](#repository-scope)
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
             │                 LangGraph                   │
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

The diagnostic workflow is implemented as a LangGraph pipeline composed of five processing nodes.

| Node | Component                        | Main Responsibility                                                                      |
|------|----------------------------------|------------------------------------------------------------------------------------------|
| 1    | **Symptom Identification Agent** | Builds a structured symptom context from a natural-language query or ML anomaly metadata |
| 2    | **CBR Memory Manager**           | Retrieves similar validated or corrected historical cases                                 |
| 3    | **Retrieval Agent**              | Performs ReAct-based evidence retrieval using Knowledge Graph, RAG, and web search        |
| 4    | **Root Cause Analysis Agent**    | Synthesizes the available evidence into an explainable RCA hypothesis                    |
| 5    | **Memory Save**                  | Persists the generated case with `pending` validation status                             |

### Symptom Identification

The Symptom Identification Agent extracts:

- Affected cell
- Relevant KPI dimensions
- KPI category
- Estimated severity

For ML-triggered investigations, structured anomaly metadata is directly mapped into the symptom context, avoiding unnecessary LLM calls.

### Case-Based Reasoning

The CBR Memory Manager retrieves similar historical investigations using multiple criteria, including:

- Symptom type
- KPI category
- Severity
- Root-cause keywords

Only **validated or corrected cases** are eligible for future retrieval.

New investigation cases are initially stored with a `pending` status and remain excluded from retrieval until expert validation.

### ReAct Retrieval

The Retrieval Agent uses a **Reasoning + Acting (ReAct)** loop to select the most appropriate knowledge source.

A deterministic Knowledge Graph query is executed before the ReAct loop to guarantee structured causal evidence.

Available retrieval sources:

| Source              | Role                                                                            |
|---------------------|---------------------------------------------------------------------------------|
| **Knowledge Graph** | Retrieves causal KPI–root cause relationships from Neo4j                        |
| **RAG**             | Retrieves relevant technical documentation using hybrid dense/sparse search     |
| **Web Search**      | External fallback for novel fault patterns when local knowledge is insufficient |


## Anomaly Detection

The system uses a hybrid anomaly detection strategy because 4G LTE KPIs exhibit heterogeneous temporal behavior.

| Condition                     | Method                     | Rationale                                          |
|-------------------------------|----------------------------|----------------------------------------------------|
| Low residual variability      | **SARIMA**                 | Stable temporal structure and daily seasonality    |
| Moderate residual variability | **Prophet**                | Robust to outliers and trend changes               |
| High residual variability     | **STL + IQR**              | Non-parametric detection for irregular series      |
| Event-based KPIs              | **Modified Z-Score / MAD** | Suitable for ratio-based and non-seasonal behavior |

The detection pipeline routes each KPI to the appropriate method according to its temporal characteristics.

---

## Key Features

- **Hybrid KPI Anomaly Detection** using SARIMA, Prophet, STL + IQR, and Modified Z-Score
- **Intent Classification** for routing natural-language requests
- **Multi-Agent Diagnostic Pipeline** orchestrated with LangGraph
- **ReAct Retrieval** for adaptive knowledge-source selection
- **Knowledge Graph Reasoning** using Neo4j causal relationships
- **Hybrid RAG** combining dense and BM25 retrieval
- **Reciprocal Rank Fusion (RRF)** for hybrid retrieval merging
- **Cross-Encoder Reranking** for final context selection
- **Case-Based Reasoning** using validated historical investigations
- **Web Search Fallback** for novel fault patterns not covered by local knowledge
- **Explainable Root Cause Analysis** with evidence traceability and confidence breakdown
- **Human-In-The-Loop Validation** with Validate / Correct / Reject / Skip actions
- **Pipeline Observability** using Arize Phoenix

---

## Tech Stack

### AI / Machine Learning

| Component | Technology |
|-----------|------------|
| Anomaly Detection | SARIMA, Prophet, STL + IQR, Modified Z-Score / MAD |
| LLM Reasoning | Large Language Models (LLMs) via API |
| Agentic Loops | ReAct (Reasoning + Acting) |
| Language | Python |

### Agentic AI

| Component | Technology |
|-----------|------------|
| Workflow Orchestration | LangGraph |
| LLM & Retrieval Components | LangChain |

### Knowledge & Retrieval

| Component | Technology |
|-----------|------------|
| Knowledge Graph | Neo4j |
| Vector Database | Qdrant |
| Sparse Retrieval | BM25 |
| Semantic Retrieval | Dense Embeddings |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Web Search Fallback | Tavily |

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

## Repository Scope

This repository focuses on the **AI/ML and reasoning components** of the RAN troubleshooting system.

The following are intentionally excluded:

- Industrial Orange Tunisia data
- Real KPI and alarm datasets
- Internal network identifiers
- Confidential technical documentation
- API keys and credentials
- Internal infrastructure configurations
- Production frontend

Synthetic or anonymized examples can be used to demonstrate the main processing pipelines without exposing industrial information.

---

## Research Areas

This project explores:

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

This project was developed as part of a Final Year Engineering Project (PFE) at the National School of Electronics and Telecommunications of Sfax (ENET'COM), in partnership with Orange Tunisia.

The repository contains only non-confidential research and software components. Industrial data and proprietary materials are not included.
