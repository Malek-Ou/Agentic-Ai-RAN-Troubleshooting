# Agentic-Ai-RAN-Troubleshooting
>
> An AI-powered platform for automated anomaly detection and root cause analysis in 4G LTE radio access networks, combining ML-based KPI monitoring, a multi-agent diagnostic pipeline, and a knowledge-augmented reasoning engine.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
---

## Overview

Since the large-scale commercial deployment of 4G LTE, network performance is no longer evaluated solely through radio coverage — service accessibility, retainability, and user experience have become critical. Traditional OSS platforms rely on manual, rule-based troubleshooting that is time-consuming and operator-dependent.

**RAN AIOps Copilot** addresses this gap by delivering:

- **Hybrid ML anomaly detection** across 4G KPIs (SARIMA, Prophet, STL+IQR, Modified Z-Score)
- **A multi-agent diagnostic pipeline** (LangGraph orchestration) with ReAct reasoning
- **Knowledge-augmented RCA** combining a Neo4j Knowledge Graph, a vector RAG store, and Case-Based Reasoning (CBR) memory
- **A Human-In-The-Loop (HITL)** validation mechanism to ensure only expert-confirmed cases populate the CBR memory
- **A NOC engineer chat interface** with real-time streaming RCA results and exportable PDF reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                           │
│   NOC Engineer ──► Chat Interface                            │
│   Network KPI Data ──► ML Anomaly Detection                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Processing Layer                         │
│                                                              │
│  Intent Classifier ──► Symptom Agent ──► CBR Memory Manager │
│                              │                    │          │
│                         Retrieval Agent ◄──────────          │
│                              │                               │
│                         RCA Analysis Agent                   │
│                              │                               │
│                    LangGraph Orchestrator                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Persistent Storage / Knowledge             │
│                                                              │
│   CBR Memory │ Vector DB (Qdrant) │ Knowledge Graph (Neo4j)  │
│                         + Web Search Fallback                │
└─────────────────────────────────────────────────────────────┘
```

### Agent Pipeline (5 Nodes)

| Node | Agent | Role |
|------|-------|------|
| 1 | **Symptom Identification Agent** | Extracts structured symptom context from query or ML anomaly |
| 2 | **CBR Memory Manager** | Retrieves similar validated past cases |
| 3 | **Retrieval Agent** | Queries Knowledge Graph + Vector DB; triggers web search fallback if needed |
| 4 | **RCA Analysis Agent** | Generates root cause, confidence score, causal chain, and recommended actions |
| 5 | **Memory Save Agent** | Persists investigation case with `pending` validation status |

### Anomaly Detection Routing

| Condition | Model | Justification |
|-----------|-------|---------------|
| Low residual variability | SARIMA | Structured series with stable 24h seasonality |
| Moderate residual variability | Prophet | Robust to outliers and trend changes |
| High residual variability | STL + IQR | Non-parametric, no distributional assumptions |
| Event-based KPIs | Statistical (MAD) | Ratio-based behavior, no exploitable seasonality |

---

## Key Features

- **Hybrid Anomaly Detection**: Four complementary detection methods routing per KPI temporal behavior, with adaptive Z-score thresholding (|Z| > 3.0)
- **Multi-Agent RCA Pipeline**: LangGraph-based orchestration with ReAct reasoning loop (Thought → Action → Observation)
- **Knowledge Graph**: Neo4j causal graph linking KPI categories, root causes, OSS alarms, and network components
- **Case-Based Reasoning (CBR)**: Semantic similarity search over validated past investigations
- **Hybrid RAG Retrieval**: Sparse (BM25) + Dense (vector) retrieval with Reciprocal Rank Fusion and cross-encoder reranking
- **Web Search Fallback**: Activated when local knowledge is insufficient
- **HITL Validation**: Expert review workflow (Validate / Correct / Reject / Skip) controlling CBR memory population
- **ML Monitoring Dashboard**: Day×Hour anomaly heatmaps, KPI rankings, severity distribution
- **Investigation Chat Interface**: Natural language queries + ML anomaly-triggered investigations, SSE streaming
- **PDF RCA Reports**: Individual and batch exportable reports with causal chains and recommended actions
- **Pipeline Observability**: Arize Phoenix tracing across all 5 agent nodes

---

## Tech Stack

### Backend
- **Python**: Core language
- **LangGraph**: Multi-agent orchestration
- **LangChain**: LLM tooling and RAG components
- **Neo4j**: Knowledge Graph (causal relationships)
- **Qdrant**: Vector database for RAG
- **SARIMA / Prophet / STL**: Time-series anomaly detection
- **Arize Phoenix**: Pipeline observability and tracing

### Frontend
- **React** 
- **Streaming SSE**: Real-time diagnostic result streaming

### DevOps / Project Management
- **Jira**: Task planning, sprint management, progress tracking
- **Docker**: Containerization
- **Git/GitHub**: Version control
---

## License

This project was developed as part of a final year engineering project (PFE) at ENET'COM Sfax in partnership with Orange Tunisia. All rights reserved.
