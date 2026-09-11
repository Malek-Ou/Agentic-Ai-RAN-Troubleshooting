"""
graph.py — LangGraph Pipeline
Huawei RAN AIOps — Agentic Pipeline

Pipeline order :
  symptom → memory_retrieve → retrieval → rca → memory_save → END

  Node 1 — symptom_identification_agent  : classifie l'input (ML anomaly ou user query)
  Node 2 — memory_retrieve_agent         : CBR — récupère les cas historiques AVANT le raisonnement
  Node 3 — retrieval_agent               : Agentic ReAct (Groq 70B) — KG + RAG + Web tools
                                           Web search est un tool ReAct, plus un node fixe.
  Node 4 — rc_analysis_agent             : LLM RCA reasoning (with CBR context already in state)
  Node 5 — memory_save_agent             : sauvegarde le RCA dans SQLite

State keys written per node :
  symptom_identification_agent → symptom_context
  memory_retrieve_agent        → similar_cases
  retrieval_agent              → rag_results, kg_result, reformulated_query,
                                  retrieval_trace, retrieval_strategy,
                                  web_search_triggered, web_search_trace
  rc_analysis_agent            → rca_result, display_answer
  memory_save_agent            → memory_saved, case_id
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Any

from agents.agent_symptom   import symptom_identification_agent
from agents.agent_memory    import memory_retrieve_agent, memory_save_agent
from agents.agent_retrieval import retrieval_agent
from agents.agent_rca       import rc_analysis_agent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class RANAIOpsState(TypedDict):
    # Inputs
    user_query          : Optional[str]
    ml_anomalies        : Optional[Any]
    ml_context          : Optional[dict]

    # Node 1 — Symptom
    symptom_context     : Optional[dict]

    # Node 2 — Memory Retrieve
    similar_cases       : Optional[list]

    # Node 3 — Retrieval (ReAct — KG + RAG + Web tools)
    rag_results         : Optional[list]
    kg_result           : Optional[dict]
    reformulated_query  : Optional[str]
    retrieval_trace     : Optional[list]
    retrieval_strategy  : Optional[str]
    web_search_triggered : Optional[bool]   # True si ReAct a appelé web_search_huawei
    web_search_trace     : Optional[list]   # trace web search (depuis rag_tools)

    # Node 4 — RCA
    rca_result          : Optional[dict]
    display_answer      : Optional[str]

    # Node 5 — Memory Save
    memory_saved        : Optional[bool]
    case_id             : Optional[int]

    # Feature flags (optional — None = default behaviour)
    disable_kg          : Optional[bool]
    disable_cbr         : Optional[bool]
    intent_hint         : Optional[str] = None

    
    memory_db_path      : Optional[str]

    # Misc
    messages            : list
    iterations          : int


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def node_symptom(state: RANAIOpsState) -> RANAIOpsState:
    logger.info("[graph] ── Node 1: symptom_identification_agent")
    t0 = time.perf_counter()
    result = symptom_identification_agent(state)
    logger.info(f"[graph] Node 1 done in {time.perf_counter()-t0:.2f}s")
    return result


def node_memory_retrieve(state: RANAIOpsState) -> RANAIOpsState:
    logger.info("[graph] ── Node 2: memory_retrieve_agent")
    t0 = time.perf_counter()
    result = memory_retrieve_agent(state)
    logger.info(f"[graph] Node 2 done in {time.perf_counter()-t0:.2f}s")
    return result


def node_retrieval(state: RANAIOpsState) -> RANAIOpsState:
    logger.info("[graph] ── Node 3: retrieval_agent")
    t0 = time.perf_counter()
    result = retrieval_agent(state)
    logger.info(f"[graph] Node 3 done in {time.perf_counter()-t0:.2f}s")
    return result


def node_rca(state: RANAIOpsState) -> RANAIOpsState:
    logger.info("[graph] ── Node 4: rc_analysis_agent")
    t0 = time.perf_counter()
    result = rc_analysis_agent(state)
    logger.info(f"[graph] Node 4 done in {time.perf_counter()-t0:.2f}s")
    return result


def node_memory_save(state: RANAIOpsState) -> RANAIOpsState:
    logger.info("[graph] ── Node 5: memory_save_agent")
    t0 = time.perf_counter()
    result = memory_save_agent(state)
    logger.info(f"[graph] Node 5 done in {time.perf_counter()-t0:.2f}s")
    return result


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def route_after_symptom(state: RANAIOpsState) -> str:
    """
    After symptom identification:
      - valid symptom   → continue to memory_retrieve
      - unknown/empty   → END
    """
    ctx = state.get("symptom_context") or {}
    symptom_type = ctx.get("symptom_type", "unknown")
 
    NON_DIAGNOSTIC = {"unknown", "definition", "chitchat", "recommendation"}
    if not ctx or symptom_type in NON_DIAGNOSTIC:
        logger.warning(
            f"[graph] Non-diagnostic symptom_type={symptom_type!r} → END "
            f"(pipeline stopped before retrieval)"
        )
        return "end"
    return "continue"


# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────

def build_graph():
    g = StateGraph(RANAIOpsState)

    # Register nodes
    g.add_node("symptom",         node_symptom)
    g.add_node("memory_retrieve", node_memory_retrieve)
    g.add_node("retrieval",       node_retrieval)
    g.add_node("rca",             node_rca)
    g.add_node("memory_save",     node_memory_save)

    # Entry point
    g.set_entry_point("symptom")

    # symptom → (conditional) → memory_retrieve | END
    g.add_conditional_edges(
        "symptom",
        route_after_symptom,
        {"continue": "memory_retrieve", "end": END},
    )

    # Linear pipeline — web search is now a ReAct tool inside node_retrieval
    g.add_edge("memory_retrieve", "retrieval")
    g.add_edge("retrieval",       "rca")
    g.add_edge("rca",             "memory_save")
    g.add_edge("memory_save",     END)

    return g.compile()


# Singleton — compiled once at import time
APP = build_graph()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def run_pipeline(
    user_query     : Optional[str]  = None,
    ml_anomalies   : Optional[Any]  = None,
    ml_context     : Optional[dict] = None,
    disable_kg     : Optional[bool] = None,
    disable_cbr    : Optional[bool] = None,
    memory_db_path : Optional[str]  = None,
    intent_hint    : Optional[str]  = None
) -> dict:
    """
    Main entry point for the RAN AIOps pipeline.

    Args:
        user_query     : free-text user question (e.g. "ALM-26235 cause?")
        ml_anomalies   : DataFrame of ML-detected anomalies
        disable_kg     : True → skip Neo4j enrichment
        disable_cbr    : True → skip CBR memory retrieve
        memory_db_path : BATCH REPORT (provisoire) — chemin SQLite optionnel
                         pour memory_retrieve_agent / memory_save_agent.
                         None (défaut) → agents.agent_memory.DB_PATH inchangé
                         (comportement live identique à avant ce paramètre).
                         Permet à un orchestrateur batch (ThreadPoolExecutor)
                         de cibler une base séparée (ex. rca_batch.db) par
                         exécution, sans variable globale partagée entre threads.

    Returns:
        Final state dict. Key fields:
          - display_answer       : formatted RCA for chatbot
          - rca_result           : full RCAResult dict
          - similar_cases        : CBR historical cases retrieved
          - memory_saved         : whether this case was saved to memory
          - retrieval_strategy   : "deterministic" | "react"
          - web_search_triggered : True si ReAct a appelé web_search_huawei
          - web_search_trace     : détail des appels web (depuis rag_tools)
    """
    initial_state: RANAIOpsState = {
        "user_query"           : user_query,
        "ml_anomalies"         : ml_anomalies,
        "symptom_context"      : None,
        "ml_context"          : ml_context,
        "similar_cases"        : None,
        "rag_results"          : None,
        "kg_result"            : None,
        "reformulated_query"   : None,
        "retrieval_trace"      : None,
        "retrieval_strategy"   : None,
        "web_search_triggered" : None,
        "web_search_trace"     : None,
        "rca_result"           : None,
        "display_answer"       : None,
        "memory_saved"         : None,
        "case_id"              : None,
        "intent_hint"          : intent_hint,
        "messages"             : [],
        "iterations"           : 0,
        "disable_kg"           : disable_kg,
        "disable_cbr"          : disable_cbr,
        "memory_db_path"       : memory_db_path,
    }
    return APP.invoke(initial_state)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ── TEST 1 — User query (web search via ReAct tool si insuffisant) ──
    print("\n" + "═"*60)
    print("TEST 1 — gt11 : Random eNodeB restart (web tool expected via ReAct)")
    print("═"*60)
    result = run_pipeline(
        user_query="Random eNodeB restart leading to cell outage and service degradation"
    )
    print(f"Strategy            : {result.get('retrieval_strategy')}")
    print(f"Web search triggered: {result.get('web_search_triggered')}")
    print(f"RAG chunks total    : {len(result.get('rag_results') or [])}")
    web_chunks = [r for r in (result.get("rag_results") or []) if r.get("source_type") == "web"]
    print(f"  dont web chunks   : {len(web_chunks)}")
    print(f"Memory saved        : {result.get('memory_saved')}")
    print(f"Case ID             : {result.get('case_id')}")
    print(result.get("display_answer", "No answer"))

    # ── TEST 2 — User query (local suffisant → web tool non appelé) ─────
    print("\n" + "═"*60)
    print("TEST 2 — gt01 : ALM-26235 (local suffisant → web skip expected)")
    print("═"*60)
    result2 = run_pipeline(
        user_query="ALM-26235 RF Unit Maintenance Link Failure cause and solution"
    )
    print(f"Strategy            : {result2.get('retrieval_strategy')}")
    print(f"Web search triggered: {result2.get('web_search_triggered')}")
    print(f"RAG chunks total    : {len(result2.get('rag_results') or [])}")
    print(result2.get("display_answer", "No answer"))
