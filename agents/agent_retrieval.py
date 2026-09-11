import json
import logging
import re
import httpx
import os
import pickle
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pathlib import Path
from config import (
    LLM_MODEL,
    LLM_MODEL_FAST,
    LLM_TEMPERATURE,
    RAG_TOP_K,
    RAG_CANDIDATE,
    TOP_CELLS_N,
    REACT_LLM_BACKEND,
    REACT_LOCAL_MODEL,
    REACT_CLOUD_MODEL,
    REACT_CLOUD_MODEL_FAST 
)
from agents.rag_tools import (
    search_documentation as _search_documentation_raw,
    query_knowledge_graph as _query_knowledge_graph_raw,
    web_search_huawei,
    ALL_TOOLS,
)

# OPT — Cache LRU intra-process sur les appels KG et RAG identiques
# Le KG est stable pendant la durée de vie du backend.
# Le cache RAG évite les doublons intra-session (même reformulation, steps différents).
import hashlib
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_rag(query: str, technology: str, top_k: int) -> str:
    """Cache LRU sur search_documentation — clé = (query, technology, top_k)."""
    return _search_documentation_raw.invoke({"query": query, "technology": technology, "top_k": top_k})

import time
from functools import lru_cache

_KG_CACHE: dict = {}
_KG_CACHE_TTL = 300  # 5 minutes

def _cached_kg(kpi_category: str, alarm_tuple: tuple, component_tuple: tuple) -> str:
    key = (kpi_category, alarm_tuple, component_tuple)
    entry = _KG_CACHE.get(key)
    if entry and (time.time() - entry['ts']) < _KG_CACHE_TTL:
        return entry['result']
    result = _query_knowledge_graph_raw.invoke({
        "kpi_category": kpi_category,
        "alarm_keywords": list(alarm_tuple),
        "components": list(component_tuple),
    })
    _KG_CACHE[key] = {'result': result, 'ts': time.time()}
    return result


class _CachedTool:
    """Wrapper LangChain-compatible autour d'un callable caché."""
    def __init__(self, fn, cache_fn):
        self._fn     = fn
        self._cache  = cache_fn
    def invoke(self, args: dict):
        return self._fn.invoke(args)  # passthrough — cache géré dans _execute_tool

# Expose sous les mêmes noms attendus par le reste du module
search_documentation   = _search_documentation_raw
query_knowledge_graph  = _query_knowledge_graph_raw

load_dotenv(Path(__file__).parent.parent / '.env')

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OSS_PATH       = os.getenv("OSS_PATH")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# REACT CONSTANTS
# ─────────────────────────────────────────────

REACT_HARD_STEP_CEILING = 4   # OPT — ceiling 6→4 : max 4 LLM calls → ~40s max vs ~60s

_WEB_SEARCH_LINE = (
    "  web_search_huawei      — Web search (Tavily), trusted Huawei/3GPP sources only.\n"
    "                           Useful when local KG/RAG leaves a gap (unknown alarm,\n"
    "                           recent advisory, software bug not in local docs).\n"
    if "web_search_huawei" in ALL_TOOLS else ""
)

REACT_SYSTEM_PROMPT = """You are a Huawei RAN expert retrieval agent.
Your ONLY job is to gather evidence — do NOT produce root cause or diagnosis.

At each step you must respond with EXACTLY ONE of these two JSON formats:

FORMAT A — call a tool:
{{
  "type": "action",
  "thought": "why this tool, given the query and what you already know",
  "tool": "search_documentation",
  "args": {{"query": "<search terms derived from the CURRENT query/KPI hint only>", "technology": "4g"}}
}}

FORMAT B — stop when evidence is sufficient:
{{
  "type": "final",
  "thought": "why the evidence gathered so far is enough to proceed to RCA"
}}

Available tools — there is no fixed order and no tool you are required to call.
Match each tool's content against the query and decide for yourself which one(s)
are worth calling, in whichever order, as many times as you judge useful:

  query_knowledge_graph  — Neo4j causal chains, alarm relationships, validated
                           root cause paths per KPI category or alarm/component
                           keywords. MUST provide at least one arg (kpi_category,
                           alarm_keywords, or components) — never call with empty
                           args {{}}.
  search_documentation   — Qdrant hybrid RAG over Huawei technical docs: alarm
                           descriptions, troubleshooting procedures, config guides.
{web_search_line}
How to decide — stop vs continue:
  - Stop (type=final) when the evidence is both sufficient AND internally consistent:
    the KG causal paths and the documentation results point toward the same root cause.
  - Continue reasoning (do NOT stop) if ANY of these is true:
      * Documentation quality is POOR or IRRELEVANT (shown in the observation header)
        → try a different query with different terminology, or broader terms
      * Documentation quality is POOR after 2 attempts → call web_search_huawei instead
      * KG returned 0 paths for a query where alarm IDs or KPI categories were provided
        → try search_documentation before stopping
      * KG causal paths and documentation point to contradictory root causes
        → call web_search_huawei to resolve the contradiction
      * search_documentation returned 0 chunks → reformulate with synonyms or broader terms
  - Use web_search_huawei only when: local evidence quality is insufficient after retrying,
    OR when the query involves a specific firmware version or recent Huawei advisory.
  - Do NOT call the same tool with identical args twice — reformulate if retrying.

General rules:
  - Only search for evidence directly related to the CURRENT query and KPI hint.
    Do NOT reuse alarm IDs or technical terms that appear ONLY in previous observations.
  - Output ONLY valid JSON, no markdown, no explanation outside the JSON.
""".strip()


# ─────────────────────────────────────────────
# HELPERS — LLM
# ─────────────────────────────────────────────

from utils.llm_cloud import (
    call_cloud_llm_messages as _call_cloud_llm_messages,
    PROVIDER as _CLOUD_PROVIDER,
)
from utils.llm import call_llm_messages_raw as _call_local_llm_messages_raw

def _backend_label() -> str:
    if REACT_LLM_BACKEND == "local":
        return f"backend=local provider=ollama model={REACT_LOCAL_MODEL}"
    return f"backend=cloud provider={_CLOUD_PROVIDER} model={REACT_CLOUD_MODEL}"

logger.info(f"[retrieval_agent] ReAct LLM — {_backend_label()}")

def _call_llm_messages(messages: list[dict], max_tokens: int = 150) -> str:  # OPT — WAS 400
    for attempt in range(2):
        try:
            if REACT_LLM_BACKEND == "local":
                text = _call_local_llm_messages_raw(
                    messages, max_tokens=max_tokens, caller="retrieval_agent",
                    model=REACT_LOCAL_MODEL, temperature=0.0,
                )
            else:
                text = _call_cloud_llm_messages(messages, max_tokens=max_tokens, model=REACT_CLOUD_MODEL)
            if text:
                return text
        except Exception as e:
            logger.warning(f"[retrieval_agent] {REACT_LLM_BACKEND} LLM messages call failed (attempt {attempt + 1}/2): {e}")
            continue
        logger.warning(f"[retrieval_agent] {REACT_LLM_BACKEND} LLM messages call returned empty (attempt {attempt + 1}/2)")
    return ""

def _parse_llm_action(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"[retrieval_agent][react] Non-JSON LLM output: {text[:120]}")
        return {}

# ─────────────────────────────────────────────
# HELPERS — SHARED
# ─────────────────────────────────────────────

_REFORMULATION_BANNED_PHRASES = [
    "potential", "could", "might", "includes", "include",
    "solution", "troubleshooting", "this indicates", "relevant terms",
    "for huawei base stations", "be related to",
]

def _sanitize_reformulation(text: str, fallback: str, original_q: str = "") -> str:
    if not text:
        return fallback
    text = text.strip().strip('"').strip("'")
    if len(text.split()) > 20:
        logger.warning("[retrieval_agent] Reformulation too long — using fallback")
        return fallback
    lowered = text.lower()
    if any(p in lowered for p in _REFORMULATION_BANNED_PHRASES):
        logger.warning("[retrieval_agent] Reformulation looks like prose — using fallback")
        return fallback
    hallucinated = set(re.findall(r"ALM-\d+", text)) - set(re.findall(r"ALM-\d+", original_q))
    if hallucinated:
        logger.warning(f"[retrieval_agent] Alarm ID hallucinée {hallucinated} — retrait")
        for alm in hallucinated:
            text = text.replace(alm, "")
        text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────
# FIX 1 — REFORMULATION AMÉLIORÉE
# ─────────────────────────────────────────────

SYMPTOM_QUERY_EXPANSION = {
    "e-rab"         : "E-RAB Setup Failure RRC LTE accessibility",
    "erab"          : "E-RAB Setup Failure RRC LTE accessibility",
    "setup failure" : "E-RAB Setup Failure RRC Connection LTE",
    "rrc"           : "RRC Setup Failure LTE eNodeB",
    "packet loss"   : "UL DL Packet Loss LTE transport S1AP",
    "ul loss"       : "UL Packet Loss rate LTE QCI BBU RRU",
    "dl loss"       : "DL Packet Loss PDCP BLER LTE eNodeB",
    "cpri"          : "CPRI optical link failure BBU RRU ",
    "optical"       : "optical module CPRI SFP BBU RRU failure",
    "sfp"           : "SFP optical transceiver CPRI BBU fault",
    "bbu"           : "BBU board failure hardware LTE eNodeB",
    "rru"           : "RRU CPRI link failure RF hardware LTE",
    "clock"         : "IP clock synchronization failure GPS LTE",
    "synchron"      : "synchronization GPS PTP IP clock failure LTE",
    "transport"     : "transport Ethernet S1 IP link failure LTE",
    "x2"            : "X2AP interface failure handover LTE eNodeB",
    "handover"      : "handover failure X2AP mobility LTE",
    "throughput"    : "throughput degradation DL UL PDSCH PUSCH LTE",
}

FAULT_DOMAIN_EXPANSION = {
    "hardware_stability" : "restart crash watchdog hardware board reset",
    "optical_link"        : "CPRI optical fiber SFP BBU RRU",
    "signaling"           : "RRC S1AP E-RAB SCTP signaling",
    "performance"         : "throughput PRB BLER congestion scheduler",
    "radio_interference"  : "SINR RSRP interference noise power",
}

def _rule_based_reformulation(original_q: str, symptom_context: dict) -> str:
    MAX_ENRICHMENT_TERMS = 3
    lowered   = original_q.lower()
    alarm_ids = re.findall(r"ALM-\d+", original_q)
    candidates: list[str] = []
    if alarm_ids:
        candidates.append(" ".join(alarm_ids))
    for keyword, taxonomy_query in SYMPTOM_QUERY_EXPANSION.items():
        if keyword in lowered:
            candidates.append(taxonomy_query)
    fault_domain = symptom_context.get("fault_domain", "unknown")
    if fault_domain in FAULT_DOMAIN_EXPANSION:
        candidates.append(FAULT_DOMAIN_EXPANSION[fault_domain])
    kpi_category = symptom_context.get("kpi_category", "")
    if kpi_category and not candidates:
        kpi_map = {
            "accessibility"   : "E-RAB Setup Failure RRC LTE accessibility eNodeB",
            "user_experience" : "throughput latency QCI PDSCH PUSCH LTE degradation",
            "retainability"   : "call drop RRC E-RAB release failure LTE",
            "integrity"       : "BLER packet loss UL DL LTE QCI",
        }
        if kpi_category in kpi_map:
            candidates.append(kpi_map[kpi_category])
    if not candidates:
        return original_q
    seen = set()
    unique = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    enrichment = unique[:MAX_ENRICHMENT_TERMS]
    result = " | ".join([original_q] + enrichment)
    logger.info(f"[retrieval_agent] Rule-based fallback produced: '{result}'")
    return result

# ─────────────────────────────────────────────
# FIX v3 — REFORMULATION AVEC ml_context
# ─────────────────────────────────────────────


def _reformulate_query(symptom_context: dict, ml_context: Optional[dict] = None) -> str:
    symptom_type = symptom_context.get("symptom_type")
    summary      = symptom_context.get("summary", "")
    severity     = symptom_context.get("severity", "")
    original_q   = symptom_context.get("original_query", "")
    kpi_category = symptom_context.get("kpi_category", "")
    top_kpis     = symptom_context.get("top_kpis", [])
    components   = symptom_context.get("affected_components", [])
    tech_label   = symptom_context.get("technology", "4g").upper()

    # ── ML anomaly path avec ml_context prioritaire ──────────────────
    if symptom_type == "ml_anomaly" and ml_context:
        parts = []
        cell = ml_context.get("cell")
        if cell:
            parts.append(cell)
        kpi_cat = ml_context.get("kpi_category") or kpi_category
        if kpi_cat:
            parts.append(f"KPI {kpi_cat}")
        z_score = ml_context.get("max_z_score")
        if z_score:
            parts.append(f"zscore {z_score:.1f}")
        breakdown = ml_context.get("kpi_breakdown", [])
        if breakdown:
            kpi_names = [item.get("kpi", "") for item in breakdown[:3] if item.get("kpi")]
            if kpi_names:
                parts.append("top " + " ".join(kpi_names))
        if components:
            parts.extend(components)
        parts.append("LTE eNodeB")
        reformulated = " ".join(parts)
        logger.info(f"[retrieval_agent] Reformulation from ml_context: '{reformulated}'")
        return reformulated

    # ── User query path ──────────────────────────────────────────────
    if symptom_type == "user_query":
        if re.search(r"ALM-\d+", original_q):
            logger.info("[retrieval_agent] Alarm ID detected → skip reformulation")
            return original_q
        rule_based_fallback = _rule_based_reformulation(original_q, symptom_context)
        if rule_based_fallback != original_q:
            logger.info("[retrieval_agent] Rule-based sufficient — skipping LLM reformulation")
            return rule_based_fallback
        prompt = (
            "You are a Huawei RAN fault retrieval expert.\n"
            "Convert the user query into a SHORT technical keyword query for documentation search.\n\n"
            "STRICT RULES:\n"
            "- Output ONLY keywords, NO sentences, NO verbs, NO punctuation except spaces\n"
            "- Max 12 words total\n"
            "- Use exact Huawei terminology: E-RAB, CPRI, BBU, RRU, S1AP, X2AP, PDSCH, PUSCH, RBLER, QCI\n"
            "- Include: fault type + component + technology\n"
            "- When calling query_knowledge_graph, always derive alarm_keywords from the CURRENT query, not from previous observations. Do not reuse the exact same args as a previous call."
            "- Do NOT output explanations, do NOT use 'could', 'might', 'potential', 'solution'\n\n"
            "Examples:\n"
            "Input: 'E-RAB setup failure and UL packet loss LTE, what could be the cause?'\n"
            "Output: E-RAB setup failure UL packet loss LTE RRC S1AP eNodeB\n\n"
            "Input: 'Why is the CPRI link failing on my BTS?'\n"
            "Output: CPRI optical link failure BBU RRU  SFP\n\n"
            "Input: 'Cells dropping calls after upgrade'\n"
            "Output: call drop RRC E-RAB release failure LTE eNodeB upgrade\n\n"
            f"Input: '{original_q}'\n"
            "Output:"
        )
        result = _call_llm_messages([{"role": "user", "content": prompt}], max_tokens=30)
        return _sanitize_reformulation(result, rule_based_fallback, original_q)

    # ── ML anomaly path sans ml_context (fallback existant) ─────────
    kpi_map = {
        "user_experience" : "UL Packet Loss E-RAB Setup Failure DL RBLER LTE eNodeB",
        "accessibility"   : "E-RAB Setup Failure RRC Connection LTE eNodeB S1AP",
        "radio_quality"   : "DL RBLER CQI SINR interference LTE RRU BBU",
        "resource_pressure": "PRB utilization TTI congestion LTE eNodeB",
        "traffic_load"    : "DL UL traffic volume spike LTE eNodeB",
    }
    if top_kpis:
        cleaned = []
        for kpi in top_kpis[:3]:
            c = re.sub(r"[()%]", " ", str(kpi))
            c = re.sub(r"\s+", " ", c).strip()
            if c:
                cleaned.append(c)
        parts = cleaned
        if components:
            parts += components
        parts.append(f"LTE {tech_label} eNodeB" if tech_label != "4G" else "LTE eNodeB")
        reformulated = " ".join(parts)
    else:
        reformulated = kpi_map.get(kpi_category, summary[:80])
    logger.info(f"[retrieval_agent] Query reformulated from ML anomaly (deterministic): '{reformulated}'")
    return reformulated

# ─────────────────────────────────────────────
# KG HELPER
# ─────────────────────────────────────────────

def _empty_kg_result() -> dict:
    return {
        "kg_paths"        : [],
        "related_alarms"  : [],
        "affected_enodebs": [],
        "available"       : False,
    }

# ─────────────────────────────────────────────
# PATH (A) — DETERMINISTIC RETRIEVAL
# ─────────────────────────────────────────────

def _deterministic_rag_search(query: str, technology: str, components: list[str]) -> list[dict]:
    try:
        from rag.retrieval import retrieve_hybrid
        from rag.reranker import rerank
        if components:
            component_hint = ", ".join(components[:3])
            clean_query = query.strip().strip('"').strip("'")
            enriched_query = f"{clean_query} {component_hint}"
        else:
            enriched_query = query.strip().strip('"').strip("'")
        candidates = retrieve_hybrid(
            question   = enriched_query,
            top_k      = RAG_CANDIDATE,
            technology = technology if technology != "unknown" else None,
        )
        if not candidates:
            return []
        reranked = rerank(query, candidates, top_k=RAG_TOP_K)
        logger.info(f"[retrieval_agent][det] RAG: {len(reranked)} chunks")
        return reranked
    except Exception as e:
        logger.error(f"[retrieval_agent][det] RAG search failed: {e}")
        return []

def _deterministic_kg_query(symptom_context: dict) -> dict:
    kg_result = _empty_kg_result()
    try:
        from neo4j import GraphDatabase
        symptom_type   = symptom_context.get("symptom_type")
        kpi_category   = symptom_context.get("kpi_category", "")
        affected_cells = symptom_context.get("affected_cells", [])
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                if symptom_type == "ml_anomaly" and affected_cells:
                    top_cells = affected_cells[:TOP_CELLS_N]
                    result = session.run("""
                        MATCH (e:ENodeB)-[:HAS_CELL]->(c:Cell)
                        WHERE c.id IN $cells
                        RETURN c.id AS cell, e.id AS enodeb,
                               c.risk_score AS risk_score,
                               c.anomaly_count AS anomaly_count,
                               c.critical_ratio AS critical_ratio
                        LIMIT 20
                    """, cells=top_cells)
                    kg_result["affected_enodebs"] = [
                        {
                            "cell"          : r["cell"],
                            "enodeb"        : r["enodeb"],
                            "risk_score"    : r["risk_score"],
                            "anomaly_count" : r["anomaly_count"],
                            "critical_ratio": r["critical_ratio"],
                        }
                        for r in result
                    ]
                    if kpi_category:
                        result2 = session.run("""
                            MATCH (kc:KPICategory {category: $cat})
                                  -[r:HAS_ROOT_CAUSE]->(rc:RootCause)
                                  <-[:BELONGS_TO]-(a:OSSAlarm)
                            RETURN rc.label          AS root_cause_label,
                                   rc.category       AS rc_category,
                                   r.posterior_prob  AS posterior_prob,
                                   r.p_doc           AS p_doc,
                                   r.confidence      AS confidence,
                                   r.top_evidence    AS top_evidence,
                                   a.id              AS alarm,
                                   a.name            AS alarm_name,
                                   a.dominant_severity AS alarm_severity
                            ORDER BY r.posterior_prob DESC
                            LIMIT 10
                        """, cat=kpi_category)
                        alarms = []
                        for r in result2:
                            alarms.append({
                                "alarm"           : r["alarm"],
                                "alarm_name"      : r["alarm_name"],
                                "root_cause_label": r["root_cause_label"],
                                "category"        : r["rc_category"],
                                "posterior_prob"  : r["posterior_prob"],
                                "p_doc"           : r["p_doc"],
                                "confidence"      : r["confidence"],
                                "top_evidence"    : r["top_evidence"],
                                "alarm_severity"  : r["alarm_severity"],
                                "prior_prob"      : r["posterior_prob"],
                            })
                        kg_result["related_alarms"] = alarms
                        if alarms:
                            top = alarms[0]
                            kg_result["kg_paths"].append(
                                f"KPICategory({kpi_category}) "
                                f"→ HAS_ROOT_CAUSE (post={top['posterior_prob']:.3f}) "
                                f"→ RootCause({top['root_cause_label']}) "
                                f"← BELONGS_TO ← OSSAlarm({top['alarm']})"
                            )
                        for alarm in alarms[1:3]:
                            kg_result["kg_paths"].append(
                                f"KPICategory({kpi_category}) "
                                f"→ RootCause({alarm['root_cause_label']}) "
                                f"post={alarm['posterior_prob']:.3f} [{alarm['confidence']}]"
                            )
            kg_result["available"] = True
        logger.info(
            f"[retrieval_agent][det] KG OK — "
            f"{len(kg_result['kg_paths'])} paths, "
            f"{len(kg_result['related_alarms'])} alarms"
        )
    except ImportError:
        logger.warning("[retrieval_agent][det] neo4j driver absent — KG ignoré")
    except Exception as e:
        logger.warning(f"[retrieval_agent][det] KG query failed: {e} — continuing")
    return kg_result

def _run_deterministic_retrieval(symptom_context: dict, ml_context: Optional[dict] = None) -> tuple[list[dict], dict, list[str], str]:
    import concurrent.futures

    trace = ["[deterministic] Starting fixed retrieval pipeline — KG-guided mode (parallel KG+RAG)"]
    technology = symptom_context.get("technology", "4g")
    components = symptom_context.get("affected_components", [])
    kpi_cat    = symptom_context.get("kpi_category", "")

    # ── OPT : Lancer KG + reformulation en parallèle ────────────────────
    # La reformulation (LLM ou rule-based) et la requête KG sont indépendantes.
    # On les lance simultanément, puis on construit la vraie RAG query depuis
    # les résultats KG si disponibles (comme avant), sinon depuis la reformulation.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_kg    = ex.submit(_deterministic_kg_query, symptom_context)
        fut_query = ex.submit(_reformulate_query, symptom_context, ml_context)
        kg_result         = fut_kg.result()
        fallback_reformed = fut_query.result()

    trace.append(
        f"[deterministic] KG: {len(kg_result.get('kg_paths', []))} paths, "
        f"{len(kg_result.get('related_alarms', []))} alarms"
    )

    # ── STEP 2 : Build RAG query from KG root causes ─────────────────────
    alarms = kg_result.get("related_alarms", [])
    if alarms:
        # Dedup by root_cause_label (KG may return same label with identical posterior)
        seen_labels: set = set()
        top_rc: list = []
        for a in sorted(alarms, key=lambda x: x.get("posterior_prob", 0), reverse=True):
            label = a.get("root_cause_label")
            if label and label not in seen_labels:
                seen_labels.add(label)
                top_rc.append(a)
            if len(top_rc) == 3:
                break

        # Root cause labels with posterior score (helps reranker weight importance)
        rc_terms = [
            f"{a.get('root_cause_label')} ({float(a.get('posterior_prob', 0)):.2f})"
            for a in top_rc
        ]

        # OSS alarm names — deduped (often more specific than abstract RC labels)
        seen_alarms: set = set()
        alarm_terms = []
        for a in top_rc:
            name = a.get("alarm_name")
            if name and name not in seen_alarms:
                seen_alarms.add(name)
                alarm_terms.append(name)

        reformulated_q = (
            " ".join(rc_terms)
            + (" " + " ".join(alarm_terms) if alarm_terms else "")
            + f" LTE {technology} {kpi_cat}"
        )

        trace.append(
            f"[KG-guided retrieval] Root causes used: {[a.get('root_cause_label') for a in top_rc]}"
        )
        trace.append(f"[KG-guided retrieval] Generated query: '{reformulated_q}'")
        logger.info(f"[retrieval_agent][det] KG-guided RAG query: '{reformulated_q}'")

        # Structured trace entry for dashboard / Explainable AI
        trace.append({
            "type": "kg_guided_retrieval",
            "root_causes": [
                {
                    "name": a.get("root_cause_label"),
                    "probability": a.get("posterior_prob"),
                }
                for a in top_rc
            ],
            "query": reformulated_q,
        })

    else:
        # Fallback if KG empty — reuse reformulation computed in parallel above
        reformulated_q = fallback_reformed
        trace.append(f"[deterministic] KG empty — fallback to reformulation (parallel): '{reformulated_q}'")
        logger.info(f"[retrieval_agent][det] KG empty — fallback query: '{reformulated_q}'")

    # ── STEP 3 : RAG guided by KG ────────────────────────────────────────
    rag_results = _deterministic_rag_search(reformulated_q, technology, components)
    trace.append(f"[deterministic] RAG: {len(rag_results)} chunks from Qdrant")

    return rag_results, kg_result, trace, reformulated_q

# ─────────────────────────────────────────────
# REACT TOOLS
# ─────────────────────────────────────────────

TOOL_MAP = {
    "search_documentation" : search_documentation,
    "query_knowledge_graph": query_knowledge_graph,
    "web_search_huawei"    : web_search_huawei,
}
if os.getenv("DISABLE_WEB_SEARCH", "false").lower() in ("1", "true", "yes"):
    TOOL_MAP.pop("web_search_huawei", None)

def _execute_tool(tool_name: str, args: dict) -> str:
    tool_fn = TOOL_MAP.get(tool_name)
    if not tool_fn:
        return json.dumps({"error": f"unknown tool: {tool_name}"})
    try:
        # OPT — utilise le cache LRU pour KG et RAG (évite les appels dupliqués)
        if tool_name == "query_knowledge_graph":
            kpi_cat = args.get("kpi_category", "") or ""
            alarms  = tuple(sorted(args.get("alarm_keywords", []) or []))
            comps   = tuple(sorted(args.get("components", []) or []))
            cached  = _cached_kg(kpi_cat, alarms, comps)

            return cached if isinstance(cached, str) else json.dumps(cached, ensure_ascii=False)
        if tool_name == "search_documentation":
            query  = args.get("query", "")
            tech   = args.get("technology", "4g") or "4g"
            top_k  = int(args.get("top_k", 3))
            cached = _cached_rag(query, tech, top_k)
            logger.debug(f"[react][cache] RAG cache_info={_cached_rag.cache_info()}")
            return cached if isinstance(cached, str) else json.dumps(cached, ensure_ascii=False)

        result = tool_fn.invoke(args)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as e:
        logger.error(f"[retrieval_agent][react] Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})

# ─────────────────────────────────────────────
# FIX 2 — REACT EVIDENCE PARSING
# ─────────────────────────────────────────────

def _parse_react_evidence(
    tool_results   : list[dict],
    reformulated_q : str,
    technology     : str,
) -> tuple[list[dict], dict]:
    rag_results : list[dict] = []
    kg_result                = _empty_kg_result()

    for item in tool_results:
        tool   = item["tool"]
        args   = item.get("args", {})
        output = item["output"]

        if tool == "search_documentation":
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list) and parsed:
                    rag_results.extend(parsed)
                    top_score = parsed[0].get("rerank_score", "N/A")
                    top_score_fmt = f"{top_score:.3f}" if isinstance(top_score, float) else str(top_score)
                    logger.info(
                        f"[retrieval_agent][parse] search_documentation pass-through: "
                        f"{len(parsed)} chunks | top_score={top_score_fmt}"
                    )
                else:
                    raise ValueError("empty or non-list output")
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "[retrieval_agent][react] search_documentation output not JSON "
                    "— storing as raw fallback"
                )
                rag_results.append({
                    "text"        : output,
                    "source"      : "qdrant_react_raw",
                    "rerank_score": 0.5,
                    "score"       : 0.5,
                    "technology"  : technology,
                    "domain"      : "unknown",
                })
        elif tool == "query_knowledge_graph":
            try:
                parsed = json.loads(output)
                kg_result["kg_paths"]       += parsed.get("kg_paths", [])[:5]
                kg_result["related_alarms"] += parsed.get("related_alarms", [])[:3]
                for rc in parsed.get("root_causes", []):
                    kg_result["related_alarms"].append({
                        "alarm"            : rc.get("alarm_id", ""),
                        "alarm_name"       : rc.get("alarm_name", rc.get("root_cause", "")),
                        "root_cause_label" : rc.get("root_cause", ""),
                        "category"         : rc.get("category", ""),
                        "posterior_prob"   : rc.get("posterior", 0.5),
                        "prior_prob"       : rc.get("posterior", 0.5),
                        "confidence"       : rc.get("confidence", "medium"),
                        "alarm_severity"   : "",
                    })
                kg_result["available"] = True
            except (json.JSONDecodeError, AttributeError):
                logger.warning("[retrieval_agent][react] Could not parse KG output")
        elif tool == "web_search_huawei":
            rag_results.append({
                "text"        : output,
                "content"     : output,
                "source"      : "web_react",
                "source_type" : "web",
                "rerank_score": 0.5,
                "score"       : 0.5,
                "technology"  : technology,
                "domain"      : "web",
            })

    seen_chunks = set()
    deduped = []
    for chunk in rag_results:
        key = (
            chunk.get("source", ""),
            str(chunk.get("text", chunk.get("content", "")))[:80],
        )
        if key not in seen_chunks:
            seen_chunks.add(key)
            deduped.append(chunk)
    deduped.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return deduped, kg_result

def _build_kg_nudge(reformulated_q: str, kpi_hint: str, components: list) -> str:
    alarm_ids = re.findall(r"ALM-\d+", reformulated_q)
    has_kpi   = kpi_hint and kpi_hint != "unknown"
    has_comp  = bool(components)
    if not (alarm_ids or has_kpi or has_comp):
        return ""
    parts = []
    if alarm_ids:
        parts.append(f"alarm_keywords={alarm_ids}")
    if has_kpi:
        parts.append(f"kpi_category='{kpi_hint}'")
    if has_comp:
        parts.append(f"components={components[:2]}")
    hint = (
        f"Hint: the knowledge graph may contain validated causal paths for this query "
        f"({', '.join(parts)}). "
        f"Consider calling query_knowledge_graph first before searching documentation.\n\n"
    )
    return hint

def _run_react_retrieval(symptom_context: dict, ml_context: Optional[dict] = None) -> tuple[list[dict], dict, list[str], str]:
    trace = [
        "[react] START pure ReAct loop — LLM has full tool autonomy",
        f"[react] {_backend_label()}",
    ]
    tool_results = []

    reformulated_q = _reformulate_query(symptom_context, ml_context)
    trace.append(f"[react] Reformulated query: '{reformulated_q}'")

    technology  = symptom_context.get("technology", "4g")
    kpi_hint    = symptom_context.get("kpi_category", "unknown")
    components  = symptom_context.get("affected_components", [])

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT.format(web_search_line=_WEB_SEARCH_LINE)},
        {"role": "user", "content": (
            f"User query: {symptom_context.get('original_query', '')}\n"
            f"Reformulated: {reformulated_q}\n"
            f"KPI hint: {kpi_hint}\n"
            f"Components: {', '.join(components) if components else 'unknown'}\n"
            f"Technology: {technology}\n\n"
            f"{_build_kg_nudge(reformulated_q, kpi_hint, components)}"
            f"Start gathering evidence. Output JSON action."
        )},
    ]

    # kg_result initialisé ici pour que l'early-stop intra-boucle puisse y accéder
    # (mis à jour à chaque appel query_knowledge_graph dans _parse_react_evidence)
    kg_result      = _empty_kg_result()
    infra_failures = 0
    for step in range(REACT_HARD_STEP_CEILING):
        trace.append(f"[react] ── step {step + 1} ──")
        llm_output = _call_llm_messages(messages, max_tokens=150)  # OPT — ReAct JSON tient en ~80 tokens, 150 est large
        if not llm_output:
            infra_failures += 1
            trace.append(f"[react] Empty LLM response — infra failure {infra_failures}/2")
            logger.warning(f"[react] Groq unreachable — infra failure {infra_failures}/2")
            if infra_failures >= 2:
                trace.append("[react] Groq unreachable twice — stopping loop")
                logger.warning("[react] Groq unreachable — stopping loop")
                break
            continue
        action = _parse_llm_action(llm_output)
        if not action:
            trace.append("[react] Non-JSON output — feeding correction back")
            logger.warning("[react] Non-JSON LLM output — asking it to retry in valid JSON")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": "Your last output was not valid JSON. Respond with EXACTLY one JSON object — FORMAT A or FORMAT B — and nothing else.",
            })
            continue
        infra_failures = 0
        action_type = action.get("type", "action")
        thought     = action.get("thought", "")
        trace.append(f"[react] Thought: {thought}")
        if thought:
            logger.info(f"[react] Thought: {thought[:120]}")
        if action_type == "final":
            trace.append("[react] LLM signalled final — stopping (its own call)")
            logger.info("[react] final — LLM judged evidence sufficient, stopping loop")
            break
        tool_name = action.get("tool", "")
        args      = action.get("args", {})
        if tool_name not in TOOL_MAP:
            trace.append(f"[react] Unknown tool '{tool_name}' — feeding correction back")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": f"'{tool_name}' is not an available tool. Choose one of {list(TOOL_MAP.keys())} or output type=final.",
            })
            continue
        trace.append(f"[react] Calling {tool_name}({args})")
        query_preview = str(args.get("query", args.get("alarm_keywords", args.get("kpi_category", ""))))[:60]
        logger.info(f"[react] calling {tool_name} | query='{query_preview}'")
        observation = _execute_tool(tool_name, args)
        tool_results.append({"tool": tool_name, "args": args, "output": observation})
        trace.append(f"[react] Observation: {len(observation)} chars")
        # Mise à jour live de kg_result pour l'early-stop
        if tool_name == "query_knowledge_graph":
            try:
                parsed_kg = json.loads(observation)
                kg_result["kg_paths"]       += parsed_kg.get("kg_paths", [])[:5]
                kg_result["related_alarms"] += parsed_kg.get("related_alarms", [])[:3]
                kg_result["available"]       = bool(kg_result["kg_paths"])
            except Exception:
                pass
        messages.append({"role": "assistant", "content": llm_output})
        OBS_LIMIT = 3000
        if len(observation) > OBS_LIMIT:
            if tool_name == "search_documentation":
                trimmed = observation[:OBS_LIMIT]
                last_brace = trimmed.rfind("},")
                if last_brace > OBS_LIMIT // 2:
                    observation_display = trimmed[:last_brace + 1] + "]  /* truncated */"
                else:
                    observation_display = trimmed + "…"
            else:
                observation_display = observation[:OBS_LIMIT] + "…"
            logger.debug(f"[react] Observation from {tool_name} truncated ({len(observation)} → {OBS_LIMIT} chars).")
        else:
            observation_display = observation
        quality_hint = ""
        if tool_name == "search_documentation":
            try:
                parsed_obs = json.loads(observation)
                if isinstance(parsed_obs, list) and parsed_obs:
                    top_score = parsed_obs[0].get("rerank_score", None)
                    n_chunks  = len(parsed_obs)
                    if top_score is not None:
                        kg_has_paths = bool(kg_result.get("kg_paths"))
                        # OPT — early-stop agressif : seuil 3.0→1.0, et GOOD (≥1.5) stoppe directement
                        if top_score >= 1.0 and kg_has_paths:
                            # EARLY STOP : KG + RAG score ≥ 1.0 = suffisant → pas de LLM call
                            trace.append(
                                f"[react] Early-stop — KG paths + RAG score={top_score:.2f} ≥ 1.0 → sufficient"
                            )
                            logger.info(
                                f"[react] final — early-stop (KG+RAG sufficient, score={top_score:.2f})"
                            )
                            break
                        if top_score >= 1.5:
                            # GOOD sans KG — stoppe directement sans repasser par le LLM
                            trace.append(
                                f"[react] Early-stop — RAG score={top_score:.2f} GOOD (no KG) → sufficient"
                            )
                            logger.info(
                                f"[react] final — early-stop (RAG GOOD, score={top_score:.2f}, no KG)"
                            )
                            break
                        if top_score >= 5.0:
                            quality_hint = (
                                f"Evidence quality: EXCELLENT (top reranker score = {top_score:.2f}, {n_chunks} chunks). "
                                f"STOP NOW — output type=final immediately, do not call any more tools.\n"
                            )
                        elif top_score >= 1.5:
                            # Ce bloc n'est plus atteint (early-stop ci-dessus) — conservé pour cohérence
                            quality_hint = (
                                f"Evidence quality: GOOD (top reranker score = {top_score:.2f}, {n_chunks} chunks). "
                                f"Relevant results found. Output type=final now.\n"
                            )
                        elif top_score >= 0.0:
                            quality_hint = (
                                f"Evidence quality: POOR (top reranker score = {top_score:.2f}, {n_chunks} chunks). "
                                f"Results are weakly relevant — reformulate with different Huawei terminology "
                                f"or broader terms before stopping.\n"
                            )
                        else:
                            quality_hint = (
                                f"Evidence quality: IRRELEVANT (top reranker score = {top_score:.2f}, {n_chunks} chunks). "
                                f"Do NOT stop here. Try a different query angle or call web_search_huawei.\n"
                            )
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
        messages.append({
            "role": "user",
            "content": (
                f"Observation from {tool_name}:\n"
                f"{quality_hint}"
                f"{observation_display}\n\n"
                f"Output next JSON action or type=final."
            ),
        })
    else:
        trace.append(f"[react] Hit hard step ceiling ({REACT_HARD_STEP_CEILING}) without type=final")
        logger.warning(f"[react] Hard step ceiling reached ({REACT_HARD_STEP_CEILING}) — stopping")

    rag_results, kg_result = _parse_react_evidence(tool_results, reformulated_q, technology)
    trace.append(
        f"[react] DONE — {len(tool_results)} tool calls | "
        f"RAG={len(rag_results)} chunks | "
        f"KG={len(kg_result['kg_paths'])} paths | "
        f"alarms={len(kg_result['related_alarms'])}"
    )
    return rag_results, kg_result, trace, reformulated_q

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def _route(symptom_context: dict) -> str:
    symptom_type = symptom_context.get("symptom_type", "")
    if symptom_type == "ml_anomaly":
        return "deterministic"
    return "react"

# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION (LangGraph node)
# ─────────────────────────────────────────────

def retrieval_agent(state: dict) -> dict:
    """
    LangGraph node — Retrieval Agent.
    """
    symptom_context = state.get("symptom_context")
    ml_context = state.get("ml_context")   # ← récupéré depuis l'état

    if not symptom_context:
        logger.error("[retrieval_agent] No symptom_context in state")
        return {
            **state,
            "rag_results"          : [],
            "kg_result"            : _empty_kg_result(),
            "reformulated_query"   : "",
            "retrieval_trace"      : ["ERROR: no symptom_context"],
            "retrieval_strategy"   : "none",
            "web_search_triggered" : False,
            "web_search_trace"     : [],
        }

    if state.get("disable_kg"):
        logger.info("[retrieval_agent] disable_kg=True — KG skipped")
        strategy = _route(symptom_context)
        if strategy == "deterministic":
            rag_results, _, trace, reformulated_q = _run_deterministic_retrieval(symptom_context, ml_context)
        else:
            rag_results, _, trace, reformulated_q = _run_react_retrieval(symptom_context, ml_context)
        return {
            **state,
            "rag_results"          : rag_results,
            "kg_result"            : _empty_kg_result(),
            "reformulated_query"   : reformulated_q,
            "retrieval_trace"      : trace,
            "retrieval_strategy"   : strategy,
            "web_search_triggered" : False,
            "web_search_trace"     : [],
        }

    strategy = _route(symptom_context)
    logger.info(
        f"[retrieval_agent] strategy={strategy} | "
        f"type={symptom_context.get('symptom_type')} | "
        f"severity={symptom_context.get('severity')}"
    )

    if strategy == "deterministic":
        rag_results, kg_result, trace, reformulated_q = _run_deterministic_retrieval(symptom_context, ml_context)
    else:
        rag_results, kg_result, trace, reformulated_q = _run_react_retrieval(symptom_context, ml_context)

    web_triggered = any(isinstance(t, str) and "web_search_huawei" in t for t in trace)
    web_trace     = [t for t in trace if isinstance(t, str) and "web" in t.lower()]

    logger.info(
        f"[retrieval_agent] Done — "
        f"rag={len(rag_results)} chunks, "
        f"kg_paths={len(kg_result['kg_paths'])}, "
        f"kg_alarms={len(kg_result['related_alarms'])}, "
        f"web_triggered={web_triggered}"
    )

    return {
        **state,
        "rag_results"          : rag_results,
        "kg_result"            : kg_result,
        "reformulated_query"   : reformulated_q,
        "retrieval_trace"      : trace,
        "retrieval_strategy"   : strategy,
        "web_search_triggered" : web_triggered,
        "web_search_trace"     : web_trace,
    }

# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "═"*60)
    print("TEST — ML anomaly avec ml_context")
    print("═"*60)

    try:
        with open("ml/artifacts/anomalies_kg.pkl", "rb") as f:
            df_all = pickle.load(f)
        df_critical = df_all[df_all["severity"] == "critical"].head(100)
        state = {
            "ml_anomalies": df_critical,
            "ml_context": {
                "cell": "DAA_4551_C01_M32_k3",
                "kpi_category": "resource_pressure",
                "max_z_score": 28.8,
                "anomalies": 690,
                "severity": "critical",
                "kpi_breakdown": [
                    {"kpi": "L.Traffic.ActiveUser.DL.Avg", "count": 77},
                    {"kpi": "UL Spectrum Efficiency", "count": 66},
                    {"kpi": "FT_4G/LTE UL TRAFFIC VOLUME (GBYTES)", "count": 64},
                ],
            }
        }
        from agents.agent_symptom import symptom_identification_agent
        state = symptom_identification_agent(state)
        state = retrieval_agent(state)
        print(f"Strategy  : {state['retrieval_strategy']}")
        print(f"Reformulated: {state['reformulated_query']}")
        print(f"RAG chunks: {len(state['rag_results'])}")
        print(f"KG paths  : {len(state['kg_result']['kg_paths'])}")
        print("Trace:")
        for line in state["retrieval_trace"][-5:]:
            print(f"  {line}")
    except FileNotFoundError:
        print("  (pkl file not found — skip ML anomaly test)")