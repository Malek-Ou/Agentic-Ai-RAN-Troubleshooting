import logging
import re
import httpx
import pandas as pd
from typing import TypedDict, Optional
from dotenv import load_dotenv  # FIX: single import, removed duplicate

logger = logging.getLogger(__name__)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_MODEL_FAST,
    LLM_MODEL,
    SYMPTOM_CLOUD_MODEL,  
)

from rag.retrieval import retrieve_hybrid, format_context
from rag.reranker  import rerank
from utils.llm import call_llm_raw as _call_llm_raw 
from utils.llm_cloud import call_cloud_llm_raw as _call_cloud_llm_raw  
from utils.llm_cloud import call_cloud_llm_messages_raw as _call_cloud_messages_raw 

# ─────────────────────────────────────────────
# NEO4J SINGLETON
# ─────────────────────────────────────────────

_NEO4J_DRIVER = None

def _get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        from neo4j import GraphDatabase
        _NEO4J_DRIVER = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USER",  ),
                os.getenv("NEO4J_PASSWORD", "password"),
            )
        )
    return _NEO4J_DRIVER

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

load_dotenv()  # FIX: called exactly once

# Mapping kpi_category → symptom description
KPI_CATEGORY_LABELS = {
    "accessibility"     : "network accessibility degradation",
    "user_experience"   : "user experience degradation (throughput/latency)",
    "radio_quality"     : "radio quality degradation (BLER/CQI/interference)",
    "resource_pressure" : "resource pressure (PRB/TTI saturation)",
    "traffic_load"      : "abnormal traffic load (DL/UL volume spike)",
    "radio_propagation" : "radio propagation issue",
}

# Composants Huawei RAN reconnus — extraction déterministe
COMPONENT_PATTERNS = {
    "BBU"          : [r"\bBBU\b", r"\bbaseband\b"],
    "RRU"          : [r"\bRRU\b", r"\bremote radio\b"],
    "AAU"          : [r"\bAAU\b", r"\bactive antenna\b"],
    "optical_module": [r"\boptical[\s_]module\b", r"\bCPRI\b", r"\bfibre\b", r"\bfiber\b"],
    "transmission" : [r"\btransmission\b", r"\bbackhaul\b", r"\bMW\b", r"\bmicrowave\b"],
    "antenna"      : [r"\bantenna\b", r"\bRET\b", r"\btilt\b"],
    "power"        : [r"\bpower\b", r"\bPSU\b", r"\bvoltage\b", r"\bbattery\b"],
    "S1_interface" : [r"\bS1\b", r"\bS1-MME\b", r"\bS1-U\b"],
    "X2_interface" : [r"\bX2\b"],
    "core"         : [r"\bMME\b", r"\bSGW\b", r"\bPGW\b", r"\bEPC\b"],
}

def _extract_components_deterministic(text: str) -> list[str]:
    """Extraction déterministe des composants Huawei depuis le texte."""
    found = []
    for component, patterns in COMPONENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(component)
                break  # évite les doublons par composant
    return found


# ─────────────────────────────────────────────
# FAULT DOMAIN — couche orthogonale au kpi_category
# ─────────────────────────────────────────────
# kpi_category répond à "quel KPI/indicateur de performance est dégradé ?"
# (accessibility, user_experience, ...) — c'est une dimension STATISTIQUE,
# alignée sur le schéma Neo4j (:KPICategory) et utilisée par le CBR
# (agent_memory.py fingerprint/dédup) et par query_knowledge_graph (Cypher).
# On ne touche PAS à cet enum : il a des dépendances réelles en aval
# (Neo4j déjà ingéré, SQLite déjà peuplé avec 89+ cas, eval scripts).
#
# fault_domain répond à une question différente et complémentaire :
# "quelle est la NATURE du symptôme décrit ?" (panne matérielle/logicielle,
# dégradation de performance, problème de signalisation...). Cette
# dimension n'existe nulle part ailleurs dans le pipeline — elle sert
# uniquement à enrichir la query de retrieval (cf. agent_retrieval.py
# _rule_based_reformulation) sans jamais écraser le texte original de
# l'utilisateur. Mots-clés volontairement courts et ciblés (pas de mots
# génériques comme "failure"/"and"/"cell" — cf. bug _extract_alarm_keywords).
FAULT_DOMAIN_PATTERNS = {
    "hardware_stability": [
        r"\brestart(?:ed|ing|s)?\b", r"\breboot(?:ed|ing|s)?\b", r"\bcrash(?:ed|ing|es)?\b",
        r"\bwatchdog\b", r"\bmemory leak\b", r"\bhang(?:ing)?\b", r"\bfreeze[sd]?\b",
        r"\bboard failure\b", r"\bhardware fault\b",
    ],
    "optical_link": [
        r"\bcpri\b", r"\boptical\b", r"\bsfp\b", r"\bfib(?:er|re)\b",
    ],
    "signaling": [
        r"\brrc\b", r"\be-?rab\b", r"\bs1ap\b", r"\bx2ap\b", r"\bsctp\b",
        r"\bhandover\b",
    ],
    "performance": [
        r"\bthroughput\b", r"\blatency\b", r"\bcongestion\b", r"\bprb\b",
        r"\bbler\b", r"\bpacket loss\b",
    ],
    "radio_interference": [
        r"\binterferen[ck]e\b", r"\bsinr\b", r"\brsrp\b", r"\brsrq\b", r"\bnoise power\b",
    ],
}


def _detect_fault_domain(text: str) -> str:
    """
    Extraction déterministe de la NATURE du symptôme (orthogonale à
    kpi_category — voir commentaire ci-dessus). Retourne le premier
    domaine qui matche, par ordre de spécificité décroissante (hardware
    avant performance, qui est plus générique). "unknown" si aucun match —
    auquel cas le query builder retombe sur le comportement actuel.
    """
    for domain, patterns in FAULT_DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return domain
    return "unknown"


# Severity order
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}

# Technology detection patterns
TECH_PATTERNS = {
    "5g": [r"\b5g\b", r"\bgnb\b", r"\bnr\b", r"\bmmwave\b", r"\bsub-6\b"],
    "4g": [r"\b4g\b", r"\blte\b", r"\benb\b", r"\benodeb\b", r"\bepdc\b"],
    "3g": [r"\b3g\b", r"\bumts\b", r"\bwcdma\b", r"\bnodeb\b"],
}


# ─────────────────────────────────────────────
# SYMPTOM CONTEXT TYPE
# ─────────────────────────────────────────────

class SymptomContext(TypedDict):
    symptom_type       : str
    technology         : str
    severity           : str
    kpi_category       : Optional[str]
    fault_domain       : str   
    affected_cells     : list[str]
    affected_components: list[str]
    top_kpis           : list[str]  
    summary            : str
    raw_anomalies      : Optional[list]
    original_query     : Optional[str]
    route              : str


# ─────────────────────────────────────────────
# TECHNOLOGY DETECTION
# ─────────────────────────────────────────────

def _detect_technology(text: str) -> str:
    """Detect technology (4G/5G/3G) from text."""
    text_lower = text.lower()
    for tech, patterns in TECH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return tech
    # 4G default for ML anomalies (current dataset is 4G LTE)
    return "4g"


# ─────────────────────────────────────────────
# ML ANOMALY PROCESSING
# ─────────────────────────────────────────────

KPI_CATEGORY_TO_COMPONENTS = {
    "accessibility"     : ["BBU", "S1_interface"],
    "user_experience"   : ["BBU", "RRU"],
    "radio_quality"     : ["RRU", "AAU", "antenna"],
    "resource_pressure" : ["BBU"],
    "traffic_load"      : ["BBU", "transmission"],
    "radio_propagation" : ["AAU", "antenna", "RRU"],
}

def _process_ml_anomalies(anomalies: pd.DataFrame | list[dict]) -> SymptomContext:
    """
    Process ML anomaly input from anomalies_kg.pkl.

    Accepts:
      - Full DataFrame (filtered externally)
      - List of dicts (single or multiple anomalies)
      - Single dict
    """
    # Normalize to DataFrame
    if isinstance(anomalies, dict):
        df = pd.DataFrame([anomalies])
    elif isinstance(anomalies, list):
        df = pd.DataFrame(anomalies)
    else:
        df = anomalies.copy()

    if df.empty:
        logger.warning("[symptom_agent] No anomalies in input")
        return _empty_context()

    # ── Severity — take worst ────────────────
    severity_counts = df["severity"].value_counts().to_dict() if "severity" in df.columns else {}
    dominant_severity = max(severity_counts, key=lambda s: SEVERITY_ORDER.get(s, 0)) \
                        if severity_counts else "unknown"

    # ── KPI category — most frequent ────────
    kpi_category = None
    if "kpi_category" in df.columns:
        kpi_category = str(df["kpi_category"].value_counts().index[0])

    # ── Affected cells ───────────────────────
    affected_cells = []
    if "cell" in df.columns:
        affected_cells = df["cell"].unique().tolist()[:20]  # cap at 20

    # ── Top KPIs ─────────────────────────────
    top_kpis = []
    if "kpi" in df.columns:
        top_kpis = df["kpi"].value_counts().head(5).index.tolist()

    # ── Summary ──────────────────────────────
    n_anomalies  = len(df)
    n_cells      = len(affected_cells)
    cat_label    = KPI_CATEGORY_LABELS.get(kpi_category, kpi_category or "unknown")
    kpis_str     = ", ".join(top_kpis[:3])

    summary = (
        f"ML detected {n_anomalies} anomalies across {n_cells} 4G LTE cells. "
        f"Dominant category: {cat_label}. "
        f"Severity: {dominant_severity}. "
        f"Top affected KPIs: {kpis_str}."
    )

    logger.info(f"[symptom_agent] ML anomaly processed: {summary}")
    components = KPI_CATEGORY_TO_COMPONENTS.get(kpi_category, [])

    return SymptomContext(
        symptom_type        = "ml_anomaly",
        technology          = "4g",
        severity            = dominant_severity,
        kpi_category        = kpi_category,
        fault_domain        = "unknown",  # n/a — fault_domain is text-symptom based, ML anomalies have no free text
        affected_cells      = affected_cells,
        affected_components = components,
        top_kpis            = top_kpis,  # FIX K — was computed but never returned
        summary             = summary,
        raw_anomalies       = df.head(50).to_dict(orient="records"),  # top 50 for agent
        original_query      = None,
        route               = "rc_analysis",
    )


# ─────────────────────────────────────────────
# DETERMINISTIC FALLBACK HELPERS
# ─────────────────────────────────────────────

_KPI_KEYWORDS = {
    "accessibility"     : ["e-rab", "erab", "setup fail", "rrc", "attach"],
    "user_experience"   : ["rbler", "throughput", "latency", "dl", "ul", "packet loss"],
    "radio_quality"     : ["cqi", "sinr", "interference", "bler", "rsrp", "rsrq"],
    "resource_pressure" : ["prb", "tti", "congestion", "load"],
    "traffic_load"      : ["volume", "traffic", "spike"],
}

_SEVERITY_KEYWORDS = {
    "critical": ["critical", "failure", "fail", "down", "outage"],
    "high"    : ["high", "degradation", "alarm", "alm-"],
    "medium"  : ["medium", "warning", "issue"],
}


def _kg_fallback_lookup(keywords: list[str]) -> dict | None:
    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (a:OSSAlarm)
                WHERE any(kw IN $keywords
                      WHERE toLower(a.name) CONTAINS toLower(kw))
                OPTIONAL MATCH (a)-[:BELONGS_TO]->(rc:RootCause)
                              <-[:HAS_ROOT_CAUSE]-(kc:KPICategory)
                RETURN
                    coalesce(kc.category,        'user_experience') AS kpi_category,
                    coalesce(a.dominant_severity, 'high')           AS severity
                LIMIT 1
            """, keywords=keywords)

            record = result.single()
            if not record:
                return None

            return {
                "severity"             : record["severity"],
                "kpi_category"         : record["kpi_category"],
                "additional_components": [],
                "symptom_summary"      : " ".join(keywords),
            }
    except Exception as e:
        logger.warning(f"[symptom_agent] KG fallback failed: {e}")
        return None


def _deterministic_fallback(query: str) -> dict:
    # ── 1. Extraire alarm IDs ou mots-clés significatifs ────────
    alarm_ids = re.findall(r"ALM-\d{5}", query)
    keywords  = alarm_ids if alarm_ids else [
        w for w in re.findall(r"\b[A-Za-z]{4,}\b", query)
        if w.lower() not in {"cause", "solution", "about", "what", "with",
                             "failure", "high", "from", "that", "this"}
    ][:5]

    # ── 2. KG lookup ─────────────────────────────────────────────
    if keywords:
        kg_result = _kg_fallback_lookup(keywords)
        if kg_result:
            logger.info(f"[symptom_agent] Fallback via KG → {kg_result}")
            return kg_result

    # ── 3. Défauts raisonnables ──────────────────────────────────
    logger.warning("[symptom_agent] KG fallback empty → safe defaults")
    return {
        "severity"             : "high",
        "kpi_category"         : "user_experience",
        "additional_components": [],
        "symptom_summary"      : query,
    }


# ─────────────────────────────────────────────
# USER QUERY PROCESSING
# ─────────────────────────────────────────────

def _process_user_query(query: str) -> SymptomContext:
    technology = _detect_technology(query)

    # ── Étape 1 : extraction déterministe (fiable) ───────────────
    alarm_ids    = re.findall(r"ALM-\d{5}", query)
    components   = _extract_components_deterministic(query)
    fault_domain = _detect_fault_domain(query)

    # ── Étape 2 : LLM pour ce qu'on ne peut pas extraire ────────
    prompt = f"""You are a Huawei RAN expert.
Analyze this technical query and extract structured information.

Query: "{query}"
Already extracted alarm IDs: {alarm_ids}
Already extracted components: {components}

Respond ONLY with this JSON (no markdown, no explanation):
{{
  "severity": "critical|high|medium|low|unknown",
  "kpi_category": "accessibility|user_experience|radio_quality|resource_pressure|traffic_load|radio_propagation|unknown",
  "additional_components": [],
  "symptom_summary": "one sentence describing the symptom"
}}"""

    extracted = _call_llm_json(prompt)

    # ── Fallback déterministe si LLM fail ────────────────────────
    if not extracted:
        logger.warning("[symptom_agent] LLM failed → using deterministic fallback")
        extracted = _deterministic_fallback(query)

    severity   = extracted.get("severity", "unknown")
    kpi_cat    = extracted.get("kpi_category", "unknown")
    summary    = extracted.get("symptom_summary", query)

    # Fusionner composants déterministes + LLM (sans doublons)
    llm_components = extracted.get("additional_components", [])
    all_components = list(dict.fromkeys(components + llm_components))

    # Enrichir le summary avec les alarm IDs
    if alarm_ids:
        summary = f"[{', '.join(alarm_ids)}] {summary}"
    top_kpis: list[str] = []

    # Priority 1 — explicit batch marker in the prompt text
    batch_kpi_match = re.search(r"Dominant KPI category:\s*(\S+)", query)
    if batch_kpi_match:
        dominant_kpi_token = batch_kpi_match.group(1).strip().rstrip(".,")
        if dominant_kpi_token:
            top_kpis.append(dominant_kpi_token)
            logger.debug(f"[symptom_agent] FIX K+: top_kpis from batch prompt → {top_kpis}")

    # Priority 2 — LLM-identified kpi_category (plain user_query path)
    if not top_kpis and kpi_cat and kpi_cat not in ("unknown", ""):
        top_kpis.append(kpi_cat)
        logger.debug(f"[symptom_agent] FIX K+: top_kpis from kpi_cat fallback → {top_kpis}")

    return SymptomContext(
        symptom_type        = "user_query",
        technology          = technology,
        severity            = severity,
        kpi_category        = kpi_cat,
        fault_domain        = fault_domain,
        affected_cells      = [],
        affected_components = all_components,
        top_kpis            = top_kpis,  # FIX K+ — populated for batch + fallback to kpi_cat
        summary             = summary,
        raw_anomalies       = None,
        original_query      = query,
        route               = "rc_analysis",
    )


# ─────────────────────────────────────────────
# LLM JSON CALL
# ─────────────────────────────────────────────

def _call_llm_json(prompt: str) -> dict:
    """
    Wrapper JSON — appelle Groq (cloud) en priorité pour éviter le cold start
    Ollama (~19s). Fallback vers Ollama local si Groq échoue.
    """
    import json

    _SYSTEM = (
        "You are a Huawei RAN symptom classifier. "
        "Respond ONLY with valid JSON — no explanation, no markdown."
    )

    # ── Tentative 1 : Cloud 8B (fast model) ──────────────────────────
    # OPT — utilise SYMPTOM_CLOUD_MODEL 
    # par défaut du provider (70B). La classification de symptôme est un
    # JSON court (~10 champs) — le 8B est amplement suffisant et réduit
    # Node 1 de ~14s à ~2-3s.
    # Via call_cloud_llm_messages_raw avec le message system+user et model=8B.
    try:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        text = _call_cloud_messages_raw(
            messages   = messages,
            model      = SYMPTOM_CLOUD_MODEL,  
        )
        if text:
            text = re.sub(r"```json|```", "", text).strip()
            result = json.loads(text)
            logger.info(f"[symptom_agent] Cloud call succeeded (model={SYMPTOM_CLOUD_MODEL})")
            return result
    except Exception as e:
        logger.warning(f"[symptom_agent] Cloud call failed, falling back to local Ollama: {e}")

    # ── Fallback : Ollama local ────────────────────────────────────────
    for attempt in range(2):
        text = _call_llm_raw(prompt, max_tokens=300, caller="symptom_agent", model=LLM_MODEL_FAST)
        if not text:
            logger.warning(f"[symptom_agent] Local LLM call failed (attempt {attempt + 1}/2)")
            continue
        try:
            text = re.sub(r"```json|```", "", text).strip()
            return json.loads(text)
        except Exception as e:
            logger.warning(f"[symptom_agent] LLM JSON parse failed: {e}")
            break

    return {}


# ─────────────────────────────────────────────
# EMPTY CONTEXT
# ─────────────────────────────────────────────

def _empty_context() -> SymptomContext:
    # FIX: added missing affected_components key — previously caused KeyError downstream
    return SymptomContext(
        symptom_type        = "unknown",
        technology          = "unknown",
        severity            = "unknown",
        kpi_category        = None,
        fault_domain        = "unknown",
        affected_cells      = [],
        affected_components = [],   # FIX: was absent, causing TypedDict crash
        top_kpis            = [],   # FIX K
        summary             = "No symptom could be identified.",
        raw_anomalies       = None,
        original_query      = None,
        route               = "rc_analysis",
    )


# ── Mapping intent → symptom_type ───────────────────────
_INTENT_TO_SYMPTOM_TYPE: dict[str, str] = {
    "rca"           : "user_query",        # pipeline complet
    "definition"    : "definition",        # pas de RCA
    "chitchat"      : "chitchat",          # pas de RCA
    "recommendation": "recommendation",   # pas de RCA
}

# ── Réponse rapide pour les intents non-RCA ──────────────
_NON_RCA_REPLIES: dict[str, str] = {
    "chitchat"      : "Bonjour ! Je suis votre assistant RAN AIOps. "
                      "Posez-moi une question sur vos alarmes LTE, "
                      "dégradations KPI ou équipements Huawei.",
    "definition"    : None,   # géré par le LLM (mode léger)
    "recommendation": None,   # géré par le LLM (mode léger)
}

def _enrich_context_with_ml(context: dict, ml_context: dict) -> dict:
    enriched = dict(context)

    if "kpi_category" in ml_context and ml_context["kpi_category"]:
        enriched["kpi_category"] = ml_context["kpi_category"]
    if "severity" in ml_context and ml_context["severity"]:
        enriched["severity"] = ml_context["severity"]

    cell      = ml_context.get("cell", "")
    z_score   = ml_context.get("max_z_score", 0)
    anomalies = ml_context.get("anomalies", 0)
    enodeb    = ml_context.get("enodeb", "")
    breakdown = ml_context.get("kpi_breakdown", [])

    if cell and cell not in enriched.get("affected_cells", []):
        enriched["affected_cells"] = enriched.get("affected_cells", []) + [cell]

    kpi_names = [item.get("kpi","") for item in breakdown[:3] if item.get("kpi")]
    if kpi_names:
        enriched["top_kpis"] = kpi_names + enriched.get("top_kpis", [])

    extra = []
    if z_score:   extra.append(f"max Z-score={z_score:.1f}")
    if anomalies: extra.append(f"{anomalies} anomalies")
    if enodeb:    extra.append(f"eNodeB={enodeb}")
    if kpi_names: extra.append("top KPIs: " + ", ".join(kpi_names))
    if extra:
        enriched["summary"] = enriched.get("summary","") + " | " + " | ".join(extra)

    # Filtrer uniquement les clés valides du TypedDict
    valid_keys = {
        "symptom_type","technology","severity","kpi_category",
        "fault_domain","affected_cells","affected_components",
        "top_kpis","summary","raw_anomalies","original_query","route"
    }
    return SymptomContext(**{k: v for k, v in enriched.items() if k in valid_keys})


def symptom_identification_agent(state: dict) -> dict:
    import logging
    logger = logging.getLogger(__name__)

    ml_anomalies = state.get("ml_anomalies")
    user_query   = state.get("user_query")
    ml_context   = state.get("ml_context")
    intent_hint  = state.get("intent_hint")

    # ── CAS 1 : DataFrame ML présent ────────────────────────────────────
    if ml_anomalies is not None:
        logger.info("[symptom_agent] Processing ML anomaly input")
        context = _process_ml_anomalies(ml_anomalies)
        if ml_context:
            context = _enrich_context_with_ml(context, ml_context)
        return {**state, "symptom_context": context}

    # ── CAS 2 : ml_context seul (DataFrame absent/vide) ─────────────────
    if ml_context and ml_context.get("cell"):
        logger.info("[symptom_agent] ml_context only → forcing ml_anomaly")
        kpi_cat   = ml_context.get("kpi_category", "")
        severity  = ml_context.get("severity", "high")
        cell      = ml_context.get("cell", "")
        z_score   = ml_context.get("max_z_score", 0)
        anomalies = ml_context.get("anomalies", 0)
        enodeb    = ml_context.get("enodeb", "")
        breakdown = ml_context.get("kpi_breakdown", [])
        kpi_names = [item.get("kpi","") for item in breakdown[:3] if item.get("kpi")]
        components = KPI_CATEGORY_TO_COMPONENTS.get(kpi_cat, [])
        summary = (
            f"ML detected {anomalies} anomalies on cell {cell}"
            + (f" (eNodeB: {enodeb})" if enodeb else "") + ". "
            f"Category: {KPI_CATEGORY_LABELS.get(kpi_cat, kpi_cat)}. "
            f"Severity: {severity}. Max Z-score: {z_score:.1f}."
            + (f" Top KPIs: {', '.join(kpi_names)}." if kpi_names else "")
        )
        context = SymptomContext(
            symptom_type        = "ml_anomaly",
            technology          = "4g",
            severity            = severity,
            kpi_category        = kpi_cat,
            fault_domain        = "performance",
            affected_cells      = [cell] if cell else [],
            affected_components = components,
            top_kpis            = kpi_names,
            summary             = summary,
            raw_anomalies       = None,
            original_query      = user_query or f"Investigate {cell}",
            route               = "rc_analysis",
        )
        return {**state, "symptom_context": context}

    # ── CAS 3 : User query pure ──────────────────────────────────────────
    if not user_query:
        logger.error("[symptom_agent] No input provided")
        return {**state, "symptom_context": _empty_context()}

    logger.info(f"[symptom_agent] Processing user query: '{user_query}'")

    try:
        from agents.intent_router import route_intent
        if intent_hint and intent_hint in _INTENT_TO_SYMPTOM_TYPE:
            intent      = intent_hint
            intent_conf = 1.0
        else:
            decision    = route_intent(user_query, state.get("conversation_history") or [])
            intent      = decision["intent"]
            intent_conf = decision["confidence"]
    except Exception as e:
        logger.warning(f"[symptom_agent] Intent router failed ({e}) → defaulting to rca")
        intent = "rca"

    NON_RCA_INTENTS = {"chitchat", "definition", "recommendation"}
    if intent in NON_RCA_INTENTS:
        context = _empty_context()
        context["symptom_type"]   = _INTENT_TO_SYMPTOM_TYPE[intent]
        context["original_query"] = user_query
        context["severity"]       = "low"
        context["summary"]        = f"[{intent}] {user_query}"
        return {**state, "symptom_context": context}

    context = _process_user_query(user_query)
    if ml_context:
        context = _enrich_context_with_ml(context, ml_context)
    return {**state, "symptom_context": context}


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    import json
    logging.basicConfig(level=logging.INFO)

    print("\n" + "═"*60)
    print("TEST 1 — ML Anomaly Input")
    print("═"*60)

    with open("ml/artifacts/anomalies_kg.pkl", "rb") as f:
        df_all = pickle.load(f)

    df_critical = df_all[df_all["severity"] == "critical"].head(100)

    state1 = {"ml_anomalies": df_critical}
    result1 = symptom_identification_agent(state1)
    ctx1 = result1["symptom_context"]

    print(f"Symptom type  : {ctx1['symptom_type']}")
    print(f"Technology    : {ctx1['technology']}")
    print(f"Severity      : {ctx1['severity']}")
    print(f"KPI category  : {ctx1['kpi_category']}")
    print(f"Cells affected: {len(ctx1['affected_cells'])}")
    print(f"Components    : {ctx1['affected_components']}")
    print(f"Top KPIs      : {ctx1['top_kpis']}")
    print(f"Summary       : {ctx1['summary']}")
    print(f"Route         : {ctx1['route']}")

    print("\n" + "═"*60)
    print("TEST 2 — User Query Input (plain)")
    print("═"*60)

    state2 = {"user_query": "DL throughput degradation multiple cells LTE"}
    result2 = symptom_identification_agent(state2)
    ctx2 = result2["symptom_context"]

    print(f"Symptom type  : {ctx2['symptom_type']}")
    print(f"Technology    : {ctx2['technology']}")
    print(f"Severity      : {ctx2['severity']}")
    print(f"KPI category  : {ctx2['kpi_category']}")
    print(f"Top KPIs      : {ctx2['top_kpis']}")
    print(f"Summary       : {ctx2['summary']}")
    print(f"Route         : {ctx2['route']}")

    print("\n" + "═"*60)
    print("TEST 3 — Batch cell-case prompt (FIX K+)")
    print("═"*60)

    batch_query = (
        "RAN cell-level anomaly batch analysis.\n"
        "Cell: EBF_6751_C01_318_j3 (eNodeB: EBF_6751_LM)\n"
        "Anomaly count (critical+high window): 370\n"
        "Dominant KPI category: resource_pressure\n"
        "Max Z-score observed: 961.97\n"
        "Task: identify the most likely root cause for this cell "
        "and recommend corrective actions."
    )
    state3 = {"user_query": batch_query}
    result3 = symptom_identification_agent(state3)
    ctx3 = result3["symptom_context"]

    print(f"Symptom type  : {ctx3['symptom_type']}")
    print(f"KPI category  : {ctx3['kpi_category']}")
    print(f"Top KPIs      : {ctx3['top_kpis']}")
    print(f"Expected top_kpis: ['resource_pressure']  ← FIX K+ must populate this")

    print("\n" + "═"*60)
    print("TEST 4 — Empty context (error path)")
    print("═"*60)
    state4  = {}
    result4 = symptom_identification_agent(state4)
    ctx4    = result4["symptom_context"]
    print(f"affected_components present: {'affected_components' in ctx4}")
    print(f"top_kpis present           : {'top_kpis' in ctx4}")
    print(f"symptom_type               : {ctx4['symptom_type']}")

    print("\n" + "═"*60)
    print("TEST 5 — FIX K+ regex unit test (no LLM needed)")
    print("═"*60)

    test_cases = [
        ("Dominant KPI category: resource_pressure\nMax Z-score: 961", "resource_pressure"),
        ("Dominant KPI category: accessibility\nCell: XYZ",            "accessibility"),
        ("Dominant KPI category: radio_quality.",                       "radio_quality"),
        ("No batch marker here — plain user query",                     None),
    ]
    all_passed = True
    for query_text, expected in test_cases:
        m = re.search(r"Dominant KPI category:\s*(\S+)", query_text)
        got = m.group(1).strip().rstrip(".,") if m else None
        status = "✓" if got == expected else "✗"
        if got != expected:
            all_passed = False
        print(f"  {status}  input={query_text[:50]!r:<52} → got={got!r} expected={expected!r}")
    print(f"\nAll regex tests passed: {all_passed}")