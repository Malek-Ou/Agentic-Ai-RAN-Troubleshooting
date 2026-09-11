import logging
import re
import json
import httpx
import pandas as pd
from typing import TypedDict, Optional
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from config import (
    OLLAMA_CHAT_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_CELLS_N,
    RCA_CLOUD_MODEL,
)
from rag.retrieval import format_context
from utils.llm_cloud import build_rca_llm_caller
from agents.agent_memory import format_similar_cases


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Max tokens cap in web_primary mode — the JSON schema fits comfortably in
# 350 tokens (root_cause ~80t, explanation ~100t, evidence+actions ~150t,
# key_terms ~20t). Using the full LLM_MAX_TOKENS (512) when local RAG is weak
# causes the 70B to pad its output → +12s of unnecessary generation time.
_WEB_PRIMARY_MAX_TOKENS = int(os.getenv("RCA_WEB_PRIMARY_MAX_TOKENS", "500"))


# ─────────────────────────────────────────────
# OUTPUT TYPE
# ─────────────────────────────────────────────

class RCAResult(TypedDict):
    root_cause             : str
    root_cause_explanation : str        # causal chain in prose, no scores
    confidence             : float
    evidence               : dict       # {rag_sources, kg_paths, evidence_summary, evidence_items, cbr_cases_used}
                                        # evidence_items: [{text, source_type, source_name, url?}]
    affected_scope         : dict       # {global_pattern, worst_cells, enodebs}
    actions                : list[dict] # [{priority, action, source}] — source mandatory
    recommendations        : list[str]  # kept for backward compat
    next_actions           : list[str]  # kept for backward compat
    key_terms              : dict       # optional glossary {TERM: definition}
    sources_used           : list[str]  # ["qdrant", "neo4j", ...]
    retrieval_strategy     : str        # "deterministic" | "react"


# ─────────────────────────────────────────────
# STEP 1 — SCOPE ANALYSIS
# ─────────────────────────────────────────────

def _analyze_worst_cells(symptom_context: dict) -> dict:
    """
    Analyse globale + zoom sur worst cells.
    Option C : global pattern + top critical cells.
    """
    affected_cells = symptom_context.get("affected_cells", [])
    raw_anomalies  = symptom_context.get("raw_anomalies", [])

    if not raw_anomalies:
        return {
            "global_pattern" : False,
            "worst_cells"    : affected_cells[:TOP_CELLS_N],
            "total_cells"    : len(affected_cells),
            "scope_summary"  : "No ML anomaly data available.",
        }

    df = pd.DataFrame(raw_anomalies)

    required_cols = {"cell", "severity"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.warning(f"[rca_agent] _analyze_worst_cells: missing columns {missing}")
        return {
            "global_pattern" : False,
            "worst_cells"    : affected_cells[:TOP_CELLS_N],
            "total_cells"    : len(affected_cells),
            "scope_summary"  : f"Incomplete anomaly data (missing: {missing}).",
        }

    n_cells        = df["cell"].nunique()
    n_enodebs      = df["enodeb_id"].nunique() if "enodeb_id" in df.columns else 0
    global_pattern = n_cells > 10

    critical_df = df[df["severity"].isin(["critical", "high"])]
    worst_cells = (
        critical_df.groupby("cell")
        .size()
        .sort_values(ascending=False)
        .head(TOP_CELLS_N)
        .index.tolist()
    )

    worst_cell_details = []
    if worst_cells and "kpi" in df.columns:
        cell_groups = df[df["cell"].isin(worst_cells)].groupby("cell")
        for cell in worst_cells:
            if cell not in cell_groups.groups:
                continue
            cell_df = cell_groups.get_group(cell)
            top_kpi = cell_df["kpi"].value_counts().index[0] if len(cell_df) > 0 else "unknown"
            worst_cell_details.append({
                "cell"       : str(cell),
                "n_anomalies": int(len(cell_df)),
                "top_kpi"    : str(top_kpi),
            })

    scope_summary = (
        f"{'Network-wide' if global_pattern else 'Localized'} issue: "
        f"{n_cells} cells affected across {n_enodebs} eNodeBs. "
        f"Top {len(worst_cells)} critical cells identified."
    )

    return {
        "global_pattern"     : global_pattern,
        "worst_cells"        : [str(c) for c in worst_cells],
        "worst_cell_details" : worst_cell_details,
        "total_cells"        : n_cells,
        "total_enodebs"      : n_enodebs,
        "scope_summary"      : scope_summary,
    }


# ─────────────────────────────────────────────
# STEP 2 — KG ↔ RAG CORRELATION
# ─────────────────────────────────────────────

def _correlate_kg_rag(
    kg_result  : dict,
    rag_results: list[dict],
) -> dict:

    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "_", text.lower()).strip("_")

    correlated  = []
    kg_matched  = set()
    rag_matched = set()

    alarms = kg_result.get("related_alarms", [])

    raw_scores  = [r.get("rerank_score", r.get("score", 0)) for r in rag_results]
    max_score   = max(raw_scores) if raw_scores else 1.0
    min_score   = min(raw_scores) if raw_scores else 0.0
    score_range = max_score - min_score

    def _norm_score(s: float) -> float:
        if score_range == 0:
            return 0.5
        return (s - min_score) / score_range

    # Pass 1 : alarm ID + alarm name match
    for alarm in alarms:
        alarm_id_raw = alarm.get("alarm", "")
        alarm_name   = alarm.get("alarm_name", "")
        rc_label     = alarm.get("root_cause_label", "")
        prior        = float(alarm.get("prior_prob", 0.5) or 0.5)

        if not alarm_id_raw:
            alarm_id_raw = alarm_name or f"_anon_{id(alarm)}"

        search_terms = set(filter(None, [
            _norm(alarm_id_raw),
            _norm(alarm_name),
            _norm(rc_label),
            *[w for w in _norm(alarm_name).split("_") if len(w) >= 4],
        ]))

        best_match = None
        best_score = -1.0

        for i, chunk in enumerate(rag_results):
            text_norm    = _norm(chunk.get("text", chunk.get("content", "")))
            matched_term = None

            for term in search_terms:
                if term and len(term) >= 4 and term in text_norm:
                    matched_term = term
                    break

            if matched_term:
                raw         = chunk.get("rerank_score", chunk.get("score", 0))
                cross_score = round(prior * _norm_score(raw), 4)
                if cross_score > best_score:
                    best_score = cross_score
                    best_match = {
                        "alarm_id"     : alarm_id_raw,
                        "alarm_name"   : alarm_name,
                        "root_cause"   : rc_label,
                        "prior_prob"   : prior,
                        "matched_term" : matched_term,
                        "chunk_source" : chunk.get("source", "unknown"),
                        "chunk_text"   : chunk.get("text", chunk.get("content", ""))[:300],
                        "rerank_score" : round(raw, 4),
                        "cross_score"  : cross_score,
                        "match_type"   : "normalized",
                        "chunk_index"  : i,
                    }

        if best_match:
            correlated.append(best_match)
            kg_matched.add(alarm_id_raw)
            rag_matched.add(best_match["chunk_index"])

    # Pass 2 : category match pour les alarmes non matchées
    unmatched = [a for a in alarms if a.get("alarm", "") not in kg_matched]

    for alarm in unmatched:
        category = alarm.get("category", "")
        alarm_id = alarm.get("alarm", "")
        prior    = float(alarm.get("prior_prob", 0.5) or 0.5)

        if not category or len(category) < 2:
            continue

        cat_norm = _norm(category)

        for i, chunk in enumerate(rag_results):
            if i in rag_matched:
                continue

            text_norm = _norm(chunk.get("text", chunk.get("content", "")))
            if cat_norm in text_norm:
                raw         = chunk.get("rerank_score", chunk.get("score", 0))
                norm        = _norm_score(raw)
                cross_score = round(prior * norm * 0.7, 4)

                correlated.append({
                    "alarm_id"     : alarm_id,
                    "alarm_name"   : alarm.get("alarm_name", category),
                    "root_cause"   : alarm.get("root_cause_label", category),
                    "prior_prob"   : prior,
                    "matched_term" : cat_norm,
                    "chunk_source" : chunk.get("source", "unknown"),
                    "chunk_text"   : chunk.get("text", chunk.get("content", ""))[:300],
                    "rerank_score" : round(raw, 4),
                    "cross_score"  : cross_score,
                    "match_type"   : "category",
                })
                kg_matched.add(alarm_id)
                rag_matched.add(i)
                break

    correlated.sort(key=lambda x: x["cross_score"], reverse=True)

    def _alarm_key(a: dict) -> str:
        raw = a.get("alarm", "")
        return raw if raw else (a.get("alarm_name") or f"_anon_{id(a)}")

    kg_only  = [a for a in alarms if _alarm_key(a) not in kg_matched]
    rag_only = [rag_results[i] for i in range(len(rag_results)) if i not in rag_matched]

    correlation_score = round(len(kg_matched) / len(alarms), 2) if alarms else 0.0

    if correlated:
        lines = [
            f"KG-RAG cross-correlation: {len(correlated)} match(es) "
            f"out of {len(alarms)} KG alarms (score={correlation_score:.0%}).",
        ]
        for c in correlated[:3]:
            lines.append(
                f"  • [{c['match_type']}] '{c['alarm_name']}' / '{c['root_cause']}' "
                f"matched via '{c['matched_term']}' in '{c['chunk_source']}' "
                f"(prior={c['prior_prob']:.2f}, cross={c['cross_score']:.3f})"
            )
        if kg_only:
            lines.append(
                "  KG-only (no RAG doc): "
                + ", ".join(a.get("alarm_name", a.get("alarm", "")) for a in kg_only[:3])
            )
        summary = "\n".join(lines)
    else:
        summary = (
            f"No KG-RAG correlation found. "
            f"RAG: {len(rag_results)} chunks | KG: {len(alarms)} alarms. "
            f"Sources independent — lower diagnostic confidence."
        )

    logger.info(
        f"[rca_agent] KG↔RAG correlation: "
        f"{len(correlated)} matches | score={correlation_score:.0%} | "
        f"kg_unmatched={len(kg_only)} | rag_unmatched={len(rag_only)}"
    )

    return {
        "correlated"        : correlated,
        "kg_only"           : kg_only,
        "rag_only"          : [{"source": r.get("source"), "score": r.get("rerank_score")}
                                for r in rag_only[:5]],
        "correlation_score" : correlation_score,
        "summary"           : summary,
    }


# ─────────────────────────────────────────────
# STEP 3 — LLM FUSION REASONING
# ─────────────────────────────────────────────

def _strip_markdown_artifacts(text: str) -> str:
    """
    Defense-in-depth — guarantees root_cause stays plain prose even if the
    LLM ignores the system prompt's "no markdown" instruction.
    Also protects RAGAS faithfulness scoring from formatting noise.
    """
    if not text:
        return text
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^[\-\*•]\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _llm_rca_reasoning(
    symptom_context : dict,
    rag_results     : list[dict],
    kg_result       : dict,
    correlation     : dict,
    scope           : dict,
    reformulated_q  : str,
    similar_cases   : list[dict] | None = None,
) -> dict:
    """
    LLM synthesizes RAG + KG evidence into structured RCA output.

    v4 changes vs v3:
      - max_tokens_override: caps token output at _WEB_PRIMARY_MAX_TOKENS
        (default 350) when _local_rag_weak=True, avoiding unnecessary 70B
        generation time (+12s) in web_primary mode.
      - _WEB_PRIMARY_MAX_TOKENS configurable via RCA_WEB_PRIMARY_MAX_TOKENS env.
    """

    def _normalize_chunk_keys(chunks: list[dict]) -> list[dict]:
        normalized = []
        for chunk in chunks:
            if "content" in chunk and "text" not in chunk:
                chunk = {**chunk, "text": chunk["content"]}
            normalized.append(chunk)
        return normalized

    # Séparation local vs web
    local_chunks = [r for r in rag_results if r.get("source_type", "local") != "web"]
    web_chunks   = [r for r in rag_results if r.get("source_type") == "web"]

    # ── PATCH v5 : cap chunks locaux ─────────────────────────────────────────
    # gt12 remontait 25 contextes → prompt surdimensionné → 70B génère des
    # statements hors-contexte (faithfulness chute) + judge timeout (NaN).
    # 5 chunks locaux couvrent largement le besoin pour les 4 métriques Ragas.
    _MAX_LOCAL_CHUNKS = int(os.getenv("RCA_MAX_LOCAL_CHUNKS", "5"))
    if len(local_chunks) > _MAX_LOCAL_CHUNKS:
        logger.info(
            f"[rca_agent] Capping local chunks {len(local_chunks)} → {_MAX_LOCAL_CHUNKS} "
            f"(RCA_MAX_LOCAL_CHUNKS={_MAX_LOCAL_CHUNKS})"
        )
        local_chunks = local_chunks[:_MAX_LOCAL_CHUNKS]

    _normalized_local = _normalize_chunk_keys(local_chunks)
    _normalized_web   = _normalize_chunk_keys(web_chunks)

    rag_context = (
        format_context(_normalized_local[:2])
        if _normalized_local
        else "No relevant local documentation found."
    )

    if _normalized_web:
        web_lines = []
        for i, chunk in enumerate(_normalized_web[:2], 1):
            vendor      = chunk.get("vendor", "Unknown")
            official    = "official" if chunk.get("official") else "community"
            quality     = chunk.get("quality_score", 0.0)
            trust_score = chunk.get("trust_score", 0.0)
            url         = chunk.get("source", chunk.get("url", ""))
            domain      = re.sub(r"^https?://(www\.)?", "", url).split("/")[0] if url else "unknown"
            content     = chunk.get("text", chunk.get("content", ""))[:400]
            web_lines.append(
                f"[Web {i}] Source: {url}\n"
                f"  Domain: {domain} | Trust score: {trust_score:.2f} | Vendor: {vendor} | Type: {official} | Quality: {quality:.2f}\n"
                f"  Content: {content}"
            )
        web_rag_context = "\n\n".join(web_lines)
    else:
        web_rag_context = None

    # KG context block
    kg_context = "No Knowledge Graph data available."

    if kg_result.get("available"):
        kg_parts = []

        corr_summary = correlation.get("summary", "")
        if corr_summary:
            kg_parts.append(f"=== KG-RAG CORRELATION ===\n{corr_summary}")

        confirmed = correlation.get("correlated", [])[:2]
        if confirmed:
            names = [
                f"{c['alarm_name']} (prior={c['prior_prob']:.2f}, cross={c['cross_score']:.3f})"
                for c in confirmed
            ]
            kg_parts.append("Confirmed alarms: " + " | ".join(names))

        if correlation.get("kg_only"):
            only_ids = ", ".join(
                a.get("alarm_name", a.get("alarm", ""))
                for a in correlation["kg_only"][:2]
            )
            kg_parts.append(f"KG alarms without RAG confirmation: {only_ids}")

        if kg_result.get("kg_paths"):
            kg_parts.append("Causal paths: " + " | ".join(kg_result["kg_paths"][:2]))

        if kg_result.get("affected_enodebs"):
            enodeb_str = ", ".join(
                e.get("enodeb", "") for e in kg_result["affected_enodebs"][:3]
            )
            kg_parts.append(f"Affected eNodeBs: {enodeb_str}")

        if kg_parts:
            kg_context = "\n\n".join(kg_parts)

    scope_str       = scope.get("scope_summary", "")
    symptom_summary = symptom_context.get("summary", "")
    severity        = symptom_context.get("severity", "unknown")
    symptom_type    = symptom_context.get("symptom_type", "unknown")

    # CBR dans user prompt, format compressé
    cbr_context = format_similar_cases(similar_cases or [])
    cbr_section = f"\n\nHISTORICAL CASES:\n{cbr_context}" if cbr_context else ""

    if cbr_context:
        logger.info(
            f"[rca_agent] Injecting {len(similar_cases)} historical CBR cases into user prompt"
        )

    # Modèle RCA — toujours le modèle "heavy" (qualité de raisonnement
    # constante, pas de dégradation sur les cas jugés "simples").
    kg_has_paths    = bool(kg_result.get("kg_paths"))
    use_heavy_model = True
    model_to_use    = LLM_MODEL

    # ── SYSTEM PROMPT ──────────────────────────────────────────────────────
    system_prompt = (
        "You are an expert Huawei RAN (Radio Access Network) engineer specialized in Root Cause Analysis.\n"
        "You analyze network anomalies and faults using documentation evidence and knowledge graph relationships.\n"
        "Be precise, technical, and actionable.\n"
        "\n"
        # ── GROUNDING RULE (v5 — généraliste, défendable) ────────────────────
        # Base toutes les affirmations factuelles sur les preuves fournies
        # (KG/RAG/CBR). Autorise les inférences causales raisonnables à partir
        # de plusieurs preuves convergentes, mais interdit les claims spécifiques
        # (composants, commandes, seuils) absents du contexte.
        # Formulation inspirée des recommandations RAG faithfulness de RAGAS :
        # chaque statement doit être entailé par au moins une source fournie.
        "GROUNDING RULE — CRITICAL:\n"
        "Base all root-cause claims and recommended actions on the KG, RAG, and CBR\n"
        "evidence provided below. You may draw reasonable causal inferences when\n"
        "multiple pieces of evidence converge on the same conclusion, but do NOT\n"
        "introduce specific factual claims — component names, commands, thresholds,\n"
        "procedures — that are unsupported by the provided evidence.\n"
        "When evidence is insufficient to establish a specific root cause, explicitly\n"
        "state the uncertainty in root_cause (e.g. 'Evidence is insufficient to\n"
        "determine the exact root cause; the following hypotheses are supported by\n"
        "available documentation: ...') rather than filling the gap with general\n"
        "knowledge. A calibrated uncertain answer is more valuable than a confident\n"
        "unsupported one.\n"
        "\n"
        "CONFIDENCE CALIBRATION RUBRIC — set 'confidence' strictly according to these tiers:\n"
        "  0.80–1.00 : Strong evidence — KG causal paths confirmed AND ≥2 matching RAG chunks\n"
        "  0.55–0.79 : Moderate evidence — KG alarms OR RAG matches, but not both\n"
        "  0.30–0.54 : Weak evidence — RAG only, no KG paths, or partial RAG match\n"
        "  0.10–0.29 : Very weak — documentation found but query is vague or generic\n"
        "  0.00–0.09 : No evidence — do NOT use this range unless truly no content was retrieved\n"
        "\n"
        "IMPORTANT: Confidence must reflect evidence quality, not answer quality.\n"
        "For user queries with RAG chunks but no KG paths, 0.40–0.60 is expected and correct.\n"
        "\n"
        "NARRATIVE FUSION GUARD: Treat two alarms as causally linked ONLY if the KG evidence\n"
        "explicitly shows a causal path (e.g. 'AlarmA → CAUSES → AlarmB'). If no such path\n"
        "exists, diagnose each alarm independently. Do NOT invent causal chains.\n"
        "\n"
        "WEB SOURCE PRIORITY RULE: When the user prompt contains the tag [LOCAL_RAG_WEAK],\n"
        "it means local documentation did not match the query well. In this case:\n"
        "  - Each web source below carries a Trust score (0.0-1.0). Treat sources with\n"
        "    Trust >= 0.85 (vendor docs, 3GPP, ETSI, GSMA) as PRIMARY evidence, on par with\n"
        "    local documentation — NOT only support.huawei.com. Sources with Trust 0.5-0.84\n"
        "    (community/reference sites) are valid SECONDARY evidence — usable but flagged\n"
        "    as community-sourced in evidence items.\n"
        "  - If multiple web sources are present, use ALL that meaningfully support the\n"
        "    diagnosis, not just the first one — cite each by its domain.\n"
        "  - You MUST incorporate web content into root_cause, root_cause_explanation, and\n"
        "    the evidence items.\n"
        "  - The KG provides terrain alarm context (secondary effects) — use it to enrich,\n"
        "    not override.\n"
        "  - Do NOT ignore web chunks when [LOCAL_RAG_WEAK] is present.\n"
        "\n"
        "FIELD RULES:\n"
        "  root_cause             : 1-2 sentences, plain prose, state WHAT failed and WHY.\n"
        "  root_cause_explanation : 2-3 sentences explaining the causal chain — what triggered\n"
        "                           what, which component, what network impact. Mention the\n"
        "                           source by name (doc title or 'Knowledge Graph') but NO\n"
        "                           numerical scores, no rerank values, no prior probabilities.\n"
        "  evidence               : object with 'summary' (1-2 sentences, plain prose, no\n"
        "                           scores/numbers) and 'items' — a list of EVERY distinct\n"
        "                           source that supports the diagnosis. Each item:\n"
        "                             {\"text\": \"what this source confirms, one sentence\",\n"
        "                              \"source_type\": \"kg\" | \"local_doc\" | \"web\",\n"
        "                              \"source_name\": \"document title, 'Knowledge Graph', or\n"
        "                                              web domain (e.g. 'support.huawei.com')\"}\n"
        "                           Include ONE item per source actually used — do not merge\n"
        "                           multiple sources into a single item. If 2 web sources from\n"
        "                           different domains both contributed, emit 2 separate items.\n"
        "  actions                : list of exactly 3 objects, one per priority\n"
        "                           ('immediate' | 'short_term' | 'preventive'). Each object\n"
        "                           MUST have 'priority', 'action', and 'source'.\n"
        "                           'action' must be a SINGLE but DETAILED instruction: name\n"
        "                           the exact component/parameter/command involved, explain\n"
        "                           briefly HOW to do it and WHY it addresses the root cause —\n"
        "                           not a generic one-liner like 'check the hardware'.\n"
        "                           Example (good): 'Run DSP SFP on the affected RRU's optical\n"
        "                           module to read real-time TX/RX power; CPRI link failures\n"
        "                           are frequently caused by optical attenuation below -7dBm,\n"
        "                           so a sub-threshold reading confirms a physical fault on the\n"
        "                           fiber or module rather than a software issue.'\n"
        "                           Example (bad): 'Check the optical fiber connections.'\n"
        "                           'source' is MANDATORY — cite the doc name, 'Knowledge\n"
        "                           Graph', or the web domain that grounds this action.\n"
        "  key_terms              : optional dict — only include acronyms or Huawei-specific\n"
        "                           terms that appear in root_cause and may be unfamiliar.\n"
        "                           If none needed, return empty dict {}.\n"
        "\n"
        "No section headers, no '**bold**', no bullet points in root_cause or explanation.\n"
        "Respond ONLY with valid JSON. No markdown, no text outside the JSON object."
    )

    # ── USER PROMPT ────────────────────────────────────────────────────────
    evidence_quality = (
        f"EVIDENCE QUALITY: KG_paths={len(kg_result.get('kg_paths', []))} | "
        f"KG_RAG_match={correlation.get('correlation_score', 0.0):.0%} | "
        f"RAG_chunks={len(local_chunks)} | "
        f"WEB_chunks={len(web_chunks)} | "
        f"CBR_cases={len(similar_cases) if similar_cases else 0}"
    )

    # Detect weak local RAG — used to signal the LLM to prioritise web chunks
    _local_top_score = max(
        (r.get("rerank_score", r.get("score", 0)) for r in local_chunks),
        default=0.0,
    )
    _local_rag_weak = web_chunks and _local_top_score < 3.0

    web_section = ""
    if web_rag_context:
        if _local_rag_weak:
            web_priority_note = (
                "[LOCAL_RAG_WEAK] Local documentation did not match this query well.\n"
                "Web sources below are OFFICIAL Huawei documentation — treat as PRIMARY evidence.\n"
                "You MUST use web content in root_cause and root_cause_explanation.\n"
                "KG alarms describe secondary network effects — use them to enrich, not as root cause."
            )
        else:
            web_priority_note = (
                "Web sources are supplementary — prefer local Huawei documentation above.\n"
                "If local RAG and web sources contradict, trust local documentation.\n"
                "Cite web sources by domain name when used (e.g. 'According to support.huawei.com...')."
            )
        web_section = f"""

WEB DOCUMENTATION:
{web_priority_note}
{web_rag_context}"""

    # Signal injected into evidence line so the LLM sees it before reading evidence blocks
    web_chunk_info = ""
    if web_chunks:
        if _local_rag_weak:
            web_chunk_info = (
                f" | WEB_chunks={len(web_chunks)} [LOCAL_RAG_WEAK — web is PRIMARY evidence]"
            )
        else:
            web_chunk_info = f" | WEB_chunks={len(web_chunks)}"

    # ── FIX v4 — max_tokens_override ────────────────────────────────────────
    # In web_primary mode the 70B tends to pad its output because the prompt
    # is richer (web context injected). 350 tokens is sufficient for the full
    # JSON schema and saves ~12s of generation time on average.
    # In local-RAG mode we keep LLM_MAX_TOKENS (512) for full quality.
    max_tokens_override = _WEB_PRIMARY_MAX_TOKENS if _local_rag_weak else 800

    logger.info(
        f"[rca_agent] Web evidence mode: "
        f"{'LOCAL_RAG_WEAK → web PRIMARY' if _local_rag_weak else 'local sufficient → web supplementary'} "
        f"| local_top={_local_top_score:.2f} | web_chunks={len(web_chunks)}"
        + (f" | max_tokens_override={max_tokens_override}" if max_tokens_override else "")
    )

    user_prompt = f"""Perform Huawei RAN Root Cause Analysis.

SYMPTOM: {symptom_summary} | Severity: {severity} | {scope_str}
RETRIEVAL QUERY: {reformulated_q}
{evidence_quality}{web_chunk_info}

KG EVIDENCE:
{kg_context}

LOCAL RAG DOCUMENTATION (top 2 sources — highest priority):
{rag_context}{web_section}{cbr_section}

Respond ONLY with valid JSON — no markdown, no text outside the JSON object:
{{
  "root_cause": "1-2 sentences. Plain prose. State WHAT failed and WHY based on evidence.",
  "root_cause_explanation": "2-3 sentences. Explain the causal chain: what triggered what, which component failed, what the network impact was. Reference the source by name but no numerical scores.",
  "confidence": 0.0,
  "evidence": {{
    "summary": "1-2 sentences in plain prose. No scores or percentages.",
    "items": [
      {{"text": "What this source confirms, one sentence.", "source_type": "kg", "source_name": "Knowledge Graph"}},
      {{"text": "What this source confirms, one sentence.", "source_type": "local_doc", "source_name": "exact doc title"}},
      {{"text": "What this source confirms, one sentence.", "source_type": "web", "source_name": "domain, e.g. support.huawei.com"}}
    ]
  }},
  "actions": [
    {{"priority": "immediate", "action": "Single but detailed instruction: exact component/parameter/command, how to do it, and why it addresses the root cause.", "source": "doc name, 'Knowledge Graph', or web domain"}},
    {{"priority": "short_term", "action": "Single but detailed follow-up action within 24-48 hours, same level of detail.", "source": "doc name, 'Knowledge Graph', or web domain"}},
    {{"priority": "preventive", "action": "Single but detailed long-term measure to prevent recurrence, same level of detail.", "source": "doc name, 'Knowledge Graph', or web domain"}}
  ],
  "key_terms": {{
    "TERM": "Plain definition — only include if the term appears in root_cause and is Huawei-specific or a telecom acronym."
  }}
}}"""

    raw = _call_llm_json_full(
        system_prompt, user_prompt, model=model_to_use,
        rag_n=len(rag_results), web_primary=_local_rag_weak,
        heavy_local_case=use_heavy_model,
        max_tokens_override=max_tokens_override,   # FIX v4
    )

    raw = raw or {}
    llm_failed = bool(raw.pop("_llm_failed", False))
    raw.setdefault("root_cause",              "Root cause could not be determined.")
    raw.setdefault("root_cause_explanation",  "")
    raw.setdefault("confidence",              0.0)
    raw.setdefault("evidence",                {})
    raw.setdefault("actions",                 [])
    raw.setdefault("key_terms",               {})
    # backward compat
    raw.setdefault("recommendations",         [])
    raw.setdefault("next_actions",            [])
    raw.setdefault("evidence_summary",        "")   # legacy flat field

    # Guard — evidence must be a dict with summary/items
    if not isinstance(raw.get("evidence"), dict):
        raw["evidence"] = {}
    if not raw["evidence"].get("summary"):
        raw["evidence"]["summary"] = raw.get("evidence_summary", "")
    if not isinstance(raw["evidence"].get("items"), list):
        raw["evidence"]["items"] = []

    # Guard — actions must be a list
    if not isinstance(raw.get("actions"), list):
        raw["actions"] = []

    # Backward compat — convert old recommendations/next_actions to actions list
    if not raw["actions"]:
        recs = raw.get("recommendations", [])
        nxt  = raw.get("next_actions", [])
        raw["actions"] = (
            [{"priority": "immediate",  "action": a} for a in nxt] +
            [{"priority": "short_term", "action": r} for r in recs]
        )

    # Guard — every action must carry a source
    _default_action_source = "Knowledge Graph" if kg_has_paths else "internal analysis"
    for a in raw["actions"]:
        if isinstance(a, dict) and not a.get("source"):
            a["source"] = _default_action_source

    # Guard — key_terms must be a dict
    if not isinstance(raw.get("key_terms"), dict):
        raw["key_terms"] = {}

    # ── Sources used ───────────────────────────────────────────────────────
    sources_used = ["qdrant"] if local_chunks else []
    if kg_result.get("available") and (
        kg_result.get("kg_paths") or kg_result.get("related_alarms")
    ):
        sources_used.append("neo4j")
    if similar_cases:
        sources_used.append("cbr_memory")
    if web_chunks:
        sources_used.append("web")

    rag_sources = [
        {
            "source"      : r.get("source", "unknown"),
            "score"       : r.get("rerank_score", r.get("score", 0)),
            "technology"  : r.get("technology", "unknown"),
            "text_preview": r.get("text", r.get("content", ""))[:200],
        }
        for r in rag_results
    ]

    # ── Confidence re-scoring ──────────────────────────────────────────────
    raw_conf = float(raw.get("confidence", 0.0))
    llm_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
    llm_conf = min(max(llm_conf, 0.0), 1.0)

    kg_match_score = correlation.get("correlation_score", 0.0)
    rag_n          = len(rag_results)
    rag_coverage   = min(rag_n / 3.0, 1.0)

    # Calibration guardrail — floor near-zero LLM confidence when output is non-trivial
    _root_cause_trivial = len(raw.get("root_cause", "").strip()) < 30
    _has_evidence       = rag_n >= 1 or kg_has_paths
    if llm_conf < 0.15 and not _root_cause_trivial and _has_evidence and not llm_failed:
        logger.info(
            f"[rca_agent] GUARDRAIL: llm_conf={llm_conf:.3f} < 0.15 with non-trivial output "
            f"— flooring to 0.40 (rag_chunks={rag_n}, kg_paths={kg_has_paths})"
        )
        llm_conf = 0.40

    penalty = 1.0
    if not kg_has_paths:
        penalty *= 0.80
    kg_alarms = kg_result.get("related_alarms", [])[:3]
    if kg_alarms and kg_match_score < 0.20:
        penalty *= 0.85
    if rag_n == 0:
        penalty *= 0.85

    penalized_llm = llm_conf * penalty

    confidence = (
        0.5 * penalized_llm
        + 0.3 * kg_match_score
        + 0.2 * rag_coverage
    )

    if not kg_has_paths and rag_n == 0:
        confidence = min(confidence, 0.35)

    if llm_failed:
        confidence = 0.0

    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    # ── Score breakdown ────────────────────────────────────────────────────
    score_breakdown = {
        "llm":  round(0.5 * penalized_llm, 4),
        "kg":   round(0.3 * kg_match_score, 4),
        "rag":  round(0.2 * rag_coverage, 4),
        "llm_raw":      round(penalized_llm, 3),
        "kg_raw":       round(kg_match_score, 3),
        "rag_raw":      round(rag_coverage, 3),
        "rag_n":        rag_n,
        "kg_has_paths": kg_has_paths,
        "cbr_n":        len(similar_cases) if similar_cases else 0,
    }

    # ── Enrich web evidence items with real URL ────────────────────────────
    _web_url_by_domain = {}
    for chunk in web_chunks:
        _url = chunk.get("source", chunk.get("url", ""))
        if _url:
            _domain = re.sub(r"^https?://(www\.)?", "", _url).split("/")[0]
            _web_url_by_domain[_domain] = _url

    for item in raw["evidence"]["items"]:
        if not isinstance(item, dict):
            continue
        if item.get("source_type") == "web":
            _name = item.get("source_name", "")
            for _domain, _url in _web_url_by_domain.items():
                if _domain in _name or _name in _domain:
                    item["url"] = _url
                    break

    # ── Strip markdown from all free-text LLM fields ──────────────────────
    _evidence_summary_clean = _strip_markdown_artifacts(raw["evidence"].get("summary", ""))
    _evidence_items_clean = [
        {**item, "text": _strip_markdown_artifacts(item.get("text", ""))}
        for item in raw["evidence"].get("items", [])
        if isinstance(item, dict)
    ]
    _actions_clean = [
        {**a, "action": _strip_markdown_artifacts(a.get("action", ""))}
        for a in raw.get("actions", [])
        if isinstance(a, dict)
    ]
    _recommendations_clean = [
        _strip_markdown_artifacts(r) if isinstance(r, str) else r
        for r in raw.get("recommendations", [])
    ]
    _next_actions_clean = [
        _strip_markdown_artifacts(r) if isinstance(r, str) else r
        for r in raw.get("next_actions", [])
    ]

    # ── Log final ──────────────────────────────────────────────────────────
    logger.info(
        f"[rca_agent] Confidence re-scoring | "
        f"symptom_type={symptom_type} | kg_has_paths={kg_has_paths} | "
        f"llm_raw={raw_conf:.4g} → llm_norm={llm_conf:.2f} → penalized={penalized_llm:.2f} | "
        f"penalty={penalty:.2f} | kg_match={kg_match_score:.2f} | "
        f"rag_cov={rag_coverage:.2f} (n={rag_n}) | final={confidence:.4f} | "
        f"llm_failed={llm_failed} | "
        f"root_cause_len={len(raw.get('root_cause',''))} | "
        f"actions={len(raw.get('actions',[]))}"
    )
    logger.info(
        f"[EVAL] kpi={symptom_context.get('kpi_category')} | "
        f"kg_corr={correlation.get('correlation_score', 0):.0%} | "
        f"rag_n={len(rag_results)} | "
        f"top_reranker={max((r.get('rerank_score', 0) for r in rag_results), default=0):.2f} | "
        f"penalty={penalty:.3f} | "
        f"confidence={confidence:.2f}"
    )

    return {
        "root_cause"            : _strip_markdown_artifacts(
            raw.get("root_cause", "Root cause could not be determined.")
        ),
        "root_cause_explanation": _strip_markdown_artifacts(
            raw.get("root_cause_explanation", "")
        ),
        "confidence"            : confidence,
        "score_breakdown"       : score_breakdown,
        "evidence"              : {
            "rag_sources"     : rag_sources,
            "kg_paths"        : kg_result.get("kg_paths", [])[:5],
            "evidence_summary": _evidence_summary_clean,
            "evidence_items"  : _evidence_items_clean,
            "cbr_cases_used"  : len(similar_cases) if similar_cases else 0,
        },
        "affected_scope"        : scope,
        "actions"               : _actions_clean,
        "recommendations"       : _recommendations_clean,
        "next_actions"          : _next_actions_clean,
        "key_terms"             : raw.get("key_terms", {}),
        "sources_used"          : sources_used,
        "llm_failed"            : llm_failed,
    }


# ─────────────────────────────────────────────
# STEP 4 — FORMAT DISPLAY ANSWER
# ─────────────────────────────────────────────

def _confidence_banner(confidence: float) -> str:
    """
    Returns a visible banner to prepend to the display answer based on
    confidence tier. Empty string for high-confidence results (no noise).
    """
    if confidence >= 0.65:
        return ""
    elif confidence >= 0.40:
        return (
            f"⚠️ **Moderate confidence ({confidence:.0%})** — "
            f"results are based on partial evidence. "
            f"Expert review recommended before taking action.\n\n"
        )
    elif confidence >= 0.20:
        return (
            f"🔴 **Low confidence ({confidence:.0%})** — "
            f"the pipeline lacked strong corroborating evidence (KG paths, "
            f"high-scoring documentation, or validated CBR cases). "
            f"Do not act on this diagnosis without manual verification. "
            f"Flag this case for HITL review.\n\n"
        )
    else:
        return (
            f"🔴 **Very low confidence ({confidence:.0%})** — "
            f"evidence was insufficient for a reliable diagnosis. "
            f"This result should be treated as a hypothesis only. "
            f"Mandatory HITL review required.\n\n"
        )


def _format_display_answer(rca: dict, symptom_context: dict) -> str:
    """
    Formats RCA result for chatbot display.
    """
    severity = symptom_context.get("severity", "unknown")
    scope    = rca["affected_scope"]
    conf     = rca.get("confidence", 0.0)
    conf     = conf / 100.0 if conf > 1.0 else conf
    banner        = _confidence_banner(conf)
    root_cause    = rca.get("root_cause", "")
    explanation   = rca.get("root_cause_explanation", "")
    evidence_sum  = rca.get("evidence", {}).get("evidence_summary", "")
    evidence_items= rca.get("evidence", {}).get("evidence_items", [])
    actions       = rca.get("actions", [])
    key_terms     = rca.get("key_terms", {})

    # Guard — root_cause should not contain raw JSON
    if root_cause.strip().startswith("{"):
        try:
            parsed     = json.loads(root_cause)
            root_cause = parsed.get("root_cause", root_cause)
        except Exception:
            pass

    # Backward compat — convert old format if actions is empty
    if not actions:
        recs    = rca.get("recommendations", [])
        nxt     = rca.get("next_actions", [])
        actions = (
            [{"priority": "immediate",  "action": a} for a in nxt] +
            [{"priority": "short_term", "action": r} for r in recs]
        )

    lines = [
        "## Root Cause Analysis",
        "",
        f"**Severity:** {severity.upper()} | "
        f"**Confidence:** {conf:.0%} | "
        f"**Scope:** {scope.get('scope_summary', 'N/A')}",
        "",
        "---",
        "",
    ]

    if rca.get("llm_failed"):
        lines += [
            "⚠️ **Analyse indisponible** — le service LLM (Ollama) n'a pas répondu "
            "à temps. Ce résultat n'est pas un diagnostic et n'a pas été "
            "enregistré en mémoire. Merci de relancer la requête.",
            "",
            "---",
            "",
        ]

    lines += [
        "### Root Cause",
        root_cause,
        "",
    ]

    if explanation:
        lines += [
            "**Explanation**",
            explanation,
            "",
        ]

    if evidence_sum or evidence_items:
        lines.append("**Evidence**")
        if evidence_sum:
            lines.append(f"> {evidence_sum}")
        for item in evidence_items:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            name = item.get("source_name", "")
            url  = item.get("url", "")
            if url:
                src_label = f"[{name}]({url})" if name else f"[source]({url})"
            elif name:
                src_label = name
            else:
                src_label = ""
            suffix = f" — *{src_label}*" if src_label else ""
            lines.append(f"- {item['text']}{suffix}")
        lines.append("")

    # Key terms glossary — only if non-empty
    if key_terms:
        lines.append("**Key Terms**")
        for term, definition in key_terms.items():
            lines.append(f"- **{term}**: {definition}")
        lines.append("")

    # Actions — split by priority
    immediate  = [a for a in actions if a.get("priority") == "immediate"]
    short_term = [a for a in actions if a.get("priority") == "short_term"]
    preventive = [a for a in actions if a.get("priority") == "preventive"]

    if immediate:
        lines.append("### Immediate Actions")
        for a in immediate:
            src = f" *(source: {a['source']})*" if a.get("source") else ""
            lines.append(f"- {a['action']}{src}")
        lines.append("")

    if short_term:
        lines.append("### Short-Term Recommendations *(within 48h)*")
        for a in short_term:
            src = f" *(source: {a['source']})*" if a.get("source") else ""
            lines.append(f"- {a['action']}{src}")
        lines.append("")

    if preventive:
        lines.append("### Preventive Measures")
        for a in preventive:
            src = f" *(source: {a['source']})*" if a.get("source") else ""
            lines.append(f"- {a['action']}{src}")
        lines.append("")

    # Most affected cells — ML anomaly path only
    worst = scope.get("worst_cell_details", [])
    if worst:
        lines.append("### Most Affected Cells")
        for cell in worst[:3]:
            lines.append(
                f"- `{cell['cell']}` — {cell['n_anomalies']} anomalies "
                f"| top KPI: {cell['top_kpi']}"
            )
        lines.append("")

    # Footer
    cbr_count = rca.get("evidence", {}).get("cbr_cases_used", 0)
    if cbr_count:
        lines.append(f"*Memory: {cbr_count} similar past case(s) used.*")
    lines.append(f"*Sources: {', '.join(rca.get('sources_used', []))}*")

    return "\n" + banner + "\n".join(lines)


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

from utils.llm import call_llm_raw as _call_llm_raw


def _lenient_json_extract(text: str) -> dict:
    """
    Best-effort extraction when json.loads() fails (truncated/malformed JSON).
    Recovers scalar fields via simple regex; array-of-objects fields
    (evidence.items, actions) via object-fragment regex — only fully-formed
    objects before the cutpoint are recovered.
    """
    out: dict = {}

    def _scalar(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if not m:
            return None
        try:
            return m.group(1).encode().decode("unicode_escape")
        except Exception:
            return m.group(1)

    rc = _scalar("root_cause")
    if rc:
        out["root_cause"] = rc

    exp = _scalar("root_cause_explanation")
    if exp:
        out["root_cause_explanation"] = exp

    summary = _scalar("summary")

    conf_m = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', text)
    if conf_m:
        try:
            out["confidence"] = float(conf_m.group(1))
        except ValueError:
            pass

    # evidence.items — complete objects only
    item_re = re.compile(
        r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
        r'"source_type"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
        r'"source_name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    )
    items = [
        {"text": t, "source_type": st, "source_name": sn}
        for t, st, sn in item_re.findall(text)
    ]

    if summary or items:
        out["evidence"] = {"summary": summary or "", "items": items}

    # actions — complete objects only
    action_re = re.compile(
        r'\{\s*"priority"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
        r'"action"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
        r'"source"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    )
    actions = [
        {"priority": p, "action": a, "source": s}
        for p, a, s in action_re.findall(text)
    ]
    if actions:
        out["actions"] = actions

    # key_terms — best-effort block parse
    kt_block = re.search(r'"key_terms"\s*:\s*\{(.*?)\}', text, re.DOTALL)
    if kt_block:
        pairs = re.findall(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', kt_block.group(1))
        if pairs:
            out["key_terms"] = dict(pairs)

    return out


# ─────────────────────────────────────────────
# _call_llm_json_full — built here, after _lenient_json_extract is defined.
#
# build_rca_llm_caller() needs _lenient_json_extract as an argument.
# Python executes top-to-bottom: this assignment must come AFTER the
# definition of _lenient_json_extract above.
#
# v4: the returned callable now accepts max_tokens_override (keyword-only)
# which is forwarded to _try_cloud() → call_cloud_llm() → provider fn.
# ─────────────────────────────────────────────
from utils.llm_cloud import PROVIDER, _ENV_MODEL_KEYS
_rca_model_env_key = _ENV_MODEL_KEYS.get(PROVIDER, "OPENROUTER_MODEL")
os.environ.setdefault(_rca_model_env_key, RCA_CLOUD_MODEL)

_call_llm_json_full = build_rca_llm_caller(
    OLLAMA_CHAT_URL, LLM_MODEL, LLM_TEMPERATURE,
    LLM_MAX_TOKENS, _lenient_json_extract
)


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION (LangGraph node)
# ─────────────────────────────────────────────

def rc_analysis_agent(state: dict) -> dict:
    """
    LangGraph node — RCA Reasoning Agent.

    Reads  : state["symptom_context"]     → SymptomContext
             state["rag_results"]         → list[dict]  (from agent_retrieval)
             state["kg_result"]           → dict        (from agent_retrieval)
             state["retrieval_strategy"]  → str
             state["similar_cases"]       → list[dict]  (from memory_retrieve_agent)
    Writes : state["rca_result"]          → RCAResult
             state["display_answer"]      → str pour chatbot
    """
    symptom_context    = state.get("symptom_context")
    rag_results        = state.get("rag_results", [])
    kg_result          = state.get("kg_result", {})
    retrieval_strategy = state.get("retrieval_strategy", "unknown")
    similar_cases      = state.get("similar_cases", [])

    if not symptom_context:
        logger.error("[rca_agent] No symptom_context in state")
        return {**state,
                "rca_result"     : None,
                "display_answer" : "Error: No symptom context available."}

    if not rag_results and not kg_result.get("available"):
        logger.warning("[rca_agent] No evidence in state — retrieval_agent may not have run")

    logger.info(
        f"[rca_agent] Starting RCA | "
        f"type={symptom_context.get('symptom_type')} | "
        f"severity={symptom_context.get('severity')} | "
        f"strategy={retrieval_strategy} | "
        f"rag={len(rag_results)} chunks | "
        f"kg_paths={len(kg_result.get('kg_paths', []))} | "
        f"cbr_cases={len(similar_cases)}"
    )

    # Step 1 — KG ↔ RAG correlation
    correlation = _correlate_kg_rag(kg_result, rag_results)

    # Step 2 — Scope analysis (ML anomaly cells)
    scope = _analyze_worst_cells(symptom_context)

    reformulated_q = state.get(
        "reformulated_query",
        symptom_context.get("summary", ""),
    )

    # Step 3 — LLM reasoning
    rca = _llm_rca_reasoning(
        symptom_context = symptom_context,
        rag_results     = rag_results,
        kg_result       = kg_result,
        correlation     = correlation,
        scope           = scope,
        reformulated_q  = reformulated_q,
        similar_cases   = similar_cases,
    )

    rca["retrieval_strategy"] = retrieval_strategy

    # Step 4 — Format display answer
    display_answer = _format_display_answer(rca, symptom_context)

    logger.info(
        f"[rca_agent] RCA complete | "
        f"confidence={rca['confidence']:.2%} | "
        f"sources={rca['sources_used']}"
    )

    return {
        **state,
        "rca_result"     : rca,
        "display_answer" : display_answer,
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.agent_symptom   import symptom_identification_agent
    from agents.agent_retrieval import retrieval_agent
    from agents.agent_memory    import memory_retrieve_agent

    logging.basicConfig(level=logging.INFO)

    print("\n" + "═"*60)
    print("TEST 1 — User Query (ALM-26235)")
    print("═"*60)

    state = {"user_query": "ALM-26235 RF Unit Maintenance Link Failure cause and solution"}
    state = symptom_identification_agent(state)
    state = memory_retrieve_agent(state)
    state = retrieval_agent(state)
    state = rc_analysis_agent(state)

    print(f"Strategy      : {state.get('retrieval_strategy')}")
    print(f"CBR cases used: {state['rca_result'].get('evidence', {}).get('cbr_cases_used', 0)}")
    print(state["display_answer"])

    print("\n" + "═"*60)
    print("TEST 2 — ML Anomaly")
    print("═"*60)

    with open("ml/artifacts/anomalies_kg.pkl", "rb") as f:
        df_all = pickle.load(f)

    df_critical = df_all[df_all["severity"] == "critical"].head(200)

    state2 = {"ml_anomalies": df_critical}
    state2 = symptom_identification_agent(state2)
    state2 = memory_retrieve_agent(state2)
    state2 = retrieval_agent(state2)
    state2 = rc_analysis_agent(state2)

    print(f"Strategy      : {state2.get('retrieval_strategy')}")
    print(f"CBR cases used: {state2['rca_result'].get('evidence', {}).get('cbr_cases_used', 0)}")
    print(state2["display_answer"])