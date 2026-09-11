import logging
import os
import re
from typing import Optional, TypedDict
from utils.llm_cloud import call_cloud_llm_messages as _call_cloud_llm_messages

logger = logging.getLogger(__name__)

# Fast Groq model for intent classification — imported here so config.py is
# the single source of truth.  Falls back to a hardcoded safe default so the
# router keeps working even if the import fails (e.g. during unit tests that
# mock the config module).
try:
    from config import REACT_CLOUD_MODEL_FAST as _INTENT_MODEL
except ImportError:
    _INTENT_MODEL = "openai/gpt-oss-20b"


# ─────────────────────────────────────────────
# CONTRATS DE SORTIE
# ─────────────────────────────────────────────

class IntentDecision(TypedDict):
    intent: str                  # "chitchat" | "definition" | "recommendation" | "rca"
    confidence: float
    reason: str
    escalation_offer: bool
    forced: bool


# Seuil sous lequel le LLM est considéré "incertain" entre definition et rca.
# Volontairement haut (0.65) : on préfère proposer l'escalade trop souvent que pas assez.
CONFIDENCE_ESCALATION_THRESHOLD = float(os.getenv("INTENT_ESCALATION_THRESHOLD", "0.65"))

VALID_INTENTS = {"chitchat", "definition", "recommendation", "rca"}


# ─────────────────────────────────────────────
# FOLLOW-UP DETECTION — structurel pur, zéro NLP
# ─────────────────────────────────────────────

def is_rca_followup(last_intent: str | None, query: str) -> bool:
    """
    Détecte si la query est un follow-up sur une RCA déjà produite.

    Déclenchement uniquement sur signal structurel :
      - le dernier intent user était "rca" ou "rca_followup"
      - ET _fast_signal ne détecte pas un nouveau cas (pas d'ALM-ID, pas de
        mot-clé sévérité) — si c'est un nouveau cas, on laisse route_intent décider.

    Pas de NLP, pas de regex sur la query, pas de keyword list.
    Le raisonnement sur CE QU'EST la question appartient au LLM fort,
    pas à cette fonction.
    """
    if last_intent not in ("rca", "rca_followup"):
        return False
    forced, _ = _fast_signal(query)
    return not forced


# ─────────────────────────────────────────────
# SIGNAL RAPIDE — déterministe, ne peut que forcer RCA
# ─────────────────────────────────────────────

# Réutilise le pattern déjà présent ailleurs dans le code (agent_memory.py,
# agent_symptom.py, agent_retrieval.py) — on ne réinvente pas le regex.
_ALM_ID_PATTERN = re.compile(r"ALM-\d+")

# Mots-clés de sévérité — alignés sur SEVERITY_ORDER d'agent_symptom.py
# (critical/high) plus quelques synonymes opérationnels courants côté RAN.
# Volontairement large : un faux positif ici coûte juste un RCA un peu plus
# lent que nécessaire, un faux négatif coûte un vrai incident raté.
_SEVERITY_KEYWORDS = {
    "critical", "outage", "down", "offline", "unreachable",
    "service affecting", "service-affecting", "major incident",
    "panne", "panne critique", "hors service", "indisponible",
    "urgent", "emergency", "p1", "sev1", "sev-1",
}


def _fast_signal(query: str) -> tuple[bool, str]:
    """
    Signal rapide déterministe. Retourne (force_rca, reason).
    Ne retourne JAMAIS un signal qui éloignerait de RCA — uniquement
    force_rca=True ou force_rca=False (= "pas d'avis", le LLM décide).
    """
    m = _ALM_ID_PATTERN.search(query)
    if m:
        return True, f"alarm ID detected ({m.group(0)})"

    q_lower = query.lower()
    matched_keywords = [kw for kw in _SEVERITY_KEYWORDS if kw in q_lower]
    if matched_keywords:
        return True, f"severity keyword(s) detected ({', '.join(matched_keywords[:3])})"

    return False, ""


# ─────────────────────────────────────────────
# CLASSIFICATION LLM — sémantique, 4 catégories + confiance
# ─────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a Huawei RAN (4G LTE) AIOps assistant.
Classify the user's query into exactly one of these 4 categories:

- "chitchat": greetings, small talk, questions about what the assistant can do, no technical content.
- "definition": asking what something IS or means (a concept, acronym, component, KPI) — not asking to diagnose or fix anything.
- "recommendation": asking what to DO, best practices, preventive actions, optimization advice — without describing a specific active fault.
- "rca": describing or implying an actual fault, alarm, anomaly, or degraded KPI that needs root cause diagnosis — even if phrased informally, even without an alarm ID.

Respond with ONLY a JSON object, no preamble, no markdown:
{"intent": "<one of the 4 categories>", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}

Be conservative: if the query plausibly describes a real symptom or fault, even vaguely, lean toward "rca" rather than "definition".
If the query is a follow-up that depends on a previous technical answer (e.g. "which alarms relate to it?"), classify based on the technical depth implied, not just surface wording.
"""

_ROUTER_USER_TEMPLATE = """Recent conversation (most recent last):
{history}

User query: {query}

Classify this query."""


def _format_history(conversation_history: list[dict], max_turns: int = 8) -> str:
    """
    max_turns=8 par défaut (au lieu de 4) — conversation_history contient
    maintenant 2 entrées par échange (user + assistant, cf. app.py), donc 8
    entrées = ~4 échanges réels, pas 8. Garde le lookback effectif inchangé
    par rapport à l'ancien comportement (qui ne stockait que les tours user).
    """
    if not conversation_history:
        return "(none)"
    lines = []
    for turn in conversation_history[-max_turns:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_followup_context(last_result: dict) -> str:
    """
    Construit le contexte RCA structuré à injecter dans le prompt follow-up.

    Ce snapshot est le seul input du resolver Groq — il ne fait aucun appel RAG
    ni KG supplémentaire. "Frozen RCA context" : la vérité est ce qui a été
    produit par le pipeline, pas une ré-inférence.

    Champs injectés :
      - root_cause + explanation  (le cœur)
      - evidence summary + top items (ce qui prouve le root cause)
      - KG paths + related alarms (causalité structurelle)
      - actions par priorité (immédiat / court terme / préventif)
      - confidence + sources (qualité de l'evidence)
      - CBR cases used (historique validé, si présent)

    Les excluded_causes (hypothèses écartées) sont reconstruits si le LLM
    a retourné des alternatives dans key_terms ou evidence_items marqués
    "not confirmed" — sinon laissés vides (pas d'invention).
    """
    rca     = last_result.get("rca_result") or {}
    kg      = last_result.get("kg_result") or {}
    similar = last_result.get("similar_cases") or []

    conf = float(rca.get("confidence", 0.0))
    conf = conf / 100 if conf > 1.0 else conf

    # ── Root cause ───────────────────────────────────────────────────────────
    rc      = rca.get("root_cause", "(unknown)")
    rc_expl = rca.get("root_cause_explanation", "")

    # ── Evidence ─────────────────────────────────────────────────────────────
    ev        = rca.get("evidence") or {}
    ev_summary = ev.get("summary", "") or rca.get("evidence_summary", "")
    ev_items  = ev.get("items", [])
    ev_lines  = "\n".join(
        f"  - [{it.get('source_type','?')}] {it.get('source_name','?')}: "
        f"{(it.get('text') or '')[:180]}"
        for it in ev_items[:4]
    ) or "  (no structured evidence items)"

    # ── KG ───────────────────────────────────────────────────────────────────
    kg_paths = kg.get("kg_paths", [])
    kg_lines = "\n".join(f"  - {p}" for p in kg_paths[:5]) or "  (no KG paths retrieved)"
    alarms   = kg.get("related_alarms", [])
    alarm_lines = "\n".join(
        f"  - {a.get('alarm','')} — {a.get('alarm_name','')} (prior={a.get('prior_prob',0):.2f})"
        for a in alarms[:4]
    ) or "  (none)"

    # ── Actions ──────────────────────────────────────────────────────────────
    actions = rca.get("actions", [])
    action_lines = "\n".join(
        f"  [{a.get('priority','?')}] {a.get('action','')} — source: {a.get('source','?')}"
        for a in actions[:5]
    ) or "  (no structured actions)"

    # ── CBR memory ───────────────────────────────────────────────────────────
    cbr_lines = ""
    if similar:
        cbr_lines = "\nHISTORICAL CASES (validated CBR memory):\n" + "\n".join(
            f"  - Case #{c.get('id')} [{c.get('validation_status','?')}] "
            f"conf={c.get('confidence',0):.0%}: "
            f"{(c.get('corrected_root_cause') or c.get('root_cause') or '')[:150]}"
            for c in similar[:3]
        )

    # ── Key terms (glossaire) ─────────────────────────────────────────────────
    key_terms = rca.get("key_terms") or {}
    terms_lines = ""
    if key_terms:
        terms_lines = "\nKEY TERMS:\n" + "\n".join(
            f"  - {k}: {v}" for k, v in list(key_terms.items())[:6]
        )

    sources = ", ".join(rca.get("sources_used", [])) or "unknown"

    return f"""ROOT CAUSE:
  {rc}

CAUSAL EXPLANATION:
  {rc_expl or '(not provided)'}

EVIDENCE SUMMARY:
  {ev_summary or '(not provided)'}

EVIDENCE ITEMS:
{ev_lines}

KNOWLEDGE GRAPH PATHS:
{kg_lines}

RELATED ALARMS:
{alarm_lines}

RECOMMENDED ACTIONS:
{action_lines}
{cbr_lines}{terms_lines}

CONFIDENCE: {conf:.0%} | SOURCES: {sources}"""


def _llm_classify_local(prompt_messages: list[dict]) -> dict | None:
    """
    Fallback: call the local Ollama fast model (qwen2.5:3b-instruct) for
    intent classification when the cloud provider is unavailable.
    Returns a parsed dict on success, None on any failure.
    """
    import json
    import httpx

    try:
        from config import OLLAMA_CHAT_URL, LLM_MODEL_FAST
    except ImportError:
        OLLAMA_CHAT_URL = ""
        LLM_MODEL_FAST  = "qwen2.5:3b-instruct"

    try:
        resp = httpx.post(
            OLLAMA_CHAT_URL,
            json={
                "model"   : LLM_MODEL_FAST,
                "stream"  : False,
                "options" : {"temperature": 0.0, "num_predict": 120},
                "messages": prompt_messages,
            },
            timeout=15.0,   # classification is cheap — fail fast if Ollama is slow
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        intent = parsed.get("intent", "")
        if intent not in VALID_INTENTS:
            return None
        return {
            "intent"    : intent,
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "reason"    : parsed.get("reason", "local fallback"),
        }
    except Exception as e:
        logger.warning(f"[intent_router] local fallback also failed: {e}")
        return None


def _llm_classify(query: str, conversation_history: list[dict]) -> dict:
    """
    Classifies intent via cloud LLM (fast model) with a local Ollama fallback.

    Model choice rationale:
      - Cloud: _INTENT_MODEL (openai/gpt-oss-20b by default) — classification
        needs no reasoning depth, just fast JSON output.  Using the heavy 70b
        model here added ~2s of unnecessary latency per query.
      - Local fallback: qwen2.5:3b-instruct (already warm from startup) — avoids
        a hard failure when Groq is down or the API key is missing.  Returns
        confidence=0.0 / intent=rca only if BOTH cloud and local fail.
    """
    import json
    from utils.llm_cloud import call_cloud_llm_messages

    prompt = _ROUTER_USER_TEMPLATE.format(
        history=_format_history(conversation_history),
        query=query,
    )
    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    # ── 1. Cloud attempt ──────────────────────────────────────────────────────
    try:
        raw = call_cloud_llm_messages(
            messages,
            max_tokens=120,
            model=_INTENT_MODEL,
        )
        raw    = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "")
        if intent not in VALID_INTENTS:
            logger.warning(f"[intent_router] LLM returned invalid intent '{intent}', trying local fallback")
            raise ValueError(f"invalid intent: {intent}")
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        logger.debug(f"[intent_router] cloud classify OK | model={_INTENT_MODEL} | intent={intent} | conf={confidence:.2f}")
        return {"intent": intent, "confidence": confidence, "reason": parsed.get("reason", "")}
    except Exception as e:
        logger.warning(f"[intent_router] cloud classification failed ({type(e).__name__}: {e}), trying local fallback")

    # ── 2. Local Ollama fallback ──────────────────────────────────────────────
    local_result = _llm_classify_local(messages)
    if local_result is not None:
        logger.info(f"[intent_router] local fallback OK | intent={local_result['intent']} | conf={local_result['confidence']:.2f}")
        return local_result

    # ── 3. Both failed — safe default ─────────────────────────────────────────
    logger.warning("[intent_router] all classifiers failed — safe fallback to rca")
    return {"intent": "rca", "confidence": 0.0, "reason": "all LLM classifiers failed — safe fallback"}


# ─────────────────────────────────────────────
# SUIVI DE L'OFFRE D'ESCALADE (mode intermédiaire)
# ─────────────────────────────────────────────

# Réponses courtes considérées comme confirmation en langage naturel d'une
# offre d'escalade en attente. Volontairement courte et stricte : on ne veut
# PAS intercepter une vraie question juste parce qu'elle contient "oui"
# quelque part — seulement les réponses courtes et univoques.
_CONFIRMATION_PHRASES = {
    "oui", "yes", "ok", "okay", "vas-y", "vas y", "go", "lance", "lance le rca",
    "lance la rca", "lance le rca complet", "lance la rca complète",
    "run rca", "run the rca", "yes please", "oui stp", "oui svp",
}

_DECLINE_PHRASES = {
    "non", "no", "pas besoin", "no thanks", "non merci", "nope",
}


def check_pending_escalation(query: str, active_case: Optional[dict]) -> Optional[IntentDecision]:
    """
    À appeler EN PREMIER, avant route_intent(), sur chaque tour.

    Si le tour précédent a levé escalation_offer (porté dans
    active_case["pending_rca_offer"]), une réponse courte de confirmation
    doit être interceptée ici plutôt que de repartir dans le classifieur
    normal — "oui" seul est par nature inclassable par route_intent.

    Retourne un IntentDecision si interception (confirmation ou déclin),
    None si rien à intercepter (→ appeler route_intent normalement).
    """
    if not active_case or not active_case.get("pending_rca_offer"):
        return None

    q_clean = query.strip().lower().rstrip(".!?")

    if q_clean in _CONFIRMATION_PHRASES:
        return {
            "intent": "rca",
            "confidence": 1.0,
            "reason": "user confirmed pending RCA escalation offer",
            "escalation_offer": False,
            "forced": True,
        }

    if q_clean in _DECLINE_PHRASES:
        return {
            "intent": "definition",
            "confidence": 1.0,
            "reason": "user declined pending RCA escalation offer",
            "escalation_offer": False,
            "forced": False,
        }

    # Ni confirmation ni déclin clair (l'utilisateur a posé une autre question) :
    # on ne force rien, on laisse route_intent classifier normalement, mais
    # l'appelant doit quand même nettoyer pending_rca_offer côté active_case
    # pour ne pas réintercepter un tour plus tard sans rapport.
    return None


# ─────────────────────────────────────────────
# ROUTER PRINCIPAL
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# CHITCHAT FAST-PATH — déterministe, avant tout appel LLM
# ─────────────────────────────────────────────

# Patterns de salutations courtes — zéro ambiguïté opérationnelle.
# On intercepte AVANT _fast_signal et avant le LLM pour deux raisons :
#   1. Économiser ~300ms d'appel LLM sur des messages sans valeur technique.
#   2. Éviter le fallback "ambigu → rca" qui enverrait "hi" dans le pipeline
#      RCA complet (et générerait un badge "Not saved to CBR memory" sur une
#      simple salutation — bug UI rapporté en production).
# Le pattern est volontairement ancré au début de la query (^) pour ne pas
# intercepter "hi performance issue on sector 3" comme un chitchat.
_CHITCHAT_PATTERN = re.compile(
    r"^(hi|hello|hey|bonjour|salut|bonsoir|bonne nuit|merci|thanks?|"
    r"ok|okay|ça va|ca va|ciao|bye|aurevoir|au revoir|"
    r"how are you|comment ça va|comment ca va)\b",
    re.IGNORECASE,
)

_CHITCHAT_REPLIES = {
    "merci":   "Avec plaisir ! N'hésitez pas si vous avez d'autres questions.",
    "thanks":  "You're welcome! Let me know if you have other questions.",
    "thank":   "You're welcome! Let me know if you have other questions.",
    "bye":     "À bientôt !",
    "aurevoir":"À bientôt !",
    "au revoir":"À bientôt !",
}
_DEFAULT_CHITCHAT_REPLY = (
    "Bonjour ! Je suis votre assistant RAN AIOps. "
    "Posez-moi une question sur vos alarmes LTE, dégradations KPI ou équipements Huawei."
)


def _chitchat_fast_path(query: str) -> Optional[IntentDecision]:
    """
    Retourne un IntentDecision chitchat si la query est une salutation courte
    et unambiguë. Retourne None si aucun match → continuer avec route_intent.
    """
    q = query.strip()
    if len(q) > 60:
        return None   # trop long pour être un pur chitchat
    if not _CHITCHAT_PATTERN.match(q):
        return None
    q_lower = q.lower()
    reply = next(
        (v for k, v in _CHITCHAT_REPLIES.items() if q_lower.startswith(k)),
        _DEFAULT_CHITCHAT_REPLY,
    )
    logger.info(f"[intent_router] chitchat fast-path matched: {q!r}")
    return {
        "intent":           "chitchat",
        "confidence":       1.0,
        "reason":           "chitchat fast-path (greeting/farewell keyword)",
        "escalation_offer": False,
        "forced":           True,
        "reply":            reply,
    }


def route_intent(
    query: str,
    conversation_history: Optional[list[dict]] = None,
) -> IntentDecision:
    """
    Classifieur pur — aucun ajustement post-LLM sur la confidence, aucune
    heuristique basée sur l'historique des intents. À appeler APRÈS
    check_pending_escalation() (qui doit avoir retourné None) sur chaque
    tour utilisateur.

    Note : escalation_offer=True sur hésitation definition/rca est une
    décision de routage (quel pipeline lancer), pas un ajustement de
    confidence — la valeur LLM est retournée telle quelle.

    Résolution :
      0. Chitchat fast-path déterministe (salutations courtes) → retour
         immédiat sans aucun appel LLM.
      1. Signal rapide (regex ALM-ID + mots-clés sévérité) → si déclenché,
         force "rca" immédiatement, le LLM n'a pas le dernier mot.
      2. Sinon, classification LLM sémantique (conversation_history injecté
         dans le prompt pour que le LLM dispose lui-même du contexte).
      3. Si le LLM est confiant (>= seuil) → on suit sa classification, sans
         aucun ajustement externe.
      4. Si le LLM est incertain ET hésite entre definition/rca → mode
         intermédiaire : réponse légère + escalation_offer=True pour que
         l'UI propose le passage en RCA complet (géré dans app.py via
         check_pending_escalation / make_pending_offer_state).
      5. Si le LLM est incertain sur tout le reste (cas vraiment ambigu,
         pas spécifiquement definition-vs-rca) → fallback rca par défaut
         ("ambigu → rca", faux négatif coûteux > faux positif qui coûte
         juste du temps).

    Note : le contexte conversationnel est déjà transmis au LLM via
    conversation_history dans _llm_classify → _format_history. Il n'y a
    pas besoin — et il serait contre-productif — d'ajuster manuellement la
    confidence en fonction des intents précédents : c'est exactement le
    type d'heuristique silencieuse que ce classifieur cherche à éviter.
    """
    conversation_history = conversation_history or []

    # ── 0. Chitchat fast-path — before any LLM call ───────────────────────
    chitchat = _chitchat_fast_path(query)
    if chitchat is not None:
        return chitchat

    forced, reason = _fast_signal(query)
    if forced:
        logger.info(f"[intent_router] FORCED rca | {reason}")
        return {
            "intent": "rca",
            "confidence": 1.0,
            "reason": reason,
            "escalation_offer": False,
            "forced": True,
        }

    llm_result = _llm_classify(query, conversation_history)
    intent     = llm_result["intent"]
    confidence = llm_result["confidence"]
    llm_reason = llm_result["reason"]

    if confidence >= CONFIDENCE_ESCALATION_THRESHOLD:
        logger.info(f"[intent_router] {intent} | confidence={confidence:.2f} | {llm_reason}")
        return {
            "intent": intent,
            "confidence": confidence,
            "reason": llm_reason,
            "escalation_offer": False,
            "forced": False,
        }

    # Incertain. Distinction : hésitation definition<->rca (mode intermédiaire)
    # vs ambiguïté générale (fallback rca direct, pas de mode intermédiaire
    # pour chitchat/recommendation — l'intermédiaire n'a de sens que pour
    # "est-ce juste une définition ou un vrai problème ?").
    if intent in ("definition", "rca"):
        logger.info(
            f"[intent_router] uncertain definition/rca (confidence={confidence:.2f}) "
            f"→ mode intermédiaire, escalation_offer=True"
        )
        return {
            "intent": "definition",
            "confidence": confidence,
            "reason": f"low-confidence definition/rca ({llm_reason}) — offering RCA escalation",
            "escalation_offer": True,
            "forced": False,
        }

    logger.info(
        f"[intent_router] ambiguous ({intent}, confidence={confidence:.2f}) "
        f"→ fallback rca (ambigu → rca)"
    )
    return {
        "intent": "rca",
        "confidence": confidence,
        "reason": f"ambiguous classification ({llm_reason}) — safe fallback to rca",
        "escalation_offer": False,
        "forced": False,
    }


# ─────────────────────────────────────────────
# HELPERS POUR active_case (app.py)
# ─────────────────────────────────────────────

def make_pending_offer_state() -> dict:
    """À fusionner dans active_case côté app.py quand escalation_offer=True."""
    return {"pending_rca_offer": True}


def clear_pending_offer_state(active_case: dict) -> dict:
    """À appeler après tout tour qui n'est pas une confirmation/déclin pur,
    pour éviter qu'une offre d'escalade vieille de plusieurs tours traîne
    indéfiniment dans active_case."""
    return {**active_case, "pending_rca_offer": False}