import os
import re
import logging
from collections import defaultdict

from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Poids du score hybride ─────────────────────────────────────────────────────
W_DOC   = 0.70   # poids documentation Huawei
W_PRIOR = 0.30   # poids prior OSS terrain

# [FIX-1+5] Seuil appliqué sur le score BRUT (avant normalisation).
# Filtre uniquement les paires vraiment non documentées (p_doc faible + prior faible).
# Exemple filtré : Config/Software → radio_propagation brut = 0.70×0.18 + 0.30×0.0002 = 0.126
# Exemple conservé : Power Supply → radio_propagation brut = 0.70×0.25 + 0.30×0.046 = 0.189
POSTERIOR_THRESHOLD = 0.15


# ── Clé stable pour MATCH Neo4j (même fonction que ingest_oss_alarms.py) ──────
def to_id(s: str) -> str:
    """Slug stable : lowercase, alphanumérique + underscore."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ════════════════════════════════════════════════════════════════════════════════
# TABLE P_DOC
# ════════════════════════════════════════════════════════════════════════════════

P_DOC_MANUAL: dict[tuple[str, str], dict] = {
    # ── TRANSMISSION ──────────────────────────────────────────────────────────
    ("Transmission", "accessibility"): {
        "p_doc": 0.832,
        "doc_ref": "TG§1.12.1 — SCTP/S1 fault blocks RRC+E-RAB setup",
        "top_evidence": "TG:p51 | TG:p60",
    },
    ("Transmission", "user_experience"): {
        "p_doc": 0.650,
        "doc_ref": "TG§1.10.1 — TNL abnormal release → user rate degradation",
        "top_evidence": "TG:p46 | TG:p48",
    },
    ("Transmission", "traffic_load"): {
        "p_doc": 0.45,
        "doc_ref": "TG§1.11 — Transmission congestion affects traffic load",
        "top_evidence": "TG:p71",
    },
    ("Transmission", "resource_pressure"): {
        "p_doc": 0.40,
        "doc_ref": "TG§1.11 — Transmission congestion → scheduling pressure",
        "top_evidence": "TG:p71",
    },
    ("Transmission", "radio_quality"): {
        "p_doc": 0.22,
        "doc_ref": "TG§1.12 — S1/X2 degradation → indirect SINR impact",
        "top_evidence": "TG:p60",
    },
    ("Transmission", "radio_propagation"): {
        "p_doc": 0.18,
        "doc_ref": "TG§1.10 — Transport fault → X2 → coverage gap",
        "top_evidence": "TG:p48",
    },

    # ── RADIO / RF ────────────────────────────────────────────────────────────
    ("Radio/RF", "accessibility"): {
        "p_doc": 0.786,
        "doc_ref": "TG§1.4.2 — RF/CPRI fault causes cell unavailability",
        "top_evidence": "TG:p43 | TG:p46",
    },
    ("Radio/RF", "user_experience"): {
        "p_doc": 0.800,
        "doc_ref": "TG§1.9.2 — RF power/config fault → user rate degradation",
        "top_evidence": "TG:p61 | TG:p65",
    },
    ("Radio/RF", "traffic_load"): {
        "p_doc": 0.717,
        "doc_ref": "TG§1.11.1 — RF capacity limitation increases PRB load",
        "top_evidence": "TG:p61 | TG:p63",
    },
    ("Radio/RF", "resource_pressure"): {
        "p_doc": 0.68,
        "doc_ref": "TG§1.11.1 — RF capacity → PRB pressure → UE queuing",
        "top_evidence": "TG:p88",
    },
    ("Radio/RF", "radio_quality"): {
        "p_doc": 0.87,
        "doc_ref": "TG§1.5.1 — RF interference → RTWP/CQI/RBLER degradation",
        "top_evidence": "TG:p51",
    },
    ("Radio/RF", "radio_propagation"): {
        "p_doc": 0.82,
        "doc_ref": "TG§1.7.1 — RF antenna/RET/coverage issues → TA increase",
        "top_evidence": "TG:p47",
    },

    # ── POWER SUPPLY ──────────────────────────────────────────────────────────
    ("Power Supply", "accessibility"): {
        "p_doc": 0.325,
        "doc_ref": "MG — Power module fault → board reset → cell unavailability",
        "top_evidence": "MG:p126",
    },
    ("Power Supply", "user_experience"): {
        "p_doc": 0.270,
        "doc_ref": "MG — Intermittent power → RF instability → throughput loss",
        "top_evidence": "MG:p3",
    },
    ("Power Supply", "resource_pressure"): {
        "p_doc": 0.28,
        "doc_ref": "MG — Power fault → cell outage → neighbor PRB overload",
        "top_evidence": "MG:p126",
    },
    ("Power Supply", "radio_quality"): {
        "p_doc": 0.30,
        "doc_ref": "MG — Intermittent power → RF instability → signal quality",
        "top_evidence": "MG:p41",
    },
    ("Power Supply", "radio_propagation"): {
        "p_doc": 0.25,
        "doc_ref": "MG — Power fault → RRU shutdown → coverage hole → high TA",
        "top_evidence": "MG:p126",
    },
    ("Power Supply", "traffic_load"): {
        "p_doc": 0.22,
        "doc_ref": "MG — Power fault → cell down → traffic offload to neighbors",
        "top_evidence": "MG:p3",
    },

    # ── SYNCHRONIZATION ───────────────────────────────────────────────────────
    ("Synchronization", "accessibility"): {
        "p_doc": 0.316,
        "doc_ref": "MG — GNSS fault → TDD timing error → cell setup fails",
        "top_evidence": "MG:p111",
    },
    ("Synchronization", "user_experience"): {
        "p_doc": 0.271,
        "doc_ref": "MG — Clock drift → inter-cell interference → throughput loss",
        "top_evidence": "MG:p3",
    },
    ("Synchronization", "radio_quality"): {
        "p_doc": 0.58,
        "doc_ref": "MG — Clock drift → interference → SINR/CQI degradation",
        "top_evidence": "TG:p13",
    },
    ("Synchronization", "radio_propagation"): {
        "p_doc": 0.35,
        "doc_ref": "MG — Timing error → propagation timing mismatch",
        "top_evidence": "TG:p13",
    },
    ("Synchronization", "resource_pressure"): {
        "p_doc": 0.20,
        "doc_ref": "MG — Clock drift → retransmissions → scheduling pressure",
        "top_evidence": "MG:p27",
    },

    # ── CELL / SERVICE ────────────────────────────────────────────────────────
    ("Cell/Service", "accessibility"): {
        "p_doc": 0.361,
        "doc_ref": "TG§1.4 — Cell Unavailable is by definition accessibility fault",
        "top_evidence": "TG:p51",
    },
    ("Cell/Service", "user_experience"): {
        "p_doc": 0.272,
        "doc_ref": "TG§1.4 — Cell anomaly degrades service quality",
        "top_evidence": "TG:p51",
    },
    ("Cell/Service", "resource_pressure"): {
        "p_doc": 0.35,
        "doc_ref": "TG§1.4 — Cell anomaly → neighbor PRB overload",
        "top_evidence": "TG:p44",
    },
    ("Cell/Service", "traffic_load"): {
        "p_doc": 0.28,
        "doc_ref": "TG§1.4 — Cell anomaly → traffic redistribution on neighbors",
        "top_evidence": "TG:p44",
    },
    ("Cell/Service", "radio_quality"): {
        "p_doc": 0.24,
        "doc_ref": "TG§1.4 — Cell anomaly → interference → CQI/RBLER degradation",
        "top_evidence": "TG:p44",
    },
    ("Cell/Service", "radio_propagation"): {
        "p_doc": 0.20,
        "doc_ref": "TG§1.4 — Cell anomaly → coverage redistribution → TA variation",
        "top_evidence": "TG:p44",
    },

    # ── HARDWARE ──────────────────────────────────────────────────────────────
    ("Hardware", "accessibility"): {
        "p_doc": 0.412,
        "doc_ref": "TG§1.5.2 — Board/RF hardware fault → cell unavailability",
        "top_evidence": "TG:p46 | TG:p48",
    },
    ("Hardware", "user_experience"): {
        "p_doc": 0.335,
        "doc_ref": "TG§1.5 — Hardware degradation → service quality loss",
        "top_evidence": "TG:p63",
    },
    ("Hardware", "resource_pressure"): {
        "p_doc": 0.32,
        "doc_ref": "TG§1.5.2 — Board fault → reduced capacity → PRB pressure",
        "top_evidence": "TG:p46",
    },
    ("Hardware", "traffic_load"): {
        "p_doc": 0.25,
        "doc_ref": "TG§1.5.2 — Board failure → partial outage → neighbor traffic surge",
        "top_evidence": "TG:p46",
    },
    ("Hardware", "radio_quality"): {
        "p_doc": 0.50,
        "doc_ref": "TG§1.5.2 — RF hardware fault → signal quality degradation",
        "top_evidence": "TG:p45",
    },
    ("Hardware", "radio_propagation"): {
        "p_doc": 0.45,
        "doc_ref": "TG§1.5.2 — Antenna hardware fault → coverage degradation",
        "top_evidence": "TG:p45",
    },

    # ── CONFIGURATION / SOFTWARE ──────────────────────────────────────────────
    ("Configuration/Software", "accessibility"): {
        "p_doc": 0.817,
        "doc_ref": "TG§1.4.1 — Misconfiguration causes cell unavailability",
        "top_evidence": "TG:p43 | TG:p51",
    },
    ("Configuration/Software", "user_experience"): {
        "p_doc": 0.419,
        "doc_ref": "TG§1.9 — Config error (power, CA) → user rate degradation",
        "top_evidence": "TG:p61 | TG:p63",
    },
    ("Configuration/Software", "traffic_load"): {
        "p_doc": 0.40,
        "doc_ref": "TG§1.9.1 — CA policy misconfig → load imbalance",
        "top_evidence": "TG:p60",
    },
    ("Configuration/Software", "resource_pressure"): {
        "p_doc": 0.42,
        "doc_ref": "TG§1.9.1 — CA/scheduler misconfiguration → resource imbalance",
        "top_evidence": "TG:p60",
    },
    ("Configuration/Software", "radio_quality"): {
        "p_doc": 0.52,
        "doc_ref": "TG§1.9.2 — Power/antenna config fault → SINR/MCS degradation",
        "top_evidence": "TG:p60",
    },
    ("Configuration/Software", "radio_propagation"): {
        "p_doc": 0.55,
        "doc_ref": "TG§1.4.1 — Frequency/tilt misconfig → coverage hole → high TA",
        "top_evidence": "TG:p41",
    },

    # ── MONITORING / SECURITY ─────────────────────────────────────────────────
    ("Monitoring/Security", "accessibility"): {
        "p_doc": 0.455,
        "doc_ref": "TG§1.13 — IPsec/IKE failure can block S1",
        "top_evidence": "TG:p507",
    },
    ("Monitoring/Security", "user_experience"): {
        "p_doc": 0.30,
        "doc_ref": "TG§1.13 — IPsec tunnel overhead → latency → UX degradation",
        "top_evidence": "TG:p507",
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# CHARGEMENT P_DOC — stratégie MERGE (auto prioritaire + manuel en complément)
# ════════════════════════════════════════════════════════════════════════════════

def load_pdoc_table() -> tuple[dict[tuple[str, str], dict], str]:
    """
    1. Charge pdoc_auto.py (généré par build_pdoc_table_v3.py)
    2. Complète avec P_DOC_MANUAL pour les paires absentes
    3. Fallback sur P_DOC_MANUAL seul si pdoc_auto.py introuvable
    """
    pdoc_auto_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdoc_auto.py")

    if os.path.exists(pdoc_auto_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("pdoc_auto", pdoc_auto_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            auto_table = getattr(mod, "P_DOC_TABLE", None)

            if not auto_table or not isinstance(auto_table, dict):
                log.warning("pdoc_auto.py : P_DOC_TABLE vide ou absente → fallback manuel")
            else:
                # Validation structurelle sur un échantillon
                sample = list(auto_table.keys())[:5]
                if not all(
                    isinstance(k, tuple) and len(k) == 2
                    and isinstance(k[0], str) and isinstance(k[1], str)
                    for k in sample
                ):
                    log.warning("pdoc_auto.py : structure de clés invalide → fallback manuel")
                else:
                    added = sum(
                        1 for key, entry in P_DOC_MANUAL.items()
                        if key not in auto_table
                        or auto_table.setdefault(key, entry) is not None
                        and not auto_table.__contains__(key)
                    )
                    # Merge propre
                    added = 0
                    for key, entry in P_DOC_MANUAL.items():
                        if key not in auto_table:
                            auto_table[key] = entry
                            added += 1
                    log.info(
                        "✔ P_DOC chargée depuis pdoc_auto.py — %s paires auto"
                        " + %s paires manuelles ajoutées",
                        len(auto_table) - added, added,
                    )
                    return auto_table, "pdoc_auto (build_pdoc_table_v3) + manual complement"

        except Exception as e:
            log.warning("pdoc_auto.py : erreur au chargement (%s) → fallback manuel", e)

    log.info("⚠ pdoc_auto.py introuvable → table manuelle (%s paires)", len(P_DOC_MANUAL))
    return P_DOC_MANUAL, "manual expert table (fallback)"


P_DOC_TABLE, PDOC_SOURCE = load_pdoc_table()


# ════════════════════════════════════════════════════════════════════════════════
# LECTURE NEO4J
# ════════════════════════════════════════════════════════════════════════════════

def get_prior_from_neo4j(session) -> dict[str, float]:
    """Lit les prior_prob depuis les nœuds :RootCause."""
    result = session.run(
        "MATCH (rc:RootCause) RETURN rc.category AS cat, rc.prior_prob AS prior"
    ).data()
    if not result:
        log.error("❌ Aucun nœud :RootCause — lancer ingest_oss_alarms.py d'abord")
        return {}

    priors_raw = {r["cat"]: float(r["prior"]) for r in result if r["prior"] is not None}
    total = sum(priors_raw.values())
    if abs(total - 1.0) > 0.01:
        log.warning("⚠ prior_prob non normalisés (Σ=%.4f) → renormalisation", total)
        priors = {cat: round(v / total, 6) for cat, v in priors_raw.items()}
    else:
        priors = priors_raw

    log.info("Prior P(RC) depuis Neo4j :")
    for cat, p in sorted(priors.items(), key=lambda x: -x[1]):
        log.info("  %-28s  %.4f", cat, p)
    return priors


def get_kpi_categories_from_neo4j(session) -> list[str]:
    """
    Récupère les catégories KPI depuis les nœuds :KPICategory
    (label créé par ingest_kg.py via MERGE (kc:KPICategory {category: ...})).
    """
    result = session.run(
        "MATCH (kc:KPICategory) RETURN DISTINCT kc.category AS cat"
    ).data()
    cats = [r["cat"] for r in result if r["cat"]]
    if not cats:
        log.error("❌ Aucun nœud :KPICategory — lancer ingest_kg.py d'abord")
    else:
        log.info("KPICategories dans Neo4j (%s) : %s", len(cats), cats)
    return cats


def validate_rc_alignment(session) -> None:
    """Vérifie l'alignement des catégories RC entre Neo4j et P_DOC_TABLE."""
    result = session.run("MATCH (rc:RootCause) RETURN rc.category AS cat").data()
    neo4j_rc = {r["cat"] for r in result if r["cat"]}
    pdoc_rc  = {rc for (rc, _) in P_DOC_TABLE.keys()}

    missing_in_pdoc  = neo4j_rc - pdoc_rc
    missing_in_neo4j = pdoc_rc - neo4j_rc

    if missing_in_pdoc:
        log.warning(
            "⚠ RC dans Neo4j absentes de P_DOC_TABLE : %s → pas de HAS_ROOT_CAUSE",
            sorted(missing_in_pdoc),
        )
    if missing_in_neo4j:
        log.warning(
            "⚠ RC dans P_DOC_TABLE absentes de Neo4j : %s → vérifier ingest_oss_alarms.py",
            sorted(missing_in_neo4j),
        )
    if not missing_in_pdoc and not missing_in_neo4j:
        log.info("✔ Alignement RC parfait entre Neo4j et P_DOC_TABLE")


# ════════════════════════════════════════════════════════════════════════════════
# CONFIDENCE
# ════════════════════════════════════════════════════════════════════════════════

def confidence_level(posterior_brut: float, p_doc: float) -> str:
    """
    Niveau de confiance basé sur le score brut (absolu) et la qualité documentaire.
      high   → p_doc >= 0.70 ET score_brut >= 0.65
      medium → p_doc >= 0.40 ET score_brut >= 0.35
      low    → sinon
    """
    if p_doc >= 0.70 and posterior_brut >= 0.65:
        return "high"
    elif p_doc >= 0.40 and posterior_brut >= 0.35:
        return "medium"
    else:
        return "low"


# ════════════════════════════════════════════════════════════════════════════════
# CALCUL DU SCORE HYBRIDE
# ════════════════════════════════════════════════════════════════════════════════

def compute_posteriors(
    kpi_categories: list[str],
    priors: dict[str, float],
) -> list[dict]:
    """
    1. Calcul brut  : score = W_DOC × p_doc + W_PRIOR × p_prior
    2. Seuil brut   : éliminer paires < POSTERIOR_THRESHOLD (avant normalisation)
    3. Normalisation: Σ posterior_prob = 1.0 par KPI
    """
    rc_categories = list(priors.keys())

    # ── Coverage check ────────────────────────────────────────────────────────
    pdoc_kpi = {kpi for (_, kpi) in P_DOC_TABLE.keys()}
    missing_kpi = set(kpi_categories) - pdoc_kpi
    if missing_kpi:
        log.warning("⚠ KPI absents de P_DOC_TABLE : %s", sorted(missing_kpi))
    else:
        log.info("✔ Couverture KPI OK — toutes les catégories couvertes")

    p_prior_floor = min(priors.values()) * 0.1 if priors else 0.001

    # ── Calcul brut + seuil ───────────────────────────────────────────────────
    records_brut = []
    skipped_missing = 0
    skipped_threshold = 0

    for kpi_cat in kpi_categories:
        for rc_cat in rc_categories:
            key       = (rc_cat, kpi_cat)
            doc_entry = P_DOC_TABLE.get(key)

            # [FIX-2] Paire non documentée → skip (pas de floor p_doc)
            if doc_entry is None:
                skipped_missing += 1
                continue

            p_doc        = doc_entry["p_doc"]
            p_prior      = priors.get(rc_cat, p_prior_floor)
            doc_ref      = doc_entry.get("doc_ref", "")
            top_evidence = doc_entry.get("top_evidence", "")

            posterior_brut = round(W_DOC * p_doc + W_PRIOR * p_prior, 4)

            # [FIX-5] Seuil appliqué sur le score brut
            if posterior_brut < POSTERIOR_THRESHOLD:
                skipped_threshold += 1
                log.debug("Skip (brut=%.3f < %.2f) : %s → %s",
                          posterior_brut, POSTERIOR_THRESHOLD, rc_cat, kpi_cat)
                continue

            records_brut.append({
                "kpi_category":   kpi_cat,
                "rc_category":    rc_cat,
                "rc_id":          to_id(rc_cat),
                "posterior_brut": posterior_brut,
                "p_doc":          round(p_doc, 4),
                "p_prior":        round(p_prior, 4),
                "doc_ref":        doc_ref,
                "top_evidence":   top_evidence,
            })

    log.info(
        "Calcul brut : %s paires retenues | %s non documentées skippées | "
        "%s sous le seuil (%.2f)",
        len(records_brut), skipped_missing, skipped_threshold, POSTERIOR_THRESHOLD,
    )

    # ── Normalisation par KPI ─────────────────────────────────────────────────
    kpi_totals: dict[str, float] = defaultdict(float)
    for r in records_brut:
        kpi_totals[r["kpi_category"]] += r["posterior_brut"]

    records = []
    for r in records_brut:
        total = kpi_totals[r["kpi_category"]]
        if total == 0:
            log.warning(
                "⚠ KPI '%s' : somme brute = 0 — POSTERIOR_THRESHOLD trop élevé ?",
                r["kpi_category"],
            )
            continue
        posterior_norm = round(r["posterior_brut"] / total, 4)
        records.append({
            "kpi_category":   r["kpi_category"],
            "rc_category":    r["rc_category"],
            "rc_id":          r["rc_id"],
            "posterior_prob": posterior_norm,
            "posterior_raw":  r["posterior_brut"],
            "p_doc":          r["p_doc"],
            "p_prior":        r["p_prior"],
            "confidence":     confidence_level(r["posterior_brut"], r["p_doc"]),
            "source":         "huawei_doc+oss_prior",
            "doc_ref":        r["doc_ref"],
            "top_evidence":   r["top_evidence"],
        })

    log.info("✔ %s posteriors calculés et normalisés", len(records))

    # ── Résumé distribution par KPI ───────────────────────────────────────────
    log.info("")
    log.info("── Distribution RC par KPI ─────────────────────────────────────")
    for kpi_cat in sorted(kpi_categories):
        kpi_records = [r for r in records if r["kpi_category"] == kpi_cat]
        total = sum(r["posterior_prob"] for r in kpi_records)
        n_rc  = len(kpi_records)
        ok    = "✔" if abs(total - 1.0) < 0.01 and n_rc > 0 else "⚠"
        log.info("  %-22s  %s RC  Σ=%.4f %s", kpi_cat, n_rc, total, ok)
        for r in sorted(kpi_records, key=lambda x: -x["posterior_prob"])[:3]:
            log.info(
                "    %-26s  post=%.3f  p_doc=%.2f  [%s]",
                r["rc_category"], r["posterior_prob"], r["p_doc"], r["confidence"],
            )

    return records


# ════════════════════════════════════════════════════════════════════════════════
# REQUÊTE CYPHER
# ════════════════════════════════════════════════════════════════════════════════
#
# [FIX-3] Labels corrects :
#   - :KPICategory  (créé par ingest_kg.py)     MATCH sur category
#   - :RootCause    (créé par ingest_oss_alarms) MATCH sur rc.id (clé stable)
#
# [FIX-4] Entropie supprimée — aucun champ d'incertitude stocké en Neo4j.
# Idempotent : MERGE + SET → safe pour les reruns.

QUERY_HAS_ROOT_CAUSE = """
UNWIND $rows AS row
MATCH (rc:RootCause   {id: row.rc_id})
MATCH (kc:KPICategory {category: row.kpi_category})
MERGE (kc)-[r:HAS_ROOT_CAUSE]->(rc)
SET r.posterior_prob  = row.posterior_prob,
    r.posterior_raw   = row.posterior_raw,
    r.p_doc           = row.p_doc,
    r.p_prior         = row.p_prior,
    r.confidence      = row.confidence,
    r.source          = row.source,
    r.doc_ref         = row.doc_ref,
    r.top_evidence    = row.top_evidence,
    r.updated_at      = datetime()
"""


# ════════════════════════════════════════════════════════════════════════════════
# INGESTION
# ════════════════════════════════════════════════════════════════════════════════

def ingest_posteriors(driver, records: list[dict]) -> None:
    with driver.session(database=NEO4J_DATABASE) as session:
        def _write(tx, batch):
            tx.run(QUERY_HAS_ROOT_CAUSE, rows=batch)
        session.execute_write(_write, records)
    log.info("✔ %s relations HAS_ROOT_CAUSE injectées", len(records))


# ════════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def validate(driver, expected: int) -> None:
    log.info("── Validation ──────────────────────────────────────────────────")
    with driver.session(database=NEO4J_DATABASE) as s:

        n = s.run("MATCH ()-[r:HAS_ROOT_CAUSE]->() RETURN count(r) AS n").single()["n"]
        status = "✔" if n == expected else f"⚠ attendu {expected}"
        log.info("  HAS_ROOT_CAUSE total : %s %s", n, status)

        # Safeguard rc.id NULL
        n_null = s.run(
            "MATCH (rc:RootCause) WHERE rc.id IS NULL OR rc.id = '' "
            "RETURN count(rc) AS n"
        ).single()["n"]
        if n_null:
            log.error("❌ %s RootCause sans rc.id — relancer ingest_oss_alarms.py", n_null)
        else:
            log.info("  ✔ Tous les :RootCause ont un rc.id valide")

        log.info("")
        log.info("Top posteriors (:KPICategory → :RootCause) :")
        rows = s.run("""
            MATCH (kc:KPICategory)-[r:HAS_ROOT_CAUSE]->(rc:RootCause)
            RETURN kc.category AS kpi, rc.category AS rc,
                   r.posterior_prob AS post,
                   r.p_doc          AS p_doc,
                   r.confidence     AS conf,
                   r.top_evidence   AS evidence
            ORDER BY r.posterior_prob DESC LIMIT 15
        """).data()
        for r in rows:
            log.info(
                "  %-22s ← %-26s  post=%.3f  p_doc=%.2f  [%s]  ev=%s",
                r["kpi"], r["rc"], r["post"], r["p_doc"], r["conf"],
                (r["evidence"] or "")[:35],
            )

        log.info("")
        log.info("Root Cause dominante par KPI :")
        rows = s.run("""
            MATCH (kc:KPICategory)-[r:HAS_ROOT_CAUSE]->(rc:RootCause)
            WITH kc.category AS kpi, rc.category AS rc,
                 r.posterior_prob AS post, r.confidence AS conf
            ORDER BY kpi, post DESC
            WITH kpi, collect({rc:rc, post:post, conf:conf})[0] AS top
            RETURN kpi, top.rc AS top_rc, top.post AS top_post, top.conf AS conf
            ORDER BY kpi
        """).data()
        for r in rows:
            log.info(
                "  %-22s → %-26s  post=%.3f  [%s]",
                r["kpi"], r["top_rc"], r["top_post"], r["conf"],
            )


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("✔ Connexion Neo4j OK → %s", NEO4J_URI)
    log.info("── Source P_DOC : %s", PDOC_SOURCE)
    log.info("── Paires P_DOC : %s", len(P_DOC_TABLE))

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            priors         = get_prior_from_neo4j(session)
            kpi_categories = get_kpi_categories_from_neo4j(session)
            validate_rc_alignment(session)

        if not priors:
            log.error("❌ Aucun prior OSS — lancer ingest_oss_alarms.py d'abord")
            return
        if not kpi_categories:
            log.error("❌ Aucun KPICategory — lancer ingest_kg.py d'abord")
            return

        records = compute_posteriors(kpi_categories, priors)

        if not records:
            log.error(
                "❌ Aucun record après compute_posteriors.\n"
                "   Causes possibles :\n"
                "   1. POSTERIOR_THRESHOLD=%.2f trop élevé\n"
                "   2. Labels RC Neo4j != clés P_DOC_TABLE\n"
                "   3. KPICategories absentes de P_DOC_TABLE",
                POSTERIOR_THRESHOLD,
            )
            return

        ingest_posteriors(driver, records)
        validate(driver, expected=len(records))

    finally:
        driver.close()
        log.info("✔ Connexion fermée")


if __name__ == "__main__":
    main()
