import os
import gc
import logging
from datetime import timedelta

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_USER      = os.getenv("NEO4J_USER")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE")
OSS_PATH        = os.getenv("OSS_PATH")

BATCH_SIZE      = 500
TEMPORAL_WINDOW = 120    # minutes


# ════════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════════

def load_oss(path: str) -> pd.DataFrame:
    log.info("Chargement %s ...", path)
    df = pd.read_csv(path)

    cols = [
        "Name", "Category", "Alarm Category", "Severity",
        "eNodeB ID", "BBU Name", "CELL Name",
        "Occurred On (NT)", "Cleared On (NT)",
        "duration_min", "Associated Alarm Group ID",
    ]
    df = df[cols].copy()

    df["Occurred On (NT)"] = pd.to_datetime(df["Occurred On (NT)"])
    df["Cleared On (NT)"]  = pd.to_datetime(df["Cleared On (NT)"])
    df["eNodeB ID"]        = pd.to_numeric(df["eNodeB ID"], errors="coerce")
    df["duration_min"]     = pd.to_numeric(df["duration_min"], errors="coerce").round(2)

    log.info(
        "✔ %s alarmes | %s noms distincts | %s catégories",
        f"{len(df):,}", df["Name"].nunique(), df["Category"].nunique(),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — CONSTRUCTION DES RECORDS
# ════════════════════════════════════════════════════════════════════════════════

import re

def to_id(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def build_rootcause_records(df: pd.DataFrame) -> list[dict]:
    """Un RootCause par catégorie OSS avec prior_prob."""
    total = len(df)
    stats = (
        df.groupby("Category")
        .agg(count=("Name", "count"), alarm_count=("Name", "nunique"))
        .reset_index()
    )
    stats["prior_prob"] = (stats["count"] / total).round(4)

    records = []
    for _, row in stats.iterrows():
        records.append({
            "id":                to_id(row["Category"]),
            "label":             row["Category"],
            "category":          row["Category"],
            "prior_prob":        float(row["prior_prob"]),
            "alarm_count":       int(row["alarm_count"]),
            "total_occurrences": int(row["count"]),
        })

    log.info("✔ %s RootCause nodes préparés", len(records))
    return records


def build_ossalarm_records(df: pd.DataFrame) -> list[dict]:
    """Un OSSAlarm par nom d'alarme distinct."""
    grp = (
        df.groupby(["Name", "Category"])
        .agg(
            occurrence_count  = ("Name", "count"),
            avg_duration_min  = ("duration_min", "mean"),
            dominant_severity = ("Severity",
                                 lambda x: x.mode().iloc[0] if len(x) else "Unknown"),
            is_root           = ("Alarm Category",
                                 lambda x: (x == "Root alarm").sum()
                                           > (x == "Correlative alarm").sum()),
        )
        .reset_index()
    )

    records = []
    for _, row in grp.iterrows():
        records.append({
            "id":               to_id(row["Name"]),
            "name":             row["Name"],
            "category":         row["Category"],
            "root_cause_id":    to_id(row["Category"]),
            "occurrence_count": int(row["occurrence_count"]),
            "avg_duration_min": round(float(row["avg_duration_min"]), 2)
                                if pd.notna(row["avg_duration_min"]) else None,
            "dominant_severity":row["dominant_severity"],
            "is_root_alarm":    bool(row["is_root"]),
        })

    log.info("✔ %s OSSAlarm nodes préparés", len(records))
    return records


def build_documented_causes(df: pd.DataFrame) -> list[dict]:
    """
    Relations CAUSES documentées par le flag Root/Correlative de l'OSS.
    Ground truth → confidence = 'high'.
    [D] Ajoute std_delay_min, first_seen, last_seen.
    """
    root = (
        df[df["Alarm Category"] == "Root alarm"]
        [["Associated Alarm Group ID", "Name", "Occurred On (NT)"]]
        .rename(columns={"Name": "root_name", "Occurred On (NT)": "root_time"})
    )
    corr = (
        df[df["Alarm Category"] == "Correlative alarm"]
        [["Associated Alarm Group ID", "Name", "Occurred On (NT)"]]
        .rename(columns={"Name": "corr_name", "Occurred On (NT)": "corr_time"})
    )

    merged = root.merge(corr, on="Associated Alarm Group ID", how="inner")
    merged["delay_min"] = (
        (merged["corr_time"] - merged["root_time"]).dt.total_seconds() / 60
    ).abs()

    pairs = (
        merged.groupby(["root_name", "corr_name"])
        .agg(
            count         = ("delay_min", "count"),
            avg_delay_min = ("delay_min", "mean"),
            std_delay_min = ("delay_min", "std"),
            first_seen    = ("root_time", "min"),
            last_seen     = ("root_time", "max"),
        )
        .reset_index()
    )

    records = []
    for _, row in pairs.iterrows():
        records.append({
            "from_id":       to_id(row["root_name"]),
            "to_id":         to_id(row["corr_name"]),
            "count":         int(row["count"]),
            "avg_delay_min": round(float(row["avg_delay_min"]), 1),
            "std_delay_min": round(float(row["std_delay_min"]), 1)
                             if pd.notna(row["std_delay_min"]) else None,
            "first_seen":    row["first_seen"].isoformat(),
            "last_seen":     row["last_seen"].isoformat(),
            "source":        "oss_documented",
            "confidence":    "high",
        })

    log.info("✔ %s relations CAUSES documentées", len(records))
    return records


def build_temporal_causes(df: pd.DataFrame, window_min: int = TEMPORAL_WINDOW) -> list[dict]:
    """
    Relations CAUSES par co-occurrence temporelle avec fenêtre glissante.

    [A] Fenêtre glissante (O(n log n)) — plus de self-join quadratique O(n²).
        Le eNodeB avec 1398 alarmes aurait produit ~2M paires avec le merge ;
        ici on ne lit que les paires dans la fenêtre de 2h.

    [B] Uniquement les alarmes Root des deux côtés (is_root_alarm implicite
        via filtre Alarm Category == "Root alarm") → évite causalité inversée.

    [C] Seuil = quantile(0.9) → robuste selon la densité réelle du dataset.

    [D] std_delay_min + first_seen + last_seen stockés sur la relation.
    """
    log.info(
        "Calcul co-occurrences temporelles (fenêtre=%s min, rolling)...",
        window_min,
    )

    # [B] Root alarms uniquement
    root_only = (
        df[df["Alarm Category"] == "Root alarm"]
        .dropna(subset=["eNodeB ID"])
        .copy()
    )
    root_only["eNodeB ID"] = root_only["eNodeB ID"].astype(int)
    root_only = root_only.sort_values(
        ["eNodeB ID", "Occurred On (NT)"]
    ).reset_index(drop=True)

    window_td = timedelta(minutes=window_min)

    # [A] Fenêtre glissante par eNodeB
    pairs_raw: list[tuple] = []   # (alarm_a, alarm_b, delay_min, time_a)

    for _enb_id, group in tqdm(
        root_only.groupby("eNodeB ID"), desc="Rolling window par eNodeB"
    ):
        group = group.reset_index(drop=True)
        n = len(group)
        times = group["Occurred On (NT)"].tolist()
        names = group["Name"].tolist()

        for i in range(n):
            time_i = times[i]
            name_i = names[i]
            for j in range(i + 1, n):
                delta = times[j] - time_i
                if delta > window_td:
                    break           # trié par temps → on sort dès qu'on dépasse
                if names[j] != name_i:
                    pairs_raw.append((
                        name_i,
                        names[j],
                        delta.total_seconds() / 60,
                        time_i,
                    ))

    if not pairs_raw:
        log.warning("Aucune co-occurrence temporelle trouvée.")
        return []

    pairs_df = pd.DataFrame(
        pairs_raw, columns=["alarm_a", "alarm_b", "delay_min", "time_a"]
    )

    # [D] Agréger avec std + first/last seen
    agg = (
        pairs_df.groupby(["alarm_a", "alarm_b"])
        .agg(
            count         = ("delay_min", "count"),
            avg_delay_min = ("delay_min", "mean"),
            std_delay_min = ("delay_min", "std"),
            first_seen    = ("time_a", "min"),
            last_seen     = ("time_a", "max"),
        )
        .reset_index()
    )

    # [C] Seuil quantile 90%
    threshold = agg["count"].quantile(0.90)
    agg = agg[agg["count"] >= threshold].sort_values("count", ascending=False)
    log.info(
        "Seuil quantile(0.9) = %.0f co-occurrences → %s paires retenues",
        threshold, len(agg),
    )

    records = []
    for _, row in agg.iterrows():
        records.append({
            "from_id":       to_id(row["alarm_a"]),
            "to_id":         to_id(row["alarm_b"]),
            "count":         int(row["count"]),
            "avg_delay_min": round(float(row["avg_delay_min"]), 1),
            "std_delay_min": round(float(row["std_delay_min"]), 1)
                             if pd.notna(row["std_delay_min"]) else None,
            "first_seen":    row["first_seen"].isoformat(),
            "last_seen":     row["last_seen"].isoformat(),
            "source":        "temporal_corr",
            "confidence":    "medium",
        })

    log.info("✔ %s relations CAUSES temporelles retenues", len(records))
    return records

# ════════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — REQUÊTES CYPHER (VERSION CORRIGÉE)
# ════════════════════════════════════════════════════════════════════════════════

QUERY_CONSTRAINTS = [

    # ── Nodes uniqueness ──────────────────────────────────────────────────────
    "CREATE CONSTRAINT rootcause_unique IF NOT EXISTS "
    "FOR (rc:RootCause) REQUIRE rc.id IS UNIQUE",

    "CREATE CONSTRAINT ossalarm_unique IF NOT EXISTS "
    "FOR (a:OSSAlarm) REQUIRE a.id IS UNIQUE",

    # ── Relationship uniqueness ──────────────────────────────────────────────
    # Empêche les doublons :
    # (A)-[:CAUSES {source, from_id, to_id}]->(B)
    #
    # Exemple :
    # transmission_fault -> sctp_link_fault (temporal_corr)
    # restera UNIQUE même après rerun du script.
    #
    "CREATE CONSTRAINT causes_unique IF NOT EXISTS "
    "FOR ()-[r:CAUSES]-() "
    "REQUIRE (r.source, r.from_id, r.to_id) IS UNIQUE"
]


# ───────────────────────────────────────────────────────────────────────────────
# RootCause nodes
# ───────────────────────────────────────────────────────────────────────────────

QUERY_ROOTCAUSE = """
UNWIND $rows AS row

MERGE (rc:RootCause {id: row.id})

SET rc.label              = row.label,
    rc.category           = row.category,
    rc.prior_prob         = row.prior_prob,
    rc.alarm_count        = row.alarm_count,
    rc.total_occurrences  = row.total_occurrences
"""


# ───────────────────────────────────────────────────────────────────────────────
# OSSAlarm nodes + BELONGS_TO
# ───────────────────────────────────────────────────────────────────────────────

QUERY_OSSALARM = """
UNWIND $rows AS row

MERGE (a:OSSAlarm {id: row.id})

SET a.name               = row.name,
    a.category           = row.category,
    a.occurrence_count   = row.occurrence_count,
    a.avg_duration_min   = row.avg_duration_min,
    a.dominant_severity  = row.dominant_severity,
    a.is_root_alarm      = row.is_root_alarm

WITH a, row

MATCH (rc:RootCause {id: row.root_cause_id})

MERGE (a)-[:BELONGS_TO]->(rc)
"""


# ───────────────────────────────────────────────────────────────────────────────
# CAUSES relationships (VERSION ANTI-DUPLICATION)
# ───────────────────────────────────────────────────────────────────────────────
#
# Important :
# - from_id / to_id sont stockés dans la relation
# - MERGE devient totalement déterministe
# - plus de duplication après rerun
#
# Deux relations différentes restent possibles :
#   source='oss_documented'
#   source='temporal_corr'
#
# mais chacune reste unique.
#
# ───────────────────────────────────────────────────────────────────────────────

QUERY_CAUSES = """
UNWIND $rows AS row

MATCH (a:OSSAlarm {id: row.from_id})
MATCH (b:OSSAlarm {id: row.to_id})

MERGE (a)-[r:CAUSES {
    source  : row.source,
    from_id : row.from_id,
    to_id   : row.to_id
}]->(b)

SET r.count          = row.count,
    r.confidence     = row.confidence,
    r.avg_delay_min  = row.avg_delay_min,
    r.std_delay_min  = row.std_delay_min,
    r.first_seen     = row.first_seen,
    r.last_seen      = row.last_seen
"""

# ════════════════════════════════════════════════════════════════════════════════
# PARTIE 4 — INGESTION
# ════════════════════════════════════════════════════════════════════════════════

def _run_batch(session, query: str, records: list[dict], label: str) -> None:
    def _write(tx, batch):
        tx.run(query, rows=batch)
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc=label):
        session.execute_write(_write, records[i: i + BATCH_SIZE])


def setup_constraints(driver) -> None:
    log.info("Création des contraintes...")
    with driver.session(database=NEO4J_DATABASE) as s:
        for q in QUERY_CONSTRAINTS:
            s.run(q)
    log.info("✔ Contraintes OK")


def ingest_all(driver, df: pd.DataFrame) -> None:
    # Prépare les records
    rc_records  = build_rootcause_records(df)
    oa_records  = build_ossalarm_records(df)
    doc_causes  = build_documented_causes(df)
    temp_causes = build_temporal_causes(df)

    with driver.session(database=NEO4J_DATABASE) as session:
        log.info("── Passe 1 : RootCause nodes")
        _run_batch(session, QUERY_ROOTCAUSE, rc_records, "RootCause")

        log.info("── Passe 2 : OSSAlarm nodes + BELONGS_TO")
        _run_batch(session, QUERY_OSSALARM, oa_records, "OSSAlarm")

        log.info("── Passe 3 : CAUSES documentées (oss_documented)")
        _run_batch(session, QUERY_CAUSES, doc_causes, "CAUSES doc")

        log.info("── Passe 4 : CAUSES temporelles (temporal_corr)")
        _run_batch(session, QUERY_CAUSES, temp_causes, "CAUSES temp")

    gc.collect()
    log.info("✔ Ingestion terminée")


# ════════════════════════════════════════════════════════════════════════════════
# PARTIE 5 — VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def validate(driver) -> None:
    log.info("── Validation ──────────────────────────────────────────")
    with driver.session(database=NEO4J_DATABASE) as s:

        for label in ["RootCause", "OSSAlarm"]:
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            log.info("  %-14s  %s nœuds", label, f"{n:,}")

        for rel, src in [
            ("BELONGS_TO",  None),
            ("CAUSES",      "oss_documented"),
            ("CAUSES",      "temporal_corr"),
        ]:
            if src:
                q     = f"MATCH ()-[r:{rel} {{source:'{src}'}}]->() RETURN count(r) AS n"
                label = f"{rel} ({src})"
            else:
                q     = f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n"
                label = rel
            n = s.run(q).single()["n"]
            log.info("  %-35s  %s relations", label, f"{n:,}")

        log.info("")
        log.info("Top 5 alarmes causales (CAUSES sortantes) :")
        rows = s.run("""
            MATCH (a:OSSAlarm)-[r:CAUSES]->(b:OSSAlarm)
            RETURN a.name AS alarm,
                   sum(r.count) AS total_impact,
                   count(r)     AS nb_effects
            ORDER BY total_impact DESC LIMIT 5
        """).data()
        for r in rows:
            log.info(
                "  %-50s  impact=%s  effects=%s",
                r["alarm"][:50], r["total_impact"], r["nb_effects"],
            )

        log.info("")
        log.info("RootCause nodes (prior_prob) :")
        rows = s.run("""
            MATCH (rc:RootCause)
            RETURN rc.label AS label, rc.prior_prob AS prior, rc.alarm_count AS alarms
            ORDER BY rc.prior_prob DESC
        """).data()
        for r in rows:
            log.info(
                "  %-25s  prior=%.3f  alarmes_distinctes=%s",
                r["label"], r["prior"], r["alarms"],
            )


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    df = load_oss(OSS_PATH)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("✔ Connexion Neo4j OK → %s", NEO4J_URI)

    try:
        setup_constraints(driver)
        ingest_all(driver, df)
        validate(driver)
    finally:
        driver.close()
        log.info("✔ Connexion fermée")


if __name__ == "__main__":
    main()
