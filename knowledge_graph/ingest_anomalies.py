from neo4j import GraphDatabase
from tqdm import tqdm
import pandas as pd
import hashlib
import pickle
from dotenv import load_dotenv
import os
import gc
import hashlib
from datetime import datetime 
import hashlib

import logging
# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 0. Chargement .env ────────────────────────────────────────────────────────
load_dotenv()
SEV_SCORE={"low": 1, "medium": 2, "high": 3, "critical": 4}
BATCH_SIZE=5000
RESET_DB=True
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
DATA_PATH = os.getenv("DATA_PATH") 
# ── 0. Chargement anomalies_kg ────────────────────────────────────────────────
anomalies_kg = pd.read_pickle("data/anomalies_kg.pkl")
print(f"✔ anomalies_kg chargé : {len(anomalies_kg)} lignes")

# ── 1. Connexion ──────────────────────────────────────────────────────────────
DRIVER = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

DB_NAME = os.getenv("NEO4J_DATABASE")

# ── Cypher — 3 passes pour réduire la pression mémoire par transaction ─────────

# Passe 1 : nœuds légers (119 eNodeB, 878 Cell, 27 KPI)
# WITH DISTINCT évite de répéter MERGE sur nœuds déjà existants dans le batch
QUERY_NODES_LIGHT = """
UNWIND $rows AS row
WITH DISTINCT row.enodeb AS enodeb,
              row.cell AS cell,
              row.kpi AS kpi,
              row.kpi_category AS kpi_category,
              row.risk_score AS risk_score,
              row.anomaly_count AS anomaly_count,
              row.critical_ratio AS critical_ratio
MERGE (e:ENodeB {id: enodeb})
MERGE (c:Cell {id: cell})
SET c.risk_score     = risk_score,
    c.anomaly_count  = anomaly_count,
    c.critical_ratio = critical_ratio
MERGE (e)-[r:HAS_CELL]->(c)
MERGE (k:KPI {name: kpi})
MERGE (kc:KPICategory {category: kpi_category})
MERGE (k)-[:BELONGS_TO]->(kc)
"""

# Passe 2 : nœuds Anomaly + TimeWindow (283k — le plus lourd)
QUERY_ANOMALY = """
UNWIND $rows AS row
MERGE (a:Anomaly {id: row.anomaly_id})
SET a.severity     = row.severity,
    a.z_score      = row.z_score,
    a.model        = row.model,
    a.kpi_category = row.kpi_category
MERGE (t:TimeWindow {time: row.time_window})
SET t.hour = row.hour,
    t.day  = row.day
"""

# Passe 3 : relations uniquement (MATCH only, pas de MERGE de nœuds)
QUERY_RELATIONS = """
UNWIND $rows AS row
MATCH (c:Cell {id: row.cell})
MATCH (a:Anomaly {id: row.anomaly_id})
MATCH (k:KPI {name: row.kpi})
MATCH (t:TimeWindow {time: row.time_window})
MERGE (c)-[:HAS_ANOMALY]->(a)
MERGE (a)-[:INVOLVES_KPI]->(k)
MERGE (a)-[:OCCURS_AT]->(t)
"""


def normalize_time_window(t):
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return t.strftime("%Y-%m-%dT%H:%M:%S")




# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    log.info("Chargement %s ...", path)
    df = pd.read_pickle(path)
    gc.collect()

    df["enodeb_id"]    = df["enodeb_id"].astype(str)
    df["cell"]         = df["cell"].astype(str)
    df["kpi"]          = df["kpi"].astype(str)
    df["kpi_category"] = df["kpi_category"].astype(str)
    df["severity"]     = df["severity"].astype(str)
    df["model"]        = df["model"].astype(str)
    df["time_window"] = pd.to_datetime(df["time_window"])
    df["time_window_norm"] = df["time_window"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    df["anomaly_id"] = (
        df["cell"].astype(str).str.strip() + "|" +
        df["kpi"].astype(str).str.strip() + "|" +
        df["time_window_norm"].astype(str)).map(lambda x: hashlib.sha256(x.encode()).hexdigest())

    # Risk score par cellule
    cell_risk = (
        df.assign(score=df["severity"].map(SEV_SCORE))
        .groupby("cell", observed=True)
        .agg(
            risk_score=("score", "mean"),
            anomaly_count=("score", "count"),
            critical_ratio=("severity", lambda x: round((x == "critical").mean(), 4)),
        )
    )
    df = df.merge(cell_risk, on="cell", how="left")

    # tri par cell+enodeb → batches cohérents → moins de cache misses Neo4j
    df = df.sort_values(["cell", "enodeb_id"]).reset_index(drop=True)

    log.info(
        "✔ %s anomalies | %s eNodeB | %s Cell | %s KPI",
        f"{len(df):,}",
        df["enodeb_id"].nunique(),
        df["cell"].nunique(),
        df["kpi"].nunique(),
    )
    return df


def build_records(df: pd.DataFrame) -> list[dict]:
    raw = df.to_dict("records")

    records = []

    for r in raw:
        dt = r["time_window"]

        records.append({
            "enodeb":         r["enodeb_id"],
            "cell":           r["cell"],
            "kpi":            r["kpi"],
            "kpi_category":   r["kpi_category"],
            "severity":       r["severity"],
            "z_score":        round(float(r["z_score"]), 4)
                               if pd.notna(r["z_score"]) else None,
            "model":          r["model"],
            "anomaly_id":     r["anomaly_id"],
            "time_window":    dt,
            "hour":           int(dt.hour),
            "day":            dt.day_name(),
            "risk_score":     round(float(r["risk_score"]), 4),
            "anomaly_count":  int(r["anomaly_count"]),
            "critical_ratio": round(float(r["critical_ratio"]), 4),
        })

    return records

# ── Neo4j helpers ──────────────────────────────────────────────────────────────
def setup_db(driver, reset: bool) -> None:
    if reset:
        log.info("Nettoyage base...")
        with driver.session() as s:
            # Étape 1 : supprimer les relations d'abord (moins lourd que DETACH DELETE)
            log.info("  suppression relations...")
            while True:
                result = s.run(
                    "MATCH ()-[r]->() WITH r LIMIT 5000 DELETE r RETURN count(r) AS n"
                ).single()
                if result["n"] == 0:
                    break
            # Étape 2 : supprimer les nœuds (sans relations = DELETE simple, pas DETACH)
            log.info("  suppression noeuds...")
            while True:
                result = s.run(
                    "MATCH (n) WITH n LIMIT 5000 DELETE n RETURN count(n) AS n"
                ).single()
                if result["n"] == 0:
                    break
        log.info("✔ Base vidée")

    with driver.session() as s:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ENodeB)     REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cell)       REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:KPI)        REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (kc:KPICategory) REQUIRE kc.category IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Anomaly)    REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TimeWindow) REQUIRE t.time IS UNIQUE",
        ]
        indexes = [
            # index RCA uniquement (les UNIQUE constraints couvrent déjà .id/.name/.time)
            "CREATE INDEX IF NOT EXISTS FOR (a:Anomaly) ON (a.severity)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Anomaly) ON (a.kpi_category)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Cell)    ON (c.risk_score)",
        ]
        for q in constraints + indexes:
            s.run(q)

    log.info("✔ Contraintes & index OK")


def _run_pass(driver, query: str, records: list[dict], label: str) -> None:
    """Exécute une passe d'ingestion avec execute_write par batch."""
    def _write(tx, batch):
        tx.run(query, rows=batch)

    with driver.session() as session:
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc=label):
            batch = records[i : i + BATCH_SIZE]
            session.execute_write(_write, batch)

    gc.collect()


def ingest(driver, records: list[dict]) -> None:
    log.info("Ingestion %s records (batch=%s) en 3 passes...", f"{len(records):,}", BATCH_SIZE)

    # Passe 1 : ENodeB + Cell + KPI (léger — peu de noeuds distincts)
    log.info("-- Passe 1 : ENodeB / Cell / KPI")
    _run_pass(driver, QUERY_NODES_LIGHT, records, "Nodes ENodeB/Cell/KPI")

    # Passe 2 : Anomaly + TimeWindow (lourd — 283k noeuds)
    log.info("-- Passe 2 : Anomaly / TimeWindow")
    _run_pass(driver, QUERY_ANOMALY, records, "Nodes Anomaly/TW")

    # Passe 3 : relations uniquement (MATCH only, pas de MERGE de noeuds)
    log.info("-- Passe 3 : Relations")
    _run_pass(driver, QUERY_RELATIONS, records, "Relations")

    log.info("✔ Ingestion terminée")


def validate(driver) -> None:
    log.info("── Validation ──────────────────────────")
    with driver.session() as s:
        for label in ["ENodeB", "Cell", "KPI", "Anomaly", "TimeWindow"]:
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            log.info("  %-14s %8s", label, f"{n:,}")
        for rel in ["HAS_CELL", "HAS_ANOMALY", "INVOLVES_KPI", "OCCURS_AT"]:
            n = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n").single()["n"]
            log.info("  %-14s %8s", rel, f"{n:,}")


def rca_queries(driver) -> None:
    log.info("── RCA Queries ─────────────────────────")

    with driver.session() as s:
        # Top 10 cellules risquées
        rows = s.run("""
            MATCH (c:Cell)
            RETURN c.id AS cell, c.risk_score AS risk,
                   c.anomaly_count AS anomalies, c.critical_ratio AS crit_pct
            ORDER BY c.risk_score DESC LIMIT 10
        """).data()
        log.info("Top 10 cellules risquees :")
        for r in rows:
            log.info("  %s | risk=%.2f | anomalies=%s | crit=%.1f%%",
                     r["cell"], r["risk"], r["anomalies"], r["crit_pct"] * 100)

        # KPI hotspots
        rows = s.run("""
            MATCH (a:Anomaly)-[:INVOLVES_KPI]->(k:KPI)
            RETURN k.name AS kpi, count(a) AS n,
                   avg(CASE WHEN a.z_score IS NOT NULL THEN a.z_score END) AS avg_z
            ORDER BY n DESC LIMIT 10
        """).data()
        log.info("Top 10 KPI hotspots :")
        for r in rows:
            log.info("  %-50s | count=%s | avg_z=%.2f",
                     r["kpi"][:50], r["n"], r["avg_z"])

        # Temporal burst
        rows = s.run("""
            MATCH (t:TimeWindow)<-[:OCCURS_AT]-(a:Anomaly)
            RETURN t.time AS window, count(a) AS n
            ORDER BY n DESC LIMIT 5
        """).data()
        log.info("Top 5 fenetres temporelles :")
        for r in rows:
            log.info("  %s  ->  %s anomalies", r["window"], r["n"])

        # RCA eNodeB → KPI category
        rows = s.run("""
    MATCH (e:ENodeB)-[:HAS_CELL]->(c:Cell)-[:HAS_ANOMALY]->(a:Anomaly)
          -[:INVOLVES_KPI]->(k:KPI)-[:BELONGS_TO]->(kc:KPICategory)

    WHERE a.severity IN ['high', 'critical']

    RETURN e.id AS enodeb,
           kc.category AS category,
           count(a) AS impact,
           avg(CASE WHEN a.z_score IS NOT NULL THEN a.z_score END) AS score

    ORDER BY impact DESC
    LIMIT 10
""").data()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    df      = load_data(DATA_PATH)
    records = build_records(df)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("✔ Connexion Neo4j OK")
    log.info(
        "⚠ Si OOM : dans neo4j.conf → "
        "dbms.memory.heap.max_size=2G  +  dbms.memory.transaction.total.max=1G"
    )

    try:
        setup_db(driver, reset=RESET_DB)
        ingest(driver, records)
        validate(driver)
        rca_queries(driver)
    finally:
        driver.close()
        log.info("✔ Connexion fermée")


if __name__ == "__main__":
    main()
