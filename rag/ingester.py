import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    PayloadSchemaType,
    NamedVector,
    NamedSparseVector,
    SparseVector,
)

from qdrant_client.models import ScalarQuantizationConfig, ScalarQuantization, ScalarType, QuantizationConfig
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_GRPC_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    DISTANCE_METRIC,
    UPSERT_BATCH_SIZE,
    DRY_RUN,
)

from qdrant_client.models import SparseVector
from collections import Counter
import hashlib
import re
logger = logging.getLogger(__name__)


def _stable_hash(token: str) -> int:
    """MD5-based hash — stable across Python sessions (unlike built-in hash())."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % (2**24)


# ─────────────────────────────────────────────
# MAIN INGESTER
# ─────────────────────────────────────────────

def ingest_to_qdrant(records: list[dict[str, Any]]) -> None:
    """
    Ingests embedded records into Qdrant with hybrid search support.

    Each record must follow the structure from embedder.py:
    {
        "id"      : str          (chunk_id),
        "vector"  : List[float]  (768-dim dense),
        "payload" : dict         (metadata + text)
    }

    Steps:
      1. Validate records
      2. Connect to Qdrant
      3. Create collection with dense + BM25 sparse vectors
      4. Create payload indexes (including technology, domain, source_type)
      5. Upsert in batches (with per-batch error handling)
      6. Verify final count
    """
    if not records:
        logger.warning("[ingester] No records to ingest.")
        return

    _validate_records(records)

    if DRY_RUN:
        print(f"[ingester] DRY RUN — {len(records)} records validated, skipping upsert.")
        return

    client = _connect()
    _ensure_collection(client, embedding_dim=len(records[0]["vector"]))
    _create_indexes(client)
    _upsert_batches(client, records)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def _validate_records(records: list[dict[str, Any]]) -> None:
    """
    Validates structure and vector dimension before touching Qdrant.
    Catches malformed records early (e.g. Ollama partial timeout).
    Also validates that required metadata fields are present for hybrid search.
    """
    required_payload_fields = {"text", "technology", "domain", "source_type"}

    for i, r in enumerate(records):
        if "id" not in r or "vector" not in r or "payload" not in r:
            raise ValueError(
                f"[ingester] Record {i} is missing required fields (id/vector/payload)"
            )

        dim = len(r["vector"])
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"[ingester] Record {i} has wrong vector dimension: "
                f"got {dim}, expected {EMBEDDING_DIM}"
            )

        # Warn (not raise) on missing metadata — allows partial ingestion
        missing = required_payload_fields - set(r["payload"].keys())
        if missing:
            logger.warning(
                f"[ingester] Record {i} (id={r['id']}) missing payload fields: {missing} "
                f"— BM25 filter by technology/domain may not work correctly"
            )

        # Warn if text is empty — BM25 will produce empty sparse vector
        text = r["payload"].get("text", "")
        if not text or len(text.strip()) < 10:
            logger.warning(
                f"[ingester] Record {i} (id={r['id']}) has very short text ({len(text)} chars) "
                f"— BM25 sparse vector will be near-empty"
            )

    logger.info(f"[ingester] Validation passed — {len(records)} records OK")




# ─────────────────────────────────────────────
# COLLECTION SETUP
# ─────────────────────────────────────────────

def _ensure_collection(client: QdrantClient, embedding_dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count and info.points_count > 0:
            logger.info(
                f"[ingester] Collection '{COLLECTION_NAME}' exists "
                f"({info.points_count} points) — will upsert"
            )
            return  # ← collection OK, on n'y touche pas
        else:
            logger.warning(
                f"[ingester] Collection '{COLLECTION_NAME}' exists but is empty "
                f"— deleting and recreating..."
            )
            client.delete_collection(COLLECTION_NAME)
            # fall through to create_collection below

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=embedding_dim,
                distance=DISTANCE_METRIC,
                on_disk=False,          # vecteurs en RAM = recherche rapide
            )
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=True        # FIX 2 — sparse index sur disque (-50 MB)
                )
            )
        },
        on_disk_payload=True,           # FIX 2 — payload sur disque (-400 MB)
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,   
                quantile=0.99,
                always_ram=True,
            )
        ),
    )
    logger.info(
        f"[ingester] Collection '{COLLECTION_NAME}' created "
        f"(dense dim={embedding_dim}, metric={DISTANCE_METRIC}, "
        f"sparse=BM25 on_disk=True, payload on_disk=True, quant=INT8)"
    )


# PAYLOAD INDEXES
def _create_indexes(client: QdrantClient) -> None:
    """
    Creates payload indexes for fast filtered search.
    Safe for re-runs — skips already existing indexes.

    Indexes allow Qdrant to pre-filter BEFORE vector search:
      → file_type   : filter by PDF / XLSX / DOCX
      → parent_doc  : filter by source filename
      → doc_id      : filter by document group (MD5 of filename)
      → chunk_index : range queries on chunk position
      → technology  : filter by 4g / 5g / 3g / 2g / shared  ← KEY FOR AGENTIC ROUTING
      → domain      : filter by lte_ran / nr_ran / gsm_ran / shared_hw
      → source_type : filter by huawei_manual / huawei_kpi / huawei_config

    NOTE: chunk_id is intentionally NOT indexed — it's already the PointStruct.id
    and Qdrant indexes point IDs natively. Indexing it as payload would be redundant.
    """
    indexes_to_create = [
         ("file_type",        PayloadSchemaType.KEYWORD),
        ("parent_doc",       PayloadSchemaType.KEYWORD),
        ("doc_id",           PayloadSchemaType.KEYWORD),
        ("chunk_index",      PayloadSchemaType.INTEGER),
        ("technology",       PayloadSchemaType.KEYWORD),
        ("domain",           PayloadSchemaType.KEYWORD),
        ("source_type",      PayloadSchemaType.KEYWORD),
        ("doc_type",         PayloadSchemaType.KEYWORD),   # ← AJOUT (utilisé dans build_hybrid_query)
        ("tech_confidence",  PayloadSchemaType.KEYWORD),   # ← AJOUT (debug + filtrage qualité)
]

    collection_info  = client.get_collection(COLLECTION_NAME)
    existing_indexes = set(
        collection_info.payload_schema.keys()
        if collection_info.payload_schema else []
    )

    for field_name, field_type in indexes_to_create:
        if field_name in existing_indexes:
            logger.debug(f"[ingester] Index already exists: '{field_name}' — skipping")
            continue
        client.create_payload_index(
            collection_name = COLLECTION_NAME,
            field_name      = field_name,
            field_schema    = field_type,
            wait            = True,
        )
        logger.info(f"[ingester] Index created: '{field_name}'")

def _text_to_sparse(text: str) -> SparseVector:
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)
    vocab  = sorted(counts.keys())
    indices = [_stable_hash(t) for t in vocab]
    values  = [float(counts[t]) for t in vocab]
    return SparseVector(indices=indices, values=values)
# ─────────────────────────────────────────────
# UPSERT
# ─────────────────────────────────────────────

def _upsert_batches(
    client  : QdrantClient,
    records : list[dict[str, Any]],
) -> None:
    """
    Upserts records in batches with both dense and BM25 sparse vectors.

    Each PointStruct carries:
      - vector["dense"]  : List[float] from Nomic v1.5 (via Ollama)
      - vector["bm25"]   : str (raw text) — Qdrant tokenises and builds sparse vector
      - payload          : all metadata including technology, domain, source_type

    Per-batch try/except → one failed batch doesn't stop the pipeline.
    """
    total     = len(records)
    n_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    inserted  = 0
    failed    = 0

    logger.info(
        f"[ingester] Upserting {total} points | "
        f"{n_batches} batches | batch_size={UPSERT_BATCH_SIZE} | "
        f"vectors=dense+bm25"
    )


    for batch_num, i in enumerate(range(0, total, UPSERT_BATCH_SIZE)):

        batch  = records[i : i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id      = r["id"],
                vector  = {
                  "dense" : r["vector"],
                  "bm25"  : _text_to_sparse(r["payload"]["text"]),
                },
                payload = r["payload"],
          )
            for r in batch
    ]

        try:
            client.upsert(
                collection_name = COLLECTION_NAME,
                points          = points,
                wait            = True,
            )
            inserted += len(batch)

        except Exception as exc:
            failed += len(batch)
            logger.error(
                f"[ingester] Batch {batch_num + 1}/{n_batches} failed: {exc} "
                f"— {len(batch)} points skipped"
            )
            continue

        pct = inserted / total * 100
        logger.info(
            f"[ingester] Batch {batch_num + 1}/{n_batches} | "
            f"{inserted}/{total} points ({pct:.1f}%)"
        )

    print(f"\n[ingester] Upsert complete — {inserted} ingested | {failed} skipped")



