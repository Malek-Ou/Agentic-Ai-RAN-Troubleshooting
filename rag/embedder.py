import time
import hashlib
import logging
import pickle
import pathlib
import re
from typing import Any
from config import (
    OLLAMA_EMBED_URL as OLLAMA_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
)
import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# EMBEDDING CACHE
# ─────────────────────────────────────────────

CACHE_PATH = pathlib.Path("embedding_cache.pkl")


def _embed_key(text: str) -> str:
    """
    Content-based cache key — robust to pipeline changes.

    Uses normalized text hash instead of chunk_id so that:
      - same content always hits the cache, even if chunk_id changes
        (e.g. new files inserted before existing ones shift the idx counter)
      - whitespace/case differences don't create false cache misses
      - changing embedding model invalidates old cached vectors automatically
        (model name is included in the hash input)

    NOTE: this key is only used for cache lookup.
          The actual Qdrant point ID remains chunk_id (pipeline-derived).
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    content    = f"{EMBEDDING_MODEL}::{normalized}"   # model-scoped key
    return hashlib.sha256(content.encode()).hexdigest()


def _load_cache() -> dict:
    """
    Loads the embedding cache from disk.
    Returns an empty dict if no cache file exists.

    Cache format: { _embed_key(text): qdrant_record }
    Where qdrant_record = { "id": str, "vector": List[float], "payload": dict }
    """
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"[cache] Loaded {len(cache)} cached embeddings from {CACHE_PATH}")
            return cache
        except Exception as e:
            logger.warning(f"[cache] Failed to load cache ({e}) — starting fresh")
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """
    Persists the cache to disk after each successful batch.
    Called incrementally so progress is never lost on crash.
    """
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.warning(f"[cache] Failed to save cache: {e}")


# ─────────────────────────────────────────────
# MAIN EMBEDDER
# ─────────────────────────────────────────────

def embed_chunks(chunks: list[Document]) -> list[dict[str, Any]]:
    """
    Embeds a list of chunks using Ollama (nomic-embed-text).

    Returns a list of Qdrant-ready records:
    {
        "id"       : chunk_id (from metadata),
        "vector"   : List[float] (768-dim),
        "payload"  : full metadata + page_content
    }

    Filters applied before embedding:
      1. Noise filter    : chunks < 200 chars (stripped) are skipped
      2. Structured XLSX : communication_matrix / spare_parts /
                           compatibility / security_baseline are skipped
                           (Performance Counter xlsx with doc_type=kpi_data pass through)

    Cache behaviour:
      - On first run   : all chunks are embedded and cached to disk
      - On re-runs     : cached chunks are reused (no Ollama call)
      - On crash/retry : already-cached batches are skipped instantly
      - Cache file     : embedding_cache.pkl (~50-100 MB for ~11k chunks)
      - To force re-embed : delete embedding_cache.pkl
    """
    if not chunks:
        logger.warning("[embedder] No chunks to embed.")
        return []

    # ── Filter 1 : noise (too short for nomic-embed-text) ─────────────────
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 200]
    noise  = before - len(chunks)
    if noise:
        logger.warning(f"[embedder] {noise} chunks filtered (< 200 chars)")

    # ── Filter 2 : structured XLSX (must go to SQLite, not Qdrant) ────────
    # "technical_doc" intentionally excluded from this set —
    # it is the default fallback in data_loader and covers many valid PDFs.
    STRUCTURED_DOC_TYPES = {
        "communication_matrix",
        "spare_parts",
        "compatibility",
        "security_baseline",
    }
    before = len(chunks)
    chunks = [
        c for c in chunks
        if not (
            c.metadata.get("file_type") == "xlsx"
            and c.metadata.get("doc_type") in STRUCTURED_DOC_TYPES
        )
    ]
    structured = before - len(chunks)
    if structured:
        logger.warning(
            f"[embedder] {structured} structured XLSX chunks skipped "
            f"(use SQLite instead)"
        )

    if not chunks:
        logger.warning("[embedder] No chunks left after filtering.")
        return []

    # ── Load cache ─────────────────────────────────────────────────────────
    cache = _load_cache()

    # Pre-compute embed keys for all chunks — used in lookup + store
    # Avoids calling _embed_key twice per chunk
    chunk_keys = [_embed_key(c.page_content) for c in chunks]

    cache_hits = sum(1 for k in chunk_keys if k in cache)
    logger.info(
        f"[embedder] Cache status — {cache_hits}/{len(chunks)} chunks already cached "
        f"({len(chunks) - cache_hits} to embed)"
    )

    records   : list[dict[str, Any]] = []
    total      = len(chunks)
    batches    = _make_batches(list(zip(chunks, chunk_keys)), BATCH_SIZE)
    n_batches  = len(batches)

    logger.info(
        f"[embedder] Starting embedding | "
        f"{total} chunks | {n_batches} batches | model: {EMBEDDING_MODEL}"
    )

    for batch_idx, batch in enumerate(batches):

        # ── Split batch: cached vs needs embedding ─────────────────────────
        # Store (chunk, key) together — key already computed, no recalculation
        to_embed : list[tuple[Document, str]] = []

        for chunk, key in batch:
            if key in cache:
                records.append(cache[key])       # reuse cached vector
            else:
                to_embed.append((chunk, key))    # keep key alongside chunk

        # ── All chunks in this batch were cached → skip Ollama call ────────
        if not to_embed:
            _log_progress(batch_idx + 1, n_batches, len(records), cached=len(batch))
            continue

        # ── Embed the remaining chunks ─────────────────────────────────────
        chunks_only = [c for c, _ in to_embed]
        vectors = _embed_batch_with_retry(chunks_only, batch_idx, n_batches)

        if vectors is None:
            logger.error(
                f"[embedder] Batch {batch_idx + 1}/{n_batches} "
                f"permanently failed — {len(to_embed)} chunks skipped"
            )
            continue

        for (chunk, key), vector in zip(to_embed, vectors):
            record     = _build_qdrant_record(chunk, vector)
            cache[key] = record                  # key already computed — no recalculation
            records.append(record)

        _save_cache(cache)                       # flush to disk after every batch

        _log_progress(batch_idx + 1, n_batches, len(records))

    logger.info(
        f"[embedder] Done | "
        f"{len(records)}/{total} chunks embedded successfully"
    )

    _print_stats(records, total)
    return records


# ─────────────────────────────────────────────
# BATCH BUILDER
# ─────────────────────────────────────────────

def _make_batches(
    items: list,
    batch_size: int,
) -> list[list]:
    """Splits a list into fixed-size batches."""
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


# ─────────────────────────────────────────────
# EMBEDDING WITH RETRY
# ─────────────────────────────────────────────

def _embed_batch_with_retry(
    batch     : list[Document],
    batch_idx : int,
    n_batches : int,
) -> list[list[float]] | None:
    """
    Tries to embed a batch up to MAX_RETRIES times.
    Returns list of vectors on success, None on permanent failure.
    Uses exponential backoff between retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_ollama(batch)

        except Exception as exc:
            wait = RETRY_DELAY * (2 ** (attempt - 1))   # 10s → 20s → 40s → ...
            logger.warning(
                f"[embedder] Batch {batch_idx + 1}/{n_batches} | "
                f"attempt {attempt}/{MAX_RETRIES} failed: {exc} | "
                f"retrying in {wait:.0f}s"
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    return None


# ─────────────────────────────────────────────
# OLLAMA API CALL
# ─────────────────────────────────────────────

def _call_ollama(batch: list[Document]) -> list[list[float]]:
    """
    True batch embedding — 1 HTTP request for N chunks.
    Endpoint: /api/embed (Ollama >= 0.1.31)
    """
    texts = [
        f"search_document: {chunk.page_content}"
        for chunk in batch
    ]

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(
            OLLAMA_URL,
            json={
                "model" : EMBEDDING_MODEL,
                "input" : texts,
            }
        )
        response.raise_for_status()
        data = response.json()

        vectors = data.get("embeddings")

        if not vectors or len(vectors) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} vectors, got "
                f"{len(vectors) if vectors else 0}"
            )

        for i, v in enumerate(vectors):
            if len(v) != EMBEDDING_DIM:
                raise ValueError(
                    f"Chunk {i}: dim {len(v)} != {EMBEDDING_DIM}"
                )

        return vectors


# ─────────────────────────────────────────────
# QDRANT RECORD BUILDER
# ─────────────────────────────────────────────

def _build_qdrant_record(
    chunk  : Document,
    vector : list[float],
) -> dict[str, Any]:
    """
    Builds a Qdrant-ready record from a chunk and its embedding vector.
    """
    chunk_id = chunk.metadata.get("chunk_id")

    if not chunk_id:
        chunk_id = hashlib.md5(chunk.page_content.encode()).hexdigest()
        logger.warning(
            f"[embedder] chunk_id missing — generated fallback: {chunk_id}"
        )

    payload = {
        "text"            : chunk.page_content,
        "chunk_id"        : chunk_id,
        "chunk_hash"      : chunk.metadata.get("chunk_hash"),
        "doc_id"          : chunk.metadata.get("doc_id"),
        "parent_doc"      : chunk.metadata.get("parent_doc"),
        "chunk_index"     : chunk.metadata.get("chunk_index"),
        "chunk_total"     : chunk.metadata.get("chunk_total"),
        "file_type"       : chunk.metadata.get("file_type"),
        "doc_type"        : chunk.metadata.get("doc_type"),
        "source_type"     : chunk.metadata.get("source_type", "huawei_manual"),
        "token_estimate"  : chunk.metadata.get("token_estimate"),
        "embedding_model" : EMBEDDING_MODEL,
        "technology"      : chunk.metadata.get("technology"),
        "technologies"    : chunk.metadata.get("technologies"),
        "tech_confidence" : chunk.metadata.get("tech_confidence"),
        "tech_matched_on" : chunk.metadata.get("tech_matched_on"),
        "domain"          : chunk.metadata.get("domain"),
    }

    return {
        "id"      : chunk_id,
        "vector"  : vector,
        "payload" : payload,
    }


# ─────────────────────────────────────────────
# LOGGING HELPERS
# ─────────────────────────────────────────────

def _log_progress(
    batch_done : int,
    n_batches  : int,
    total_done : int,
    cached     : int = 0,
) -> None:
    pct    = (batch_done / n_batches) * 100
    suffix = f" | {cached} from cache" if cached else ""
    logger.info(
        f"[embedder] Progress: {batch_done}/{n_batches} batches "
        f"({pct:.1f}%) | {total_done} records built{suffix}"
    )


def _print_stats(records: list[dict], total_chunks: int) -> None:
    embedded  = len(records)
    skipped   = total_chunks - embedded
    skip_rate = (skipped / total_chunks * 100) if total_chunks else 0

    print(
        f"\n[embedder] Stats\n"
        f"  Total chunks   : {total_chunks}\n"
        f"  Embedded       : {embedded}\n"
        f"  Skipped        : {skipped} ({skip_rate:.1f}%)\n"
        f"  Model          : {EMBEDDING_MODEL}\n"
        f"  Dimension      : {EMBEDDING_DIM}\n"
        f"  Batch size     : {BATCH_SIZE}\n"
        f"  Cache file     : {CACHE_PATH} "
        f"({'exists' if CACHE_PATH.exists() else 'not found'})\n"
    )
