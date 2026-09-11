from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
import hashlib
import re
from config import SPLIT_CONFIGS, MIN_CHUNK_SIZE

from tech_classifier import classify_technology, classify_domain

HUAWEI_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",
    "\nStep ", "\nProcedure ",
    "\nAlarm ID:", "\nAlarm Name:",
    "\nParameter:", "\nCommand:",
    "\n\n", "\n", " ", "",
]

STANDARD_SEPARATORS = ["\n\n", "\n", " ", ""]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def split_documents(docs: list[Document]) -> list[Document]:

    groups = defaultdict(list)
    for doc in docs:
        groups[doc.metadata.get("file_type", "pdf")].append(doc)

    all_chunks = []

    for file_type, group_docs in groups.items():

        cfg = SPLIT_CONFIGS.get(file_type, SPLIT_CONFIGS["pdf"])

        group_docs = sorted(group_docs, key=lambda x: x.metadata.get("file_name", ""))

        separators = (
            HUAWEI_SEPARATORS
            if file_type in ("pdf", "docx", "doc")
            else STANDARD_SEPARATORS
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=separators,
            length_function=len,
            add_start_index=True,
        )

        raw_chunks = splitter.split_documents(group_docs)

        # 1. dedup (avec normalisation)
        unique_chunks = _deduplicate_raw(raw_chunks)

        # 2. filter AVANT enrich
        filtered = [
            c for c in unique_chunks
            if len(c.page_content.strip()) >= MIN_CHUNK_SIZE
            and not _is_noise(c.page_content)
        ]

        # 3. enrich APRÈS filter
        enriched = _enrich_metadata(filtered)

        print(f"[splitter] {file_type:5} | {len(group_docs):5} docs → {len(enriched):6} chunks")

        all_chunks.extend(enriched)

    print(f"\n[splitter] TOTAL → {len(all_chunks)} chunks ")
    _print_stats(all_chunks)

    return all_chunks


# ─────────────────────────────────────────────
# NOISE FILTER
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# NOISE FILTER — amélioré
# ─────────────────────────────────────────────

def _is_noise(text: str) -> bool:
    """
    Filtre les chunks sans valeur sémantique.
    - < 4 mots → trop court
    - unique_ratio < 0.3 → répétition (headers boilerplate)
    - trop de pipes → table brute sans texte
    """
    words = text.split()

    if len(words) < 4:
        return True

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.3:
        return True

    if text.count("|") > len(words) * 0.5:
        return True

    return False


# ─────────────────────────────────────────────
# METADATA ENRICHMENT
# ─────────────────────────────────────────────

def _enrich_metadata(chunks: list[Document]) -> list[Document]:

    # FIX — chunk_total basé sur grouping (robuste aux refactors futurs)
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c.metadata.get("file_name", "unknown")].append(c)
    counts = {k: len(v) for k, v in grouped.items()}

    cursor = defaultdict(int)

    for c in chunks:

        src  = c.metadata.get("file_name", "unknown")
        text = c.page_content
        idx  = cursor[src]

        content_fingerprint = hashlib.md5(text.encode()).hexdigest()[:10]
        raw_id = f"{src}_{idx}_{content_fingerprint}"

        c.metadata["chunk_id"]       = hashlib.md5(raw_id.encode()).hexdigest()
        c.metadata["chunk_hash"]     = hashlib.sha256(text.encode()).hexdigest()
        c.metadata["parent_doc"]     = src
        c.metadata["doc_id"]         = hashlib.md5(src.encode()).hexdigest()
        c.metadata["chunk_index"]    = idx
        c.metadata["chunk_total"]    = counts[src]
        c.metadata["token_estimate"] = round(len(text) / 3.5)

        result = classify_technology(filename=src, text=text)

        c.metadata["technology"]      = result.primary
        c.metadata["technologies"]    = result.technologies
        c.metadata["tech_confidence"] = result.confidence
        c.metadata["tech_matched_on"] = result.matched_on

        doc_type = c.metadata.get("doc_type") or "technical_doc"
        c.metadata["domain"]      = classify_domain(result.primary, doc_type)
        c.metadata["source_type"] = "huawei_manual"

        cursor[src] += 1

    return chunks


# ─────────────────────────────────────────────
# DEDUP avec normalisation
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalise le texte avant hash — évite faux négatifs dus au whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _deduplicate_raw(chunks: list[Document]) -> list[Document]:

    seen   = set()
    unique = []

    for c in chunks:
        h = hashlib.sha256(_normalize(c.page_content).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)

    removed = len(chunks) - len(unique)
    if removed:
        print(f"[splitter] {removed} duplicates removed")

    return unique


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def _print_stats(chunks: list[Document]):

    if not chunks:
        return

    sizes = [len(c.page_content) for c in chunks]

    print(
        f"[splitter] Stats → "
        f"min: {min(sizes)} | "
        f"max: {max(sizes)} | "
        f"avg: {sum(sizes)//len(sizes)} chars"
    )
