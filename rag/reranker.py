"""
reranker.py — Cross-encoder reranker for Huawei RAN RAG pipeline
Model: cross-encoder/ms-marco-MiniLM-L-6-v2

Flow:
  Qdrant hybrid retrieval (top CANDIDATE_POOL)
      ↓
  Cross-encoder reranker (GPU si dispo, sinon CPU)
      ↓
  Top K final results
      ↓
  LLM Generator

OPTIMISATIONS v2:
  - Auto-détection GPU (torch.cuda) → CPU fallback propre
  - Device loggé au chargement pour diagnostic immédiat
  - Aucun changement d'interface — drop-in replacement
"""

import logging
import threading

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker      = None
_reranker_lock = threading.Lock()


def _detect_device() -> str:
    """
    Retourne 'cuda' si une GPU NVIDIA est disponible, sinon 'cpu'.
    Ne lève jamais d'exception — torch peut ne pas être installé.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[reranker] GPU détectée : {gpu_name} → device=cuda")
            return "cuda"
    except Exception:
        pass
    logger.info("[reranker] Aucune GPU détectée → device=cpu")
    return "cpu"


def _get_reranker():
    """
    Lazy loading thread-safe.
    Charge le modèle une seule fois sur le device optimal (GPU > CPU).
    Retourne None si sentence_transformers n'est pas installé.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    from sentence_transformers import CrossEncoder

                    device = _detect_device()
                    logger.info(f"[reranker] Chargement du modèle : {RERANKER_MODEL} sur {device}")

                    _reranker = CrossEncoder(
                        RERANKER_MODEL,
                        max_length=512,
                        device=device,          # ← GPU si dispo, CPU sinon
                    )
                    logger.info(f"[reranker] Modèle chargé sur {device.upper()}")

                except ImportError as e:
                    logger.warning(
                        f"[reranker] sentence_transformers non disponible ({e}) "
                        "— reranking désactivé"
                    )
                    _reranker = False
                except Exception as e:
                    logger.error(f"[reranker] Échec du chargement : {e}")
                    _reranker = False

    return _reranker if _reranker is not False and _reranker is not None else None


# ─────────────────────────────────────────────
# MAIN RERANK FUNCTION
# ─────────────────────────────────────────────

def rerank(
    question: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Reranks retrieval results using a cross-encoder.

    Le cross-encoder lit (question, chunk) ensemble et score la pertinence
    plus précisément que la similarité vectorielle seule.

    Args:
        question : question originale de l'utilisateur
        results  : liste de dicts issus de retrieve_hybrid()
                   chaque dict doit avoir un champ "text"
        top_k    : nombre de résultats à retourner après reranking

    Returns:
        Liste reranked triée par score décroissant.
        Chaque dict reçoit un champ "rerank_score".
        Si le reranker est indisponible, retourne les top_k sans reranking.
    """
    if not results:
        return []

    if len(results) <= 1:
        for r in results:
            r.setdefault("rerank_score", 0.0)
        return results

    reranker = _get_reranker()
    if reranker is None:
        logger.warning("[reranker] Indisponible — retour des top candidats sans reranking")
        for r in results[:top_k]:
            r.setdefault("rerank_score", 0.0)
        return results[:top_k]

    # Construire les paires (question, texte_chunk) pour le cross-encoder
    pairs = [(question, r.get("text", r.get("content", ""))) for r in results]

    logger.info(
        f"[reranker] Scoring {len(pairs)} candidats "
        f"→ retour top {top_k}"
    )

    # Scorer toutes les paires
    scores = reranker.predict(pairs)

    # Attacher le score rerank à chaque résultat
    for result, score in zip(results, scores):
        result["rerank_score"] = round(float(score), 4)

    # Trier par score décroissant
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    # Garder top_k
    final = reranked[:top_k]

    if final:
        logger.info(
            f"[reranker] Top résultat : score={final[0]['rerank_score']} "
            f"| source : {final[0].get('source', 'unknown')}"
        )

    return final


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)

    from retrieval import retrieve_hybrid

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
               "ALM-26230 BBU CPRI optical module fault cause and solution"

    print(f"\nQuery: {question}\n")

    # Step 1 — récupérer les candidats
    candidates = retrieve_hybrid(question, top_k=100)
    print(f"Récupéré {len(candidates)} candidats depuis Qdrant\n")

    # Step 2 — reranker
    reranked = rerank(question, candidates, top_k=5)

    print("─── Résultats reranked ───")
    for i, r in enumerate(reranked, 1):
        print(
            f"[{i}] rerank: {r.get('rerank_score', 'N/A'):+.4f} "
            f"| vector: {r.get('score', 'N/A')} "
            f"| {r.get('source', 'unknown')}"
        )
        print(f"     {r.get('text', r.get('content', ''))[:200]}...")
        print()