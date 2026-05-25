"""
rag.py — Dual-source RAG retrieval (lease + state law) backed by ChromaDB.

Retrieval pipeline (two stages):
  Stage 1 — bi-encoder (ChromaDB built-in all-MiniLM-L6-v2)
    Fast semantic lookup; fetches _FETCH_K candidates per source.
    Good recall, but imprecise — topically unrelated chunks can slip
    through when their vector happens to land near the query.

  Stage 2 — cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
    Takes each (query, chunk) pair together and predicts a relevance
    probability. Filters chunks below _RERANK_THRESHOLD before they
    reach the LLM. Downloads ~66 MB on first use; cached locally.

Public API:
  get_retriever(user_id, state)              → {"lease": collection, "law": collection}
  retrieve_context(question, user_id, state) → {"lease_chunks": [...], "law_chunks": [...], "question": ...}
  format_context(retrieved)                  → prompt-ready string
"""

import math
import pathlib

import chromadb

_VECTORSTORE_PATH = str(pathlib.Path(__file__).parent.parent / "vectorstore")
_TOP_K = 3        # max unique results to surface per source
_FETCH_K = 9      # raw results to fetch before dedup (3× headroom)
_MIN_SCORE = 0.0  # bi-encoder floor; dedup + reranker handle precision

# ── Cross-encoder reranker settings ──────────────────────────────────────────
_RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_THRESHOLD = 0.05   # sigmoid probability; 0.05 ≈ "not clearly irrelevant"
                            # raise to 0.3-0.5 to be more aggressive

# Lazy-loaded singleton — avoids downloading the model until first query
_cross_encoder = None


def _get_cross_encoder():
    """Load (and cache) the cross-encoder model on first call."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        print(f"[rag] Loading cross-encoder '{_RERANK_MODEL}' (first-time download ~66 MB)…")
        _cross_encoder = CrossEncoder(_RERANK_MODEL)
        print("[rag] Cross-encoder ready.")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rerank(question: str, chunks: list[tuple], threshold: float = _RERANK_THRESHOLD) -> list[tuple]:
    """
    Re-rank a list of retrieved chunks using a cross-encoder relevance model.

    The cross-encoder reads the (question, chunk) pair together and outputs a
    relevance probability via sigmoid.  Chunks below `threshold` are dropped.

    Falls back to the original bi-encoder ordering if the model is unavailable.

    Args:
        question:  The user's plain-English question.
        chunks:    List of (text, metadata, score) tuples from _deduplicate().
        threshold: Minimum sigmoid probability to keep a chunk (default 0.05).

    Returns:
        Filtered, relevance-sorted list of at most _TOP_K chunks.
    """
    if not chunks:
        return []

    try:
        encoder = _get_cross_encoder()
        pairs = [(question, chunk[0][:800]) for chunk in chunks]   # 800 chars ≈ 200 tokens
        raw_scores = encoder.predict(pairs, show_progress_bar=False)

        # Convert logits → probabilities for interpretable thresholding
        probs = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

        # Sort by probability descending, then filter
        ranked = sorted(zip(chunks, probs), key=lambda x: x[1], reverse=True)
        relevant = [(c, p) for c, p in ranked if p >= threshold]

        if relevant:
            print(
                f"[rag] Cross-encoder kept {len(relevant)}/{len(chunks)} chunks "
                f"(threshold={threshold:.2f})"
            )
            return [c for c, _ in relevant[:_TOP_K]]

        # Nothing passed the threshold — return the single best chunk anyway
        # so the LLM always has something to cite from
        print(
            f"[rag] Cross-encoder: no chunk above {threshold:.2f}; "
            f"returning top-1 (p={ranked[0][1]:.3f})"
        )
        return [ranked[0][0]]

    except Exception as exc:
        print(f"[rag] Cross-encoder reranking failed ({exc}); using bi-encoder order.")
        return chunks[:_TOP_K]


def _text_fingerprint(text: str, length: int = 150) -> str:
    """Normalised prefix used to detect near-duplicate chunks."""
    return " ".join(text.lower().split())[:length]


def _deduplicate(chunks: list[tuple]) -> list[tuple]:
    """
    Remove duplicate and near-duplicate chunks from a result list.

    Strategy:
      1. Sort by score descending so the best copy of a duplicate wins.
      2. Keep a chunk only if its normalised text prefix hasn't been seen.
      3. Return at most _TOP_K unique, relevant chunks.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c[2], reverse=True)
    seen: set[str] = set()
    unique: list[tuple] = []
    for chunk in sorted_chunks:
        if chunk[2] < _MIN_SCORE:
            continue  # skip low-relevance results
        fp = _text_fingerprint(chunk[0])
        if fp not in seen:
            seen.add(fp)
            unique.append(chunk)
        if len(unique) >= _TOP_K:
            break
    return unique


def _load_collection(collection_name: str):
    """
    Return a ChromaDB collection, or None if it doesn't exist / is empty.
    """
    client = chromadb.PersistentClient(path=_VECTORSTORE_PATH)
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return None
    collection = client.get_collection(collection_name)
    if collection.count() == 0:
        return None
    return collection


def _query_collection(collection, question: str, source_type: str) -> list[tuple]:
    """
    Query a ChromaDB collection and return top-k results as plain tuples.

    For lease collections:     (text, page_num, score)
    For state-law collections: (text, state_code, score)

    ChromaDB returns distances (lower = more similar); we convert to a
    similarity score as  score = 1 - distance  (clamped to [0, 1]).
    """
    # Cap fetch at the actual collection size to avoid ChromaDB errors
    n = min(_FETCH_K, collection.count())
    if n == 0:
        return []

    results = collection.query(query_texts=[question], n_results=n)

    raw: list[tuple] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = float(max(0.0, 1.0 - dist))
        if source_type == "lease":
            page = meta.get("page", "?") if meta else "?"
            raw.append((doc, page, score))
        else:
            section = meta.get("state", meta.get("url", "?")) if meta else "?"
            raw.append((doc, section, score))

    return _deduplicate(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_retriever(user_id: str, state: str) -> dict:
    """
    Load ChromaDB collections for a user's lease and their state's tenant law.

    Args:
        user_id: Identifies the ChromaDB lease collection ("lease_{user_id}").
        state:   Two-letter state code ("CA", "NY", etc.).

    Returns:
        {"lease": <collection or None>, "law": <collection or None>}
        Either value may be None if the collection hasn't been ingested yet.
    """
    lease_col = _load_collection(f"lease_{user_id}")
    law_col = _load_collection(f"state_law_{state.lower()}")

    if lease_col is None:
        print(f"[rag] Warning: collection 'lease_{user_id}' not found or empty.")
    if law_col is None:
        print(f"[rag] Warning: collection 'state_law_{state.lower()}' not found or empty.")

    return {"lease": lease_col, "law": law_col}


def retrieve_context(question: str, user_id: str, state: str) -> dict:
    """
    Retrieve the most relevant chunks from both the lease and state law.

    Args:
        question: The tenant's question in plain English.
        user_id:  Identifies the lease collection.
        state:    Two-letter state code.

    Returns:
        {
            "lease_chunks": [(text, page_num, score), ...],  # up to TOP_K entries
            "law_chunks":   [(text, section,  score), ...],  # up to TOP_K entries
            "question":     question
        }
    """
    collections = get_retriever(user_id, state)

    # Stage 1 — bi-encoder retrieval + deduplication
    lease_chunks: list[tuple] = []
    if collections["lease"] is not None:
        lease_chunks = _query_collection(collections["lease"], question, "lease")

    law_chunks: list[tuple] = []
    if collections["law"] is not None:
        law_chunks = _query_collection(collections["law"], question, "law")

    # Stage 2 — cross-encoder reranking (filters topically irrelevant chunks)
    lease_chunks = _rerank(question, lease_chunks)
    law_chunks   = _rerank(question, law_chunks)

    return {
        "lease_chunks": lease_chunks,
        "law_chunks": law_chunks,
        "question": question,
    }


def format_context(retrieved: dict) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM prompt.

    Args:
        retrieved: The dict returned by retrieve_context().

    Returns:
        A multi-line string with clearly labelled sections.
    """
    lines = []

    lines.append("FROM YOUR LEASE:")
    lines.append("-" * 40)
    if retrieved["lease_chunks"]:
        for i, (text, page, score) in enumerate(retrieved["lease_chunks"], start=1):
            lines.append(f"[Chunk {i} | Page {page} | Relevance {score:.2f}]")
            lines.append(text.strip())
            lines.append("")
    else:
        lines.append(
            "(LEASE: No relevant sections found. "
            "This topic may not be addressed in the tenant's lease, "
            "or may use different terminology.)"
        )
        lines.append("")

    lines.append("FROM YOUR STATE LAW:")
    lines.append("-" * 40)
    if retrieved["law_chunks"]:
        for i, (text, section, score) in enumerate(retrieved["law_chunks"], start=1):
            lines.append(f"[Chunk {i} | {section} | Relevance {score:.2f}]")
            lines.append(text.strip())
            lines.append("")
    else:
        lines.append(
            "(STATE LAW: No relevant statute sections found for this question.)"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    question = "What are the rules about late fees and security deposit?"
    user_id = "test"
    state = "CA"

    print(f"Question: {question}\n")
    retrieved = retrieve_context(question, user_id=user_id, state=state)
    print(f"Lease chunks:     {len(retrieved['lease_chunks'])}")
    print(f"Law chunks:       {len(retrieved['law_chunks'])}\n")
    print(format_context(retrieved))

    if not retrieved["lease_chunks"] and not retrieved["law_chunks"]:
        print("No chunks found. Make sure you have ingested a lease and state law first.")
        print("  python app/ingest.py            # ingest dummy lease")
        print("  python data/scrape_laws.py CA   # ingest CA law")
        sys.exit(1)
