"""
rag.py — Dual-source RAG retrieval (lease + state law) backed by ChromaDB.

ChromaDB manages its own embeddings (all-MiniLM-L6-v2) so we query it
directly rather than routing through LlamaIndex's VectorStoreIndex, which
would require a separate embedding model (OpenAI / HuggingFace).

Public API:
  get_retriever(user_id, state)              → {"lease": collection, "law": collection}
  retrieve_context(question, user_id, state) → {"lease_chunks": [...], "law_chunks": [...], "question": ...}
  format_context(retrieved)                  → prompt-ready string
"""

import pathlib

import chromadb

_VECTORSTORE_PATH = str(pathlib.Path(__file__).parent.parent / "vectorstore")
_TOP_K = 3        # max unique results to surface per source
_FETCH_K = 9      # raw results to fetch before dedup (3× headroom)
_MIN_SCORE = 0.0  # drop only negative-clamped scores; deduplication handles quality


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

    lease_chunks = []
    if collections["lease"] is not None:
        lease_chunks = _query_collection(collections["lease"], question, "lease")

    law_chunks = []
    if collections["law"] is not None:
        law_chunks = _query_collection(collections["law"], question, "law")

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
