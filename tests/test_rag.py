"""
test_rag.py — Tests for the ingestion pipeline and RAG query engine.
"""
import pathlib
import tempfile

import pytest
from reportlab.pdfgen import canvas

from app.ingest import ingest_lease, ingest_state_law
from app.rag import retrieve_context, format_context, get_retriever
from app.chat import ask, generate_answer, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_pdf(text_lines: list[str]) -> pathlib.Path:
    """Create a single-page PDF containing the given lines and return its path."""
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / "test_lease.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.setFont("Helvetica", 12)
    y = 720
    for line in text_lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return pdf_path


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

def test_ingest_returns_nonzero_chunk_count():
    """Ingesting a simple lease PDF should produce at least one chunk."""
    pdf_path = _make_temp_pdf([
        "LEASE AGREEMENT",
        "Tenant agrees to pay $1,500 per month.",
        "The lease term is 12 months starting January 1, 2025.",
        "Late fees of $50 apply after a 5-day grace period.",
        "No pets are allowed on the premises without written consent.",
    ])
    chunk_count = ingest_lease(str(pdf_path), user_id="pytest")
    assert chunk_count > 0, "Expected at least one chunk to be ingested"


def test_ingest_returns_int():
    """ingest_lease should always return an integer."""
    pdf_path = _make_temp_pdf(["Short lease text for type check."])
    result = ingest_lease(str(pdf_path), user_id="pytest_type")
    assert isinstance(result, int)


def test_ingest_idempotent():
    """Ingesting the same PDF twice should not raise and should return same count."""
    pdf_path = _make_temp_pdf([
        "Idempotency test lease.",
        "Tenant shall maintain the unit in good condition.",
    ])
    count_first = ingest_lease(str(pdf_path), user_id="pytest_idem")
    count_second = ingest_lease(str(pdf_path), user_id="pytest_idem")
    assert count_first == count_second, "Re-ingesting the same PDF should produce the same chunk count"


# ---------------------------------------------------------------------------
# RAG retrieval tests
# ---------------------------------------------------------------------------

_RAG_USER_ID = "pytest_rag"
_RAG_STATE = "CA"

_LEASE_LINES = [
    "LEASE AGREEMENT — RAG TEST",
    "Monthly rent is $1,800, due on the 1st of each month.",
    "A security deposit of $3,600 (two months rent) is required.",
    "Late fees of $75 apply if rent is not received by the 5th.",
    "The landlord must give 24-hour notice before entering the unit.",
    "No smoking is permitted anywhere on the premises.",
    "Tenant may not sublet without written consent of the landlord.",
    "The lease term is 12 months commencing February 1, 2025.",
]


@pytest.fixture(scope="module")
def rag_setup():
    """Ingest a test lease + CA state law once for all RAG tests."""
    pdf_path = _make_temp_pdf(_LEASE_LINES)
    ingest_lease(str(pdf_path), user_id=_RAG_USER_ID)
    ingest_state_law(_RAG_STATE)
    return {"user_id": _RAG_USER_ID, "state": _RAG_STATE}


def test_retrieve_context_returns_both_sources(rag_setup):
    """retrieve_context should return non-empty chunks from both lease and state law."""
    result = retrieve_context(
        question="What is the security deposit and what are my rights?",
        user_id=rag_setup["user_id"],
        state=rag_setup["state"],
    )
    assert "lease_chunks" in result
    assert "law_chunks" in result
    assert "question" in result
    assert len(result["lease_chunks"]) > 0, "Expected lease chunks"
    assert len(result["law_chunks"]) > 0, "Expected state law chunks"


def test_retrieve_context_chunk_structure(rag_setup):
    """Each chunk should be a 3-tuple: (text, page_or_section, score)."""
    result = retrieve_context(
        question="late fee notice entry",
        user_id=rag_setup["user_id"],
        state=rag_setup["state"],
    )
    for chunk in result["lease_chunks"] + result["law_chunks"]:
        assert len(chunk) == 3, "Each chunk must be a (text, metadata, score) tuple"
        text, meta, score = chunk
        assert isinstance(text, str) and len(text) > 0
        assert isinstance(score, float)


def test_format_context_contains_section_headers(rag_setup):
    """format_context output must include both required section headers."""
    result = retrieve_context(
        question="Can my landlord enter without notice?",
        user_id=rag_setup["user_id"],
        state=rag_setup["state"],
    )
    formatted = format_context(result)
    assert "FROM YOUR LEASE:" in formatted
    assert "FROM YOUR STATE LAW:" in formatted


def test_get_retriever_returns_dict(rag_setup):
    """get_retriever should return a dict with 'lease' and 'law' keys."""
    retrievers = get_retriever(rag_setup["user_id"], rag_setup["state"])
    assert "lease" in retrievers
    assert "law" in retrievers
    assert retrievers["lease"] is not None
    assert retrievers["law"] is not None


# ---------------------------------------------------------------------------
# Answer generation tests
# ---------------------------------------------------------------------------

def test_ask_returns_nonempty_answer(rag_setup):
    """ask() should return a non-empty answer string grounded in the context."""
    result = ask(
        question="What is the late fee policy in my lease?",
        user_id=rag_setup["user_id"],
        state=rag_setup["state"],
    )

    assert "answer" in result
    assert "lease_sources" in result
    assert "law_sources" in result
    assert "question" in result

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0, "Expected a non-empty answer"

    answer_lower = result["answer"].lower()
    assert "lease" in answer_lower or "law" in answer_lower, (
        "Answer should reference 'lease' or 'law'"
    )


def test_ask_response_structure(rag_setup):
    """ask() sources should be lists of dicts with expected keys."""
    result = ask(
        question="Can my landlord enter the unit without notice?",
        user_id=rag_setup["user_id"],
        state=rag_setup["state"],
    )
    for source in result["lease_sources"]:
        assert "text" in source and "page" in source
    for source in result["law_sources"]:
        assert "text" in source and "section" in source


def test_system_prompt_is_string():
    """SYSTEM_PROMPT must be a non-empty string."""
    assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 0
