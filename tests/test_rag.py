"""
test_rag.py — Tests for the ingestion pipeline and RAG query engine.
"""
import pathlib
import tempfile

import pytest
from reportlab.pdfgen import canvas

from app.ingest import ingest_lease


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
