"""
ingest.py — Load and chunk lease PDFs into ChromaDB.

Pipeline:
  1. Extract text per page with PyMuPDF (fitz)
  2. Fall back to pytesseract OCR for pages with no extractable text
  3. Chunk with LlamaIndex SentenceSplitter (512 tokens, 64 overlap)
  4. Upsert into ChromaDB collection "lease_{user_id}"
"""

import os
import pathlib
import tempfile

import chromadb
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text_from_page(page: fitz.Page) -> str:
    """Return text from a fitz page, falling back to OCR if the page is image-only."""
    text = page.get_text().strip()
    if text:
        return text

    # OCR fallback: render page to image and run tesseract
    pix = page.get_pixmap(dpi=200)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pix.save(tmp_path)
        img = Image.open(tmp_path)
        text = pytesseract.image_to_string(img).strip()
    finally:
        os.unlink(tmp_path)
    return text


def _extract_pages(pdf_path: str) -> list[dict]:
    """Return a list of {page_num, text} dicts for every page in the PDF."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = _extract_text_from_page(page)
            if text:
                pages.append({"page_num": page_num, "text": text})
    return pages


def _chunk_pages(pages: list[dict], filename: str) -> list[Document]:
    """Convert raw page text into LlamaIndex Documents with metadata."""
    documents = []
    for p in pages:
        documents.append(
            Document(
                text=p["text"],
                metadata={
                    "source": "lease",
                    "filename": filename,
                    "page": p["page_num"],
                },
            )
        )

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def _get_collection(user_id: str) -> chromadb.Collection:
    """Return (or create) the ChromaDB collection for this user."""
    db_path = pathlib.Path(__file__).parent.parent / "vectorstore"
    db_path.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(f"lease_{user_id}")


def _upsert_nodes(collection: chromadb.Collection, nodes, filename: str) -> int:
    """Upsert LlamaIndex nodes into ChromaDB. Returns the number of chunks added."""
    ids, documents, metadatas = [], [], []
    for i, node in enumerate(nodes):
        chunk_id = f"{filename}::chunk::{i}"
        ids.append(chunk_id)
        documents.append(node.get_content())
        metadatas.append(node.metadata)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return len(ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_lease(pdf_path: str, user_id: str = "default") -> int:
    """
    Ingest a lease PDF into ChromaDB.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        user_id:  Identifier used to namespace the ChromaDB collection.

    Returns:
        Number of chunks stored.
    """
    pdf_path = str(pathlib.Path(pdf_path).resolve())
    filename = pathlib.Path(pdf_path).name

    print(f"[ingest] Reading {filename} ...")
    pages = _extract_pages(pdf_path)
    print(f"[ingest] Extracted text from {len(pages)} page(s)")

    nodes = _chunk_pages(pages, filename)
    print(f"[ingest] Split into {len(nodes)} chunk(s)")

    collection = _get_collection(user_id)
    count = _upsert_nodes(collection, nodes, filename)
    print(f"[ingest] Upserted {count} chunk(s) → collection 'lease_{user_id}'")

    return count


def ingest_statutes(statutes_dir: str):
    """Load pre-scraped state law text files and upsert into the vector store."""
    pass


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from reportlab.pdfgen import canvas

    # Build a tiny 1-page PDF in /tmp
    tmp_pdf = pathlib.Path(tempfile.mkdtemp()) / "dummy_lease.pdf"
    c = canvas.Canvas(str(tmp_pdf))
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "LEASE AGREEMENT")
    c.drawString(72, 700, "Tenant agrees to pay $1,500/month rent on the 1st of each month.")
    c.drawString(72, 680, "The lease term is 12 months commencing January 1, 2025.")
    c.drawString(72, 660, "Late fees of $50 apply after a 5-day grace period.")
    c.save()

    print(f"\nDummy PDF created at: {tmp_pdf}")
    chunks = ingest_lease(str(tmp_pdf), user_id="test")
    print(f"\nResult: {chunks} chunk(s) ingested.")
    sys.exit(0 if chunks > 0 else 1)
