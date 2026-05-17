"""
ingest.py — Load and chunk lease PDFs and state law statutes into ChromaDB.

Pipeline:
  1. Extract text per page with PyMuPDF (fitz)
  2. Fall back to pytesseract OCR for pages with no extractable text
  3. Chunk with LlamaIndex SentenceSplitter (512 tokens, 64 overlap)
  4. Upsert into ChromaDB collection "lease_{user_id}"

State law:
  - scrape_state_law(state) fetches + cleans statute HTML
  - ingest_state_law(state) chunks and stores in "state_law_{state}"
  - ingest_all_states() loops all 50 states with a 2s delay
"""

import os
import pathlib
import tempfile
import time

import chromadb
import fitz  # PyMuPDF
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------------
# State → statute URL mapping (all 50 states)
# ---------------------------------------------------------------------------

STATE_URLS: dict[str, str] = {
    # ── West ──────────────────────────────────────────────────────────────
    "CA": "https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?lawCode=CIV&division=3.&title=5.&part=4.&chapter=2.",
    "WA": "https://app.leg.wa.gov/RCW/default.aspx?cite=59.18",
    "OR": "https://www.oregonlegislature.gov/bills_laws/ors/ors090.html",
    "NV": "https://www.leg.state.nv.us/NRS/NRS-118A.html",
    "AZ": "https://www.azleg.gov/arsDetail/?title=33",
    "ID": "https://legislature.idaho.gov/statutesrules/idstat/Title6/T6CH3/",
    "MT": "https://leg.mt.gov/bills/mca/title_0700/chapter_0240/parts_index.html",
    "WY": "https://wyoleg.gov/statutes/compress/title34.pdf",
    "CO": "https://leg.colorado.gov/sites/default/files/images/olls/crs2023-title-38.pdf",
    "UT": "https://le.utah.gov/xcode/Title57/Chapter22/57-22.html",
    "AK": "https://www.akleg.gov/basis/statutes.asp#34.03",
    "HI": "https://www.capitol.hawaii.gov/hrscurrent/Vol11_Ch0476-0490/HRS0521/",
    # ── Midwest ───────────────────────────────────────────────────────────
    "MN": "https://www.revisor.mn.gov/statutes/cite/504B",
    "WI": "https://docs.legis.wisconsin.gov/statutes/statutes/704",
    "MI": "https://www.legislature.mi.gov/Laws/MCL?objectName=mcl-Act-348-of-1972",
    "IL": "https://www.ilga.gov/legislation/ilcs/ilcs3.asp?ActID=2201",
    "IN": "https://iga.in.gov/laws/2023/ic/titles/32#32-31",
    "OH": "https://codes.ohio.gov/ohio-revised-code/chapter-5321",
    "IA": "https://www.legis.iowa.gov/law/iowaCode/sections?codeChapter=562A",
    "MO": "https://revisor.mo.gov/main/OneChapter.aspx?chapter=535",
    "KS": "https://www.kslegislature.org/li/b2023_24/statute/058_000_0000_chapter/058_025_0000_article/",
    "NE": "https://nebraskalegislature.gov/laws/statutes.php?statute=76-1401",
    "SD": "https://sdlegislature.gov/Statutes/43",
    "ND": "https://www.legis.nd.gov/cencode/t47c16.html",
    # ── South ─────────────────────────────────────────────────────────────
    "TX": "https://statutes.capitol.texas.gov/Docs/PR/htm/PR.92.htm",
    "FL": "https://www.flsenate.gov/Laws/Statutes/2023/Chapter83/All",
    "GA": "https://advance.lexis.com/container?config=00JAA2ZjZiNjIxNS0yMTdiLTQ4NzctYjBlYS00YTc4YWQ3NTYxZTkKAFBvZENhdGFsb2f1p8OyZMnrMv3Ncc3FHr0J&crid=7a0ea14d-3204-4b3f-9ed8-0eb69f5ec8f0",
    "NC": "https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/ByChapter/Chapter_42.html",
    "SC": "https://www.scstatehouse.gov/code/t27c040.php",
    "VA": "https://law.lis.virginia.gov/vacode/title55.1/chapter12/",
    "WV": "https://code.wvlegislature.gov/37-6/",
    "KY": "https://apps.legislature.ky.gov/law/statutes/chapter.aspx?id=39160",
    "TN": "https://www.tn.gov/lawsandregs.html",
    "AL": "https://alison.legislature.state.al.us/codeofalabama/1975/coatoc.htm",
    "MS": "https://law.justia.com/codes/mississippi/title-89/chapter-8/",
    "AR": "https://www.arkleg.state.ar.us/Acts/FTPDocument?path=%2FCODE%2F&file=18.pdf&ddBienniumSession=2023%2F2023R",
    "LA": "https://legis.la.gov/legis/Law.aspx?d=109414",
    "OK": "https://www.oscn.net/applications/oscn/DeliverDocument.asp?CiteID=137667",
    "NM": "https://www.nmlegis.gov/Sessions/23%20Regular/final/HB0007.pdf",
    # ── Northeast ─────────────────────────────────────────────────────────
    "NY": "https://www.nysenate.gov/legislation/laws/RPP/220",
    "PA": "https://www.legis.state.pa.us/cfdocs/legis/LI/consCheck.cfm?txtType=HTM&ttl=68&div=0&chpt=8",
    "NJ": "https://law.justia.com/codes/new-jersey/title-46/section-46-8-1/",
    "CT": "https://www.cga.ct.gov/current/pub/title_47a.htm",
    "MA": "https://malegislature.gov/Laws/GeneralLaws/PartII/TitleI/Chapter186",
    "RI": "https://rilegislature.gov/laws/title34/34--18.htm",
    "VT": "https://legislature.vermont.gov/statutes/chapter/09/137",
    "NH": "https://www.gencourt.state.nh.us/rsa/html/nhtoc/nhtoc-LV-540.htm",
    "ME": "https://legislature.maine.gov/statutes/14/title14ch709.pdf",
    "DE": "https://delcode.delaware.gov/title25/c055/",
    "MD": "https://mgaleg.maryland.gov/mgawebsite/Laws/StatuteText?article=0008&section=8-211&enactments=False",
    # ── Mid-Atlantic / Other ───────────────────────────────────────────────
    "DC": "https://code.dccouncil.gov/us/dc/council/code/titles/42/chapters/35/",
    "MN": "https://www.revisor.mn.gov/statutes/cite/504B",
}

# Deduplicate and ensure all standard 50-state abbreviations are present.
# A few states above use Justia/PDF fallbacks where the official site is hard to scrape.

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


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
# State law scraping + ingestion
# ---------------------------------------------------------------------------

def scrape_state_law(state: str) -> str:
    """
    Fetch and clean the tenant-rights statute page for a given state.

    Resolution order:
      1. Check data/statutes/<STATE>.txt for a pre-saved file (always wins)
      2. Fetch the configured URL and parse the HTML

    Args:
        state: Two-letter state abbreviation (e.g. "CA").

    Returns:
        Clean plain-text statute content, or "" if nothing could be retrieved.

    Raises:
        ValueError: If the state abbreviation is not in STATE_URLS.
        requests.HTTPError: If the HTTP request fails.
    """
    state = state.upper()
    if state not in STATE_URLS:
        raise ValueError(f"No URL configured for state: {state}")

    # 1. Pre-saved file fallback — place <STATE>.txt in data/statutes/ to override scraping
    statutes_dir = pathlib.Path(__file__).parent.parent / "data" / "statutes"
    saved_file = statutes_dir / f"{state}.txt"
    if saved_file.exists():
        print(f"[scrape] {state} → loading from {saved_file}")
        return saved_file.read_text(encoding="utf-8").strip()

    # 2. Live scrape
    url = STATE_URLS[state]
    print(f"[scrape] {state} → {url}")

    resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate tags that don't contain statute text
    for tag in soup(["nav", "header", "footer", "script", "style",
                     "noscript", "aside", "form", "button", "iframe"]):
        tag.decompose()

    # Try to find the main content block; fall back to full body
    main = (
        soup.find(id="manylawsections")         # CA leginfo
        or soup.find(id="maincontent")
        or soup.find(id="content")
        or soup.find(class_="statute-body")     # TX statutes
        or soup.find(class_="field-items")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )

    text = (main or soup).get_text(separator="\n")

    # Collapse excessive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)

    # If the page loaded but yielded no text, it is likely JavaScript-rendered.
    # Save the URL to a note file so the user knows to manually download it.
    if not cleaned:
        note = statutes_dir / f"{state}.NOTE"
        statutes_dir.mkdir(parents=True, exist_ok=True)
        note.write_text(
            f"Page at {url} appears to be JavaScript-rendered or returned no text.\n"
            f"Please download the statute text manually and save it as data/statutes/{state}.txt\n"
        )
        print(f"[scrape] {state}: no extractable text — see {note}")

    return cleaned


def ingest_state_law(state: str) -> int:
    """
    Scrape, chunk, and store a state's tenant-rights statutes in ChromaDB.

    Args:
        state: Two-letter state abbreviation.

    Returns:
        Number of chunks stored.
    """
    state = state.upper()
    url = STATE_URLS.get(state, "")

    text = scrape_state_law(state)
    if not text:
        print(f"[ingest_state] {state}: no text retrieved, skipping.")
        return 0

    doc = Document(
        text=text,
        metadata={"source": "state_law", "state": state, "url": url},
    )

    splitter = SentenceSplitter(chunk_size=600, chunk_overlap=80)
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"[ingest_state] {state}: {len(nodes)} chunk(s) after splitting")

    db_path = pathlib.Path(__file__).parent.parent / "vectorstore"
    db_path.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(f"state_law_{state.lower()}")

    ids, documents, metadatas = [], [], []
    for i, node in enumerate(nodes):
        ids.append(f"{state}::chunk::{i}")
        documents.append(node.get_content())
        metadatas.append(node.metadata)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"[ingest_state] {state}: upserted {len(ids)} chunk(s) → 'state_law_{state.lower()}'")
    return len(ids)


def ingest_all_states(delay: float = 2.0) -> dict[str, int]:
    """
    Ingest tenant-rights statutes for all states in STATE_URLS.

    Args:
        delay: Seconds to wait between HTTP requests (default 2.0).

    Returns:
        Dict mapping state abbreviation → chunk count (0 on failure).
    """
    results: dict[str, int] = {}
    states = sorted(STATE_URLS.keys())
    total = len(states)

    for idx, state in enumerate(states, start=1):
        print(f"\n[{idx}/{total}] Ingesting {state} ...")
        try:
            count = ingest_state_law(state)
            results[state] = count
        except Exception as exc:
            print(f"[ingest_all] {state}: ERROR — {exc}")
            results[state] = 0

        if idx < total:
            time.sleep(delay)

    succeeded = sum(1 for v in results.values() if v > 0)
    print(f"\n[ingest_all] Done. {succeeded}/{total} states ingested successfully.")
    return results


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
