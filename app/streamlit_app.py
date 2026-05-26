"""
streamlit_app.py — Main Lease Lens web interface.

Run with:
    streamlit run app/streamlit_app.py
"""

import pathlib
import tempfile
import uuid
from typing import Any

import chromadb
import fitz
import streamlit as st

from app.ingest import ingest_lease, ingest_state_law
from app.chat import ask
from app.utils import scan_illegal_clauses, get_state
from app.documents import (
    generate_deposit_letter,
    generate_repair_notice,
    generate_moveout_checklist,
    save_letter_pdf,
    STATE_DEPOSIT_LAW,
    STATE_HABITABILITY_LAW,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lease Lens",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

STARTER_QUESTIONS = [
    "Can my landlord enter without notice?",
    "What can they deduct from my deposit?",
    "Am I responsible for this repair?",
    "How much notice do I need to give?",
]

_VECTORSTORE_PATH = str(pathlib.Path(__file__).parent.parent / "vectorstore")

# ---------------------------------------------------------------------------
# Cached resource: ChromaDB client (shared across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_chroma_client() -> chromadb.PersistentClient:
    """Single ChromaDB client reused for the lifetime of the Streamlit process."""
    pathlib.Path(_VECTORSTORE_PATH).mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=_VECTORSTORE_PATH)


def _collection_exists(name: str) -> bool:
    client = _get_chroma_client()
    return any(c.name == name for c in client.list_collections())


def _state_law_indexed(state: str) -> bool:
    return _collection_exists(f"state_law_{state.lower()}")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "messages": [],
        "user_id": str(uuid.uuid4())[:8],
        "state": get_state(),
        "lease_loaded": False,
        "lease_chunk_count": 0,
        "lease_text": "",
        "scan_results": None,
        "pending_question": None,
        "letter_text": "",
        "letter_pdf_path": None,
        "repair_notice_text": "",
        "repair_notice_pdf_path": None,
        "checklist_pdf_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Source deduplication helper
# ---------------------------------------------------------------------------

def _dedup_sources(sources: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
    """Return a deduplicated list of source dicts, keeping the first occurrence of each unique text."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for s in sources:
        # Use first 150 normalised chars as a fingerprint
        fp = " ".join(s.get(text_key, "").lower().split())[:150]
        if fp and fp not in seen:
            seen.add(fp)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "connection" in msg or "refused" in msg or "connect" in msg:
        return (
            "**Local AI model is not running.**\n\n"
            "Start it in a terminal with:\n```\nollama serve\n```"
        )
    if "not found" in msg or "does not exist" in msg:
        return (
            "**Your lease hasn't been uploaded yet.**\n\n"
            "Upload your lease PDF in the sidebar to get started."
        )
    return f"**Something went wrong:** {exc}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<h1 style="font-family: Playfair Display, serif; font-size: 22px; font-weight: 900; color: #F9F9F7; letter-spacing: -0.01em;">🏠 Lease Lens</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-family: Inter, sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 0.2em; color: #A3A3A3; margin-top: 0;">Tenant Rights Assistant</p>', unsafe_allow_html=True)
    st.divider()

    # State selector
    default_idx = STATES.index(st.session_state.state) if st.session_state.state in STATES else 4
    selected_state = st.selectbox(
        "Your state",
        options=STATES,
        index=default_idx,
        help="Used to look up your state's tenant protection laws.",
    )
    st.session_state.state = selected_state

    st.divider()

    # PDF uploader
    st.markdown("**Upload your lease**")
    uploaded_file = st.file_uploader(
        "Drop your lease PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        suffix = pathlib.Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Detect scanned-only PDFs before full ingestion
            with fitz.open(tmp_path) as doc:
                raw_text = "\n\n".join(page.get_text() for page in doc)
                is_scanned = len(raw_text.strip()) < 100

            if is_scanned:
                st.info(
                    "This PDF appears to be scanned. "
                    "Using OCR — this may take 30 seconds."
                )

            with st.spinner("Ingesting lease..."):
                chunk_count = ingest_lease(tmp_path, user_id=st.session_state.user_id)

            st.session_state.lease_loaded = True
            st.session_state.lease_chunk_count = chunk_count
            st.session_state.lease_text = raw_text

        except Exception as exc:
            st.error(f"Failed to ingest lease: {exc}")
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    if st.session_state.lease_loaded:
        st.success(f"✓ Lease loaded ({st.session_state.lease_chunk_count} chunks)")

        # State law availability check
        if not _state_law_indexed(st.session_state.state):
            with st.spinner(f"Loading {st.session_state.state} state law for the first time..."):
                try:
                    ingest_state_law(st.session_state.state)
                    st.toast(f"✓ {st.session_state.state} state law indexed", icon="📜")
                except Exception as exc:
                    st.warning(f"Could not load {st.session_state.state} state law: {exc}")

        if st.button("🔍 Scan for issues", use_container_width=True):
            with st.spinner("Scanning for problematic clauses..."):
                st.session_state.scan_results = scan_illegal_clauses(
                    st.session_state.lease_text,
                    state=st.session_state.state,
                )
            if st.session_state.scan_results:
                st.toast(
                    f"Found {len(st.session_state.scan_results)} issue(s) — see Scan results tab",
                    icon="⚠️",
                )
            else:
                st.toast("No issues found", icon="✅")
    else:
        st.info("Upload your lease PDF above to get started.")

    st.divider()

    # Clear conversation
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.caption(f"Session ID: `{st.session_state.user_id}`")

# ---------------------------------------------------------------------------
# Newsprint CSS
# ---------------------------------------------------------------------------

def inject_newsprint_css():
    return """
    /* ── SECONDARY BACKGROUND OVERRIDE (config.toml sets #111111 for sidebar,
          restore light for non-sidebar secondary elements) ── */
    .stExpander,
    [data-testid="stExpander"],
    [data-testid="stForm"],
    [data-testid="stVerticalBlock"] [data-testid="stHorizontalBlock"],
    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    [data-baseweb="input"],
    [data-baseweb="textarea"] {
        background-color: #F9F9F7 !important;
        color: #111111 !important;
    }
    /* Sidebar text: override config.toml's textColor which affects ALL text */
    [data-testid="stSidebar"] * {
        color: #F9F9F7 !important;
    }
    [data-testid="stSidebar"] label {
        color: #A3A3A3 !important;
    }
    [data-testid="stSidebar"] code {
        color: #525252 !important;
    }

    /* ── GLOBAL BACKGROUND ── */
    .stApp, .main, .block-container {
        background-color: #F9F9F7 !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4' viewBox='0 0 4 4'%3E%3Cpath fill='%23111111' fill-opacity='0.04' d='M1 3h1v1H1V3zm2-2h1v1H3V1z'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    .stAppHeader {
        background: transparent !important;
    }

    /* ── MASTHEAD ── */
    .masthead-container {
        border-top: 4px solid #CC0000;
        border-bottom: 1px solid #111111;
        padding: 20px 0 14px;
        margin-bottom: 24px;
        width: 100%;
        max-width: 100%;
    }
    .masthead-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .masthead-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(52px, 9vw, 96px);
        font-weight: 900;
        color: #111111;
        letter-spacing: -0.03em;
        line-height: 0.95;
        margin: 0;
    }
    .masthead-vol {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #737373;
        letter-spacing: 0.1em;
        padding-bottom: 6px;
    }
    .masthead-sub {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.35em;
        color: #737373;
        margin-top: 6px;
    }

    /* ── BODY TYPOGRAPHY ── */
    .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: 'Lora', serif !important;
        font-size: 15px !important;
        line-height: 1.75 !important;
        color: #111111 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Playfair Display', serif !important;
        color: #111111 !important;
        font-weight: 700 !important;
    }
    .stMarkdown h1 { font-size: 32px !important; font-weight: 900 !important; }
    .stMarkdown h2 { font-size: 22px !important; }
    .stMarkdown h3 { font-size: 18px !important; }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #111111 !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: #737373 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 20px !important;
        border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #CC0000 !important;
        border-bottom: 2px solid #CC0000 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #CC0000 !important;
        height: 2px !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #111111 !important;
        height: 1px !important;
    }

    /* ── CHAT MESSAGES ── */
    [data-testid="stChatMessage"] {
        border-radius: 0 !important;
        background: transparent !important;
    }
    [data-testid="stChatMessage"][data-role="user"] {
        border: 1px solid #111111 !important;
        background: #F9F9F7 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] {
        border-left: 4px solid #111111 !important;
        background: #F5F5F0 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatMessage"] p {
        font-family: 'Lora', serif !important;
        font-size: 15px !important;
        color: #111111 !important;
    }

    /* ── CHAT INPUT ── */
    .stChatInput textarea,
    [data-testid="stChatInputTextArea"] {
        border: none !important;
        border-bottom: 2px solid #111111 !important;
        border-radius: 0 !important;
        background: transparent !important;
        font-family: 'Lora', serif !important;
        font-size: 15px !important;
        color: #111111 !important;
        box-shadow: none !important;
    }
    .stChatInput {
        background: transparent !important;
        border: none !important;
        border-top: 1px solid #E5E5E0 !important;
        border-radius: 0 !important;
    }

    /* ── INFO / HINT BOXES ── */
    .stInfo, [data-testid="stInfoBox"],
    div[class*="stAlert"] {
        border: 1px solid #111111 !important;
        border-left: 4px solid #111111 !important;
        border-radius: 0 !important;
        background: #F5F5F0 !important;
        color: #111111 !important;
        font-family: 'Lora', serif !important;
    }
    .stInfo p, [data-testid="stInfoBox"] p {
        color: #111111 !important;
    }

    /* ── BUTTONS (main area) ── */
    .main .stButton button,
    [data-testid="stMainBlockContainer"] .stButton button {
        border: 1px solid #111111 !important;
        background: #111111 !important;
        color: #F9F9F7 !important;
        border-radius: 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        transition: all 0.2s !important;
        padding: 8px 20px !important;
    }
    .main .stButton button:hover {
        background: #F9F9F7 !important;
        color: #111111 !important;
    }

    /* ── EXPANDERS (sources) ── */
    .stExpander {
        border: 1px solid #E5E5E0 !important;
        border-radius: 0 !important;
        background: transparent !important;
    }
    .stExpander summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: #737373 !important;
    }

    /* ── SIDEBAR TOGGLE BUTTON ── */
    [data-testid="collapsedControl"],
    button[kind="header"],
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #111111 !important;
        border: 1px solid #444444 !important;
        color: #F9F9F7 !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #F9F9F7 !important;
        stroke: #F9F9F7 !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #111111 !important;
        border-right: 1px solid #333333 !important;
        border-bottom: 1px solid #333333 !important;
        width: 40px !important;
        height: 40px !important;
    }
    .stSidebar button svg path,
    [data-testid="collapsedControl"] svg path {
        stroke: #F9F9F7 !important;
        fill: none !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: #CC0000 !important;
        border-color: #CC0000 !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222222 !important;
    }
    [data-testid="stSidebar"] * { color: #F9F9F7 !important; }
    [data-testid="stSidebar"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: #A3A3A3 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        border: 1px solid #404040 !important;
        border-radius: 0 !important;
        background: #1a1a1a !important;
    }
    [data-testid="stSidebar"] .stFileUploader {
        border: 1px solid #404040 !important;
        border-radius: 0 !important;
        background: #1a1a1a !important;
        padding: 8px !important;
    }
    [data-testid="stSidebar"] .stButton button {
        border: 1px solid #F9F9F7 !important;
        background: transparent !important;
        color: #F9F9F7 !important;
        border-radius: 0 !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 12px !important;
    }
    [data-testid="stSidebar"] hr { border-color: #333333 !important; }
    [data-testid="stSidebar"] code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        color: #525252 !important;
        background: transparent !important;
    }

    /* ── REMOVE STREAMLIT BRANDING ── */
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }
    """

st.markdown(
    f'<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
    f"<style>{inject_newsprint_css()}</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Newspaper masthead
# ---------------------------------------------------------------------------

st.markdown("""
<div class="masthead-container">
  <div class="masthead-row">
    <span class="masthead-title">LEASE LENS</span>
    <span class="masthead-vol">VOL. 1.0</span>
  </div>
  <div class="masthead-sub">Tenant Rights Intelligence System</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main area — three tabs
# ---------------------------------------------------------------------------

tab_chat, tab_scan, tab_docs = st.tabs([
    "Ask your lease",
    "Scan results",
    "Documents",
])

# ── Tab 1: Chat ──────────────────────────────────────────────────────────────

with tab_chat:
    if not st.session_state.lease_loaded:
        st.markdown("""
<span style="display:inline-block;background:#111111;color:#F9F9F7;font-family:'Inter',sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;padding:4px 10px;border-radius:0;margin-bottom:12px;">01. Get Started</span>
<div style="font-family:'Playfair Display',serif;font-size:24px;font-weight:700;color:#111111;margin-top:4px;margin-bottom:10px;line-height:1.2;">Welcome to Lease Lens 👋</div>
<p style="font-family:'Lora',serif;font-size:15px;line-height:1.75;color:#111111;margin-bottom:20px;">Upload your lease PDF in the sidebar to get started. Once uploaded, you can ask plain-English questions about your rights and obligations.</p>
""", unsafe_allow_html=True)
        st.markdown("""
<div style="margin-top:20px;border-left:3px solid #111111;padding-left:16px;margin-left:0;">
<p style="font-family:'Inter',sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:0.15em;color:#737373;margin:0 0 10px 0;">What Lease Lens can help with</p>
<p style="font-family:'Lora',serif;font-size:15px;line-height:2;color:#111111;margin:0;">
— Understanding your security deposit rights<br>
— Knowing when your landlord can (and can't) enter<br>
— Figuring out who's responsible for repairs<br>
— Checking if any lease clauses may be unenforceable<br>
— Understanding notice requirements for moving out
</p>
</div>
""", unsafe_allow_html=True)

    else:
        # Starter question pills (only before any conversation)
        if not st.session_state.messages:
            st.markdown("**Suggested questions — click to ask:**")
            cols = st.columns(2)
            for i, q in enumerate(STARTER_QUESTIONS):
                if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

        # Render existing chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    sources = msg["sources"]
                    lease_srcs = _dedup_sources(sources.get("lease") or [])
                    law_srcs = _dedup_sources(sources.get("law") or [])
                    if lease_srcs or law_srcs:
                        with st.expander("Sources"):
                            if lease_srcs:
                                st.markdown("**From your lease:**")
                                for s in lease_srcs:
                                    st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                            else:
                                st.markdown("_Your lease did not contain a directly relevant section for this question._")
                            if law_srcs:
                                st.markdown(f"**From {st.session_state.state} state law:**")
                                for s in law_srcs:
                                    st.markdown(f"- _{s['text'][:120]}..._")

        # Resolve prompt: pending starter click takes priority over chat input
        prompt: str | None = st.session_state.pop("pending_question", None)
        chat_input = st.chat_input("Ask a question about your lease...")
        if chat_input:
            prompt = chat_input

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # State-law warning before generating
                if not _state_law_indexed(st.session_state.state):
                    st.warning(
                        f"{st.session_state.state} state law not yet indexed. "
                        "Answers will be based on your lease only."
                    )

                with st.spinner("Checking your lease and state law..."):
                    try:
                        result = ask(
                            prompt,
                            user_id=st.session_state.user_id,
                            state=st.session_state.state,
                        )
                        answer = result["answer"]
                        sources = {
                            "lease": result["lease_sources"],
                            "law": result["law_sources"],
                        }
                    except Exception as exc:
                        answer = _classify_error(exc)
                        sources = {}

                st.markdown(answer)
                lease_srcs = _dedup_sources(sources.get("lease") or [])
                law_srcs = _dedup_sources(sources.get("law") or [])
                with st.expander("Sources"):
                    if lease_srcs:
                        st.markdown("**From your lease:**")
                        for s in lease_srcs:
                            st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                    else:
                        st.markdown(
                            "_Your lease did not contain a directly relevant section "
                            "for this question._"
                        )
                    if law_srcs:
                        st.markdown(f"**From {st.session_state.state} state law:**")
                        for s in law_srcs:
                            st.markdown(f"- _{s['text'][:120]}..._")
                    elif not lease_srcs:
                        st.markdown(
                            "_No matching state law sections found either. "
                            "Consider contacting your landlord in writing or a tenant rights attorney._"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        # Legal disclaimer info box
        st.info(
            "Lease Lens uses AI to help you understand your rights. "
            "Always verify important decisions with a licensed attorney.",
            icon="⚖️",
        )

# ── Tab 2: Scan results ───────────────────────────────────────────────────────

with tab_scan:
    if not st.session_state.lease_loaded:
        st.info(
            "Upload your lease and click **🔍 Scan for issues** in the sidebar to see results here."
        )

    elif st.session_state.scan_results is None:
        st.info(
            "Click **🔍 Scan for issues** in the sidebar to check your lease for problematic clauses."
        )

    elif len(st.session_state.scan_results) == 0:
        st.success(
            "No obviously problematic clauses were detected. "
            "This scan checks for common patterns only — consider a full attorney review."
        )

    else:
        results = st.session_state.scan_results
        high = [r for r in results if r["severity"] == "high"]
        medium = [r for r in results if r["severity"] == "medium"]

        st.markdown(
            f"### Found {len(results)} potentially unenforceable "
            f"clause{'s' if len(results) != 1 else ''}"
        )
        col1, col2 = st.columns(2)
        col1.metric("🔴 High severity", len(high))
        col2.metric("🟡 Medium severity", len(medium))
        st.divider()

        for i, finding in enumerate(results, start=1):
            badge = "🔴 HIGH" if finding["severity"] == "high" else "🟡 MEDIUM"
            with st.container(border=True):
                st.markdown(f"**{i}. {finding['issue']}** &nbsp; `{badge}`")
                st.markdown(f"**Why this may be illegal:** {finding['why_illegal']}")
                with st.expander("Show lease text"):
                    st.markdown(f"_{finding['clause_text'][:400]}_")

                st.caption("➡️ Use the **📄 Documents** tab to generate a deposit demand or repair notice.")

        st.divider()
        st.caption(
            "This is an automated scan, not legal advice. "
            "Consult a tenant rights attorney to confirm enforceability in your jurisdiction."
        )

# ── Tab 3: Documents ─────────────────────────────────────────────────────────

with tab_docs:
    from datetime import date as _date, timedelta as _td

    st.markdown("### 📄 Legal Document Generator")
    st.markdown(
        "Generate state-specific tenant rights documents — each one cites your state's exact statute."
    )

    subtab_deposit, subtab_repair, subtab_checklist = st.tabs([
        "💰 Deposit Demand",
        "🔧 Repair Notice",
        "📋 Move-Out Checklist",
    ])

    # ── Sub-tab 1: Security deposit demand letter ─────────────────────────────

    with subtab_deposit:
        st.markdown("#### 💰 Security Deposit Demand Letter")
        st.markdown(
            "Demand return of your security deposit citing the exact statute for your state."
        )

        dep_law = STATE_DEPOSIT_LAW.get(st.session_state.state, {})
        if dep_law:
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Return deadline", f"{dep_law['return_deadline']} days")
                c2.metric("Max penalty", f"{dep_law['penalty']}× deposit")
                c3.metric("Certified mail?", "Yes" if dep_law["certified_mail_required"] else "No")
                st.caption(f"Governing statute: **{dep_law['statute_citation']}**")

        st.divider()

        with st.form("deposit_letter_form"):
            st.markdown("**Your information**")
            col_t1, col_t2 = st.columns(2)
            dl_tenant_name = col_t1.text_input("Your full name", placeholder="Jane Smith", key="dl_tname")
            dl_tenant_addr = col_t2.text_area(
                "Your current mailing address", height=88, key="dl_taddr",
                placeholder="456 Oak Ave, Apt 2\nSan Francisco, CA 94102",
            )

            st.markdown("**Landlord information**")
            col_l1, col_l2 = st.columns(2)
            dl_ll_name = col_l1.text_input("Landlord / property manager name", placeholder="Pacific Properties LLC", key="dl_llname")
            dl_ll_addr = col_l2.text_area("Landlord mailing address", height=88, key="dl_lladdr",
                placeholder="789 Market St, Suite 100\nSan Francisco, CA 94103",
            )

            st.markdown("**Rental details**")
            dl_prop_addr = st.text_input("Rental property address", key="dl_propaddr",
                placeholder="123 Main St, Apt 4B, San Francisco, CA 94101",
            )
            col_d1, col_d2, col_d3 = st.columns(3)
            dl_move_out = col_d1.date_input("Move-out date", value=_date.today() - _td(days=30), key="dl_moveout")
            dl_deposit = col_d2.number_input("Security deposit ($)", min_value=0.0, step=50.0, format="%.2f", key="dl_deposit")
            dl_state = col_d3.selectbox("State", options=STATES,
                index=STATES.index(st.session_state.state) if st.session_state.state in STATES else 4,
                key="dl_state",
            )
            dl_deductions = st.text_area(
                "Deductions claimed by landlord (one per line — leave blank if none)",
                height=90, key="dl_deductions",
                placeholder="$200 carpet cleaning\n$150 paint touch-up",
            )
            dl_submitted = st.form_submit_button("✉️ Generate Demand Letter", use_container_width=True, type="primary")

        if dl_submitted:
            if not dl_tenant_name.strip() or not dl_ll_name.strip():
                st.warning("Please fill in at least your name and the landlord's name.")
            else:
                params = {
                    "tenant_name": dl_tenant_name.strip(),
                    "tenant_address": dl_tenant_addr.strip(),
                    "landlord_name": dl_ll_name.strip(),
                    "landlord_address": dl_ll_addr.strip(),
                    "property_address": dl_prop_addr.strip(),
                    "move_out_date": str(dl_move_out),
                    "deposit_amount": dl_deposit,
                    "deductions_claimed": [d.strip() for d in dl_deductions.splitlines() if d.strip()],
                    "state": dl_state,
                }
                with st.spinner("Drafting your demand letter — this may take 20–30 seconds..."):
                    try:
                        st.session_state.letter_text = generate_deposit_letter(params)
                        st.session_state.letter_pdf_path = save_letter_pdf(
                            st.session_state.letter_text, filename="deposit_demand_letter.pdf"
                        )
                        st.toast("Letter generated!", icon="✉️")
                    except Exception as exc:
                        st.error(_classify_error(exc))
                        st.session_state.letter_text = ""
                        st.session_state.letter_pdf_path = None

        if st.session_state.letter_text:
            st.divider()
            st.markdown("#### Your demand letter")
            st.text_area("dl_output", value=st.session_state.letter_text, height=480, label_visibility="collapsed")
            if st.session_state.letter_pdf_path:
                st.download_button(
                    "⬇️ Download PDF", type="primary", use_container_width=True,
                    data=pathlib.Path(st.session_state.letter_pdf_path).read_bytes(),
                    file_name="deposit_demand_letter.pdf", mime="application/pdf",
                )
            st.caption("⚖️ AI-generated. Review carefully and consult a tenant rights attorney before sending.")

    # ── Sub-tab 2: Repair demand notice ──────────────────────────────────────

    with subtab_repair:
        st.markdown("#### 🔧 Repair Demand Notice")
        st.markdown(
            "Send a formal written notice demanding repairs. "
            "This triggers the statutory deadline and preserves your right to rent withholding or repair-and-deduct."
        )

        hab_law = STATE_HABITABILITY_LAW.get(st.session_state.state, {})
        if hab_law:
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Repair deadline", f"{hab_law['repair_deadline_days']} days")
                remedies_short = ", ".join(
                    r.replace("rent withholding", "Withhold rent")
                     .replace("repair and deduct", "R&D")
                     .replace("lease termination", "Terminate")
                    for r in hab_law["tenant_remedies"]
                )
                c2.metric("Your remedies", remedies_short)
                c3.metric("Can withhold rent?", "Yes" if hab_law["rent_withhold_threshold"] > 0 else "No")
                st.caption(f"Governing statute: **{hab_law['statute_citation']}**")

        st.divider()

        with st.form("repair_notice_form"):
            st.markdown("**Your information**")
            col_rt1, col_rt2 = st.columns(2)
            rn_tenant_name = col_rt1.text_input("Your full name", placeholder="Jane Smith", key="rn_tname")
            rn_tenant_addr = col_rt2.text_area("Your current mailing address", height=88, key="rn_taddr",
                placeholder="456 Oak Ave, Apt 2\nSan Francisco, CA 94102",
            )

            st.markdown("**Landlord information**")
            col_rl1, col_rl2 = st.columns(2)
            rn_ll_name = col_rl1.text_input("Landlord / property manager name", placeholder="Pacific Properties LLC", key="rn_llname")
            rn_ll_addr = col_rl2.text_area("Landlord mailing address", height=88, key="rn_lladdr",
                placeholder="789 Market St, Suite 100\nSan Francisco, CA 94103",
            )

            rn_prop_addr = st.text_input("Rental property address", key="rn_propaddr",
                placeholder="123 Main St, Apt 4B, San Francisco, CA 94101",
            )

            col_ri1, col_ri2 = st.columns([2, 1])
            rn_issue = col_ri1.text_area(
                "Describe the repair issue in detail",
                height=110, key="rn_issue",
                placeholder="The ceiling in the master bedroom has been leaking since a rainstorm on March 15. "
                            "Water drips onto the floor during rain. I reported this verbally on March 16.",
            )
            rn_reported = col_ri2.date_input(
                "Date you first reported it", value=_date.today() - _td(days=14), key="rn_reported"
            )
            rn_state = col_ri2.selectbox("State", options=STATES,
                index=STATES.index(st.session_state.state) if st.session_state.state in STATES else 4,
                key="rn_state",
            )

            rn_submitted = st.form_submit_button("🔧 Generate Repair Notice", use_container_width=True, type="primary")

        if rn_submitted:
            if not rn_tenant_name.strip() or not rn_ll_name.strip() or not rn_issue.strip():
                st.warning("Please fill in your name, landlord's name, and a description of the repair issue.")
            else:
                rn_params = {
                    "tenant_name": rn_tenant_name.strip(),
                    "tenant_address": rn_tenant_addr.strip(),
                    "landlord_name": rn_ll_name.strip(),
                    "landlord_address": rn_ll_addr.strip(),
                    "property_address": rn_prop_addr.strip(),
                    "issue_description": rn_issue.strip(),
                    "date_first_reported": str(rn_reported),
                    "state": rn_state,
                }
                with st.spinner("Drafting your repair notice — this may take 20–30 seconds..."):
                    try:
                        st.session_state.repair_notice_text = generate_repair_notice(rn_params)
                        st.session_state.repair_notice_pdf_path = save_letter_pdf(
                            st.session_state.repair_notice_text, filename="repair_demand_notice.pdf"
                        )
                        st.toast("Repair notice generated!", icon="🔧")
                    except Exception as exc:
                        st.error(_classify_error(exc))
                        st.session_state.repair_notice_text = ""
                        st.session_state.repair_notice_pdf_path = None

        if st.session_state.repair_notice_text:
            st.divider()
            st.markdown("#### Your repair notice")
            st.text_area("rn_output", value=st.session_state.repair_notice_text, height=480, label_visibility="collapsed")
            if st.session_state.repair_notice_pdf_path:
                st.download_button(
                    "⬇️ Download PDF", type="primary", use_container_width=True,
                    data=pathlib.Path(st.session_state.repair_notice_pdf_path).read_bytes(),
                    file_name="repair_demand_notice.pdf", mime="application/pdf",
                )
            st.info(
                "📮 **Send this notice via certified mail, return receipt requested.** "
                "Keep a copy. The statutory repair clock starts on the date of delivery.",
                icon="📬",
            )

    # ── Sub-tab 3: Move-out checklist ─────────────────────────────────────────

    with subtab_checklist:
        st.markdown("#### 📋 Move-Out Condition Checklist")
        st.markdown(
            "Generate a room-by-room PDF checklist documenting your unit's condition at move-out. "
            "Includes your state's **normal wear & tear** definition to protect your deposit."
        )

        with st.form("checklist_form"):
            col_c1, col_c2 = st.columns(2)
            cl_tenant = col_c1.text_input("Your name (optional)", placeholder="Jane Smith", key="cl_tenant")
            cl_addr = col_c2.text_input("Property address (optional)", key="cl_addr",
                placeholder="123 Main St, Apt 4B",
            )

            col_c3, col_c4, col_c5 = st.columns(3)
            cl_state = col_c3.selectbox("State", options=STATES,
                index=STATES.index(st.session_state.state) if st.session_state.state in STATES else 4,
                key="cl_state",
            )
            cl_move_in = col_c4.date_input("Move-in date", value=_date.today() - _td(days=365), key="cl_movein")
            cl_prop_type = col_c5.selectbox("Property type", options=["apartment", "house"], key="cl_proptype")

            cl_submitted = st.form_submit_button("📋 Generate Checklist PDF", use_container_width=True, type="primary")

        if cl_submitted:
            cl_params = {
                "state": cl_state,
                "move_in_date": str(cl_move_in),
                "property_type": cl_prop_type,
                "tenant_name": cl_tenant.strip() or None,
                "property_address": cl_addr.strip() or None,
            }
            with st.spinner("Building your checklist PDF..."):
                try:
                    st.session_state.checklist_pdf_path = generate_moveout_checklist(cl_params)
                    st.toast("Checklist ready!", icon="📋")
                except Exception as exc:
                    st.error(f"Could not generate checklist: {exc}")
                    st.session_state.checklist_pdf_path = None

        if st.session_state.checklist_pdf_path:
            st.success("✓ Your checklist is ready — download it below.")
            st.download_button(
                "⬇️ Download Checklist PDF", type="primary", use_container_width=True,
                data=pathlib.Path(st.session_state.checklist_pdf_path).read_bytes(),
                file_name="moveout_checklist.pdf", mime="application/pdf",
            )
            st.divider()
            st.markdown(
                "**Tips for using this checklist:**\n"
                "- Walk through each room **before** handing over keys\n"
                "- Take timestamped photos for every item you mark as damaged\n"
                "- Give your landlord a copy and keep one for yourself\n"
                "- If your landlord disputes any charge, compare against the normal wear & tear definition"
            )
            st.caption("⚖️ This checklist is informational. Consult a tenant rights attorney for disputes.")
