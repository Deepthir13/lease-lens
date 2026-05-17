"""
streamlit_app.py — Main Lease Lens web interface.

Run with:
    streamlit run app/streamlit_app.py
"""

import pathlib
import tempfile
import uuid

import chromadb
import fitz
import streamlit as st

from app.ingest import ingest_lease, ingest_state_law
from app.chat import ask
from app.utils import scan_illegal_clauses, get_state

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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

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
    st.markdown("## 🏠 Lease Lens")
    st.caption("Your private tenant rights assistant — runs entirely on your machine.")
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
# Main area — two tabs
# ---------------------------------------------------------------------------

tab_chat, tab_scan = st.tabs(["💬 Ask your lease", "🔎 Scan results"])

# ── Tab 1: Chat ──────────────────────────────────────────────────────────────

with tab_chat:
    if not st.session_state.lease_loaded:
        st.markdown("### Welcome to Lease Lens 👋")
        st.markdown(
            "Upload your lease PDF in the sidebar to get started. "
            "Once uploaded, you can ask plain-English questions about your rights and obligations."
        )
        st.markdown("**What Lease Lens can help with:**")
        st.markdown(
            "- Understanding your security deposit rights\n"
            "- Knowing when your landlord can (and can't) enter\n"
            "- Figuring out who's responsible for repairs\n"
            "- Checking if any lease clauses may be unenforceable\n"
            "- Understanding notice requirements for moving out"
        )

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
                    if sources.get("lease") or sources.get("law"):
                        with st.expander("Sources"):
                            if sources.get("lease"):
                                st.markdown("**From your lease:**")
                                for s in sources["lease"]:
                                    st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                            if sources.get("law"):
                                st.markdown(
                                    f"**From {st.session_state.state} state law:**"
                                )
                                for s in sources["law"]:
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
                if sources.get("lease") or sources.get("law"):
                    with st.expander("Sources"):
                        if sources.get("lease"):
                            st.markdown("**From your lease:**")
                            for s in sources["lease"]:
                                st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                        if sources.get("law"):
                            st.markdown(
                                f"**From {st.session_state.state} state law:**"
                            )
                            for s in sources["law"]:
                                st.markdown(f"- _{s['text'][:120]}..._")

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

                st.button(
                    "Generate dispute letter",
                    key=f"dispute_{i}",
                    use_container_width=True,
                    disabled=True,
                )
                st.caption("Dispute letter generation coming in Phase 5.")

        st.divider()
        st.caption(
            "This is an automated scan, not legal advice. "
            "Consult a tenant rights attorney to confirm enforceability in your jurisdiction."
        )
