"""
streamlit_app.py — Main Lease Lens web interface.

Run with:
    streamlit run app/streamlit_app.py
"""

import tempfile
import pathlib
import uuid

import streamlit as st

from app.ingest import ingest_lease
from app.chat import ask
from app.utils import scan_illegal_clauses, summarize_scan, get_state

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
# State abbreviations
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
        "active_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

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
        with st.spinner("Ingesting lease..."):
            # Save to a temp file then ingest
            suffix = pathlib.Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                chunk_count = ingest_lease(tmp_path, user_id=st.session_state.user_id)
                st.session_state.lease_loaded = True
                st.session_state.lease_chunk_count = chunk_count

                # Also read the raw text for the clause scanner
                import fitz
                with fitz.open(tmp_path) as doc:
                    st.session_state.lease_text = "\n\n".join(
                        page.get_text() for page in doc
                    )
            except Exception as exc:
                st.error(f"Failed to ingest lease: {exc}")
            finally:
                pathlib.Path(tmp_path).unlink(missing_ok=True)

    if st.session_state.lease_loaded:
        st.success(f"✓ Lease loaded ({st.session_state.lease_chunk_count} chunks)")

        if st.button("🔍 Scan for issues", use_container_width=True):
            with st.spinner("Scanning for problematic clauses..."):
                st.session_state.scan_results = scan_illegal_clauses(
                    st.session_state.lease_text,
                    state=st.session_state.state,
                )
    else:
        st.info("Upload your lease PDF above to get started.")

    st.divider()
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
        # Starter question pills
        if not st.session_state.messages:
            st.markdown("**Suggested questions — click to ask:**")
            cols = st.columns(2)
            for i, q in enumerate(STARTER_QUESTIONS):
                if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                    st.session_state.active_question = q

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
                                st.markdown("**From {state} state law:**".format(
                                    state=st.session_state.state
                                ))
                                for s in sources["law"]:
                                    st.markdown(f"- _{s['text'][:120]}..._")

        # Handle starter question click or chat input
        prompt = st.session_state.pop("active_question", None)
        chat_input = st.chat_input("Ask a question about your lease...")
        if chat_input:
            prompt = chat_input

        if prompt:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
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
                        answer = f"Sorry, something went wrong: {exc}"
                        sources = {}

                st.markdown(answer)
                if sources.get("lease") or sources.get("law"):
                    with st.expander("Sources"):
                        if sources.get("lease"):
                            st.markdown("**From your lease:**")
                            for s in sources["lease"]:
                                st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                        if sources.get("law"):
                            st.markdown(f"**From {st.session_state.state} state law:**")
                            for s in sources["law"]:
                                st.markdown(f"- _{s['text'][:120]}..._")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

# ── Tab 2: Scan results ───────────────────────────────────────────────────────

with tab_scan:
    if not st.session_state.lease_loaded:
        st.info("Upload your lease and click **Scan for issues** in the sidebar to see results here.")

    elif st.session_state.scan_results is None:
        st.info("Click **🔍 Scan for issues** in the sidebar to check your lease for problematic clauses.")

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
            f"### Found {len(results)} potentially unenforceable clause{'s' if len(results) != 1 else ''}"
        )
        col1, col2 = st.columns(2)
        col1.metric("🔴 High severity", len(high))
        col2.metric("🟡 Medium severity", len(medium))
        st.divider()

        for i, finding in enumerate(results, start=1):
            is_high = finding["severity"] == "high"
            border_color = "#ff4b4b" if is_high else "#ffa500"
            badge = "🔴 HIGH" if is_high else "🟡 MEDIUM"

            with st.container(border=True):
                st.markdown(f"**{i}. {finding['issue']}** &nbsp; `{badge}`")
                st.markdown(f"**Why this may be illegal:** {finding['why_illegal']}")
                with st.expander("Show lease text"):
                    st.markdown(f"_{finding['clause_text'][:400]}_")

                if st.button(
                    "Generate dispute letter",
                    key=f"dispute_{i}",
                    use_container_width=True,
                    disabled=True,
                ):
                    pass  # Phase 5 placeholder
                st.caption("Dispute letter generation coming in Phase 5.")

        st.divider()
        st.caption(
            "This is an automated scan, not legal advice. "
            "Consult a tenant rights attorney to confirm enforceability in your jurisdiction."
        )
