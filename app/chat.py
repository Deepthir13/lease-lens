"""
chat.py — Answer generation layer + Streamlit chat UI for Lease Lens.

Answer pipeline:
  ask(question, user_id, state)
    → retrieve_context()        # dual ChromaDB lookup
    → format_context()          # build prompt context string
    → generate_answer()         # Ollama / Mistral inference
    → structured response dict
"""

import ollama
import streamlit as st

from app.rag import retrieve_context, format_context
from app.utils import get_ollama_model, get_state

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a tenant rights assistant. You answer questions strictly from two "
    "sources: (1) the tenant's lease document and (2) their state's tenant law. "
    "Always cite your source by saying 'Your lease says...' or 'Your state law "
    "says...'. If the lease and law conflict, always flag this clearly and explain "
    "which one takes precedence. Never give general legal advice — only answer from "
    "the provided context. End every answer with: 'Note: This is informational only. "
    "Consult a tenant rights attorney for your specific situation.'"
)

# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(question: str, context: str, model: str = "mistral") -> str:
    """
    Generate an answer using Ollama.

    Args:
        question: The tenant's plain-English question.
        context:  The formatted context string from format_context().
        model:    Ollama model name (default: "mistral").

    Returns:
        The generated answer string.
    """
    user_message = (
        f"Here is the relevant context retrieved from your lease and state law:\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str, user_id: str, state: str) -> dict:
    """
    Full RAG + LLM pipeline: retrieve → format → generate.

    Args:
        question: The tenant's question.
        user_id:  Identifies the lease ChromaDB collection.
        state:    Two-letter state code.

    Returns:
        {
            "answer":        str,
            "lease_sources": [{"text": ..., "page": ...}, ...],
            "law_sources":   [{"text": ..., "section": ...}, ...],
            "question":      str,
        }
    """
    retrieved = retrieve_context(question, user_id=user_id, state=state)
    context = format_context(retrieved)
    model = get_ollama_model()
    answer = generate_answer(question, context, model=model)

    lease_sources = [
        {"text": text, "page": page}
        for text, page, _ in retrieved["lease_chunks"]
    ]
    law_sources = [
        {"text": text, "section": section}
        for text, section, _ in retrieved["law_chunks"]
    ]

    return {
        "answer": answer,
        "lease_sources": lease_sources,
        "law_sources": law_sources,
        "question": question,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.title("Lease Lens")
    st.caption("Ask questions about your lease and tenant rights.")

    # Sidebar config
    with st.sidebar:
        st.header("Settings")
        user_id = st.text_input("User ID", value="default")
        state = st.text_input("State (2-letter code)", value=get_state()).upper()
        st.markdown("---")
        st.caption("Make sure you've ingested your lease and state law before asking questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your lease..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(prompt, user_id=user_id, state=state)
                    response = result["answer"]

                    # Show source citations in an expander
                    if result["lease_sources"] or result["law_sources"]:
                        with st.expander("Sources"):
                            if result["lease_sources"]:
                                st.markdown("**From your lease:**")
                                for s in result["lease_sources"]:
                                    st.markdown(f"- Page {s['page']}: _{s['text'][:120]}..._")
                            if result["law_sources"]:
                                st.markdown("**From state law:**")
                                for s in result["law_sources"]:
                                    st.markdown(f"- {s['section']}: _{s['text'][:120]}..._")
                except Exception as exc:
                    response = f"Error generating answer: {exc}"

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
