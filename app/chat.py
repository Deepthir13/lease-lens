"""
chat.py — Streamlit chat interface for Lease Lens.
"""
import streamlit as st


def main():
    st.title("Lease Lens")
    st.caption("Ask questions about your lease and tenant rights.")

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
            response = "RAG pipeline not yet connected."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
