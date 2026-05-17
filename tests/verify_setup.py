"""
verify_setup.py — Confirms all dependencies and Ollama connectivity are working.
"""
import sys


def check_imports():
    try:
        import llama_index.core
        print("  llama_index.core   OK")
    except ImportError as e:
        print(f"  llama_index.core   FAILED: {e}")
        sys.exit(1)

    try:
        import chromadb
        print("  chromadb           OK")
    except ImportError as e:
        print(f"  chromadb           FAILED: {e}")
        sys.exit(1)

    try:
        import fitz  # PyMuPDF
        print("  fitz (pymupdf)     OK")
    except ImportError as e:
        print(f"  fitz (pymupdf)     FAILED: {e}")
        sys.exit(1)

    try:
        import streamlit
        print("  streamlit          OK")
    except ImportError as e:
        print(f"  streamlit          FAILED: {e}")
        sys.exit(1)


def check_ollama():
    try:
        import ollama
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        reply = response["message"]["content"].strip()
        print(f"  ollama (mistral)   OK  →  model replied: \"{reply}\"")
    except Exception as e:
        print(f"  ollama (mistral)   FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\nChecking imports...")
    check_imports()

    print("\nChecking Ollama connectivity...")
    check_ollama()

    print("\n✓ All dependencies OK\n")
