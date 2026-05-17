# Lease Lens

Lease Lens is a local RAG (Retrieval-Augmented Generation) chatbot that answers tenant questions from their lease and state law.

## Overview

Upload your lease PDF and ask plain-English questions. Lease Lens retrieves the relevant clauses from your lease alongside applicable state statutes, then uses a locally-running LLM (via Ollama) to generate accurate, grounded answers — no data leaves your machine.

## Features

- **Local-first**: Powered by Ollama (e.g. Mistral) — your documents stay on your device
- **Dual knowledge base**: Combines your lease with pre-loaded state tenant law
- **RAG pipeline**: Uses ChromaDB + LlamaIndex for fast semantic retrieval
- **Simple UI**: Streamlit-based chat interface

## Project Structure

```
lease-lens/
├── app/
│   ├── __init__.py
│   ├── ingest.py       # PDF ingestion and chunking
│   ├── rag.py          # RAG query engine
│   ├── chat.py         # Streamlit chat UI
│   └── utils.py        # Shared utilities
├── data/
│   ├── leases/         # User-uploaded lease PDFs
│   └── statutes/       # Pre-scraped state law texts
├── vectorstore/        # ChromaDB persistent storage
├── tests/
│   └── test_rag.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull mistral
   ```

2. Clone this repo and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

4. Run the app:
   ```bash
   streamlit run app/chat.py
   ```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_MODEL` | Ollama model to use | `mistral` |
| `STATE` | Two-letter state code for statute lookup | `CA` |

## Running Tests

```bash
pytest tests/
```
