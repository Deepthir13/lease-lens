# Lease Lens 🏠

**Lease Lens** is a fully local, privacy-first tenant rights assistant. Upload your lease PDF, ask plain-English questions, scan for illegal clauses, and generate state-specific legal documents — all powered by a locally-running LLM. Your documents never leave your machine.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Deepthir13/lease-lens/actions/workflows/test.yml/badge.svg)

---

## What it does

| Feature | Description |
|---|---|
| 💬 **Ask your lease** | RAG-powered Q&A grounded in your lease + your state's tenant statutes |
| 🔎 **Clause scanner** | Detects 8 categories of potentially unenforceable clauses (entry notice waivers, ESA bans, excessive late fees, etc.) |
| 💰 **Deposit demand letter** | Generates a formal demand letter citing your state's exact return deadline and penalty multiplier |
| 🔧 **Repair notice** | Drafts a written notice that starts the statutory repair clock and lists your remedies |
| 📋 **Move-out checklist** | Room-by-room condition PDF with your state's normal wear & tear definition |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.11 and 3.12 |
| [Ollama](https://ollama.com) | latest | Must be running locally |
| Mistral 7B | — | `ollama pull mistral` |
| Tesseract OCR | latest | Only needed for scanned PDFs |

Install Tesseract on macOS:
```bash
brew install tesseract
```
On Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Deepthir13/lease-lens.git
cd lease-lens

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env               # edit if you want a different model or default state

# 5. Start Ollama and pull the model
ollama serve &
ollama pull mistral

# 6. (Optional) Pre-index state law for your state
python data/scrape_laws.py CA      # replace CA with your state code

# 7. Run the app
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Select your state** in the sidebar (auto-detected from `.env`)
2. **Upload your lease PDF** — text-based or scanned (OCR runs automatically)
3. **Ask questions** in the chat tab, e.g.:
   - *"Can my landlord enter without notice?"*
   - *"What can they deduct from my deposit?"*
4. **Scan for issues** — click the sidebar button to run the clause scanner
5. **Generate documents** in the Documents tab:
   - Deposit demand letter → download as PDF
   - Repair demand notice → download as PDF
   - Move-out checklist → download as PDF

---

## Screenshots

> _Screenshots will be added after the first stable release._

---

## Project Structure

```
lease-lens/
├── app/
│   ├── __init__.py
│   ├── ingest.py        # PDF ingestion (PyMuPDF + OCR) + state law scraper
│   ├── rag.py           # Dual-source RAG retrieval (lease + state law)
│   ├── chat.py          # LLM answer generation (Ollama/Mistral)
│   ├── utils.py         # Illegal clause scanner + env helpers
│   ├── documents.py     # Deposit letter, repair notice, move-out checklist
│   └── streamlit_app.py # Main web UI
├── data/
│   ├── statutes/        # Cached state law texts (auto-populated)
│   └── scrape_laws.py   # One-time setup: index all 50 states
├── tests/
│   ├── conftest.py      # CI mocks (Ollama + HTTP)
│   └── test_rag.py      # Ingestion, retrieval, and generation tests
├── vectorstore/         # ChromaDB persistent storage (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | [Ollama](https://ollama.com) + [Mistral 7B](https://mistral.ai) (fully local) |
| **Embeddings** | ChromaDB built-in (`all-MiniLM-L6-v2`) |
| **Vector store** | [ChromaDB](https://www.trychroma.com) persistent client |
| **Document chunking** | [LlamaIndex](https://www.llamaindex.ai) `SentenceSplitter` |
| **PDF parsing** | [PyMuPDF](https://pymupdf.readthedocs.io) + [Pytesseract](https://github.com/madmaze/pytesseract) (OCR fallback) |
| **UI** | [Streamlit](https://streamlit.io) |
| **PDF generation** | [ReportLab](https://www.reportlab.com) |
| **State law data** | Scraped from official state legislature websites |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `STATE` | `CA` | Default state for tenant law lookup |

---

## Running Tests

Tests mock both Ollama and HTTP calls, so no live server is needed:

```bash
pytest tests/ -v
```

---

## Supported States

All 50 US states are supported for:
- Security deposit return deadlines and penalties
- Implied warranty of habitability repair deadlines
- Tenant remedies (rent withholding, repair-and-deduct, lease termination)

State law data is sourced from official state legislature websites and cached in `data/statutes/`.

---

## Roadmap

- [ ] Dispute letter auto-linked to scan findings
- [ ] Multi-language support
- [ ] Lease comparison (old vs. renewal)
- [ ] Mobile-optimised layout

---

## Disclaimer

Lease Lens uses AI to help you understand your rights. **It is not a substitute for legal advice.** Always verify important decisions with a licensed tenant rights attorney in your jurisdiction.

---

## License

MIT — see [LICENSE](LICENSE) for details.
