
# 📄 PaperPilot — RAG-powered Document Q&A

[![CI](https://github.com/21Mandar/PaperPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/21Mandar/PaperPilot/actions/workflows/ci.yml)

Ask questions about a PDF and get **grounded, page-cited answers** from a local
open-weights LLM. PaperPilot ingests a PDF, retrieves the most relevant passages
via dense vector search (FAISS), and generates answers constrained to that
retrieved context — so every answer is traceable back to a page.

**🔗 Live demo:** https://huggingface.co/spaces/21Mandar/PaperPilot

![PaperPilot UI](data/hf_ss.png)

## Features

- **Grounded answers** — responses are restricted to retrieved passages, cite
  page numbers like `(p. 9)`, and abstain when the document doesn't cover the
  question.
- **Dense semantic retrieval** — `all-MiniLM-L6-v2` embeddings + FAISS, not
  keyword search.
- **Local open-weights LLM** — runs entirely on-device via Hugging Face
  Transformers (no API keys, no data leaving the machine).
- **Measured, not asserted** — a built-in eval harness reports recall@k and
  compares prompt strategies (`src/eval.py`).
- **Agentic mode (optional)** — a Claude-driven tool-use agent (`src/agent.py`)
  that can issue further searches for multi-part questions and double-check a
  page before citing it, instead of a single fixed retrieve-then-generate pass.
- **CI on every push/PR** — GitHub Actions runs the test suite and fails the
  build if retrieval quality (recall@1) regresses (`.github/workflows/ci.yml`).
- **One-click deploy** — Dockerized and running on Hugging Face Spaces.

## How it works

```
PDF ──► chunk + embed ──► FAISS index ──► dense retrieval ──► LLM (grounded) ──► answer + page citations
        (all-MiniLM-L6-v2)                 (top-k)            (Qwen2.5-Instruct)
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).
- **Vector store:** FAISS `IndexFlatIP` (exact search, no infra).
- **Generation:** Hugging Face Transformers with a local open-weights instruct
  model (Qwen2.5). Integrated/orchestrated — **not** fine-tuned.
- **Grounding:** the system prompt restricts answers to retrieved passages,
  requires page citations, and abstains when context is insufficient.

## Measured results

On a 12-question hand-labeled eval set (`data/eval.json`), each question tagged
with the page(s) that actually contain its answer:

| metric | value |
|--------|-------|
| recall@1 | 0.83 |
| recall@3 | 1.00 |
| recall@5 | 1.00 |
| retrieval latency (median) | ~31 ms |

A single few-shot exemplar raised the answer **citation rate from 0% → 100%**
versus zero-shot — a concrete, measured prompt-engineering result (`src/eval.py`).

> `recall@k` here is hit-rate@k over page-level labels (stable across
> re-chunking, unlike chunk IDs).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.ingest                      # build the FAISS index from data/sample.pdf
python -m src.rag "How are radioactive materials transported?"
python -m src.eval --with-generation      # recall@k + zero/few-shot comparison
streamlit run app/main.py                 # the UI
```

## Agentic mode (optional, Claude API)

`src/agent.py` is an alternate backend: instead of one fixed retrieve-then-generate
pass, Claude drives a tool-use loop over the same FAISS index — issuing further
searches for multi-part questions and double-checking a page's full text before
citing it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.agent "How are radioactive materials transported?"
```

Or from the Streamlit UI: switch the sidebar **Backend** to "Agentic (Claude API)".

- **Model:** defaults to `claude-opus-4-8`; override with the `PAPERPILOT_AGENT_MODEL`
  env var (e.g. `claude-haiku-4-5` for a cheaper/faster option) or the sidebar's
  "Claude model" field.
- **Cost:** pay-per-token, no ongoing free tier. A typical query (a couple of
  tool round-trips) runs roughly $0.02–0.03 on Opus 4.8, ~$0.003–0.005 on
  Haiku 4.5 — cents for a full local test/demo session.
- **Local-only by design:** this mode is not wired into the Docker image or the
  public Hugging Face Space — it only runs where *you* export your own
  `ANTHROPIC_API_KEY` (your terminal, or Streamlit running on your own machine).
  That keeps the key off of any unauthenticated public endpoint. A "bring your
  own key" field for the public demo is a possible future addition, not part of
  this mode today.

## Run with Docker

```bash
docker build -t paperpilot .
docker run -p 7860:7860 paperpilot
# open http://localhost:7860
```

The image installs a CPU-only torch build, bakes in the prebuilt FAISS index,
pre-downloads the models, and defaults to the lighter `Qwen/Qwen2.5-0.5B-Instruct`
(via the `PAPERPILOT_LLM` env var) for responsiveness on a free CPU host. Locally
the code defaults to the stronger `Qwen2.5-1.5B-Instruct`.

## Deploy to Hugging Face Spaces

The YAML frontmatter at the top of this file configures the Space (Docker SDK,
port 7860). Create a Space (**SDK: Docker**) and push:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main
```

Binary files (`*.pdf`, `*.index`) are tracked with Git LFS, which the Space
requires for binary storage.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

- **`test`** — installs dependencies and runs the `pytest` suite (`tests/`).
- **`eval-gate`** — runs `python -m src.eval` (retrieval-only, uses the
  committed FAISS index — no model download) and fails the build if recall@1
  drops below 0.75 (`scripts/check_eval_regression.py`), a margin below the
  measured 0.83 baseline. The metrics file is uploaded as a build artifact on
  every run.

Neither job loads the local Qwen model or calls the Claude API — both stay
fast, free, and secret-free.

## Project layout

```
src/ingest.py                     load PDF → chunk → embed → build FAISS index
src/rag.py                        retrieve top-k → grounded generation (zero/few-shot)
src/agent.py                      agentic backend (Claude tool-use loop, optional)
src/eval.py                       recall@k + prompt-comparison harness → results/
app/main.py                       Streamlit UI
tests/                            pytest suite (ingest, rag, agent)
scripts/check_eval_regression.py  CI recall@1 regression gate
.github/workflows/ci.yml          GitHub Actions: test + eval-gate
data/                             sample.pdf + eval.json
index/                            prebuilt FAISS index + chunk metadata
```

## Limitations

- Text-based PDFs only; scanned/image PDFs would need an OCR step before
  ingestion.
- The sample eval set is small (12 questions) — enough to demonstrate the
  metric, not to make strong quantitative claims.
- On a free CPU Space, generation with a 1.5B+ model is slow; the deploy uses a
  0.5B model to stay responsive, trading some answer quality for latency.

## Notes

- On Apple Silicon, `torch`/`transformers` must be imported **before** `faiss`
  (OpenMP initialization order) or moving the model to MPS segfaults — see the
  comment in `src/rag.py`.
