---
title: PaperPilot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📄 PaperPilot — RAG-powered Document Q&A

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

## Project layout

```
src/ingest.py   load PDF → chunk → embed → build FAISS index
src/rag.py      retrieve top-k → grounded generation (zero/few-shot)
src/eval.py     recall@k + prompt-comparison harness → results/
app/main.py     Streamlit UI
data/           sample.pdf + eval.json
index/          prebuilt FAISS index + chunk metadata
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
